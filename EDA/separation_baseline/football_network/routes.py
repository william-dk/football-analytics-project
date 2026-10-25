"""Step 2.0 route layer: normalized route frames plus a one-row-per-route table.

Coordinate convention (verified on the five-game sample): x/y are already
offense-right on every play, while absolute_yardline_number stays in the
original system on left-moving plays and dir/o are never flipped. Headings are
therefore derived from x/y, never from dir.

Route frame (x_norm, y_norm): origin at the receiver's frame-1 (snap proxy)
position, +x downfield, +y toward the receiver's own sideline (away from the
QB's frame-1 y, the ball proxy). Translation and mirroring preserve distances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAY_KEY = ["game_id", "play_id"]
ROUTE_KEY = PLAY_KEY + ["nfl_id"]
ROUTE_ROLES = ("Targeted Receiver", "Other Route Runner")
FPS = 10
MOTION_SPEED = 1.0      # yd/s at frame 1; above this the receiver is in motion
HEADING_MIN_SPEED = 1.0  # yd/s; headings below this are dominated by position noise
SUPP_COLUMNS = [
    "season", "week", "down", "yards_to_go", "quarter", "offense_formation",
    "receiver_alignment", "route_of_targeted_receiver", "play_action", "dropback_type",
    "pass_result", "pass_length", "team_coverage_man_zone", "team_coverage_type",
    "defenders_in_the_box", "play_nullified_by_penalty", "yards_gained",
    "expected_points_added",
]


def split_phases(tracking: pd.DataFrame) -> pd.DataFrame:
    """Add input_end_frame and is_input; output frames follow the throw."""
    tracking = tracking.copy()
    last = tracking.groupby(PLAY_KEY).frame_id.transform("max")
    tracking["input_end_frame"] = last - tracking.num_frames_output
    tracking["is_input"] = tracking.frame_id.le(tracking.input_end_frame)
    return tracking


def route_frames(tracking: pd.DataFrame) -> pd.DataFrame:
    """Return normalized pre-throw frames for every route runner."""
    tracking = split_phases(tracking)
    if tracking.duplicated(ROUTE_KEY + ["frame_id"]).any():
        raise ValueError("Duplicate game/play/player/frame keys")
    inp = tracking.loc[tracking.is_input]
    qb = (inp.loc[inp.player_role.eq("Passer") & inp.frame_id.eq(1)]
          .drop_duplicates(PLAY_KEY).set_index(PLAY_KEY).y.rename("qb_y0"))
    rr = inp.loc[inp.player_role.isin(ROUTE_ROLES)].sort_values(ROUTE_KEY + ["frame_id"]).copy()
    rr = rr.join(qb, on=PLAY_KEY)
    g = rr.groupby(ROUTE_KEY, sort=False)
    rr["x0"], rr["y0"] = g.x.transform("first"), g.y.transform("first")
    # Outside = away from the ball proxy; plays without a passer keep the raw side.
    rr["side"] = np.sign(rr.y0 - rr.qb_y0).replace(0, 1).fillna(1)
    rr["x_norm"] = rr.x - rr.x0
    rr["y_norm"] = (rr.y - rr.y0) * rr.side
    rr["t"] = (rr.frame_id - g.frame_id.transform("first")) / FPS
    rr["vx"] = g.x_norm.diff() * FPS
    rr["vy"] = g.y_norm.diff() * FPS
    heading = np.arctan2(rr.vy, rr.vx)
    heading = heading.where(rr.s.ge(HEADING_MIN_SPEED))
    rr["heading_sin"], rr["heading_cos"] = np.sin(heading), np.cos(heading)
    turn = heading.groupby([rr[k] for k in ROUTE_KEY], sort=False).diff()
    rr["turn_rate"] = np.degrees(np.arctan2(np.sin(turn), np.cos(turn))) * FPS
    return rr[ROUTE_KEY + ["frame_id", "t", "player_name", "player_position", "player_role",
                           "x_norm", "y_norm", "vx", "vy", "s", "a",
                           "heading_sin", "heading_cos", "turn_rate",
                           "x", "y", "side", "qb_y0", "play_direction",
                           "absolute_yardline_number", "input_end_frame"]]


def route_table(frames: pd.DataFrame, supplementary: pd.DataFrame) -> pd.DataFrame:
    """One row per route, joined to play context; labels kept only for targets."""
    g = frames.groupby(ROUTE_KEY, sort=False)
    first = g.first()
    routes = pd.DataFrame({
        "player_name": first.player_name,
        "player_position": first.player_position,
        "is_targeted": first.player_role.eq("Targeted Receiver"),
        "n_frames": g.size(),
        "throw_time_s": g.t.max(),
        "in_motion_at_snap": first.s.gt(MOTION_SPEED),
        "lateral_from_ball_yd": (first.y - first.qb_y0).abs(),
        "sideline_dist_yd": np.minimum(first.y, 53.3 - first.y),
        "end_x_norm": g.x_norm.last(),
        "end_y_norm": g.y_norm.last(),
        "max_speed": g.s.max(),
        "max_abs_turn_rate": g.turn_rate.apply(lambda v: v.abs().max()),
    }).reset_index()
    supp = supplementary.loc[supplementary.season.eq(2023), PLAY_KEY + SUPP_COLUMNS]
    if supp.duplicated(PLAY_KEY).any():
        raise ValueError("Duplicate game/play keys in supplementary data")
    routes = routes.merge(supp, on=PLAY_KEY, how="left", validate="many_to_one", indicator=True)
    if not routes._merge.eq("both").all():
        raise ValueError("Route plays missing from supplementary data")
    routes = routes.drop(columns="_merge")
    # The source route label describes only the targeted receiver.
    routes["route_label"] = routes.route_of_targeted_receiver.where(routes.is_targeted)
    return routes.drop(columns="route_of_targeted_receiver")
