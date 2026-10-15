"""Frame-level nearest-DB baseline. Distances use observed x/y in yards."""
from __future__ import annotations

import numpy as np
import pandas as pd

DB_POSITIONS = frozenset({"CB", "DB", "S", "FS", "SS"})
PLAY_KEY = ["game_id", "play_id"]
FRAME_KEY = PLAY_KEY + ["frame_id"]
ROW_KEY = FRAME_KEY + ["nfl_id"]


def nearest_defenders(tracking: pd.DataFrame) -> pd.DataFrame:
    """Return one row per observed route runner/frame, with two nearest groups.

    The comparison group is supplied defensive-coverage players, not all 11
    defenders. DB distances use only CB/DB/S/FS/SS within that group. Ties
    resolve to the smaller NFL ID. Missing comparison groups stay missing.
    Raises on duplicate keys or nonfinite coordinates; never silently imputes.
    """
    required = ROW_KEY + ["player_name", "player_position", "player_role",
                           "player_side", "x", "y"]
    missing = set(required) - set(tracking.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if tracking.duplicated(ROW_KEY).any():
        raise ValueError("Duplicate game/play/player/frame keys")
    if not np.isfinite(tracking[["x", "y"]].to_numpy()).all():
        raise ValueError("Nonfinite coordinates")
    receivers = tracking.loc[
        tracking.player_side.eq("Offense")
        & tracking.player_role.isin(["Targeted Receiver", "Other Route Runner"]),
        required,
    ].copy()
    receivers["is_targeted"] = receivers.player_role.eq("Targeted Receiver")
    defenders = tracking.loc[
        tracking.player_side.eq("Defense")
        & tracking.player_role.eq("Defensive Coverage"),
        FRAME_KEY + ["nfl_id", "player_position", "x", "y"],
    ].rename(columns={"nfl_id": "defender_id", "player_position": "defender_position",
                      "x": "defender_x", "y": "defender_y"})
    receivers = receivers.reset_index(drop=True)
    receivers["_row"] = np.arange(len(receivers))
    pairs = receivers.merge(defenders, on=FRAME_KEY, how="inner", validate="many_to_many")
    pairs["distance"] = np.hypot(pairs.x - pairs.defender_x, pairs.y - pairs.defender_y)
    pairs = pairs.sort_values(["distance", "defender_id"], kind="stable")
    result = receivers.set_index("_row")
    for name, candidates in [
        ("coverage", pairs),
        ("db", pairs.loc[pairs.defender_position.isin(DB_POSITIONS)]),
    ]:
        closest = candidates.drop_duplicates("_row").set_index("_row")
        result[f"nearest_{name}_id"] = closest.defender_id.reindex(result.index).astype("Int64")
        result[f"nearest_{name}_position"] = closest.defender_position.reindex(result.index)
        result[f"separation_{name}_yards"] = closest.distance.reindex(result.index)
        counts = candidates.groupby("_row").size()
        result[f"n_{name}_observed"] = counts.reindex(result.index, fill_value=0)
    return result.reset_index(drop=True).rename(columns={"nfl_id": "receiver_id"})
