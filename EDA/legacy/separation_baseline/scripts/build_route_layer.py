"""Build the Step 2.0 route layer and run its validation checks.

Defaults use the repository's five-game sample; pass --tracking for the full
file. Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_network.routes import route_frames, route_table

parser = argparse.ArgumentParser()
parser.add_argument("--tracking", type=Path, default=Path("EDA/tracking_5_games.csv"))
parser.add_argument("--supplementary", type=Path,
                    default=Path("EDA/Supplementary Data Analysis/supplementary_data.csv"))
parser.add_argument("--out", type=Path, default=Path("EDA/separation_baseline/outputs/route_layer_sample"))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

tracking = pd.read_csv(args.tracking)
supp = pd.read_csv(args.supplementary, low_memory=False)
frames = route_frames(tracking)
routes = route_table(frames, supp)
frames.to_csv(args.out / "route_frames.csv.gz", index=False, float_format="%.4f")
routes.to_csv(args.out / "routes.csv", index=False)

# Validation: frame-1 origin, downfield first second, label-consistent mirroring,
# and QB depth behind the line of scrimmage under the verified convention.
early = frames[frames.t.le(1.0)].groupby(["game_id", "play_id", "nfl_id"]).x_norm.last()
lab = routes.dropna(subset=["route_label"]).set_index(["game_id", "play_id", "nfl_id"])
lab = lab.join(early.rename("x_1s"))
qb = tracking[tracking.player_role.eq("Passer") & tracking.frame_id.eq(1)]
los = np.where(qb.play_direction.eq("left"), 120 - qb.absolute_yardline_number,
               qb.absolute_yardline_number)
# Catch point (last output frame) in each target's route frame, for validation only.
start = frames[frames.t.eq(0)].set_index(["game_id", "play_id", "nfl_id"])[["x", "y", "side"]]
catch = (tracking.sort_values("frame_id").groupby(["game_id", "play_id", "nfl_id"])[["x", "y"]].last()
         .join(start, rsuffix="0", how="inner"))
lab["catch_y_norm"] = ((catch.y - catch.y0) * catch.side).reindex(lab.index)
toward_sideline = lambda r, col: float((lab.loc[lab.route_label.eq(r), col] > 0).mean())
LATERAL = ["OUT", "CORNER", "FLAT", "IN", "SLANT", "POST"]
checks = {
    "routes": len(routes), "targeted_routes": int(routes.is_targeted.sum()),
    "plays": int(routes[["game_id", "play_id"]].drop_duplicates().shape[0]),
    "origin_max_abs_yd": float(frames[frames.t.eq(0)][["x_norm", "y_norm"]].abs().max().max()),
    "share_targets_downfield_after_1s_excl_screen_flat": float(
        lab.loc[~lab.route_label.isin(["SCREEN", "FLAT"]), "x_1s"].gt(0).mean()),
    # At the throw many CORNERs are still in their inside-leaning stem (truncation).
    "share_toward_sideline_at_throw": {r: toward_sideline(r, "end_y_norm") for r in LATERAL},
    "share_toward_sideline_at_catch": {r: toward_sideline(r, "catch_y_norm") for r in LATERAL},
    "qb_depth_behind_los_median_yd_by_direction": (
        pd.Series(los - qb.x.to_numpy(), index=qb.play_direction).groupby(level=0).median().round(2).to_dict()),
    "share_in_motion_at_snap": float(routes.in_motion_at_snap.mean()),
}
(args.out / "checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
