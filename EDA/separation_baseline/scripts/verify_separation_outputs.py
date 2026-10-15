"""Reconcile generated frames with summaries and refresh the local report."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_separation_baseline import write_report, svg_case


def main():
    out = Path("outputs/separation_baseline")
    stats = json.loads((out/"audit_and_results.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(out/"receiver_play_metrics.csv")
    assert not metrics.duplicated(["game_id","play_id","receiver_id"]).any()
    eligible = metrics.loc[metrics.eligible_summary]
    wr = eligible.loc[eligible.player_position.eq("WR")]
    targets = wr.loc[wr.is_targeted]
    assert len(eligible) == stats["eligible_receiver_plays"]
    for name, rows in [("all_receivers",eligible),("wr_all_routes",wr),("wr_targets",targets)]:
        players = pd.read_csv(out/f"players_{name}.csv")
        assert players.receiver_plays.sum() == len(rows)
        weighted = np.average(players.mean_input_end_db_yards, weights=players.receiver_plays)
        assert abs(weighted-rows.input_end_db_yards.mean()) < 1e-9
    assert metrics.loc[~metrics.is_targeted,"target_route_label"].isna().all()
    census=0
    for path in sorted((out/"frames").glob("week_*.csv.gz")):
        frame = pd.read_csv(path, usecols=["game_id","play_id","receiver_id","frame_id","input_end_frame",
            "separation_db_yards","seconds_to_input_end"])
        census+=len(frame)
        assert np.allclose(frame.seconds_to_input_end,(frame.frame_id-frame.input_end_frame)/10)
        terminal=frame.loc[frame.frame_id.eq(frame.input_end_frame)]
        compared=terminal.merge(metrics,on=["game_id","play_id","receiver_id"],validate="one_to_one")
        assert np.allclose(compared.separation_db_yards,compared.input_end_db_yards,equal_nan=True,atol=1e-8)
    assert census == stats["receiver_frames"]
    for case in stats["cases"]:
        assert (out/case["svg"]).exists()
        distance = pd.read_csv(out/case["svg"].replace(".svg","_distances.csv"))
        assert distance.frame_id.is_monotonic_increasing
        tracking = pd.read_csv(out/case["svg"].replace(".svg","_tracking.csv"))
        receiver_id = int(distance.receiver_id.iloc[0])
        row = metrics.loc[metrics.game_id.eq(case["game_id"]) & metrics.play_id.eq(case["play_id"])
                          & metrics.receiver_id.eq(receiver_id)].iloc[0]
        (out/case["svg"]).write_text(svg_case(tracking,distance,receiver_id,row),encoding="utf-8")
    write_report(out,stats,wr,targets,pd.read_csv(out/"source_preview.csv"))
    checks={"frame_census":census,"summary_keys_unique":True,"player_counts_reconcile":True,
        "weighted_player_means_reconcile":True,"untargeted_route_labels_missing":True,
        "all_input_end_distances_reconcile":True,"time_offsets_reconcile":True,
        "case_artifacts_exist":True,"geometry_contract":"scripts/verify_separation_baseline.py"}
    (out/"output_validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    main()
