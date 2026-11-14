"""Collect the headline numbers of one or more runs into a table.

    python compare_runs.py                       # every run under runs/
    python compare_runs.py --runs v1 v2 abl_z8   # chosen runs, in this order

Writes runs/comparison.csv and prints a compact table. Missing stages leave blanks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def collect(run_dir: Path) -> dict:
    """Headline numbers of one run (None where a stage has not been run)."""
    row: dict = {"run": run_dir.name}
    summary = read_json(run_dir / "summary.json")
    if summary:
        args, cfg = summary["args"], summary["model_config"]
        row.update(
            kinds="+".join(summary.get("kinds", ["full"])), inputs=" ".join(args["inputs"]), z_dim=cfg["z_dim"],
            window=f"{args['win_min']}-{args['win_max']}", readout=cfg.get("readout", "cls"),
            decoder=cfg.get("decoder", "transformer") + (f"/{cfg['basis']}" if cfg.get("basis") else ""),
            vel_weight=args.get("vel_weight", 0.0), whole_prob=args.get("whole_prob", 0.0), lr=args["lr"],
            epochs=summary["epochs_run"],
            steps=summary.get("gradient_steps"), minutes=round(summary["minutes"], 1),
            parameters=summary["parameters"])
    report = read_json(run_dir / "eval_report.json")
    if report:
        full = report.get("full", report)                      # first-version reports had no kind level
        test = full.get("test", {})
        whole = test.get("whole_route_from_z", {})
        row.update(
            whole_route_yd=whole.get("median_yd"),
            whole_route_pca16_yd=test.get("whole_route_pca16_visible_fit", {}).get("median_yd"),
            roughness_ratio=(whole["roughness_model"] / whole["roughness_actual"]
                             if "roughness_model" in whole else None),
            velocity_rmse=whole.get("velocity_rmse_yd_s"))
        for name in ("random_window_10", "break_window_15"):
            w = test.get(name, {})
            row[f"{name}_model"] = w.get("model_median_yd")
            row[f"{name}_line"] = w.get("line_median_yd")
            row[f"{name}_pca"] = w.get("pca_median_yd")
        knn = full.get("route_type_knn", report.get("route_type_knn", {}))
        row["route_type_z"] = knn.get("z", {}).get("accuracy")
        row["route_type_pca16"] = knn.get("pca16_padded", {}).get("accuracy")
        row["route_type_positions"] = knn.get("positions_at_6_fixed_times", {}).get("accuracy")
        if "pre" in report:
            row["pre_whole_route_yd"] = report["pre"].get("test", {}).get("whole_route_from_z", {}).get("median_yd")
            row["assigned_type_accuracy"] = report.get("assigned_route_types", {}).get(
                "out_of_fold_accuracy_on_targeted")
    ident = read_json(run_dir / "identification" / "identification_report.json")
    if ident:
        overall = ident["overall_route_weighted"]
        for name in ("z", "positions", "pca16", "speed"):
            row[f"lda_{name}_1"] = overall.get(f"{name}_balanced_top1")
            row[f"lda_{name}_10"] = overall.get(f"{name}_multi_10")
        row["lda_chance"] = overall.get("chance_top1")
    for population in ("targeted", "all"):
        style = read_json(run_dir / "style" / population / "identification.json")
        if not style:
            continue
        tag = "tgt" if population == "targeted" else "all"
        row[f"{tag}_candidates"] = style["candidates"]
        for name, res in style["identification"].items():
            for n in ("1", "10", "all"):
                if n in res:
                    row[f"{tag}_{name}_{n}"] = res[n]["top1"]
        z = style["identification"].get("z", {})
        row[f"{tag}_z_team_10"] = z.get("10", {}).get("team")
        row[f"{tag}_chance_team"] = z.get("chance", {}).get("team")
        var = style.get("variance_route_weighted", {})
        for key in ("between_receiver_share", "game_share", "position_part", "team_part", "individual_part"):
            row[f"{tag}_{key}"] = var.get(key)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--runs", nargs="+", default=None, help="run names (default: all with a model)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    names = args.runs or sorted(p.name for p in args.runs_dir.iterdir() if (p / "model.pt").exists())
    table = pd.DataFrame([collect(args.runs_dir / name) for name in names])
    out = args.out or args.runs_dir / "comparison.csv"
    table.to_csv(out, index=False)
    show = [c for c in ("run", "kinds", "inputs", "z_dim", "window", "decoder", "steps", "whole_route_yd",
                        "whole_route_pca16_yd", "break_window_15_model", "break_window_15_pca", "route_type_z",
                        "lda_z_1", "lda_z_10", "lda_positions_10", "tgt_z_1", "tgt_z_all", "tgt_pca16_all",
                        "all_z_1", "all_z_10", "all_z_all", "all_pca16_all") if c in table.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(table[show].to_string(index=False))
    print(f"\nFull table: {out}")


if __name__ == "__main__":
    main()
