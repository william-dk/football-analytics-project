"""End-to-end pipeline: tracking tables -> sequences -> route embedding -> receiver analysis.

    python run_pipeline.py --name v2                 # run every stage that has no output yet
    python run_pipeline.py --name v2 --from evaluate # redo evaluation and everything after it
    python run_pipeline.py --name quick --epochs 3   # extra options go to train.py

Stages
  build           build_sequences.py   data/sequences.npz, routes_meta.csv, build_report.json
  test            unit tests
  train           train.py             runs/<name>/model.pt, history.csv, summary.json
  evaluate        evaluate.py          runs/<name>/eval_report.json, embeddings.csv, reconstructions.png
  identify        identify_receivers.py  runs/<name>/identification/   (targeted routes, per route type)
  style-targeted  player_style.py      runs/<name>/style/targeted/     (targeted routes, labeled types)
  style-all       player_style.py      runs/<name>/style/all/          (all routes run, assigned types)
  summary         headline numbers     runs/<name>/pipeline_summary.json

A stage is skipped when its output already exists, unless it is at or after --from, or --force.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("build", "test", "train", "evaluate", "identify", "style-targeted", "style-all", "summary")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="v2", help="run folder under --runs-dir")
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    p.add_argument("--from", dest="start", choices=STAGES, default=None, help="redo this stage and all later ones")
    p.add_argument("--only", nargs="+", choices=STAGES, default=None, help="run just these stages")
    p.add_argument("--force", action="store_true", help="redo every selected stage")
    return p.parse_known_args()


def run(script: str, *args: str) -> None:
    command = [sys.executable, script, *args]
    print(f"\n>>> {' '.join(command)}", flush=True)
    tick = time.time()
    subprocess.run(command, cwd=HERE, check=True)
    print(f"<<< {script} finished in {(time.time() - tick) / 60:.1f} min", flush=True)


def main() -> None:
    args, train_args = parse_args()
    run_dir = args.runs_dir / args.name
    data, runs = ["--data-dir", str(args.data_dir)], ["--runs-dir", str(args.runs_dir)]
    target = ["--run-dir", str(run_dir)] + data
    outputs = {
        "build": args.data_dir / "sequences.npz",
        "train": run_dir / "summary.json",                     # written when training has finished
        "evaluate": run_dir / "eval_report.json",
        "identify": run_dir / "identification" / "identification_report.json",
        "style-targeted": run_dir / "style" / "targeted" / "identification.json",
        "style-all": run_dir / "style" / "all" / "identification.json",
    }
    commands = {
        "build": lambda: run("build_sequences.py", "--out-dir", str(args.data_dir), "--overwrite"),
        "test": lambda: run("-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"),
        "train": lambda: run("train.py", "--name", args.name, *data, *runs, "--overwrite", *train_args),
        "evaluate": lambda: run("evaluate.py", *target),
        "identify": lambda: run("identify_receivers.py", *target),
        "style-targeted": lambda: run("player_style.py", *target, "--population", "targeted"),
        "style-all": lambda: run("player_style.py", *target, "--population", "all"),
    }
    redo_from = STAGES.index(args.start) if args.start else len(STAGES)
    started = time.time()
    for i, stage in enumerate(STAGES):
        if args.only and stage not in args.only:
            continue
        if stage == "summary":
            from compare_runs import collect
            summary = collect(run_dir)
            with open(run_dir / "pipeline_summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print("\n=== summary ===")
            for key, value in summary.items():
                if value is not None:
                    print(f"{key:32s} {value:.4f}" if isinstance(value, float) else f"{key:32s} {value}")
            continue
        done = stage in outputs and outputs[stage].exists()
        if done and not args.force and i < redo_from and not (args.only and stage in args.only):
            print(f"--- {stage}: output exists ({outputs[stage].name}), skipped")
            continue
        if stage == "test" and not args.force and i < redo_from and not args.only and outputs["train"].exists():
            print("--- test: skipped (the run is already trained; use --only test to run the tests)")
            continue
        commands[stage]()
    print(f"\nPipeline finished in {(time.time() - started) / 60:.1f} min. Results are in {run_dir}")


if __name__ == "__main__":
    main()
