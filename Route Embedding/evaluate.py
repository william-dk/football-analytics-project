"""Evaluate a trained route autoencoder.

1. Knocked-out windows of fixed width (5, 10, 15 frames) at random interior positions, and windows
   centred on the measured break: model fill versus straight-line fill, in yards.
2. Whole-route reconstruction from z with nothing knocked out.
3. Route-type check: kNN on z versus kNN on raw positions at six fixed times (train -> val/test).
4. Embeddings for every route (embeddings.csv) and an example figure (reconstructions.png).

    python evaluate.py --run-dir runs/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from masking import (centered_windows, interior_windows, mean_distance, straight_line_fill,
                     window_frames, window_mask)
from model import RouteAutoencoder, run_batches

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--widths", nargs="+", type=int, default=[5, 10, 15])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def load(run_dir: Path, data_dir: Path):
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = RouteAutoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with np.load(data_dir / "sequences.npz") as d:
        data = {k: d[k] for k in d.files}
    meta = pd.read_csv(data_dir / "routes_meta.csv", low_memory=False)
    return ckpt, model, data, meta


class Runner:
    """Holds scaled tensors and turns model output back into yards."""

    def __init__(self, ckpt, model, data):
        self.model = model
        channels = list(ckpt["channels"])
        self.in_idx = [channels.index(c) for c in ckpt["inputs"]]
        self.xy_out = [ckpt["targets"].index("x"), ckpt["targets"].index("y")]
        self.scales = np.asarray(ckpt["scales"], dtype=np.float32)
        self.pos_scale = float(self.scales[channels.index("x")])
        self.raw_xy = data["X"][..., :2].astype(float)            # yards
        self.X = torch.from_numpy(data["X"] / self.scales)
        self.lengths = data["lengths"].astype(np.int64)
        self.steps = self.X.shape[1]
        self.valid = torch.from_numpy(np.arange(self.steps)[None, :] < self.lengths[:, None])

    def predict(self, rows: np.ndarray, knocked_out: np.ndarray):
        idx = torch.from_numpy(rows)
        pred, z = run_batches(self.model, self.X[idx][..., self.in_idx], self.valid[idx],
                              torch.from_numpy(knocked_out))
        return pred[..., self.xy_out].numpy() * self.pos_scale, z.numpy()


def window_comparison(runner: Runner, rows: np.ndarray, starts: np.ndarray, width: int, labels) -> dict:
    ok = starts >= 0
    rows, starts, labels = rows[ok], starts[ok], np.asarray(labels)[ok]
    pred_xy, _ = runner.predict(rows, window_mask(starts, width, runner.steps))
    truth = window_frames(runner.raw_xy[rows], starts, width)
    model_err = mean_distance(window_frames(pred_xy, starts, width), truth)
    line_err = mean_distance(straight_line_fill(runner.raw_xy[rows], starts, width), truth)
    by_label = (pd.DataFrame(dict(label=labels, model=model_err, line=line_err)).dropna(subset=["label"])
                .groupby("label").median().round(3).to_dict("index"))
    return dict(routes=int(ok.sum()), model_median_yd=float(np.median(model_err)),
                line_median_yd=float(np.median(line_err)), model_mean_yd=float(model_err.mean()),
                line_mean_yd=float(line_err.mean()), share_model_better=float(np.mean(model_err < line_err)),
                by_label_median_yd=by_label)


def fixed_time_positions(raw_xy: np.ndarray, lengths: np.ndarray, frames=(5, 10, 15, 20, 25, 30)) -> np.ndarray:
    """Baseline route description: x, y at fixed times; routes that already ended keep their last point."""
    cols = []
    for f in frames:
        idx = np.minimum(f, lengths - 1)
        cols.append(raw_xy[np.arange(len(raw_xy)), idx])
    return np.concatenate(cols, axis=1)


def knn_route_type(features: np.ndarray, labels: np.ndarray, split: np.ndarray, eval_splits) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    has = pd.notna(labels)
    train = has & (split == "train")
    test = has & np.isin(split, eval_splits)
    scaler = StandardScaler().fit(features[train])
    knn = KNeighborsClassifier(n_neighbors=15).fit(scaler.transform(features[train]), labels[train])
    pred = knn.predict(scaler.transform(features[test]))
    return dict(routes=int(test.sum()), accuracy=float(accuracy_score(labels[test], pred)),
                macro_f1=float(f1_score(labels[test], pred, average="macro")))


def plot_examples(runner: Runner, meta: pd.DataFrame, out_path: Path, seed: int, width: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    test = meta[meta.split.eq("test") & meta.route_label.notna()]
    labels = sorted(test.route_label.unique())
    fig, axes = plt.subplots(3, 4, figsize=(14, 12))
    for ax, label in zip(axes.flat, labels):
        pool = test[test.route_label.eq(label) & test.has_break & (test.length >= width + 2)]
        pool = pool if len(pool) else test[test.route_label.eq(label) & (test.length >= width + 2)]
        r = pool.iloc[rng.integers(len(pool))]
        row = np.array([int(r.row)])
        center = int(round(r.break_t_s * 10)) if r.has_break and pd.notna(r.break_t_s) else int(r.length) // 2
        start = centered_windows(runner.lengths[row], [center], width)
        filled, _ = runner.predict(row, window_mask(start, width, runner.steps))
        whole, _ = runner.predict(row, np.zeros((1, runner.steps), dtype=bool))
        n, s0 = int(r.length), int(start[0])
        truth = runner.raw_xy[int(r.row), :n]
        line = straight_line_fill(runner.raw_xy[row], start, width)[0]
        ax.plot(truth[:, 1], truth[:, 0], color="0.6", lw=3, label="actual route")
        ax.plot(truth[s0:s0 + width, 1], truth[s0:s0 + width, 0], "k.", ms=6, label="knocked-out frames")
        ax.plot(whole[0, :n, 1], whole[0, :n, 0], color="tab:green", lw=1, label="rebuilt from z (no knockout)")
        ax.plot(filled[0, s0:s0 + width, 1], filled[0, s0:s0 + width, 0], color="tab:blue", lw=2,
                label="model fill")
        ax.plot(line[:, 1], line[:, 0], color="tab:orange", ls="--", lw=1.5, label="straight-line fill")
        ax.set_title(f"{label}  ({r.player_name})", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Test-split routes: {width}-frame window centred on the break "
                 "(x: lateral yd, + toward own sideline; y: downfield yd)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt, model, data, meta = load(args.run_dir, args.data_dir)
    runner = Runner(ckpt, model, data)
    split = data["split"]
    labels = meta.route_label.to_numpy(dtype=object)
    report: dict = {"checkpoint_epoch": ckpt["epoch"], "inputs": ckpt["inputs"], "targets": ckpt["targets"]}

    for sp in args.splits:
        rows = np.flatnonzero(split == sp)
        res: dict = {}
        for w in args.widths:
            starts = interior_windows(runner.lengths[rows], w, np.random.default_rng(args.seed + w))
            res[f"random_window_{w}"] = window_comparison(runner, rows, starts, w, labels[rows])
        brk = meta.loc[rows]
        has = brk.has_break.to_numpy(bool) & brk.break_t_s.notna().to_numpy()
        centers = np.where(has, np.round(brk.break_t_s.fillna(0).to_numpy() * 10), 0).astype(np.int64)
        inside = has & (centers < runner.lengths[rows])
        for w in (10, 15):
            starts = centered_windows(runner.lengths[rows], centers, w)
            starts[~inside] = -1
            res[f"break_window_{w}"] = window_comparison(runner, rows, starts, w, labels[rows])
        whole, _ = runner.predict(rows, np.zeros((len(rows), runner.steps), dtype=bool))
        valid = runner.valid[torch.from_numpy(rows)].numpy()
        dist = np.linalg.norm(whole - runner.raw_xy[rows], axis=-1)
        per_route = (dist * valid).sum(1) / valid.sum(1)
        res["whole_route_from_z"] = dict(routes=len(rows), median_yd=float(np.median(per_route)),
                                         mean_yd=float(per_route.mean()),
                                         p90_yd=float(np.quantile(per_route, 0.9)))
        report[sp] = res

    all_rows = np.arange(len(meta))
    _, z = runner.predict(all_rows, np.zeros((len(all_rows), runner.steps), dtype=bool))
    emb = meta[["row", "game_id", "play_id", "nfl_id", "player_name", "position", "team", "route_label",
                "week", "split", "length"]].copy()
    for j in range(z.shape[1]):
        emb[f"z{j}"] = z[:, j]
    emb.to_csv(args.run_dir / "embeddings.csv", index=False)

    baseline = fixed_time_positions(runner.raw_xy, runner.lengths)
    report["route_type_knn"] = {
        "z": knn_route_type(z, labels, split, args.splits),
        "positions_at_6_fixed_times": knn_route_type(baseline, labels, split, args.splits),
    }
    if not args.no_plots:
        plot_examples(runner, meta, args.run_dir / "reconstructions.png", args.seed)

    with open(args.run_dir / "eval_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    for sp in args.splits:
        print(f"\n[{sp}] window fill error, median yards (model vs straight line, share model better)")
        for name, r in report[sp].items():
            if name.startswith(("random_window", "break_window")):
                print(f"  {name:18s} n={r['routes']:5d}  model {r['model_median_yd']:.3f}  "
                      f"line {r['line_median_yd']:.3f}  better {r['share_model_better']:.0%}")
        w = report[sp]["whole_route_from_z"]
        print(f"  whole route rebuilt from z: median {w['median_yd']:.3f} yd, p90 {w['p90_yd']:.3f} yd")
    k = report["route_type_knn"]
    print(f"\nRoute-type kNN on {'+'.join(args.splits)}: z acc {k['z']['accuracy']:.3f} / macro-F1 "
          f"{k['z']['macro_f1']:.3f}; fixed-time positions acc {k['positions_at_6_fixed_times']['accuracy']:.3f}"
          f" / macro-F1 {k['positions_at_6_fixed_times']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
