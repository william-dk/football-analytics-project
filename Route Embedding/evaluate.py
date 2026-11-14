"""Evaluate a trained route autoencoder.

For each sequence kind in the data ("full" targeted routes, "pre" pre-throw paths of all runners):
1. Knocked-out windows of fixed width (5, 10, 15 frames) at random interior positions, and windows
   centred on the measured break: model fill versus straight-line fill and versus a linear
   16-number code fitted to the visible frames (PCA fill), in yards.
2. Whole-route reconstruction from z with nothing knocked out, its smoothness, and the same
   16-number budget spent on PCA (the linear alternative to the autoencoder).
3. Route-type check: kNN on z versus kNN on raw positions at six fixed times (train -> val/test).
Then, for the whole data set:
4. Embeddings for every sequence (embeddings.csv). Pre-throw sequences also get an assigned route
   type read from z: untargeted runners have no label, so a classifier trained on the labeled
   targeted routes names their route. Targeted routes get out-of-fold predictions (folds by game).
5. An example figure (reconstructions.png).

    python evaluate.py --run-dir runs/v2
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
from route_sequences import smooth_positions

HERE = Path(__file__).resolve().parent
EMB_COLUMNS = ["row", "kind", "game_id", "play_id", "nfl_id", "player_name", "position", "team", "is_targeted",
               "route_label", "week", "split", "length"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--widths", nargs="+", type=int, default=[5, 10, 15])
    p.add_argument("--folds", type=int, default=5, help="game folds for the assigned route types")
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
    if "kind" not in data:                       # targeted-only data from the first version
        data["kind"] = np.full(len(data["lengths"]), "full")
        meta["kind"], meta["is_targeted"] = "full", True
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
        self.kind = data["kind"]

    def predict(self, rows: np.ndarray, knocked_out: np.ndarray):
        """Positions in yards (N, steps, 2) and z (N, z_dim) for the given rows, in their order."""
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(self.lengths[rows], kind="stable")       # similar lengths batch together
        idx = torch.from_numpy(rows[order])
        pred, z = run_batches(self.model, self.X[idx][..., self.in_idx], self.valid[idx],
                              torch.from_numpy(knocked_out[order]))
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        return pred[..., self.xy_out].numpy()[back] * self.pos_scale, z.numpy()[back]

    def embed(self, rows: np.ndarray) -> np.ndarray:
        return self.predict(rows, np.zeros((len(rows), self.steps), dtype=bool))[1]


def window_comparison(runner: Runner, rows: np.ndarray, starts: np.ndarray, width: int, labels,
                      pca: "RoutePCA | None" = None) -> dict:
    ok = starts >= 0
    rows, starts, labels = rows[ok], starts[ok], np.asarray(labels)[ok]
    knocked = window_mask(starts, width, runner.steps)
    pred_xy, _ = runner.predict(rows, knocked)
    truth = window_frames(runner.raw_xy[rows], starts, width)
    errors = dict(model=mean_distance(window_frames(pred_xy, starts, width), truth),
                  line=mean_distance(straight_line_fill(runner.raw_xy[rows], starts, width), truth))
    if pca is not None:
        visible = runner.valid.numpy()[rows] & ~knocked
        errors["pca"] = mean_distance(window_frames(pca.fit(runner.raw_xy[rows], visible)[1], starts, width), truth)
    by_label = (pd.DataFrame(dict(label=labels, **errors)).dropna(subset=["label"])
                .groupby("label").median().round(3).to_dict("index"))
    out = dict(routes=int(ok.sum()), share_model_better=float(np.mean(errors["model"] < errors["line"])),
               by_label_median_yd=by_label)
    for name, err in errors.items():
        out[f"{name}_median_yd"], out[f"{name}_mean_yd"] = float(np.median(err)), float(err.mean())
    if pca is not None:
        out["share_model_better_than_pca"] = float(np.mean(errors["model"] < errors["pca"]))
    return out


def route_error(pred_xy: np.ndarray, true_xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean Euclidean error per route over its real frames (yards)."""
    return (np.linalg.norm(pred_xy - true_xy, axis=-1) * valid).sum(1) / valid.sum(1)


def roughness(xy: np.ndarray, valid: np.ndarray) -> float:
    """Mean size of the second difference of the path (yards per frame squared): a jitter measure."""
    second = xy[:, 2:] - 2 * xy[:, 1:-1] + xy[:, :-2]
    return float(np.linalg.norm(second, axis=-1)[valid[:, 2:]].mean())


def step_velocity_rmse(pred_xy: np.ndarray, true_xy: np.ndarray, valid: np.ndarray, fps: float = 10.0) -> float:
    """RMS error of the frame-to-frame velocity (yards per second)."""
    diff = (np.diff(pred_xy, axis=1) - np.diff(true_xy, axis=1)) * fps
    pair = valid[:, 1:] & valid[:, :-1]
    return float(np.sqrt((diff ** 2).sum(-1)[pair].mean()))


def smooth_routes(xy: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """[1,4,6,4,1]/16 smoothing of each route's real frames (the smoother used for the inputs)."""
    out = xy.copy()
    for i, n in enumerate(lengths):
        out[i, :n] = smooth_positions(xy[i, :n])
    return out


def padded_positions(raw_xy: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """(N, 2 * steps) positions; frames after the route ends hold its last position."""
    idx = np.minimum(np.arange(raw_xy.shape[1])[None, :], lengths[:, None] - 1)
    return np.take_along_axis(raw_xy, idx[:, :, None], axis=1).reshape(len(raw_xy), -1)


def resampled_positions(raw_xy: np.ndarray, lengths: np.ndarray, points: int = 32) -> np.ndarray:
    """(N, 2 * points) positions at `points` equally spaced times between the first and last frame."""
    out = np.empty((len(raw_xy), points, 2))
    for i, n in enumerate(lengths):
        t = np.linspace(0, n - 1, points)
        for j in range(2):
            out[i, :, j] = np.interp(t, np.arange(n), raw_xy[i, :n, j])
    return out.reshape(len(raw_xy), -1)


def resampled_to_frames(flat: np.ndarray, lengths: np.ndarray, steps: int) -> np.ndarray:
    """Inverse of resampled_positions: back to one position per real frame (zeros in the padding)."""
    pts = flat.reshape(len(flat), -1, 2)
    out = np.zeros((len(flat), steps, 2))
    for i, n in enumerate(lengths):
        t = np.linspace(0, n - 1, pts.shape[1])
        for j in range(2):
            out[i, :n, j] = np.interp(np.arange(n), t, pts[i, :, j])
    return out


class RoutePCA:
    """A linear 16-number route code: principal components of routes in absolute time (frames
    after a route ends hold its last position when the components are estimated)."""

    def __init__(self, raw_xy: np.ndarray, lengths: np.ndarray, fit_rows: np.ndarray, components: int = 16):
        from sklearn.decomposition import PCA
        self.steps = raw_xy.shape[1]
        self.pca = PCA(n_components=components).fit(padded_positions(raw_xy[fit_rows], lengths[fit_rows]))
        self.mean = self.pca.mean_.reshape(self.steps, 2)
        self.basis = self.pca.components_.reshape(components, self.steps, 2)
        self.var = self.pca.explained_variance_

    def codes(self, raw_xy: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Coefficients of the padded route (the plain PCA projection)."""
        return self.pca.transform(padded_positions(raw_xy, lengths))

    def fit(self, raw_xy: np.ndarray, visible: np.ndarray, noise_yd: float = 0.05):
        """Coefficients fitted to the visible frames only (least squares with the components'
        variances as a prior, noise_yd per coordinate), and the route they rebuild.
        Returns (codes (N, K), positions (N, steps, 2))."""
        m = visible.astype(float)
        gram = np.einsum("nt,ktc,ltc->nkl", m, self.basis, self.basis)
        rhs = np.einsum("nt,ktc,ntc->nk", m, self.basis, raw_xy - self.mean)
        gram += np.diag(noise_yd ** 2 / self.var)[None]
        codes = np.linalg.solve(gram, rhs[..., None])[..., 0]
        return codes, self.mean[None] + np.einsum("nk,ktc->ntc", codes, self.basis)


def resampled_pca(raw_xy: np.ndarray, lengths: np.ndarray, fit_rows: np.ndarray, components: int = 16):
    """PCA after stretching every route to the same number of points; the route length is used to
    map back, as the decoder is told the length. Returns (codes, reconstruction in yards)."""
    from sklearn.decomposition import PCA
    flat = resampled_positions(raw_xy, lengths)
    pca = PCA(n_components=components).fit(flat[fit_rows])
    codes = pca.transform(flat)
    return codes, resampled_to_frames(pca.inverse_transform(codes), lengths, raw_xy.shape[1])


def fixed_time_positions(raw_xy: np.ndarray, lengths: np.ndarray, frames=(5, 10, 15, 20, 25, 30)) -> np.ndarray:
    """Baseline route description: x, y at fixed times; routes that already ended keep their last point."""
    cols = []
    for f in frames:
        idx = np.minimum(f, lengths - 1)
        cols.append(raw_xy[np.arange(len(raw_xy)), idx])
    return np.concatenate(cols, axis=1)


def knn_route_type(features: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    """kNN (k = 15) route-type accuracy; train and test are boolean row masks."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    has = pd.notna(labels)
    train, test = train & has, test & has
    scaler = StandardScaler().fit(features[train])
    knn = KNeighborsClassifier(n_neighbors=15).fit(scaler.transform(features[train]), labels[train])
    pred = knn.predict(scaler.transform(features[test]))
    return dict(routes=int(test.sum()), accuracy=float(accuracy_score(labels[test], pred)),
                macro_f1=float(f1_score(labels[test], pred, average="macro")))


def route_type_classifier():
    """Classifier that names a route from its z."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))


def assign_route_types(z: np.ndarray, labels: np.ndarray, games: np.ndarray, folds: int, seed: int):
    """Route type for every row from z. Labeled rows get out-of-fold predictions (whole games held
    out), so no route's own label is used to name it; unlabeled rows use a classifier trained on
    all labeled rows. Returns (assigned label, its probability, out-of-fold accuracy)."""
    from sklearn.model_selection import GroupKFold
    labeled = pd.notna(labels)
    assigned = np.full(len(z), None, dtype=object)
    prob = np.full(len(z), np.nan)
    rows = np.flatnonzero(labeled)
    order = np.random.default_rng(seed).permutation(len(rows))
    rows = rows[order]
    for tr, te in GroupKFold(n_splits=folds).split(rows, groups=games[rows]):
        clf = route_type_classifier().fit(z[rows[tr]], labels[rows[tr]])
        p = clf.predict_proba(z[rows[te]])
        assigned[rows[te]], prob[rows[te]] = clf.classes_[p.argmax(1)], p.max(1)
    accuracy = float(np.mean(assigned[labeled] == labels[labeled]))
    rest = np.flatnonzero(~labeled)
    if len(rest):
        clf = route_type_classifier().fit(z[labeled], labels[labeled])
        p = clf.predict_proba(z[rest])
        assigned[rest], prob[rest] = clf.classes_[p.argmax(1)], p.max(1)
    return assigned, prob, accuracy


def plot_examples(runner: Runner, meta: pd.DataFrame, out_path: Path, seed: int, width: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    test = meta[meta.kind.eq("full") & meta.split.eq("test") & meta.route_label.notna()]
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
    split, kind = data["split"], data["kind"]
    labels = meta.route_label.to_numpy(dtype=object)
    report: dict = {"checkpoint_epoch": ckpt["epoch"], "inputs": ckpt["inputs"], "targets": ckpt["targets"],
                    "trained_on_kinds": ckpt.get("kinds", ["full"])}

    all_rows = np.arange(len(meta))
    nothing = np.zeros((len(all_rows), runner.steps), dtype=bool)
    whole_all, z = runner.predict(all_rows, nothing)
    valid_all = runner.valid.numpy()
    positions = fixed_time_positions(runner.raw_xy, runner.lengths)

    for k in [k for k in ("full", "pre") if (kind == k).any()]:
        of_kind = kind == k
        fit_rows = np.flatnonzero(of_kind & (split == "train"))
        pca = RoutePCA(runner.raw_xy, runner.lengths, fit_rows)
        pca_codes = pca.codes(runner.raw_xy, runner.lengths)
        time_codes, time_recon = resampled_pca(runner.raw_xy, runner.lengths, fit_rows)
        report[k] = {}
        for sp in args.splits:
            rows = np.flatnonzero(of_kind & (split == sp))
            res: dict = {}
            for w in args.widths:
                starts = interior_windows(runner.lengths[rows], w, np.random.default_rng(args.seed + w))
                res[f"random_window_{w}"] = window_comparison(runner, rows, starts, w, labels[rows], pca)
            brk = meta.loc[rows]
            has = brk.has_break.to_numpy(bool) & brk.break_t_s.notna().to_numpy()
            centers = np.where(has, np.round(brk.break_t_s.fillna(0).to_numpy() * 10), 0).astype(np.int64)
            inside = has & (centers < runner.lengths[rows])
            if inside.any():
                for w in (10, 15):
                    starts = centered_windows(runner.lengths[rows], centers, w)
                    starts[~inside] = -1
                    res[f"break_window_{w}"] = window_comparison(runner, rows, starts, w, labels[rows], pca)
            truth, valid, whole = runner.raw_xy[rows], valid_all[rows], whole_all[rows]
            per_route = route_error(whole, truth, valid)
            smoothed = route_error(smooth_routes(whole, runner.lengths[rows]), truth, valid)
            res["whole_route_from_z"] = dict(
                routes=len(rows), median_yd=float(np.median(per_route)), mean_yd=float(per_route.mean()),
                p90_yd=float(np.quantile(per_route, 0.9)),
                median_yd_after_smoothing=float(np.median(smoothed)),
                roughness_model=roughness(whole, valid), roughness_actual=roughness(truth, valid),
                velocity_rmse_yd_s=step_velocity_rmse(whole, truth, valid))
            recons = {"padded": pca.pca.inverse_transform(pca_codes[rows]).reshape(len(rows), runner.steps, 2),
                      "visible_fit": pca.fit(truth, valid)[1], "resampled": time_recon[rows]}
            for name, recon in recons.items():
                err = route_error(recon, truth, valid)
                res[f"whole_route_pca16_{name}"] = dict(median_yd=float(np.median(err)), mean_yd=float(err.mean()),
                                                        p90_yd=float(np.quantile(err, 0.9)))
            report[k][sp] = res
        train, test = of_kind & (split == "train"), of_kind & np.isin(split, args.splits)
        report[k]["route_type_knn"] = {
            "z": knn_route_type(z, labels, train, test),
            "positions_at_6_fixed_times": knn_route_type(positions, labels, train, test),
            "pca16_padded": knn_route_type(pca_codes, labels, train, test),
            "pca16_resampled": knn_route_type(time_codes, labels, train, test),
        }

    emb = meta[[c for c in EMB_COLUMNS if c in meta.columns]].copy()
    if (kind == "pre").any():
        pre = np.flatnonzero(kind == "pre")
        assigned, prob, accuracy = assign_route_types(z[pre], labels[pre], meta.game_id.to_numpy()[pre],
                                                      args.folds, args.seed)
        emb["assigned_label"], emb["assigned_prob"] = None, np.nan
        emb.loc[pre, "assigned_label"], emb.loc[pre, "assigned_prob"] = assigned, prob
        counts = pd.Series(assigned).value_counts()
        report["assigned_route_types"] = {
            "classifier": "kNN (k = 15, distance-weighted) on standardized z of pre-throw sequences",
            "out_of_fold_accuracy_on_targeted": accuracy, "folds": args.folds,
            "sequences": int(len(pre)), "counts": {str(k): int(v) for k, v in counts.items()}}
    for j in range(z.shape[1]):
        emb[f"z{j}"] = z[:, j]
    emb.to_csv(args.run_dir / "embeddings.csv", index=False)

    if not args.no_plots:
        plot_examples(runner, meta, args.run_dir / "reconstructions.png", args.seed)

    with open(args.run_dir / "eval_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    for k in [k for k in ("full", "pre") if k in report]:
        for sp in args.splits:
            print(f"\n[{k} / {sp}] window fill error, median yards")
            for name, r in report[k][sp].items():
                if name.startswith(("random_window", "break_window")):
                    print(f"  {name:18s} n={r['routes']:5d}  model {r['model_median_yd']:.3f}  "
                          f"line {r['line_median_yd']:.3f}  PCA fill {r['pca_median_yd']:.3f}  "
                          f"model better than line {r['share_model_better']:.0%}, "
                          f"than PCA {r['share_model_better_than_pca']:.0%}")
            w = report[k][sp]["whole_route_from_z"]
            print(f"  whole route rebuilt from z: median {w['median_yd']:.3f} yd (p90 {w['p90_yd']:.3f}); "
                  f"PCA-16 padded {report[k][sp]['whole_route_pca16_padded']['median_yd']:.3f}, "
                  f"fitted to real frames {report[k][sp]['whole_route_pca16_visible_fit']['median_yd']:.3f}, "
                  f"time-stretched {report[k][sp]['whole_route_pca16_resampled']['median_yd']:.3f}; "
                  f"roughness model/actual {w['roughness_model']:.3f}/{w['roughness_actual']:.3f}; "
                  f"velocity RMSE {w['velocity_rmse_yd_s']:.2f} yd/s")
        knn = report[k]["route_type_knn"]
        print(f"[{k}] route-type kNN on {'+'.join(args.splits)}: " + "; ".join(
            f"{name} acc {v['accuracy']:.3f} / F1 {v['macro_f1']:.3f}" for name, v in knn.items()))
    if "assigned_route_types" in report:
        a = report["assigned_route_types"]
        print(f"\nAssigned route types for {a['sequences']} pre-throw sequences; out-of-fold accuracy on "
              f"targeted routes {a['out_of_fold_accuracy_on_targeted']:.3f}")


if __name__ == "__main__":
    main()
