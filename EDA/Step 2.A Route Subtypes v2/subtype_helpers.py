"""Within-label route geometry clustering; outcomes are loaded only after fitting."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20261025
KEY = ["game_id", "play_id", "nfl_id"]
GEOM = ["catch_x", "catch_y", "max_depth_yd", "total_turn_deg", "retreat_yd", "break_depth_yd"]
TIMING = ["throw_time_s", "ball_flight_s", "break_t_s", "speed_before", "speed_after"]
METRICS = ["sep_at_throw", "sep_at_arrival", "sep_change_flight"]
ROOT = Path(__file__).resolve().parent


def log(message):
    print(message, flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode(raw, fitted=None):
    """Refit all imputation, scaling and PCA inside each game bootstrap."""
    if fitted is None:
        medians = np.zeros(raw.shape[1])
        for j in range(raw.shape[1]):
            finite = raw[np.isfinite(raw[:, j]), j]
            medians[j] = np.median(finite) if len(finite) else 0.0
        imputed = np.where(np.isfinite(raw), raw, medians)
        scaler = StandardScaler().fit(imputed)
        z = scaler.transform(imputed)
        pca = PCA(n_components=.95, svd_solver="full").fit(z[:, :24])
        fitted = (medians, scaler, pca)
    medians, scaler, pca = fitted
    z = scaler.transform(np.where(np.isfinite(raw), raw, medians))
    return np.column_stack([pca.transform(z[:, :24]), z[:, 24:]]), fitted


def mixture(x, k, seed, n_init=10):
    model = GaussianMixture(k, covariance_type="diag", random_state=seed,
                            n_init=n_init, max_iter=500, reg_covar=1e-4, tol=1e-3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x)
    return model, [str(w.message) for w in caught]


def arc_shape(xy):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    unique = np.r_[True, np.diff(arc) > 0]
    at = np.linspace(0, arc[-1], 12)
    return np.column_stack([np.interp(at, arc[unique], xy[unique, j])
                            for j in range(2)]).ravel(), float(arc[-1])


def stability(raw, games, labels, k, reps, rng):
    if k == 1:
        return {"mean_ari": 1., "ari_sd": 0., "aris": [1.] * reps,
                "resamples": reps, "nonconverged_refits": 0, "note": "k=1 has trivially perfect ARI; no refit needed"}
    unique = np.unique(games)
    by_game = {g: np.flatnonzero(games == g) for g in unique}
    scores, nonconverged = [], 0
    for b in range(reps):
        drawn = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_game[g] for g in drawn])
        overlap = np.unique(indices)
        xb, prep = encode(raw[indices])
        model, _ = mixture(xb, k, int(rng.integers(2**31 - 1)), n_init=10)
        nonconverged += int(not model.converged_)
        pred = model.predict(encode(raw[overlap], prep)[0])
        scores.append(float(adjusted_rand_score(labels[overlap], pred)))
        if (b + 1) % 10 == 0:
            log(f"  stability {b + 1}/{reps}: mean ARI {np.mean(scores):.3f}")
    return {"mean_ari": float(np.mean(scores)), "ari_sd": float(np.std(scores)),
            "aris": scores, "resamples": reps, "nonconverged_refits": nonconverged}


def name_clusters(frame, assignments):
    order = []
    for c in np.unique(assignments):
        g = frame.iloc[np.flatnonzero(assignments == c)]
        # Use break depth only when detected for at least half the subtype.
        depth = g.break_depth_yd.median() if g.break_depth_yd.notna().mean() >= .5 else g.catch_x.median()
        order.append((float(depth), float(g.catch_y.median()), int(c)))
    order.sort()
    mapping, names, used = {}, {}, {}
    label = frame.route_label.iloc[0]
    for rank, (depth, lateral, c) in enumerate(order, 1):
        band = ("shallow" if label == "CROSS" else "short") if depth < 10 else ("intermediate" if depth < 20 else "deep")
        turn = frame.iloc[np.flatnonzero(assignments == c)].total_turn_deg.median()
        turn_name = "outward" if turn > 20 else ("inward" if turn < -20 else "straight")
        base = f"{label}_{band}_{turn_name}"
        used[base] = used.get(base, 0) + 1
        mapping[c] = rank
        names[c] = base if used[base] == 1 else f"{base}_{used[base]}"
    return mapping, names


def separation_bootstrap(frame, reps, rng):
    clusters = sorted(frame.cluster_id.unique())
    pairs = list(itertools.combinations(clusters, 2))
    games = frame.game_id.to_numpy()
    unique = np.unique(games)
    by_game = {g: np.flatnonzero(games == g) for g in unique}
    cid = frame.cluster_id.to_numpy()
    vals = frame[METRICS[:2]].to_numpy(float)
    samples = {(a, b, m): [] for a, b in pairs for m in range(2)}
    def med(a):
        a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else np.nan
    for _ in range(reps):
        indices = np.concatenate([by_game[g] for g in rng.choice(unique, len(unique), replace=True)])
        for a, b in pairs:
            for m in range(2):
                samples[a, b, m].append(med(vals[indices[cid[indices] == b], m]) -
                                        med(vals[indices[cid[indices] == a], m]))
    result = []
    for a, b in pairs:
        for m, metric in enumerate(METRICS[:2]):
            observed = med(vals[cid == b, m]) - med(vals[cid == a, m])
            distribution = np.array(samples[a, b, m])
            finite = distribution[np.isfinite(distribution)]
            lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            result.append({"label": frame.route_label.iloc[0], "subtype_a": frame.loc[frame.cluster_id.eq(a), "provisional_subtype"].iloc[0],
                           "subtype_b": frame.loc[frame.cluster_id.eq(b), "provisional_subtype"].iloc[0],
                           "metric": metric, "difference_b_minus_a_yd": observed, "ci_low_yd": lo,
                           "ci_high_yd": hi, "valid_bootstraps": len(finite), "bootstrap_resamples": reps,
                           "meets_separation_rule": bool(len(finite) == reps and (lo > 0 or hi < 0) and abs(observed) >= .5)})
    return result


def pointwise_median(paths):
    end = max(p[-1, 0] for p in paths)
    grid = np.arange(0, end + .001, .1)
    values = np.full((len(paths), len(grid), 2), np.nan)
    for i, p in enumerate(paths):
        use = (grid >= p[0, 0] - 1e-8) & (grid <= p[-1, 0] + 1e-8)
        for j in range(2):
            values[i, use, j] = np.interp(grid[use], p[:, 0], p[:, j + 1])
    keep = np.sum(np.isfinite(values[:, :, 0]), axis=0) >= math.ceil(len(paths) / 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values[:, keep], axis=0)
    return grid[keep], median


def draw_paths(ax, frame, paths, color, rng, limit=200):
    chosen = rng.choice(len(frame), min(limit, len(frame)), replace=False)
    all_paths = [paths[k] for k in frame[KEY].itertuples(index=False, name=None)]
    for i in chosen:
        p = all_paths[i]
        ax.plot(p[:, 1], p[:, 2], color=color, alpha=.08, linewidth=.6)
        throw = frame.iloc[i].throw_time_s
        if np.isfinite(throw) and p[0, 0] <= throw <= p[-1, 0]:
            ax.scatter(np.interp(throw, p[:, 0], p[:, 1]), np.interp(throw, p[:, 0], p[:, 2]),
                       color=color, s=5, alpha=.25, marker="x")
    grid, med = pointwise_median(all_paths)
    ax.plot(med[:, 0], med[:, 1], color=color, linewidth=2.5,
            label=f"{frame.provisional_subtype.iloc[0]} (n={len(frame):,})")
    return len(grid)


def style_axes(ax):
    ax.set_xlabel("Downfield from receiver snap position (yd)")
    ax.set_ylabel("Lateral toward receiver's own sideline (yd)")
    ax.axhline(0, color="gray", linewidth=.5)
    ax.axvline(0, color="gray", linewidth=.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.15)


def cards(routes, paths, output, rng):
    dest = output / "cluster_cards"
    dest.mkdir(exist_ok=True)
    colors = ["#176b8f", "#c66b29", "#63853e", "#8a5c91", "#6a6a6a"]
    for label, whole in routes.groupby("route_label", sort=True):
        overview, ax = plt.subplots(figsize=(10, 7))
        for i, (subtype, g) in enumerate(whole.groupby("provisional_subtype", sort=True)):
            color = colors[i % len(colors)]
            fig, one = plt.subplots(figsize=(10, 7))
            draw_paths(one, g, paths, color, rng)
            style_axes(one)
            depth = g.break_depth_yd.median() if g.break_depth_yd.notna().any() else np.nan
            one.set_title(f"{subtype} | n={len(g):,} | median break={depth:.1f} yd\n"
                          f"median catch=({g.catch_x.median():.1f}, {g.catch_y.median():.1f}) yd; "
                          f"throw={g.throw_time_s.median():.1f}s; arrival sep={g.sep_at_arrival.median():.2f} yd")
            one.legend(loc="best")
            fig.text(.5, .015, "Thin: up to 200 full paths; x: each sampled throw; thick: median at actual snap time (0.1s grid, >=50% support).",
                     ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, 1))
            fig.savefig(dest / f"{subtype}.png", dpi=140)
            plt.close(fig)
            draw_paths(ax, g, paths, color, rng, limit=70)
        style_axes(ax)
        ax.set_title(f"{label}: provisional route geometry subtypes | n={len(whole):,}")
        ax.legend(loc="best")
        overview.text(.5, .015, "Source: 2023 targeted routes, full input + output paths; medians use actual seconds since snap; x marks throws.",
                      ha="center", fontsize=8)
        overview.tight_layout(rect=(0, .04, 1, 1))
        overview.savefig(dest / f"{label}_overview.png", dpi=140)
        plt.close(overview)
        log(f"Cards saved: {label}")



import sys
sys.path.insert(0, str(ROOT.parent / "Step 2.0 Route Dataset"))
from build_break_features import cohort, clean, resolve_recorded_path
