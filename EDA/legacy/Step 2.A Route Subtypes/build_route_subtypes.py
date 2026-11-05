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
GEOM = ["catch_x", "catch_y", "max_depth", "break_depth_yd", "break_angle_deg", "no_break"]
TIMING = ["throw_time_s", "ball_flight_s", "break_time_s", "speed_before_break", "speed_after_break"]
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


def mixture(x, k, seed, n_init=5):
    model = GaussianMixture(k, covariance_type="full", random_state=seed,
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


def load_geometry(input_dir):
    # Explicit allow-list excludes outcomes and separation even from the route table.
    cols = KEY + ["is_targeted", "route_label", "break_detected", "break_depth_yd",
                  "break_angle_deg", "player_position", "team_coverage_man_zone"] + TIMING
    all_routes = pd.read_csv(input_dir / "routes.csv", usecols=cols, low_memory=False)
    routes = all_routes.loc[all_routes.is_targeted.eq(True) & all_routes.route_label.notna()].copy()
    routes = routes.sort_values(KEY).reset_index(drop=True)
    if routes.duplicated(KEY).any():
        raise ValueError("Duplicate targeted route keys")
    frames = pd.read_csv(input_dir / "target_full_paths.csv", low_memory=False,
                         usecols=KEY + ["frame_id", "t", "phase", "x_norm", "y_norm"])
    frames = frames.merge(routes[KEY], on=KEY, how="inner", validate="many_to_one")
    if frames.duplicated(KEY + ["frame_id"]).any():
        raise ValueError("Duplicate route/frame keys")
    if not np.isfinite(frames[["t", "x_norm", "y_norm"]].to_numpy()).all():
        raise ValueError("Missing/nonfinite path coordinates or times")
    paths = {}
    for key, group in frames.groupby(KEY, sort=False):
        group = group.sort_values(["t", "frame_id"])
        if (np.diff(group.t.to_numpy()) <= 0).any():
            raise ValueError(f"Non-increasing path time: {key}")
        paths[key] = group[["t", "x_norm", "y_norm"]].to_numpy()
    shape, length, catch_x, catch_y, max_depth = [], [], [], [], []
    for key in routes[KEY].itertuples(index=False, name=None):
        if key not in paths:
            raise ValueError(f"Missing full path: {key}")
        p = paths[key]
        sh, le = arc_shape(p[:, 1:])
        shape.append(sh)
        length.append(le)
        catch_x.append(p[-1, 1])
        catch_y.append(p[-1, 2])
        max_depth.append(p[:, 1].max())
    routes["path_length_yd"] = length
    routes["catch_x"], routes["catch_y"], routes["max_depth"] = catch_x, catch_y, max_depth
    routes["no_break"] = (~routes.break_detected.eq(True)).astype(int)
    routes.loc[routes.no_break.eq(1), ["break_depth_yd", "break_angle_deg"] + TIMING[2:]] = np.nan
    routes["too_short"] = routes.path_length_yd < 1.0
    raw = np.column_stack([np.array(shape), routes[GEOM].to_numpy(float)])
    return routes, paths, raw, {"routes_table_rows": len(all_routes),
                               "all_targeted_routes": int(all_routes.is_targeted.eq(True).sum()),
                               "excluded_unlabeled_targeted_routes": int((all_routes.is_targeted.eq(True) & all_routes.route_label.isna()).sum()),
                               "full_path_frames_used": len(frames),
                               "full_path_phases": frames.phase.value_counts().to_dict()}


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
        model, _ = mixture(xb, k, int(rng.integers(2**31 - 1)), n_init=3)
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
        base = f"{label}_{band}"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT.parent / "Step 2.0 Route Dataset" / "data")
    parser.add_argument("--supplementary", type=Path, default=ROOT.parent / "Supplementary Data Analysis" / "supplementary_data.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [args.input_dir / f for f in ["routes.csv", "target_full_paths.csv", "target_separation_summary.csv"]] + [args.supplementary]
    hashes = {str(p.resolve()): digest(p) for p in sources}
    log("Loading geometry; separation outcomes will be loaded after all clustering")
    routes, paths, raw, counts = load_geometry(args.input_dir)
    routes["cluster_id"] = 0
    routes["provisional_subtype"] = routes.route_label
    routes["gmm_probability"] = np.nan
    routes["gmm_log_likelihood"] = np.nan
    routes["outlier"] = False
    diagnostics, bic_rows = {}, []
    for li, (label, whole) in enumerate(routes.groupby("route_label", sort=True)):
        index = whole.index[~whole.too_short].to_numpy()
        fit = routes.loc[index]
        rng = np.random.default_rng(SEED + li * 1009)
        log(f"Fitting {label}: {len(index)} eligible / {len(whole)} routes")
        if not len(index):
            diagnostics[label] = {"chosen_k": 1, "reason": "all routes too short", "mean_ari": 1., "silhouette": None, "bic": {}}
            continue
        x, prep = encode(raw[index])
        models, bics, fitting_warnings = {}, {}, {}
        for k in ([1] if len(whole) < 200 or len(index) < 200 else range(1, 5)):
            model, caught = mixture(x, k, SEED + li * 1009 + k)
            models[k], bics[k], fitting_warnings[k] = model, float(model.bic(x)), caught
            predicted = model.predict(x)
            sizes = np.bincount(predicted, minlength=k)
            bic_rows.append({"label": label, "k": k, "bic": bics[k], "converged": bool(model.converged_),
                             "iterations": int(model.n_iter_), "cluster_sizes": json.dumps(sizes.tolist()),
                             "size_rule_pass": bool(k == 1 or (sizes.min() >= 100 and sizes.min() >= .1 * len(whole)))})
            log(f"  k={k}: BIC={bics[k]:.1f}, sizes={sizes.tolist()}, converged={model.converged_}")
        best = min(bics, key=bics.get)
        selected = best
        while selected > 1:
            sizes = np.bincount(models[selected].predict(x), minlength=selected)
            if sizes.min() >= 100 and sizes.min() >= .1 * len(whole):
                break
            selected -= 1
        model = models[selected]
        assignments = model.predict(x)
        mapping, names = name_clusters(fit, assignments)
        routes.loc[index, "cluster_id"] = [mapping[int(c)] for c in assignments]
        routes.loc[index, "provisional_subtype"] = [names[int(c)] for c in assignments] if selected > 1 else label
        routes.loc[index, "gmm_probability"] = model.predict_proba(x).max(axis=1)
        ll = model.score_samples(x)
        routes.loc[index, "gmm_log_likelihood"] = ll
        # Exact ceil(1%) per eligible label, deterministic route-key tie breaking.
        outliers = index[np.argsort(ll, kind="stable")[:math.ceil(.01 * len(index))]]
        routes.loc[outliers, "outlier"] = True
        sil = None
        if selected > 1:
            km = KMeans(selected, n_init=20, random_state=SEED).fit(x)
            sil = float(silhouette_score(x, km.labels_, sample_size=min(2000, len(x)), random_state=SEED))
        stable = stability(raw[index], fit.game_id.to_numpy(), assignments, selected, 50, rng)
        diagnostics[label] = {"chosen_k": selected, "bic_winner_k": best, "bic": bics,
                              "silhouette": sil, "shape_pca_components": int(prep[2].n_components_),
                              "shape_pca_variance": float(prep[2].explained_variance_ratio_.sum()),
                              "fitted_routes": len(index), "label_routes": len(whole),
                              "detected_break_share": float(fit.break_detected.mean()),
                              "all_missing_break_features": [c for c in ["break_depth_yd", "break_angle_deg"] if fit[c].isna().all()],
                              "fitting_warnings": fitting_warnings, "selected_model_converged": bool(model.converged_),
                              "size_reduction": bool(selected != best), **stable}
        log(f"Selected {label} k={selected}; stability ARI={stable['mean_ari']:.3f}")
        # Persist fitting diagnostics to expose progress during long runs.
        (output / "clustering_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    log("All clustering complete. Now loading separation for comparisons.")
    sep = pd.read_csv(args.input_dir / "target_separation_summary.csv", low_memory=False,
                      usecols=["game_id", "play_id", "receiver_nfl_id"] + METRICS)
    sep = sep.rename(columns={"receiver_nfl_id": "nfl_id"})
    counts["separation_summary_rows"] = len(sep)
    counts["missing_arrival_separation_all_targets"] = int(sep.sep_at_arrival.isna().sum())
    routes = routes.merge(sep, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not routes._merge.eq("both").all():
        raise ValueError("Separation summary missing route keys")
    routes = routes.drop(columns="_merge")
    comparisons, decisions = [], {}
    for li, (label, whole) in enumerate(routes.groupby("route_label", sort=True)):
        info = diagnostics[label]
        eligible = whole.loc[~whole.too_short]
        diffs = separation_bootstrap(eligible, 1000, np.random.default_rng(SEED + 100000 + li)) if info["chosen_k"] > 1 else []
        comparisons.extend(diffs)
        if len(whole) < 200 or len(eligible) < 200:
            decision, reason = "keep label", "insufficient sample (<200 eligible or total routes)"
        elif info["chosen_k"] == 1:
            decision, reason = "keep label", ("BIC split failed minimum subtype size; reduced to k=1" if info.get("size_reduction") else "BIC favors one cluster")
        elif info["mean_ari"] < .6:
            decision, reason = "keep label", "unstable: mean game-bootstrap ARI <0.6"
        elif any(d["meets_separation_rule"] for d in diffs):
            decision, reason = "split", "stable clustering and >=0.5 yd median separation difference with bootstrap 95% CI excluding zero"
        else:
            decision, reason = "keep label", "no subtype pair meets both separation difference and interval thresholds"
        decisions[label] = {"decision": decision, "reason": reason}
        log(f"Decision {label}: {decision} ({reason})")
    routes["route_subtype"] = [r.provisional_subtype if decisions[r.route_label]["decision"] == "split" and not r.too_short else r.route_label
                               for r in routes.itertuples()]
    routes["flags"] = [";".join(f for f, yes in [("too_short", r.too_short), ("outlier", r.outlier)] if yes)
                       for r in routes.itertuples()]
    outcols = KEY + ["route_label", "cluster_id", "provisional_subtype", "gmm_probability", "route_subtype", "flags"]
    routes[outcols].to_csv(output / "route_subtypes.csv", index=False)
    rows = []
    for (label, subtype), group in routes.groupby(["route_label", "provisional_subtype"], sort=True):
        info = diagnostics[label]
        row = {"label": label, "subtype": subtype, "n": len(group), "share_of_label": len(group) / sum(routes.route_label.eq(label)),
               "chosen_k": info["chosen_k"], "bic_table": json.dumps(info["bic"]), "silhouette": info["silhouette"],
               "stability_ari": info["mean_ari"], "unstable": info["mean_ari"] < .6,
               "too_short_n": int(group.too_short.sum()), "outlier_n": int(group.outlier.sum()),
               "break_detected_n": int(group.break_detected.sum()), **decisions[label]}
        for col in GEOM[:-1] + TIMING:
            row[f"median_{col}"] = group[col].median() if group[col].notna().any() else np.nan
            row[f"n_{col}"] = int(group[col].notna().sum())
        for pos in ["WR", "TE", "RB"]:
            row[f"share_{pos}"] = float(group.player_position.eq(pos).mean())
        row["share_other_position"] = float((~group.player_position.isin(["WR", "TE", "RB"])).mean())
        for coverage in ["MAN_COVERAGE", "ZONE_COVERAGE"]:
            row[f"share_{coverage}"] = float(group.team_coverage_man_zone.eq(coverage).mean())
        row["share_missing_coverage"] = float(group.team_coverage_man_zone.isna().mean())
        for metric in METRICS:
            values = group[metric].dropna()
            q = values.quantile([.25, .5, .75])
            row.update({f"{metric}_n": len(values), f"{metric}_median": q.loc[.5], f"{metric}_q25": q.loc[.25],
                        f"{metric}_q75": q.loc[.75], f"{metric}_iqr": q.loc[.75] - q.loc[.25]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "subtype_summary.csv", index=False)
    pd.DataFrame(bic_rows).to_csv(output / "bic_diagnostics.csv", index=False)
    compare_cols = ["label", "subtype_a", "subtype_b", "metric", "difference_b_minus_a_yd", "ci_low_yd", "ci_high_yd", "valid_bootstraps", "bootstrap_resamples", "meets_separation_rule"]
    pd.DataFrame(comparisons, columns=compare_cols).to_csv(output / "separation_comparisons.csv", index=False)
    supp = pd.read_csv(args.supplementary, usecols=["game_id", "play_id", "play_description"], low_memory=False)
    supp = supp.drop_duplicates()
    if supp.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Conflicting supplementary descriptions for a play")
    review = routes.loc[routes.outlier, KEY + ["route_label", "provisional_subtype", "gmm_log_likelihood"]].rename(
        columns={"route_label": "label", "provisional_subtype": "nearest_subtype", "gmm_log_likelihood": "log_likelihood"})
    review = review.merge(supp, on=["game_id", "play_id"], how="left", validate="many_to_one")
    review.sort_values(["label", "log_likelihood"]).to_csv(output / "outlier_review.csv", index=False)
    descriptor_cols = KEY + ["route_label", "path_length_yd"] + GEOM + TIMING
    routes[descriptor_cols].to_csv(output / "route_descriptors.csv", index=False)
    cards(routes, paths, output, np.random.default_rng(SEED + 999999))
    unchanged = all(digest(p) == hashes[str(p.resolve())] for p in sources)
    counts.update({"expected_labeled_targeted_routes": 14105, "labeled_targeted_routes": len(routes),
                   "expected_count_matches": len(routes) == 14105, "routes_per_label": routes.route_label.value_counts().to_dict(),
                   "too_short_total": int(routes.too_short.sum()), "too_short_per_label": routes.groupby("route_label").too_short.sum().to_dict(),
                   "every_route_assigned_exactly_once": bool(not routes.duplicated(KEY).any() and routes.route_subtype.notna().all()),
                   "output_rows": len(routes), "chosen_k_per_label": {l: d["chosen_k"] for l, d in diagnostics.items()},
                   "outlier_count": len(review), "missing_arrival_separation": int(routes.sep_at_arrival.isna().sum()),
                   "outlier_descriptions_missing": int(review.play_description.isna().sum()),
                   "input_sha256": hashes, "inputs_unchanged": unchanged, "seed": SEED,
                   "stability_resamples": 50, "separation_bootstrap_resamples": 1000,
                   "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                   "decisions": decisions})
    (output / "validation.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    split = [l for l in decisions if decisions[l]["decision"] == "split"]
    lines = ["# Step 2.A — within-label route subtypes", "", f"2023 targeted routes: **{len(routes):,}**, across {routes.game_id.nunique()} games. Seed {SEED}. "
             f"Full input + output geometry only was used for fitting; separation was joined afterward.", "", "## Step 3 decisions", ""]
    for label in sorted(split):
        subnames = sorted(routes.loc[routes.route_label.eq(label) & ~routes.too_short, "provisional_subtype"].unique())
        lines.append(f"- **{label}: split** into {', '.join(subnames)} (ARI {diagnostics[label]['mean_ari']:.3f}).")
        for d in comparisons:
            if d["label"] == label and d["meets_separation_rule"]:
                lines.append(f"  - {d['subtype_b']} minus {d['subtype_a']}, {d['metric']}: **{d['difference_b_minus_a_yd']:+.2f} yd** "
                             f"(95% CI {d['ci_low_yd']:+.2f} to {d['ci_high_yd']:+.2f}).")
    if not split:
        lines.append("No labels meet the split rule; retain all original labels for Step 3.")
    indicator_caveats = []
    for label in split:
        for subtype, g in routes.loc[routes.route_label.eq(label) & ~routes.too_short].groupby("provisional_subtype"):
            if g.break_detected.mean() >= .95:
                indicator_caveats.append(f"{subtype}: {g.break_detected.sum():,}/{len(g):,} detected breaks")
    lines += ["", "## Labels retained", ""]
    for label in sorted(set(decisions) - set(split)):
        d = diagnostics[label]
        lines.append(f"- **{label}**: k={d['chosen_k']}, ARI={d['mean_ari']:.3f}; {decisions[label]['reason']}.")
    lines += ["", "## Caveats and review", "",
              f"- Only {routes.break_detected.sum():,}/{len(routes):,} routes have detected breaks. Names use median catch depth unless at least half a subtype has a detected break. Names are provisional geometry bands (<10, 10–<20, >=20 yd), not verified football categories.",
              "- Shape uses 12 arc-length points, standardized within label and reduced to 95% PCA variance. Geometry is also standardized; all-missing break columns use zero for fitting only. k is BIC-selected, then reduced until every subtype has >=100 routes and >=10% of its whole label.",
              "- ARI uses 50 game-resampled refits (including preprocessing), matching each overlapping route once. ARI for k=1 is trivially 1. Silhouette is a k-means cross-check, not the GMM assignment's silhouette.",
              "- Retaining a label after a minimum-size failure does not establish uniform geometry. Rare detected-break indicators can produce small mixture components that force k back to 1.",
              "- Separation intervals use 1,000 paired game resamples per label and all subtype pairs; intervals are exploratory and unadjusted for multiple comparisons. Separation associations may reflect depth, position, coverage or other context; they are not causal effects.",
              f"- {counts['missing_arrival_separation']:,} labeled routes lack arrival separation "
              f"({counts['missing_arrival_separation_all_targets']:,} across all targets, including two unlabeled routes); each summary reports available counts. "
              f"{counts['too_short_total']} paths <1 yd retain their original label (cluster_id=0, no fitted probability).",
              f"- **{len(review)} outliers** (ceil of 1% per fitted label) remain in all analyses and appear in outlier_review.csv for manual review. Low likelihood alone does not establish a mislabel.",
              "- Cards show up to 200 full paths and throw markers. The median uses actual 0.1-second snap-time bins without stretching or endpoint extrapolation, ending when fewer than half the routes remain.",
              f"- Input SHA-256 checks: {'unchanged' if unchanged else 'FAILED'}. All {len(routes):,} routes are assigned exactly once. Detailed BIC, stability and pairwise comparisons accompany this report."]
    if indicator_caveats:
        lines.insert(lines.index("## Caveats and review") + 2,
                     "- **Break-detection dependence:** " + "; ".join(indicator_caveats) +
                     ". The accepted split partly reflects the required break-detection feature, so review the geometry cards before giving these groups football names.")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged or not counts["every_route_assigned_exactly_once"] or not counts["expected_count_matches"]:
        raise AssertionError("Validation failed; see validation.json")
    log(f"Complete: {len(routes):,} routes, splits={split}, outliers={len(review)}; inputs unchanged.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
