"""Can we identify a receiver by how he runs a certain route?

Within each route type (supplementary label), a classifier sees one route and must say which
receiver ran it. Candidates are receivers with at least --min-per-half routes of that type in
both odd and even weeks. Two folds: train on odd weeks and test on even weeks, then the reverse,
so every prediction is for a week the classifier never saw. The classifier is shrinkage LDA on
standardized features with equal priors.

Main metric: balanced accuracy (mean of each receiver's recall), whose chance level is exactly
1/K. Plain accuracy is also reported, but it rewards guessing the receivers with the most
routes: with shuffled labels it already beat 1/K on ANGLE (16% vs 9%).

Feature sets
  z          16-number route embedding from the autoencoder
  z+speed    z plus the three speed numbers below
  positions  x, y at 0.5, 1.0, ..., 3.0 s after the snap (12 numbers)
  speed      max speed, mean speed, speed at 1.0 s (how fast, not how)
  timing     route length and throw frame (set by the QB and the play, not the receiver)

Controls
  permutation null  z with training labels shuffled (--permutations times): p-value for z
  within-team       choose only among the true receiver's teammates (same scheme and play calls)
  pre-throw z       z re-encoded with the post-throw frames knocked out (the path while adjusting
                    to the ball is hidden), compared with full z on the same routes
  autoencoder       accuracy on routes from games the autoencoder trained on vs held-out games
  logistic          multinomial logistic regression instead of LDA (z only)
Several routes: combine n routes of the same unknown receiver by summing log-probabilities
(player-balanced, so chance is 1/K).

    python identify_receivers.py --run-dir runs/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import Runner, fixed_time_positions, load

HERE = Path(__file__).resolve().parent
FEATURE_SETS = ("z", "z+speed", "positions", "speed", "timing")
MULTI_SETS = ("z", "z+speed", "positions", "speed")
TEAM_SETS = ("z", "z+speed", "positions", "speed")
MULTI_N = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--min-per-half", type=int, default=5, help="routes of the type in odd AND even weeks")
    p.add_argument("--min-candidates", type=int, default=10, help="skip route types with fewer receivers")
    p.add_argument("--max-tail", type=int, default=15, help="pre-throw control: max post-throw frames")
    p.add_argument("--permutations", type=int, default=50)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--signature-label", default="HITCH")
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def lda(n_classes: int):
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto", priors=np.full(n_classes, 1 / n_classes)))


def logistic(n_classes: int):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))


def out_of_fold(make_model, F, y, half, n_classes, shuffle_rng=None) -> np.ndarray:
    """Class probabilities for every route from the model trained on the other half-season."""
    probs = np.zeros((len(y), n_classes))
    for train_half in ("odd", "even"):
        tr = np.flatnonzero(half == train_half)
        te = np.flatnonzero(half != train_half)
        labels = y[tr].copy()
        if shuffle_rng is not None:
            shuffle_rng.shuffle(labels)
        model = make_model(len(np.unique(labels))).fit(F[tr], labels)   # equal priors over present receivers
        probs[np.ix_(te, model.classes_)] = model.predict_proba(F[te])
    return probs


def topk_hits(probs, y, k: int) -> np.ndarray:
    return (np.argsort(-probs, axis=1, kind="stable")[:, :k] == y[:, None]).any(axis=1)


def balanced(hits, y, n_classes: int) -> float:
    """Mean recall over receivers that have routes here (chance = 1/K for any guessing rule)."""
    den = np.bincount(y, minlength=n_classes).astype(float)
    num = np.bincount(y, weights=hits.astype(float), minlength=n_classes)
    return float(np.mean(num[den > 0] / den[den > 0]))


def balanced_game_ci(hits, y, games, n_classes, rng, n_boot: int) -> list[float]:
    """95% interval for balanced accuracy, resampling whole games."""
    uniq, inv = np.unique(games, return_inverse=True)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    mult = np.zeros((n_boot, len(uniq)))
    np.add.at(mult, (np.arange(n_boot)[:, None], draws), 1)
    weights = mult[:, inv]
    onehot = np.eye(n_classes)[y]
    num, den = (weights * hits) @ onehot, weights @ onehot
    recall = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    acc = np.nanmean(recall, axis=1)
    return [float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))]


def frequency_hits(y, half, n_classes, k: int) -> np.ndarray:
    hits = np.zeros(len(y), dtype=bool)
    for train_half in ("odd", "even"):
        tr, te = half == train_half, half != train_half
        top = np.argsort(-np.bincount(y[tr], minlength=n_classes), kind="stable")[:k]
        hits[te] = np.isin(y[te], top)
    return hits


def within_team(probs, y, route_team, cand_team) -> dict:
    """Choose only among the route's team's candidates; player-balanced accuracy and chance."""
    hits = np.full(len(y), np.nan)
    chance = np.full(len(y), np.nan)
    for i in range(len(y)):
        same = np.flatnonzero(cand_team == route_team[i])
        if len(same) < 2 or y[i] not in same:
            continue
        hits[i] = same[np.argmax(probs[i, same])] == y[i]
        chance[i] = 1 / len(same)
    ok = ~np.isnan(hits)
    if not ok.any():
        return dict(routes=0, players=0, accuracy=float("nan"), chance=float("nan"))
    frame = pd.DataFrame(dict(y=y[ok], hit=hits[ok], chance=chance[ok])).groupby("y").mean()
    return dict(routes=int(ok.sum()), players=len(frame), accuracy=float(frame.hit.mean()),
                chance=float(frame.chance.mean()))


def multi_route(probs, y, n: int, rng, reps: int = 200) -> float:
    """Accuracy when n routes of the same unknown receiver are combined (player-balanced)."""
    logp = np.log(np.clip(probs, 1e-12, None))
    accs = []
    for player in np.unique(y):
        rows = np.flatnonzero(y == player)
        if len(rows) < n:
            continue
        picks = np.array([rng.choice(rows, size=n, replace=False) for _ in range(reps)])
        accs.append(np.mean(np.argmax(logp[picks].sum(axis=1), axis=1) == player))
    return float(np.mean(accs))


def knocked_after_throw(runner: Runner, rows, throw_index) -> np.ndarray:
    t = np.arange(runner.steps)[None, :]
    return (t > throw_index[rows][:, None]) & (t < runner.lengths[rows][:, None])


def decode_mean(runner: Runner, z_mean: np.ndarray, length: int) -> np.ndarray:
    valid = torch.from_numpy(np.arange(runner.steps)[None, :] < length)
    with torch.no_grad():
        out = runner.model.decode(torch.from_numpy(z_mean[None].astype(np.float32)), valid)
    return out[0, :length][:, runner.xy_out].numpy() * runner.pos_scale


def plot_accuracy(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = table.route_type.tolist()
    x = np.arange(len(labels))
    colors = {"z": "tab:blue", "z+speed": "tab:purple", "positions": "tab:green", "speed": "tab:orange",
              "timing": "tab:gray"}
    width = 0.16
    fig, ax = plt.subplots(figsize=(13, 5))
    for j, name in enumerate(FEATURE_SETS):
        vals = table[f"{name}_balanced_top1"].to_numpy()
        kwargs = {}
        if name == "z":
            ci = np.array(table["z_balanced_top1_ci"].tolist())
            kwargs = dict(yerr=np.vstack([vals - ci[:, 0], ci[:, 1] - vals]), capsize=3)
        ax.bar(x + (j - 2) * width, vals, width, label=name, color=colors[name], **kwargs)
    ax.scatter(x, table.chance_top1, marker="_", s=1100, color="black", zorder=5, label="chance = 1/K")
    ax.set_xticks(x, [f"{lab}\nK={k}" for lab, k in zip(labels, table.candidates)])
    ax.set_ylabel("balanced top-1 accuracy, held-out half-season")
    ax.set_title("Which receiver ran this route? One route; candidates ran this route type in both halves")
    ax.legend(ncol=6, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_multi(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = table.routes / table.routes.sum()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, color in zip(MULTI_SETS, ("tab:blue", "tab:purple", "tab:green", "tab:orange")):
        ax.plot(MULTI_N, [(table[f"{name}_multi_{n}"] * w).sum() for n in MULTI_N], "o-", color=color, label=name)
    ax.axhline((table.chance_top1 * w).sum(), color="black", ls="--", label="chance")
    ax.set_xlabel("routes of the same type from the unknown receiver")
    ax.set_ylabel("balanced top-1 accuracy (route-weighted mean)")
    ax.set_xticks(MULTI_N)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_signatures(runner, meta, z, players: pd.DataFrame, label: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    of_type = np.flatnonzero(meta.route_label.eq(label).to_numpy())
    league = decode_mean(runner, z[of_type].mean(0), int(np.median(runner.lengths[of_type])))
    top = players[players.route_type.eq(label)].sort_values("z_recall", ascending=False).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax, (_, p) in zip(axes.flat, top.iterrows()):
        mine = of_type[meta.nfl_id.to_numpy()[of_type] == p.nfl_id]
        for r in mine:
            n = runner.lengths[r]
            ax.plot(runner.raw_xy[r, :n, 1], runner.raw_xy[r, :n, 0], color="0.8", lw=0.8)
        sig = decode_mean(runner, z[mine].mean(0), int(np.median(runner.lengths[mine])))
        ax.plot(league[:, 1], league[:, 0], "k--", lw=1.5, label=f"league mean {label}")
        ax.plot(sig[:, 1], sig[:, 0], color="tab:blue", lw=2.5, label="decoded mean of his z")
        ax.set_title(f"{p.player_name} ({p.team}, {p.position}): {p.routes} routes, "
                     f"identified {p.z_recall:.0%}", fontsize=9)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{label}: most identifiable receivers. Gray = his actual routes; "
                 "x = lateral yd (+ toward his sideline), y = downfield yd", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.run_dir / "identification"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt, model, data, meta = load(args.run_dir, args.data_dir)
    runner = Runner(ckpt, model, data)
    emb = pd.read_csv(args.run_dir / "embeddings.csv", low_memory=False)
    if not np.array_equal(emb.row.to_numpy(), meta.row.to_numpy()):
        raise SystemExit("embeddings.csv and routes_meta.csv are not in the same row order")
    zcols = [c for c in emb.columns if c[0] == "z" and c[1:].isdigit()]
    z = emb[zcols].to_numpy(float)

    X, lengths = data["X"], runner.lengths
    speed = np.where(np.arange(X.shape[1])[None, :] < lengths[:, None], X[..., 4], np.nan)
    speed_feats = np.c_[np.nanmax(speed, 1), np.nanmean(speed, 1),
                        X[np.arange(len(X)), np.minimum(10, lengths - 1), 4]]
    features = {
        "z": z,
        "z+speed": np.c_[z, speed_feats],
        "positions": fixed_time_positions(runner.raw_xy, lengths),
        "speed": speed_feats,
        "timing": np.c_[lengths, meta.throw_index.to_numpy()].astype(float),
    }
    throw_index = meta.throw_index.to_numpy()
    tail = lengths - 1 - throw_index
    half = np.where(meta.week.to_numpy() % 2 == 1, "odd", "even")
    route_label = meta.route_label.to_numpy(dtype=object)
    nfl_id = meta.nfl_id.to_numpy()
    primary_team = meta.groupby("nfl_id").team.agg(lambda t: t.mode().iloc[0])
    name_of = meta.groupby("nfl_id").player_name.first()
    pos_of = meta.groupby("nfl_id").position.agg(lambda t: t.mode().iloc[0])
    rng = np.random.default_rng(args.seed)

    results, players, skipped = [], [], {}
    for label in sorted(pd.unique(route_label[pd.notna(route_label)])):
        sel = route_label == label
        counts = (pd.DataFrame(dict(nfl_id=nfl_id[sel], half=half[sel])).groupby(["nfl_id", "half"]).size()
                  .unstack(fill_value=0).reindex(columns=["odd", "even"], fill_value=0))
        cands = np.sort(counts.index[(counts.odd >= args.min_per_half)
                                     & (counts.even >= args.min_per_half)].to_numpy())
        if len(cands) < args.min_candidates:
            skipped[label] = int(len(cands))
            continue
        rows = np.flatnonzero(sel & np.isin(nfl_id, cands))
        y = np.searchsorted(cands, nfl_id[rows])
        k, h, games = len(cands), half[rows], meta.game_id.to_numpy()[rows]
        cand_team = primary_team.reindex(cands).to_numpy()
        route_team = meta.team.to_numpy()[rows]
        res = dict(route_type=label, candidates=k, routes=len(rows), chance_top1=1 / k, chance_top5=min(1.0, 5 / k))
        probs, hits = {}, {}
        for name in FEATURE_SETS:
            probs[name] = out_of_fold(lda, features[name][rows], y, h, k)
            hits[name] = topk_hits(probs[name], y, 1)
            res[f"{name}_top1"] = float(hits[name].mean())
            res[f"{name}_balanced_top1"] = balanced(hits[name], y, k)
            res[f"{name}_balanced_top5"] = balanced(topk_hits(probs[name], y, 5), y, k)
            res[f"{name}_balanced_top1_ci"] = balanced_game_ci(hits[name], y, games, k, rng, args.bootstrap)
        res["frequency_top1"] = float(frequency_hits(y, h, k, 1).mean())
        null = np.array([balanced(topk_hits(out_of_fold(lda, z[rows], y, h, k, shuffle_rng=rng), y, 1), y, k)
                         for _ in range(args.permutations)])
        res["z_null_mean"], res["z_null_95"] = float(null.mean()), float(np.quantile(null, 0.95))
        res["z_p_value"] = float((1 + np.sum(null >= res["z_balanced_top1"])) / (1 + len(null)))
        res["z_logistic_balanced_top1"] = balanced(topk_hits(out_of_fold(logistic, z[rows], y, h, k), y, 1), y, k)
        for name in TEAM_SETS:
            wt = within_team(probs[name], y, route_team, cand_team)
            res[f"within_team_{name}"] = wt["accuracy"]
            res["within_team_routes"], res["within_team_players"] = wt["routes"], wt["players"]
            res["within_team_chance"] = wt["chance"]
        short = tail[rows] <= args.max_tail
        _, z_pre = runner.predict(rows[short], knocked_after_throw(runner, rows[short], throw_index))
        res["prethrow_routes"] = int(short.sum())
        res["z_full_same_routes_balanced_top1"] = balanced(topk_hits(
            out_of_fold(lda, z[rows][short], y[short], h[short], k), y[short], 1), y[short], k)
        res["z_prethrow_balanced_top1"] = balanced(topk_hits(
            out_of_fold(lda, z_pre, y[short], h[short], k), y[short], 1), y[short], k)
        ae_train = meta.split.to_numpy()[rows] == "train"
        res["z_top1_autoencoder_train_games"] = float(hits["z"][ae_train].mean())
        res["z_top1_autoencoder_holdout_games"] = float(hits["z"][~ae_train].mean())
        res["autoencoder_holdout_routes"] = int((~ae_train).sum())
        for name in MULTI_SETS:
            for n in MULTI_N:
                res[f"{name}_multi_{n}"] = multi_route(probs[name], y, n, rng)
        results.append(res)
        top5 = topk_hits(probs["z"], y, 5)
        for j, pid in enumerate(cands):
            mine = y == j
            players.append(dict(route_type=label, nfl_id=int(pid), player_name=name_of[pid], position=pos_of[pid],
                                team=primary_team[pid], routes=int(mine.sum()), candidates=k, chance=1 / k,
                                z_recall=float(hits["z"][mine].mean()), z_top5_recall=float(top5[mine].mean()),
                                z_speed_recall=float(hits["z+speed"][mine].mean()),
                                speed_recall=float(hits["speed"][mine].mean())))
        ci = res["z_balanced_top1_ci"]
        print(f"{label:6s} K={k:3d} n={len(rows):5d} | balanced top-1: z {res['z_balanced_top1']:.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}] p={res['z_p_value']:.3f} | z+speed {res['z+speed_balanced_top1']:.3f} "
              f"| positions {res['positions_balanced_top1']:.3f} | speed {res['speed_balanced_top1']:.3f} "
              f"| timing {res['timing_balanced_top1']:.3f} | chance {1 / k:.3f}", flush=True)

    table = pd.DataFrame(results)
    player_table = pd.DataFrame(players)
    w = table.routes / table.routes.sum()
    overall = {f"{name}_balanced_top1": float((table[f"{name}_balanced_top1"] * w).sum()) for name in FEATURE_SETS}
    overall.update({f"{name}_balanced_top5": float((table[f"{name}_balanced_top5"] * w).sum())
                    for name in ("z", "z+speed")})
    overall.update({f"{name}_multi_{n}": float((table[f"{name}_multi_{n}"] * w).sum())
                    for name in MULTI_SETS for n in MULTI_N})
    tw = table.within_team_routes / table.within_team_routes.sum()
    overall.update({f"within_team_{name}": float((table[f"within_team_{name}"] * tw).sum()) for name in TEAM_SETS})
    pw = table.prethrow_routes / table.prethrow_routes.sum()
    overall.update({
        "chance_top1": float((table.chance_top1 * w).sum()),
        "chance_top5": float((table.chance_top5 * w).sum()),
        "z_null_mean": float((table.z_null_mean * w).sum()),
        "z_logistic_balanced_top1": float((table.z_logistic_balanced_top1 * w).sum()),
        "within_team_chance": float((table.within_team_chance * tw).sum()),
        "z_full_same_routes_balanced_top1": float((table.z_full_same_routes_balanced_top1 * pw).sum()),
        "z_prethrow_balanced_top1": float((table.z_prethrow_balanced_top1 * pw).sum()),
        "routes": int(table.routes.sum()),
        "route_types": table.route_type.tolist(),
    })
    report = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  overall_route_weighted=overall, skipped_route_types=skipped, by_route_type=results)
    with open(out_dir / "identification_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    flat = table.copy()
    for col in [c for c in flat.columns if c.endswith("_ci")]:
        flat[col] = flat[col].map(lambda v: f"{v[0]:.3f}-{v[1]:.3f}")
    flat.to_csv(out_dir / "identification_by_route_type.csv", index=False)
    player_table.sort_values(["route_type", "z_recall"], ascending=[True, False]).to_csv(
        out_dir / "player_recall.csv", index=False)
    plot_accuracy(table, out_dir / "identification_accuracy.png")
    plot_multi(table, out_dir / "identification_multi_route.png")
    if args.signature_label in set(table.route_type):
        plot_signatures(runner, meta, z, player_table, args.signature_label,
                        out_dir / f"signature_routes_{args.signature_label}.png")
    print("\nRoute-weighted:", json.dumps({k: round(v, 3) for k, v in overall.items() if isinstance(v, float)}))
    print("Skipped route types (candidates):", skipped)


if __name__ == "__main__":
    main()
