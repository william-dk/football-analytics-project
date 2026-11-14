"""How much of a route is the receiver? Identification, variance split and style metrics.

Within each route type the random-effects model of player_model.py is fitted to a feature set
(the embedding z or a baseline). Everything is cross-validated by half-season: the model and each
receiver's profile come from odd weeks and are tested on even weeks, then the reverse.

Populations (--population)
  targeted  targeted routes, full path, supplementary route labels
  all       every route runner's pre-throw path; the route type is the one evaluate.py assigned
            from z (untargeted runners have no label)

Outputs in <run-dir>/style/<population>/
  identification.json           accuracy of naming the receiver from n routes, per feature set,
                                with within-team and within-position controls and a label-shuffle null
  identification_by_type.csv    the same per route type (n routes of that type)
  variance.csv                  share of the feature variance that lies between receivers, split
                                into position, team and individual parts, per route type
  metric_stability.csv          the same split for plain route metrics (speed, depth, braking, ...):
                                which simple numbers are a property of the receiver
  reliability.csv               odd-week vs even-week agreement of receiver profiles per style axis
  scheme.json                   can the team be named from routes of receivers never seen in training?
  player_style.csv              distinctiveness and consistency per receiver and route type
  receivers.csv                 one row per receiver: season profile and how often he is identified
  figures: identification_curve.png, variance_split.png, style_axes.png, style_map.png

    python player_style.py --run-dir runs/v2 --population all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate import RoutePCA, Runner, fixed_time_positions, load
from player_model import (average_noise, bag_noise, bag_scores, cells, fit_type_model, player_posteriors,
                          project, remove_group_means, route_scores, variance_components)

HERE = Path(__file__).resolve().parent
FEATURE_SETS = ("z", "pca16", "positions", "speed", "route64")
DESCRIPTION = {
    "z": "embedding z (16)",
    "pca16": "PCA of the route (16)",
    "positions": "x, y at 6 fixed times (12)",
    "speed": "max, mean, speed at 1.0 s (3)",
    "route64": "x, y every 0.2 s to 6.4 s (64)",
}
COLORS = {"z": "#1f77b4", "pca16": "#2ca02c", "positions": "#8c564b", "speed": "#ff7f0e", "route64": "#7f7f7f"}
DEFAULTS = {"targeted": dict(min_per_half=15, bag_sizes=(1, 3, 5, 10, 20), map_routes=40),
            "all": dict(min_per_half=40, bag_sizes=(1, 3, 5, 10, 20, 30, 50), map_routes=150)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--population", choices=("targeted", "all"), default="all")
    p.add_argument("--condition", choices=("type", "type+alignment"), default="type",
                   help="what 'a certain route' means: the route type, or the route type run from the "
                        "same alignment (outermost, slot, backfield). The second is a control: it asks "
                        "whether receivers are told apart only by where they line up. Its outputs go to "
                        "style/<population>_alignment/")
    p.add_argument("--min-per-half", type=int, default=None,
                   help="candidates need this many routes (any type) in odd AND even weeks")
    p.add_argument("--min-type-receivers", type=int, default=20,
                   help="a route type is used when this many receivers have 2+ routes of it in each half")
    p.add_argument("--features", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    p.add_argument("--bag-sizes", nargs="+", type=int, default=None)
    p.add_argument("--reps", type=int, default=20, help="random route sets per receiver, size and fold")
    p.add_argument("--permutations", type=int, default=2, help="label-shuffle refits for the z null")
    p.add_argument("--min-style-routes", type=int, default=8, help="routes of a type to list a receiver")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- population

def load_population(run_dir: Path, data_dir: Path, population: str, condition: str = "type") -> dict:
    """Rows, labels and feature sets of one population. Player identity is used only here, never
    by the autoencoder."""
    ckpt, model, data, meta = load(run_dir, data_dir)
    runner = Runner(ckpt, model, data)
    emb = pd.read_csv(run_dir / "embeddings.csv", low_memory=False)
    if not np.array_equal(emb.row.to_numpy(), meta.row.to_numpy()):
        raise SystemExit("embeddings.csv and routes_meta.csv are not in the same row order")
    kind = data["kind"]
    if population == "targeted":
        types = meta.route_label.to_numpy(dtype=object)
        rows = np.flatnonzero((kind == "full") & pd.notna(types))
    else:
        if "assigned_label" not in emb.columns:
            raise SystemExit("The data has no pre-throw sequences; rebuild with both kinds and re-run evaluate.py")
        types = emb.assigned_label.to_numpy(dtype=object)
        rows = np.flatnonzero((kind == "pre") & pd.notna(types))
    if condition == "type+alignment":
        types = types.copy()
        types[rows] = [f"{t} {a}" for t, a in zip(types[rows], meta.alignment.fillna("unknown").to_numpy()[rows])]
    zcols = [c for c in emb.columns if c[0] == "z" and c[1:].isdigit()]
    X, lengths = data["X"], runner.lengths
    speed = np.where(np.arange(X.shape[1])[None, :] < lengths[:, None], X[..., 4], np.nan)
    of_kind = np.flatnonzero(kind == ("full" if population == "targeted" else "pre"))
    fit_rows = of_kind[data["split"][of_kind] == "train"]
    features = {
        "z": emb[zcols].to_numpy(float)[rows],
        "pca16": RoutePCA(runner.raw_xy, lengths, fit_rows).codes(runner.raw_xy[rows], lengths[rows]),
        "positions": fixed_time_positions(runner.raw_xy, lengths)[rows],
        "speed": np.c_[np.nanmax(speed, 1), np.nanmean(speed, 1),
                       X[np.arange(len(X)), np.minimum(10, lengths - 1), 4]][rows],
        "route64": fixed_time_positions(runner.raw_xy, lengths, frames=range(2, 66, 2))[rows],
    }
    sub = meta.iloc[rows].reset_index(drop=True)
    # One team and position per receiver (his most frequent), so they are receiver-level facts.
    primary_team = sub.groupby("nfl_id").team.agg(lambda t: t.mode().iloc[0])
    primary_pos = sub.groupby("nfl_id").position.agg(lambda t: t.mode().iloc[0])
    return dict(
        runner=runner, meta=sub, rows=rows, features=features, types=types[rows],
        player=sub.nfl_id.to_numpy(), game=sub.game_id.to_numpy(), half=np.where(sub.week.to_numpy() % 2 == 1, "odd", "even"),
        team=primary_team.reindex(sub.nfl_id).to_numpy(), position=primary_pos.reindex(sub.nfl_id).to_numpy(),
        split=sub.split.to_numpy(), name=sub.groupby("nfl_id").player_name.first(),
        primary_team=primary_team, primary_pos=primary_pos)


def subset(pop: dict, mask: np.ndarray) -> dict:
    """The population restricted to some routes (only the per-route label arrays are cut)."""
    return {**pop, **{k: pop[k][mask] for k in ("types", "player", "game", "half", "team", "position", "split")}}


def choose_sets(pop: dict, min_per_half: int, min_type_receivers: int, min_type_routes: int = 200):
    """Candidate receivers (enough routes in both halves) and usable route types (enough receivers
    with repeated routes of the type, and enough such routes, in each half)."""
    counts = (pd.DataFrame(dict(p=pop["player"], h=pop["half"])).groupby(["p", "h"]).size()
              .unstack(fill_value=0).reindex(columns=["odd", "even"], fill_value=0))
    cands = np.sort(counts.index[(counts.odd >= min_per_half) & (counts.even >= min_per_half)].to_numpy())
    per_type = pd.DataFrame(dict(t=pop["types"], p=pop["player"], h=pop["half"])).groupby(["t", "h", "p"]).size()
    repeat = per_type[per_type >= 2]
    stats = pd.DataFrame(dict(receivers=repeat.groupby(level=["t", "h"]).size(),
                              routes=repeat.groupby(level=["t", "h"]).sum()))
    ok = ((stats.receivers >= min_type_receivers) & (stats.routes >= min_type_routes)).groupby(level="t").sum()
    types = sorted(ok.index[ok == 2])               # the type must qualify in both half-seasons
    return cands, types


# ----------------------------------------------------------------------------- one fold

class Fold:
    """Models fitted on one set of routes and everything needed to score another set."""

    def __init__(self, F, types, player, game, cand_idx, train, test, type_list, n_cand):
        self.n_cand, self.game = n_cand, game
        self.post: dict = {}
        self.models: dict = {}
        self.v = np.zeros_like(np.asarray(F, dtype=float))
        usable = np.zeros(len(F), dtype=bool)
        for t in type_list:
            of_type = types == t
            tr = train & of_type
            try:
                model = fit_type_model(F[tr], player[tr], game[tr])
            except ValueError:
                continue
            c = tr & (cand_idx >= 0)
            mean, var, _ = player_posteriors(model, project(model, F[c]), cand_idx[c], game[c], n_cand)
            self.models[t], self.post[t] = model, (mean, var)
            te = test & of_type
            self.v[te] = project(model, F[te])
            usable |= te
        self.rows = np.flatnonzero(usable & (cand_idx >= 0))      # test routes of candidates
        self.types, self.truth = types[self.rows], cand_idx[self.rows]

    def single_scores(self) -> np.ndarray:
        """(test routes, candidates) log-likelihoods for one route at a time."""
        out = np.empty((len(self.rows), self.n_cand), dtype=np.float32)
        for t, (mean, var) in self.post.items():
            sel = np.flatnonzero(self.types == t)
            if len(sel):
                out[sel] = route_scores(self.models[t], mean, var, self.v[self.rows[sel]])
        return out

    def bag(self, local: np.ndarray) -> np.ndarray:
        """Scores (candidates,) for a set of test routes given by positions in self.rows. Routes
        of each type are averaged and scored by that type's model; types add up."""
        total = np.zeros(self.n_cand)
        types = self.types[local]
        for t in pd.unique(types):
            pick = self.rows[local[types == t]]
            mean, var = self.post[t]
            total += bag_scores(mean, var, self.v[pick].mean(0), bag_noise(self.models[t], self.game[pick]))
        return total


def restricted_hit(scores: np.ndarray, truth: int, group: np.ndarray) -> float:
    """Top-1 hit when only candidates in the truth's group (team or position) may be named;
    NaN when the truth has no other candidate in his group."""
    same = np.flatnonzero(group == group[truth])
    if len(same) < 2:
        return np.nan
    return float(same[np.argmax(scores[same])] == truth)


def evaluate_feature(F, pop, cand_idx, type_list, cand_team, cand_pos, bag_sizes, reps, rng,
                     shuffle: bool = False, by_type: bool = False) -> dict:
    """Cross-validated identification for one feature set.

    Returns per-receiver hit rates: res[n] is a DataFrame indexed by candidate with columns
    top1, team, position (NaN where undefined), for n in bag_sizes plus "all" (every route of the
    held-out half as one set). res["single"] has one row per test route."""
    n_cand = len(cand_team)
    types, player, half, game = pop["types"], pop["player"], pop["half"], pop["game"]
    sums = {n: np.zeros((n_cand, 3)) for n in list(bag_sizes) + ["all"]}
    cnts = {n: np.zeros((n_cand, 3)) for n in list(bag_sizes) + ["all"]}
    single, type_rows = [], []
    for train_half in ("odd", "even"):
        train, test = half == train_half, half != train_half
        labels = player
        if shuffle:                                   # break the receiver-route link in training
            labels = player.copy()
            for t in type_list:
                sel = np.flatnonzero(train & (types == t))
                labels[sel] = labels[rng.permutation(sel)]
        cands = pop["cands"]
        idx = np.where(np.isin(labels, cands), np.minimum(np.searchsorted(cands, labels), n_cand - 1), -1)
        fold = Fold(F, types, labels, game, idx, train, test, type_list, n_cand)
        fold.truth = cand_idx[fold.rows]              # test labels are never shuffled
        S = fold.single_scores()
        order = np.argsort(-S, axis=1, kind="stable")
        rank = (order == fold.truth[:, None]).argmax(1)
        single.append(pd.DataFrame(dict(
            row=fold.rows, cand=fold.truth, type=fold.types, top1=rank == 0, top5=rank < 5, rank=rank,
            team=[restricted_hit(S[i], fold.truth[i], cand_team) for i in range(len(S))],
            position=[restricted_hit(S[i], fold.truth[i], cand_pos) for i in range(len(S))])))
        by_cand = pd.Series(np.arange(len(fold.rows))).groupby(fold.truth).apply(lambda s: s.to_numpy())
        for c, local in by_cand.items():
            for n in list(bag_sizes) + ["all"]:
                if n == 1:
                    continue                          # taken from the single-route table below
                if n == "all":
                    draws = [local]
                elif len(local) >= n:
                    draws = [rng.choice(local, size=n, replace=False) for _ in range(reps)]
                else:
                    continue
                for pick in draws:
                    s = fold.bag(pick)
                    hit = [float(np.argmax(s) == c), restricted_hit(s, c, cand_team),
                           restricted_hit(s, c, cand_pos)]
                    ok = ~np.isnan(hit)
                    sums[n][c] += np.nan_to_num(hit)
                    cnts[n][c] += ok
            if by_type:
                local_types = fold.types[local]
                for t in pd.unique(local_types):
                    mine = local[local_types == t]
                    if len(mine) >= 10:
                        hits = [np.argmax(fold.bag(rng.choice(mine, size=10, replace=False))) == c
                                for _ in range(min(reps, 10))]
                        type_rows.append(dict(type=t, cand=c, n=10, top1=float(np.mean(hits))))
    single = pd.concat(single, ignore_index=True)
    res = {"single": single}
    one = single.groupby("cand")[["top1", "team", "position"]].mean()
    res[1] = one.reindex(np.arange(n_cand))
    for n in sums:
        if n == 1:
            continue
        with np.errstate(invalid="ignore"):
            res[n] = pd.DataFrame(sums[n] / cnts[n], columns=["top1", "team", "position"])
    if by_type:
        res["by_type"] = pd.DataFrame(type_rows)
    return res


def summarize(res: dict, bag_sizes, n_cand: int, chance_team: float, chance_pos: float) -> dict:
    """Receiver-balanced accuracy with a 95% interval (receivers as the sampling unit)."""
    out = {}
    for n in list(bag_sizes) + ["all"]:
        table = res[n]
        entry = {"receivers": int(table.top1.notna().sum())}
        for col in ("top1", "team", "position"):
            vals = table[col].dropna().to_numpy()
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
            entry[col] = float(vals.mean()) if len(vals) else float("nan")
            entry[f"{col}_ci"] = [float(vals.mean() - 1.96 * se), float(vals.mean() + 1.96 * se)] if len(vals) > 1 else None
        out[str(n)] = entry
    single = res["single"]
    out["1"]["top5"] = float(single.groupby("cand").top5.mean().mean())
    out["1"]["median_rank_percentile"] = float(np.median(single["rank"]) / n_cand)
    out["chance"] = {"top1": 1 / n_cand, "top5": 5 / n_cand, "team": chance_team, "position": chance_pos}
    return out


def paired_gap(res_a: dict, res_b: dict, steps) -> dict:
    """Accuracy of feature set a minus feature set b on the same receivers, with a 95% interval
    (receivers as the sampling unit). Both sets are scored on the same held-out routes."""
    out = {}
    for n in steps:
        diff = (res_a[n].top1 - res_b[n].top1).dropna().to_numpy()
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        out[str(n)] = dict(gap=float(diff.mean()), ci=[float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                           receivers=int(len(diff)))
    return out


# ----------------------------------------------------------------------------- descriptive analyses

def half_season_products(f, codes, team_of, half, n_players):
    """Agreement between half-season averages, summed over feature columns.

    same   weighted mean of (a receiver's odd-week average) . (his own even-week average):
           what is stable about one receiver. Noise is independent between the two sets of
           games, so it averages out instead of adding to the estimate.
    mate   the same product for two DIFFERENT receivers on one team (one's odd weeks with the
           other's even weeks): what teammates share, i.e. the team's part.
    Returns (same, mate)."""
    odd, even = half == "odd", half == "even"
    n_odd = np.bincount(codes[odd], minlength=n_players).astype(float)
    n_even = np.bincount(codes[even], minlength=n_players).astype(float)
    m_odd, m_even = np.zeros((n_players, f.shape[1])), np.zeros((n_players, f.shape[1]))
    np.add.at(m_odd, codes[odd], f[odd])
    np.add.at(m_even, codes[even], f[even])
    both = (n_odd > 0) & (n_even > 0)
    m_odd, m_even = m_odd[both] / n_odd[both, None], m_even[both] / n_even[both, None]
    w = (n_odd * n_even / np.maximum(n_odd + n_even, 1))[both]
    own = (m_odd * m_even).sum(1)
    same = float((w * own).sum() / w.sum())
    teams, team_codes = np.unique(team_of[both], return_inverse=True)
    s_odd, s_even = np.zeros((len(teams), f.shape[1])), np.zeros((len(teams), f.shape[1]))
    np.add.at(s_odd, team_codes, m_odd * w[:, None])
    np.add.at(s_even, team_codes, m_even * w[:, None])
    num = (s_odd * s_even).sum() - (w ** 2 * own).sum()
    den = (np.bincount(team_codes, weights=w) ** 2).sum() - (w ** 2).sum()
    return same, float(num / den)


def variance_split(F, pop, type_list, with_axes: bool = True) -> pd.DataFrame:
    """Per route type: how the feature variance splits into a stable between-receiver part, a
    game-level part and play-to-play variation, and how the between-receiver part splits into
    position, team and the individual remainder (see half_season_products)."""
    rows = []
    for t in type_list:
        sel = pop["types"] == t
        f, player, game, half = F[sel], pop["player"][sel], pop["game"][sel], pop["half"][sel]
        ids, codes = np.unique(player, return_inverse=True)
        if len(ids) < 40:
            continue
        team_of = pd.Series(pop["team"][sel]).groupby(codes).first().to_numpy()
        centred = f - f.mean(0)
        total = float(centred.var(0).sum())
        same, _ = half_season_products(centred, codes, team_of, half, len(ids))
        adjusted = remove_group_means(f, [pop["position"][sel]])
        same_pos, mate = half_season_products(adjusted, codes, team_of, half, len(ids))
        game_var = variance_components(f, player, game)[1]
        row = dict(
            route_type=t, routes=int(sel.sum()), receivers=int(len(ids)),
            between_receiver_share=same / total, game_share=float(game_var.sum()) / total,
            play_share=1 - same / total - float(game_var.sum()) / total,
            position_share_of_total=(same - same_pos) / total, team_share_of_total=mate / total,
            individual_share_of_total=(same_pos - mate) / total,
            position_part=(same - same_pos) / same, team_part=mate / same,
            individual_part=(same_pos - mate) / same)
        if with_axes:
            model = fit_type_model(f, player, game)
            row.update(axis1_lambda=float(model.lam[0]), axis2_lambda=float(model.lam[1]),
                       axis3_lambda=float(model.lam[2]), axis1_game=float(model.game[0]),
                       axes_with_signal=int((model.lam > 0.02).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def weighted(table: pd.DataFrame, columns) -> dict:
    """Route-weighted means over route types. The three parts of the between-receiver share are
    ratios of the weighted shares, so a route type with a tiny share cannot dominate them."""
    w = table.routes / table.routes.sum()
    out = {c: float((table[c] * w).sum()) for c in columns if c in table.columns}
    if "between_receiver_share" in out:
        for part in ("position", "team", "individual"):
            out[f"{part}_part"] = out[f"{part}_share_of_total"] / out["between_receiver_share"]
    return out


SHARES = ("between_receiver_share", "game_share", "play_share", "position_share_of_total", "team_share_of_total",
          "individual_share_of_total", "axis1_lambda", "axis1_game")


def metric_stability(pop: dict, type_list) -> pd.DataFrame:
    """Which plain route metrics are a property of the receiver? For each metric: the share of its
    variance (within a route type) that is a stable difference between receivers, how that share
    splits, and how many routes it takes to measure a receiver's value with reliability 0.7."""
    desc = route_descriptors(pop)
    rows = []
    for col in desc.columns:
        res = weighted(variance_split(desc[[col]].to_numpy(float), pop, type_list, with_axes=False), SHARES)
        share = res["between_receiver_share"]
        res["routes_for_reliability_0.7"] = (0.7 / 0.3) * (1 - share) / share if share > 0 else float("inf")
        rows.append(dict(metric=col, **res))
    return pd.DataFrame(rows).sort_values("between_receiver_share", ascending=False)


def reliability(F, pop, type_list, min_routes: int = 5, axes: int = 4) -> pd.DataFrame:
    """Do receiver profiles replicate? Style axes come from one half-season; the correlation
    across receivers between their odd-week and even-week averages on each axis is then measured
    (both directions averaged). `predicted_r` is what the fitted model expects for these samples."""
    rows = []
    for t in type_list:
        sel = pop["types"] == t
        f, player, half, game = F[sel], pop["player"][sel], pop["half"][sel], pop["game"][sel]
        counts = pd.DataFrame(dict(p=player, h=half)).groupby(["p", "h"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=["odd", "even"], fill_value=0)
        keep = counts.index[(counts.odd >= min_routes) & (counts.even >= min_routes)].to_numpy()
        if len(keep) < 20:
            continue
        kept = np.isin(player, keep)
        idx = np.searchsorted(keep, player[kept])
        corr, pred, fitted = np.zeros(axes), np.zeros(axes), 0
        for fit_half in ("odd", "even"):
            try:
                model = fit_type_model(f[half == fit_half], player[half == fit_half], game[half == fit_half])
            except ValueError:
                continue
            v = project(model, f[kept])
            avg, gain = {}, {}
            for h in ("odd", "even"):
                m = half[kept] == h
                n = np.bincount(idx[m], minlength=len(keep)).astype(float)
                avg[h] = np.zeros((len(keep), v.shape[1]))
                np.add.at(avg[h], idx[m], v[m])
                avg[h] /= n[:, None]
                cell, cell_player, _ = cells(idx[m], game[kept][m])
                sq = np.bincount(cell_player, weights=np.bincount(cell).astype(float) ** 2, minlength=len(keep))
                gain[h] = model.lam / (model.lam + average_noise(model, n, sq))      # (receivers, axes)
            for j in range(axes):
                corr[j] += np.corrcoef(avg["odd"][:, j], avg["even"][:, j])[0, 1]
                pred[j] += float(np.sqrt(gain["odd"][:, j] * gain["even"][:, j]).mean())
            fitted += 1
        for j in range(axes if fitted else 0):
            rows.append(dict(route_type=t, axis=j + 1, receivers=len(keep),
                             mean_routes_per_half=float(counts.loc[keep].to_numpy().mean()),
                             split_half_r=float(corr[j] / fitted), predicted_r=float(pred[j] / fitted)))
    return pd.DataFrame(rows)


def scheme_signal(F, pop, type_list, rng, bag_sizes=(1, 10, 30), reps: int = 20) -> dict:
    """Name the TEAM from routes of receivers the model never saw.

    Receivers of each team are split in two groups. Team profiles are built from one group and
    tested on the other group's routes, so a correct call cannot come from recognising the
    receiver; it has to come from what the team's receivers share (scheme, quarterback, coaching)."""
    teams = np.sort(pd.unique(pop["team"]))
    team_idx = np.searchsorted(teams, pop["team"])
    players = pd.DataFrame(dict(p=pop["player"], team=team_idx)).drop_duplicates("p")
    group = {}
    for _, members in players.groupby("team"):
        ids = rng.permutation(members.p.to_numpy())
        group.update({p: i % 2 for i, p in enumerate(ids)})
    side = np.array([group[p] for p in pop["player"]])
    out = {n: [] for n in bag_sizes}
    for train_side in (0, 1):
        train, test = side == train_side, side != train_side
        fold = Fold(F, pop["types"], team_idx, pop["game"], team_idx, train, test, type_list, len(teams))
        S = fold.single_scores()
        by_team = pd.Series(np.arange(len(fold.rows))).groupby(fold.truth).apply(lambda s: s.to_numpy())
        for c, local in by_team.items():
            for n in bag_sizes:
                if n == 1:
                    out[n].append(float(np.mean(S[local].argmax(1) == c)))
                elif len(local) >= n:
                    out[n].append(float(np.mean([np.argmax(fold.bag(rng.choice(local, n, replace=False))) == c
                                                 for _ in range(reps)])))
    return {"teams": int(len(teams)), "chance": 1 / len(teams),
            "top1_by_routes": {str(n): float(np.mean(v)) for n, v in out.items() if v}}


def player_style(F, pop, type_list, min_routes: int) -> pd.DataFrame:
    """Distinctiveness and consistency per receiver and route type, in units of the typical
    play-to-play spread (all weeks). Features are first centred on the receiver's position group
    within the route type, so a running back is compared with running backs.

    distinctiveness  size of the receiver's estimated effect (posterior mean): how far his average
                     route of this type sits from the average route of his position
    distance_debiased  the raw distance of his average from that average with the part expected
                     from sampling noise removed (noisier, not shrunk)
    consistency      spread of his routes around his own average, relative to the league-typical
                     spread (1 = typical, below 1 = more repeatable)"""
    rows = []
    for t in type_list:
        sel = pop["types"] == t
        f, player, game = remove_group_means(F[sel], [pop["position"][sel]]), pop["player"][sel], pop["game"][sel]
        try:
            model = fit_type_model(f, player, game)
        except ValueError:
            continue
        v = project(model, f)
        d = v.shape[1]
        ids, codes = np.unique(player, return_inverse=True)
        mean, _, counts = player_posteriors(model, v, codes, game, len(ids))
        avg = np.zeros((len(ids), d))
        np.add.at(avg, codes, v)
        avg /= counts[:, None]
        cell, cell_player, _ = cells(codes, game)
        sq = np.bincount(cell_player, weights=np.bincount(cell).astype(float) ** 2, minlength=len(ids))
        noise = average_noise(model, counts, sq).sum(1)           # expected squared size of pure noise
        spread = np.zeros(len(ids))
        np.add.at(spread, codes, ((v - avg[codes]) ** 2).sum(1))
        for i, pid in enumerate(ids):
            n = int(counts[i])
            if n < min_routes:
                continue
            rows.append(dict(
                route_type=t, nfl_id=int(pid), routes=n,
                distinctiveness=float(np.linalg.norm(mean[i])),
                distance_debiased=float(np.sqrt(max((avg[i] ** 2).sum() - noise[i], 0.0))),
                consistency=float(spread[i] / ((n - 1) * d)),
                axis1=float(mean[i, 0]), axis2=float(mean[i, 1]), axis3=float(mean[i, 2])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def decode_z(runner: Runner, z: np.ndarray, length: int) -> np.ndarray:
    valid = torch.from_numpy(np.arange(runner.steps)[None, :] < length)
    with torch.no_grad():
        out = runner.model.decode(torch.from_numpy(z[None].astype(np.float32)), valid)
    return out[0, :length][:, runner.xy_out].numpy() * runner.pos_scale


def describe_axis(v_axis: np.ndarray, descriptors: pd.DataFrame) -> str:
    """Name the route descriptor a style axis follows most closely."""
    corr = descriptors.apply(lambda col: np.corrcoef(v_axis, col)[0, 1])
    best = corr.abs().idxmax()
    return f"r = {corr[best]:+.2f} with {best}"


def route_descriptors(pop: dict) -> pd.DataFrame:
    runner, rows = pop["runner"], pop["rows"]
    xy, lengths = runner.raw_xy[rows], runner.lengths[rows]
    speed = pop["features"]["speed"]
    last = xy[np.arange(len(rows)), lengths - 1]
    at = lambda f: xy[np.arange(len(rows)), np.minimum(f, lengths - 1)]
    valid = np.arange(xy.shape[1])[None, :] < lengths[:, None]
    accel = runner.X[torch.from_numpy(rows)][..., 5].numpy() * runner.scales[5]      # signed, yd/s^2
    steps = np.linalg.norm(np.diff(xy, axis=1), axis=-1) * valid[:, 1:]
    return pd.DataFrame({
        "max speed": speed[:, 0], "mean speed": speed[:, 1], "speed at 1.0 s": speed[:, 2],
        "hardest braking": -np.where(valid, accel, np.inf).min(1),
        "depth at 1.5 s": at(15)[:, 0], "lateral at 1.5 s": at(15)[:, 1],
        "final depth": last[:, 0], "final lateral": last[:, 1],
        "path straightness": np.linalg.norm(last, axis=1) / np.maximum(steps.sum(1), 1e-9),
        "duration": lengths.astype(float)})


def plot_style_axes(pop: dict, type_list, path: Path, axes: int = 3, max_types: int = 4,
                    position: str = "WR") -> None:
    """Decode the style axes of one position group, so the axes show how wide receivers differ
    from each other and not how they differ from tight ends or running backs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    group = pop["position"] == position
    counts = pd.Series(pop["types"][group]).value_counts()
    show = [t for t in counts.index if t in type_list][:max_types]
    desc = route_descriptors(pop)
    z = pop["features"]["z"]
    fig, axs = plt.subplots(len(show), axes, figsize=(4.6 * axes, 4.3 * len(show)), squeeze=False)
    for i, t in enumerate(show):
        sel = (pop["types"] == t) & group
        model = fit_type_model(z[sel], pop["player"][sel], pop["game"][sel])
        v = project(model, z[sel])
        back = np.linalg.inv(model.transform)                    # row j: z change per unit of axis j
        length = int(np.median(pop["runner"].lengths[pop["rows"][sel]]))
        for j in range(axes):
            ax = axs[i, j]
            step = 2 * np.sqrt(model.lam[j])                     # two between-receiver standard deviations
            for sign, color in ((-1, "#d62728"), (1, "#1f77b4")):
                path_xy = decode_z(pop["runner"], model.mean + sign * step * back[j], length)
                ax.plot(path_xy[:, 1], path_xy[:, 0], color=color, lw=2.2,
                        label=f"{'+' if sign > 0 else '-'}2 SD receiver")
                ticks = np.arange(0, length, 5)
                ax.plot(path_xy[ticks, 1], path_xy[ticks, 0], ".", color=color, ms=7)
            mid = decode_z(pop["runner"], model.mean, length)
            ax.plot(mid[:, 1], mid[:, 0], "k--", lw=1, label="league average")
            share = model.lam[j] / (model.lam[j] + model.game[j] + model.noise[j])
            ax.set_title(f"{t}, style axis {j + 1}: {share:.0%} of its variance is the receiver\n"
                         f"{describe_axis(v[:, j], desc[sel])}", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"What separates one {position} from another on the same route: decoded style axes "
                 "(dots every 0.5 s; x = lateral yd, y = downfield yd)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_identification(summary: dict, bag_sizes, path: Path, population: str, n_cand: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [str(n) for n in bag_sizes] + ["all"]
    ticks = [str(n) for n in bag_sizes] + ["half\nseason"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, key, title in ((axs[0], "top1", f"among all {n_cand} candidates"),
                           (axs[1], "position", "among receivers of his position"),
                           (axs[2], "team", "among his own teammates")):
        for name, res in summary.items():
            ax.plot(range(len(steps)), [res[n][key] for n in steps], "o-", color=COLORS[name],
                    label=DESCRIPTION[name], lw=2.4 if name == "z" else 1.4)
            if name == "z":
                ci = np.array([res[n][f"{key}_ci"] for n in steps], dtype=float)
                ax.fill_between(range(len(steps)), ci[:, 0], ci[:, 1], color=COLORS[name], alpha=0.15)
        chance = next(iter(summary.values()))["chance"][key]
        ax.axhline(chance, color="black", ls="--", lw=1, label=f"chance ({chance:.1%})")
        ax.set_xticks(range(len(steps)), ticks)
        ax.set_xlabel("routes from the unknown receiver")
        ax.set_ylabel("receiver named correctly")
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    label = "targeted routes, labeled route types" if population == "targeted" else "all routes run, assigned route types"
    fig.suptitle(f"Who ran these routes? ({label}; mixed route types, from a half-season the model never saw)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_variance(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = table.sort_values("routes", ascending=False)
    x = np.arange(len(table))
    share = table.between_receiver_share.to_numpy()
    parts = [("position (WR / TE / RB)", "position_part", "#9467bd"), ("team", "team_part", "#ff7f0e"),
             ("individual", "individual_part", "#1f77b4")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bottom = np.zeros(len(table))
    for label, col, color in parts:
        vals = share * table[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(x, [f"{t}\n{n:,} routes" for t, n in zip(table.route_type, table.routes)], fontsize=8)
    ax.set_ylabel("share of embedding variance between receivers")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Within a route type, how much of the embedding is a stable difference between receivers? "
                 "(the rest varies from game to game and play to play)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_style_map(receivers: pd.DataFrame, path: Path, min_routes: int, top: int = 16) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = receivers.dropna(subset=["distinctiveness", "consistency"])
    data = data[data.routes >= min_routes]
    fig, ax = plt.subplots(figsize=(9.5, 7))
    colors = {"WR": "#1f77b4", "TE": "#2ca02c", "RB": "#d62728", "FB": "#7f7f7f"}
    for pos, grp in data.groupby("position"):
        ax.scatter(grp.distinctiveness, grp.consistency, s=10 + grp.routes / 8, alpha=0.6,
                   color=colors.get(pos, "0.5"), label=pos)
    named = pd.concat([data.nlargest(top // 2, "distinctiveness"), data.nsmallest(top // 4, "consistency"),
                       data.nlargest(top // 8, "consistency"),
                       data.nlargest(top // 8, "identified_10_routes")]).drop_duplicates("nfl_id")
    for i, (_, r) in enumerate(named.sort_values("consistency").iterrows()):
        ax.annotate(r.player_name, (r.distinctiveness, r.consistency), fontsize=7,
                    xytext=(4, 4 if i % 2 == 0 else -9), textcoords="offset points")
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel("distinctiveness: distance of his average route from the average of his position, same route type\n"
                  "(route-weighted over his route types; units of play-to-play spread)")
    ax.set_ylabel("consistency: spread of his routes around his own average (1 = league typical)")
    ax.set_title(f"Receiver style map: receivers with at least {min_routes} routes (marker size = routes)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    cfg = DEFAULTS[args.population]
    min_per_half = args.min_per_half or cfg["min_per_half"]
    bag_sizes = tuple(args.bag_sizes or cfg["bag_sizes"])
    out_dir = args.run_dir / "style" / (args.population + ("_alignment" if args.condition != "type" else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pop = load_population(args.run_dir, args.data_dir, args.population, args.condition)
    cands, type_list = choose_sets(pop, min_per_half, args.min_type_receivers)
    pop["cands"] = cands
    cand_idx = np.where(np.isin(pop["player"], cands), np.searchsorted(cands, pop["player"]), -1)
    cand_team = pop["primary_team"].reindex(cands).to_numpy()
    cand_pos = pop["primary_pos"].reindex(cands).to_numpy()
    chance_team = float(np.mean([1 / (cand_team == t).sum() for t in cand_team if (cand_team == t).sum() > 1]))
    chance_pos = float(np.mean([1 / (cand_pos == t).sum() for t in cand_pos if (cand_pos == t).sum() > 1]))
    print(f"{args.population}: {len(pop['rows'])} routes, {len(np.unique(pop['player']))} receivers, "
          f"{len(cands)} candidates (>= {min_per_half} routes per half), route types {type_list}", flush=True)

    results, summary = {}, {}
    for name in args.features:
        results[name] = evaluate_feature(pop["features"][name], pop, cand_idx, type_list, cand_team, cand_pos,
                                         bag_sizes, args.reps, rng, by_type=(name == "z"))
        summary[name] = summarize(results[name], bag_sizes, len(cands), chance_team, chance_pos)
        line = "  ".join(f"{n}: {summary[name][str(n)]['top1']:.1%}" for n in list(bag_sizes) + ["all"])
        print(f"{name:10s} top-1 by routes  {line}", flush=True)
    null = []
    for _ in range(args.permutations):
        res = evaluate_feature(pop["features"]["z"], pop, cand_idx, type_list, cand_team, cand_pos,
                               bag_sizes[:4], args.reps, rng, shuffle=True)
        null.append({str(n): float(res[n].top1.mean()) for n in bag_sizes[:4]})
    z_single = results["z"]["single"]
    ae_train = pop["split"][z_single.row.to_numpy()] == "train"
    per_half = z_single.groupby("cand").size() / 2            # every candidate is tested in both folds
    report = dict(
        population=args.population, condition=args.condition, routes=int(len(pop["rows"])), receivers=int(len(np.unique(pop["player"]))),
        candidates=int(len(cands)), min_per_half=min_per_half, route_types=type_list, bag_sizes=list(bag_sizes),
        routes_per_half_season={"mean": float(per_half.mean()), "median": float(per_half.median())},
        candidates_by_position={str(k): int(v) for k, v in pd.Series(cand_pos).value_counts().items()},
        features={k: DESCRIPTION[k] for k in args.features}, identification=summary,
        z_minus={name: paired_gap(results["z"], results[name], list(bag_sizes) + ["all"])
                 for name in args.features if name != "z"},
        z_shuffled_labels=({k: float(np.mean([n[k] for n in null])) for k in null[0]} if null else None),
        z_single_route_top1={"autoencoder_train_games": float(z_single.top1[ae_train].mean()),
                             "autoencoder_heldout_games": float(z_single.top1[~ae_train].mean())})

    z = pop["features"]["z"]
    variance = variance_split(z, pop, type_list)
    variance.to_csv(out_dir / "variance.csv", index=False)
    report["variance_route_weighted"] = weighted(variance, SHARES)
    report["variance_other_features"] = {
        name: weighted(variance_split(pop["features"][name], pop, type_list, with_axes=False), SHARES)
        for name in args.features if name != "z"}
    metrics = metric_stability(pop, type_list)
    metrics.to_csv(out_dir / "metric_stability.csv", index=False)
    report["metric_stability"] = metrics.set_index("metric")[
        ["between_receiver_share", "individual_part", "routes_for_reliability_0.7"]].round(4).to_dict("index")
    rel = reliability(z, pop, type_list)
    rel.to_csv(out_dir / "reliability.csv", index=False)
    report["reliability_by_axis"] = rel.groupby("axis")[["split_half_r", "predicted_r"]].mean().round(3).to_dict("index")
    scheme = {name: scheme_signal(pop["features"][name], pop, type_list, rng) for name in ("z", "speed")
              if name in args.features}
    report["scheme"] = scheme
    with open(out_dir / "scheme.json", "w", encoding="utf-8") as handle:
        json.dump(scheme, handle, indent=2)

    by_type = results["z"]["by_type"]
    type_table = (z_single.groupby(["type", "cand"]).top1.mean().groupby("type").agg(["mean", "size"])
                  .rename(columns={"mean": "top1_1_route", "size": "receivers"}))
    if len(by_type):
        type_table["top1_10_routes"] = by_type.groupby("type").top1.mean()
        type_table["receivers_with_10_routes"] = by_type.groupby("type").cand.nunique()
    type_table["routes"] = z_single.groupby("type").size()
    type_table["chance"] = 1 / len(cands)
    type_table.reset_index().rename(columns={"type": "route_type"}).to_csv(
        out_dir / "identification_by_type.csv", index=False)

    style = player_style(z, pop, type_list, args.min_style_routes)
    style.insert(2, "player_name", pop["name"].reindex(style.nfl_id).to_numpy())
    style.insert(3, "team", pop["primary_team"].reindex(style.nfl_id).to_numpy())
    style.insert(4, "position", pop["primary_pos"].reindex(style.nfl_id).to_numpy())
    style.sort_values(["route_type", "distinctiveness"], ascending=[True, False]).to_csv(
        out_dir / "player_style.csv", index=False)
    weight = lambda g, col: float(np.average(g[col], weights=g.routes))
    receivers = style.groupby("nfl_id").apply(lambda g: pd.Series(dict(
        routes=int(g.routes.sum()), route_types=len(g), distinctiveness=weight(g, "distinctiveness"),
        consistency=weight(g, "consistency"))), include_groups=False).reset_index()
    receivers.insert(1, "player_name", pop["name"].reindex(receivers.nfl_id).to_numpy())
    receivers.insert(2, "team", pop["primary_team"].reindex(receivers.nfl_id).to_numpy())
    receivers.insert(3, "position", pop["primary_pos"].reindex(receivers.nfl_id).to_numpy())
    for n in bag_sizes:
        hit = pd.Series(results["z"][n].top1.to_numpy(), index=cands)
        receivers[f"identified_{n}_route{'s' if n > 1 else ''}"] = hit.reindex(receivers.nfl_id).to_numpy()
    receivers["identified_half_season"] = pd.Series(results["z"]["all"].top1.to_numpy(),
                                                    index=cands).reindex(receivers.nfl_id).to_numpy()
    receivers.sort_values("distinctiveness", ascending=False).to_csv(out_dir / "receivers.csv", index=False)
    # Is a style metric a stable trait? Compare it between the two half-seasons.
    halves = {h: player_style(z[pop["half"] == h], subset(pop, pop["half"] == h), type_list, 5)
              .set_index(["route_type", "nfl_id"]) for h in ("odd", "even")}
    both = halves["odd"].join(halves["even"], lsuffix="_odd", rsuffix="_even", how="inner")
    report["metric_stability_odd_vs_even"] = {
        "cells": int(len(both)),
        **{m: float(np.corrcoef(both[f"{m}_odd"], both[f"{m}_even"])[0, 1])
           for m in ("distinctiveness", "distance_debiased", "consistency")}}

    with open(out_dir / "identification.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if not args.no_plots:
        plot_identification(summary, bag_sizes, out_dir / "identification_curve.png", args.population, len(cands))
        plot_variance(variance, out_dir / "variance_split.png")
        plot_style_axes(pop, type_list, out_dir / "style_axes.png")
        plot_style_map(receivers, out_dir / "style_map.png", cfg["map_routes"])
    for name, gaps in report["z_minus"].items():
        print(f"z minus {name:10s}", "  ".join(
            f"{n}: {g['gap']:+.1%} [{g['ci'][0]:+.1%}, {g['ci'][1]:+.1%}]" for n, g in gaps.items()))
    print("\nshuffled-label null (z):", report["z_shuffled_labels"])
    print("variance (route-weighted):", {k: round(v, 3) for k, v in report["variance_route_weighted"].items()})
    print(metrics[["metric", "between_receiver_share", "position_part", "team_part", "individual_part",
                   "routes_for_reliability_0.7"]].round(3).to_string(index=False))
    print("reliability by axis:", report["reliability_by_axis"])
    print("scheme:", {k: {n: round(a, 3) for n, a in v["top1_by_routes"].items()} for k, v in scheme.items()},
          "chance", round(next(iter(scheme.values()))["chance"], 3) if scheme else None)
    print("metric stability:", {k: round(v, 3) if isinstance(v, float) else v
                                for k, v in report["metric_stability_odd_vs_even"].items()})


if __name__ == "__main__":
    main()
