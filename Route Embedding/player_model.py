"""Random-effects model of how receivers differ within one route type (NumPy).

Within a route type, a route's feature vector f (the embedding z, or any baseline description) is

    f = type mean + receiver effect + game effect + play-to-play variation
    receiver effect ~ N(0, B)   the same for every route a receiver runs of this type
    game effect     ~ N(0, C)   shared by his routes of this type in one game (opponent, game
                                plan, field, weather)
    play-to-play    ~ N(0, E)   coverage, play call, the throw, noise

This is the model behind linear discriminant analysis, extended so that receivers with few routes
and routes clustered in games are handled properly (in speaker identification the receiver-only
version is known as probabilistic LDA). The features are rotated so that the covariance within a
receiver becomes the identity and B becomes diagonal: coordinate j ("style axis" j) has
between-receiver variance lam[j], game variance game[j] and play variance noise[j], with
game[j] + noise[j] close to 1. Axes with lam = 0 carry no identity and drop out of every score.

B is estimated from products of a receiver's averages over two disjoint sets of games. Noise is
independent across games, so this is unbiased whatever the game effects are; a one-way ANOVA
would count shared game effects as receiver differences. Its eigenvalues are then measured on
receivers that did not choose the axes (regularized_between), so weak axes are not overrated.

A receiver's average over n routes, n_g of them in game g, has noise variance
    nu = noise / n + game * sum(n_g^2) / n^2
His effect then has posterior mean lam / (lam + nu) * average and variance lam * nu / (lam + nu),
and a new set of routes with average a and noise nu' has likelihood N(a; mean, variance + nu').
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TypeModel:
    mean: np.ndarray          # (d,) feature mean of the route type
    transform: np.ndarray     # (d, d): v = (f - mean) @ transform are the style-axis coordinates
    lam: np.ndarray           # (d,) between-receiver variance per axis, descending
    game: np.ndarray          # (d,) game-level variance per axis
    noise: np.ndarray         # (d,) play-level variance per axis
    players: int              # receivers with at least one route
    routes: int


def group_sums(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((n_groups, values.shape[1]))
    np.add.at(out, codes, values)
    return out


def dense(codes: np.ndarray) -> tuple[np.ndarray, int]:
    uniq, inv = np.unique(codes, return_inverse=True)
    return inv, len(uniq)


def cells(player: np.ndarray, game: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Receiver-game cells. player must be dense 0..P-1. Returns (cell of each route, receiver of
    each cell, rank of each cell's game among that receiver's games)."""
    order = np.lexsort((game, player))
    p, g = player[order], game[order]
    new = np.r_[True, (p[1:] != p[:-1]) | (g[1:] != g[:-1])]
    cell_sorted = np.cumsum(new) - 1
    cell = np.empty(len(player), dtype=np.int64)
    cell[order] = cell_sorted
    cell_player = p[new]
    first = np.r_[True, cell_player[1:] != cell_player[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(cell_player)), 0))
    return cell, cell_player, np.arange(len(cell_player)) - start


def within_covariance(F: np.ndarray, player: np.ndarray) -> np.ndarray:
    """Pooled covariance of routes around their receiver's mean (receivers with >= 2 routes), with
    Ledoit-Wolf shrinkage toward a scaled identity."""
    from sklearn.covariance import ledoit_wolf
    counts = np.bincount(player)
    resid = F - (group_sums(F, player, len(counts)) / np.maximum(counts, 1)[:, None])[player]
    multi = counts[player] >= 2
    dof = int(multi.sum() - (counts >= 2).sum())
    if dof < 2 * F.shape[1]:
        raise ValueError("too few repeated routes to estimate the within-receiver covariance")
    cov, _ = ledoit_wolf(resid[multi], assume_centered=True)
    return cov * (multi.sum() / dof)      # the residuals lose one degree of freedom per receiver


def game_set_averages(V: np.ndarray, player: np.ndarray, game: np.ndarray):
    """Each receiver's averages over two disjoint sets of his games (his games in order,
    alternating). Returns (weights (P',), a (P', d), b (P', d)) for receivers with both sets."""
    n_players = int(player.max()) + 1
    cell, _, cell_rank = cells(player, game)
    side = (cell_rank % 2)[cell]
    na = np.bincount(player[side == 0], minlength=n_players).astype(float)
    nb = np.bincount(player[side == 1], minlength=n_players).astype(float)
    both = (na > 0) & (nb > 0)
    if both.sum() < 6:
        raise ValueError("too few receivers with routes in two games")
    a = group_sums(V[side == 0], player[side == 0], n_players)[both] / na[both, None]
    b = group_sums(V[side == 1], player[side == 1], n_players)[both] / nb[both, None]
    return (na * nb / np.maximum(na + nb, 1))[both], a, b


def cross_between(V: np.ndarray, player: np.ndarray, game: np.ndarray) -> np.ndarray:
    """Between-receiver covariance of centred features V from products of the two game-set
    averages. Unbiased entry by entry; returns a symmetric (d, d) matrix."""
    w, a, b = game_set_averages(V, player, game)
    cross = (a * w[:, None]).T @ b / w.sum()
    return (cross + cross.T) / 2


def regularized_between(V: np.ndarray, player: np.ndarray, game: np.ndarray, splits: int = 20,
                        seed: int = 0) -> np.ndarray:
    """Between-receiver covariance with honest eigenvalues.

    The eigenvalues of cross_between are too spread out: its axes adapt to the noise, so the
    largest look larger than they are. Here the receivers are split in two at random; the axes
    come from one half and the variance along each axis is measured on the other half. The result
    is averaged over many splits, so an axis without real signal averages to about zero. (This is
    the NERCOME estimator of Abadir, Distaso and Zikes, applied to the cross-game products.)"""
    w, a, b = game_set_averages(V, player, game)
    products = w[:, None, None] * (a[:, :, None] * b[:, None, :] + b[:, :, None] * a[:, None, :]) / 2
    rng = np.random.default_rng(seed)
    total, used = np.zeros((V.shape[1], V.shape[1])), 0
    for _ in range(splits):
        first = rng.random(len(w)) < 0.5
        if first.sum() < 3 or (~first).sum() < 3:
            continue
        for axes_from, measured_on in ((first, ~first), (~first, first)):
            _, vecs = np.linalg.eigh(products[axes_from].sum(0) / w[axes_from].sum())
            other = products[measured_on].sum(0) / w[measured_on].sum()
            along = np.einsum("ij,ik,kj->j", vecs, other, vecs)
            total += (vecs * along) @ vecs.T
            used += 1
    if not used:
        raise ValueError("too few receivers to split")
    return total / used


def nested_components(V: np.ndarray, player: np.ndarray, game: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column of V: (game-level variance, play-level variance) within receivers, by the
    method of moments for unbalanced nested data. player must be dense."""
    n_players = int(player.max()) + 1
    cell, cell_player, _ = cells(player, game)
    n_cells = len(cell_player)
    n_c = np.bincount(cell, minlength=n_cells).astype(float)
    n_p = np.bincount(player, minlength=n_players).astype(float)
    cell_mean = group_sums(V, cell, n_cells) / n_c[:, None]
    player_mean = group_sums(V, player, n_players) / np.maximum(n_p, 1)[:, None]
    dof_play = len(V) - n_cells
    within_player = ((V - player_mean[player]) ** 2).sum(0) / max(len(V) - (n_p > 0).sum(), 1)
    if dof_play < 30:                              # almost no repeated routes within a game
        return np.zeros(V.shape[1]), within_player
    play = ((V - cell_mean[cell]) ** 2).sum(0) / dof_play
    ss_game = (n_c[:, None] * (cell_mean - player_mean[cell_player]) ** 2).sum(0)
    sq = np.bincount(cell_player, weights=n_c ** 2, minlength=n_players)
    denom = len(V) - (sq[n_p > 0] / n_p[n_p > 0]).sum()
    if denom <= 0:
        return np.zeros(V.shape[1]), within_player
    game_var = np.maximum((ss_game - (n_cells - (n_p > 0).sum()) * play) / denom, 0.0)
    return game_var, play


def fit_type_model(F: np.ndarray, player: np.ndarray, game: np.ndarray) -> TypeModel:
    """Fit the model to the routes of one type. player and game hold any integer ids."""
    F = np.asarray(F, dtype=float)
    player, n_players = dense(player)
    if n_players < 3:
        raise ValueError("need at least three receivers")
    mean = F.mean(0)
    evals, evecs = np.linalg.eigh(within_covariance(F, player))
    whiten = evecs / np.sqrt(np.maximum(evals, 1e-12))            # (f - mean) @ whiten has covariance I
    U = (F - mean) @ whiten
    lam, rot = np.linalg.eigh(regularized_between(U, player, game))
    order = np.argsort(-lam)
    lam, rot = np.maximum(lam[order], 0.0), rot[:, order]
    game_var, play_var = nested_components(U @ rot, player, game)
    play_var = np.maximum(play_var, 1e-6)          # constant features (the origin) have no variance at all
    return TypeModel(mean=mean, transform=whiten @ rot, lam=lam, game=game_var, noise=play_var,
                     players=n_players, routes=len(F))


def project(model: TypeModel, F: np.ndarray) -> np.ndarray:
    return (np.asarray(F, dtype=float) - model.mean) @ model.transform


def average_noise(model: TypeModel, n, sq) -> np.ndarray:
    """Noise variance of an average over n routes whose per-game counts have squared sum sq.
    n and sq may be scalars or (P,) arrays; returns (d,) or (P, d). n = 0 gives infinity."""
    n = np.asarray(n, dtype=float)[..., None]
    sq = np.asarray(sq, dtype=float)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = model.noise / n + model.game * sq / n ** 2
    return np.where(n > 0, nu, np.inf)


def player_posteriors(model: TypeModel, v: np.ndarray, player: np.ndarray, game: np.ndarray, n_players: int):
    """Posterior of each receiver's effect from his routes v (already projected). player holds
    indices 0..n_players-1; receivers without routes keep the prior N(0, lam).
    Returns (mean (P, d), var (P, d), counts (P,))."""
    counts = np.bincount(player, minlength=n_players).astype(float)
    sq = np.zeros(n_players)
    if len(player):
        cell, cell_player, _ = cells(player, game)
        sq = np.bincount(cell_player, weights=np.bincount(cell).astype(float) ** 2, minlength=n_players)
    avg = group_sums(v, player, n_players) / np.maximum(counts, 1)[:, None]
    nu = average_noise(model, counts, sq)
    gain = model.lam[None, :] / (model.lam[None, :] + nu)        # 0 where the receiver has no routes
    return gain * avg, model.lam[None, :] * (1 - gain), counts


def bag_noise(model: TypeModel, game: np.ndarray) -> np.ndarray:
    """Noise variance (d,) of the average of a set of routes played in the given games."""
    _, per_game = np.unique(game, return_counts=True)
    return average_noise(model, len(game), float((per_game ** 2).sum()))


def bag_scores(post_mean: np.ndarray, post_var: np.ndarray, bag_mean: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Log-likelihood (up to a constant shared by all receivers) that a set of routes of this type
    with average bag_mean and noise variance nu was run by each receiver. Returns (P,)."""
    var = post_var + nu[None, :]
    return -0.5 * (((bag_mean[None, :] - post_mean) ** 2) / var + np.log(var)).sum(1)


def route_scores(model: TypeModel, post_mean: np.ndarray, post_var: np.ndarray, v: np.ndarray,
                 chunk: int = 512) -> np.ndarray:
    """bag_scores for many single routes at once. Returns (N, P)."""
    var = post_var + (model.noise + model.game)[None, :]
    const = -0.5 * np.log(var).sum(1)
    out = np.empty((len(v), len(post_mean)))
    for i in range(0, len(v), chunk):
        diff = v[i:i + chunk, None, :] - post_mean[None, :, :]
        out[i:i + chunk] = const[None, :] - 0.5 * (diff ** 2 / var[None, :, :]).sum(-1)
    return out


def variance_components(F: np.ndarray, player: np.ndarray, game: np.ndarray):
    """Per feature column: (between-receiver, game-level, play-level) variance. The between-receiver
    part uses disjoint game sets (see cross_between) and may be slightly negative by chance."""
    F = np.asarray(F, dtype=float)
    player, _ = dense(player)
    centred = F - F.mean(0)
    between = np.diag(cross_between(centred, player, game))
    game_var, play_var = nested_components(centred, player, game)
    return between, game_var, play_var


def remove_group_means(F: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Least-squares removal of additive group means (for example position and team) from F."""
    F = np.asarray(F, dtype=float)
    cols = [np.ones((len(F), 1))]
    for g in groups:
        codes, n = dense(g)
        cols.append(np.eye(n)[codes][:, 1:])                      # drop one level per factor
    design = np.concatenate(cols, axis=1)
    coef, *_ = np.linalg.lstsq(design, F, rcond=None)
    return F - design @ coef
