"""Is the embedding the limit? A supervised ceiling for receiver identification.

player_style.py names the receiver from z with a linear, Gaussian random-effects model. Here
classifiers are trained on the receiver labels to tell receivers apart, and compared with it on
exactly the same task: profiles from one half-season, tested on the other; the same candidates,
routes, route types and sets of n routes. Only z is learned without labels; both readouts use the
labels of the training half-season.

Supervised models (--models)
  mlp_z        MLP on z (the frozen embedding) and the route type
  mlp_all      MLP on z, the PCA code, positions at six times and speed
  mlp_positions, mlp_pca16, mlp_speed   the same MLP on each baseline description alone
  tf_scratch   Transformer encoder on the raw sequence, trained from scratch for this task
  tf_finetune  the same encoder started from the autoencoder's weights and fine-tuned
References: the random-effects model of player_model.py on z (re_z), on the features of mlp_all
(re_all), and on z with ONE receiver profile shared by all route types (re_z_shared; z is
centred on its route type first). The last one separates two reasons a classifier can win:
it learns from labels, and it pools a receiver's routes of every type.
Control (--shuffle-control): mlp_z trained with the receiver labels shuffled within route type
must fall to chance.

How the comparison is kept fair
  - "How he runs a route", not "which routes he runs". A classifier would otherwise learn how
    often each receiver runs each route type. Training adds log p(receiver | route type) to the
    logits (logit adjustment), so the raw logits it is scored with are log-likelihoods of the
    route given the receiver, up to a constant: the quantity the random-effects model scores.
  - n routes are combined by adding these scores (independent evidence) or, when that works
    better on held-out games of the training half, by averaging the probabilities.
  - The stopping time and the MLP size are chosen on games held out from the training
    half-season; the model is then retrained on the whole half-season. Nothing is tuned on the
    test half.
  - Every method is scored on the same sets of routes, so differences are paired.

How much data the classifiers need (--train-frac): every method learns from a random share of
the training half-season's routes and is tested on the same full half-season as before.

Scores are cached in <run-dir>/supervised/<population>/, so models can be trained in separate
processes (--no-eval) and compared afterwards.

    python supervised_ceiling.py --run-dir runs/v2 --population all
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from masking import training_knockout
from model import RouteAutoencoder
from player_style import DEFAULTS, Fold, choose_sets, load_population, restricted_hit
from train import epoch_batches

HERE = Path(__file__).resolve().parent
MODELS = ("mlp_z", "mlp_all", "mlp_positions", "mlp_pca16", "mlp_speed", "tf_scratch", "tf_finetune")
LABELS = {
    "re_z": "z + random-effects model",
    "re_all": "all features + random-effects model",
    "re_z_shared": "z + random-effects model, one receiver profile across route types",
    "mlp_z_shuffled": "control: classifier on z, receiver labels shuffled",
    "mlp_z": "classifier on z",
    "mlp_all": "classifier on z and all hand-made descriptions",
    "mlp_positions": "classifier on x, y at six fixed times",
    "mlp_pca16": "classifier on the PCA code",
    "mlp_speed": "classifier on speed only",
    "tf_scratch": "Transformer trained on receiver labels, from scratch",
    "tf_finetune": "Transformer trained on receiver labels, from the autoencoder",
}
# Lines of the figure, in legend order. z has one colour under both readouts.
COLORS = {"mlp_z": "#1f77b4", "re_z": "#1f77b4", "tf_finetune": "#d62728", "tf_scratch": "#e377c2",
          "mlp_positions": "#8c564b", "mlp_pca16": "#2ca02c", "mlp_speed": "#ff7f0e"}
REFERENCES = ("re_z", "re_all", "re_z_shared")
CONTROL = "mlp_z_shuffled"
MLP_GRID = [dict(hidden=256, dropout=0.2, weight_decay=0.01), dict(hidden=256, dropout=0.5, weight_decay=0.01),
            dict(hidden=256, dropout=0.5, weight_decay=0.1), dict(hidden=64, dropout=0.2, weight_decay=0.01)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--population", choices=("targeted", "all"), default="all")
    p.add_argument("--condition", choices=("type", "type+alignment"), default="type",
                   help="as in player_style.py: 'type+alignment' treats a route type run from another "
                        "alignment as a different route (outputs go to supervised/<population>_alignment/)")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--folds", nargs="+", default=["odd", "even"], choices=("odd", "even"),
                   help="the half-season the model is trained on")
    p.add_argument("--epochs-mlp", type=int, default=60, help="longest MLP training tried")
    p.add_argument("--epochs-tf", type=int, default=30, help="longest Transformer training tried")
    p.add_argument("--inner-frac", type=float, default=0.2, help="share of training games held out for tuning")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="learn from this random share of the training half-season's routes (all methods; "
                        "outputs go to supervised/<population>_frac<percent>/)")
    p.add_argument("--reps", type=int, default=20, help="random route sets per receiver, size and fold")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--force", action="store_true", help="retrain even when cached scores exist")
    p.add_argument("--shuffle-control", action="store_true",
                   help="also train mlp_z with receiver labels shuffled within route type")
    p.add_argument("--no-eval", action="store_true", help="train and cache only")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- models

class FeatureClassifier(nn.Module):
    """MLP on a fixed feature vector and the route type."""

    def __init__(self, n_features: int, n_types: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.n_types = n_types
        self.net = nn.Sequential(nn.Linear(n_features + n_types, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, features: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, nn.functional.one_hot(type_id, self.n_types).float()], dim=1))


class RouteClassifier(nn.Module):
    """The autoencoder's Transformer encoder with a classification head. It reads the [CLS] state
    and the average of the frame states, so it is not held to the 16-number bottleneck."""

    def __init__(self, backbone: RouteAutoencoder, n_types: int, n_classes: int, type_dim: int = 16,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        width = backbone.config["d_model"]
        self.type_emb = nn.Embedding(n_types, type_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * width + type_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x, valid, knocked_out, type_id) -> torch.Tensor:
        states = self.backbone.enc_norm(self.backbone.hidden_states(x, valid, knocked_out))
        frames = (states[:, 1:] * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        pooled = self.drop(torch.cat([states[:, 0], frames], dim=1))
        return self.head(torch.cat([pooled, self.type_emb(type_id)], dim=1))


class FeatureData:
    """Standardized features for every route of the population."""

    def __init__(self, features: np.ndarray, fit_rows: np.ndarray):
        mean, std = features[fit_rows].mean(0), features[fit_rows].std(0) + 1e-9
        self.F = torch.from_numpy(((features - mean) / std).astype(np.float32))
        self.n_features = self.F.shape[1]

    def batches(self, rows: np.ndarray, rng: np.random.Generator | None, batch_size: int = 512):
        order = rng.permutation(rows) if rng is not None else rows
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    def logits(self, model, rows: np.ndarray, type_id: torch.Tensor, rng=None) -> torch.Tensor:
        idx = torch.from_numpy(rows)
        return model(self.F[idx], type_id[idx])


class SequenceData:
    """The encoder's input channels for every route of the population."""

    def __init__(self, runner, rows: np.ndarray):
        idx = torch.from_numpy(rows)
        self.X = runner.X[idx][..., runner.in_idx].contiguous()
        self.valid = runner.valid[idx]
        self.lengths = runner.lengths[rows]

    def batches(self, rows: np.ndarray, rng: np.random.Generator | None, batch_size: int = 256):
        if rng is None:                                     # prediction: similar lengths together
            order = rows[np.argsort(self.lengths[rows], kind="stable")]
            return [order[i:i + 1024] for i in range(0, len(order), 1024)]
        return epoch_batches(rows, self.lengths, batch_size, rng)

    def logits(self, model, rows: np.ndarray, type_id: torch.Tensor, rng=None) -> torch.Tensor:
        """With rng (training) half of the routes get a knocked-out window, as augmentation."""
        used = int(self.lengths[rows].max())
        idx = torch.from_numpy(rows)
        if rng is None:
            knocked = torch.zeros(len(rows), used, dtype=torch.bool)
        else:
            knocked = torch.from_numpy(training_knockout(self.lengths[rows], rng, used, whole_prob=0.5))
        return model(self.X[idx, :used], self.valid[idx, :used], knocked, type_id[idx])


# ----------------------------------------------------------------------------- training

def log_prior(type_id: np.ndarray, y: np.ndarray, n_types: int, n_classes: int, alpha: float = 1.0) -> torch.Tensor:
    """log p(receiver | route type) from the training routes, (types, receivers), smoothed."""
    counts = np.zeros((n_types, n_classes))
    np.add.at(counts, (type_id, y), 1.0)
    counts += alpha
    return torch.from_numpy(np.log(counts / counts.sum(1, keepdims=True)).astype(np.float32))


@torch.no_grad()
def predict(model, data, rows: np.ndarray, type_id: torch.Tensor) -> np.ndarray:
    """Raw logits (no prior) for the given rows, in their order."""
    model.eval()
    out = np.empty((len(rows), 0), dtype=np.float32)
    position = {int(r): i for i, r in enumerate(rows)}
    for batch in data.batches(rows, None):
        logits = data.logits(model, batch, type_id).numpy()
        if out.shape[1] == 0:
            out = np.empty((len(rows), logits.shape[1]), dtype=np.float32)
        out[[position[int(r)] for r in batch]] = logits
    return out


def balanced_top1(scores: np.ndarray, truth: np.ndarray) -> float:
    hit = pd.Series(scores.argmax(1) == truth)
    return float(hit.groupby(truth).mean().mean())


def train_model(model, data, y, type_id, prior, rows, epochs: int, groups: list[dict], seed: int,
                val_rows: np.ndarray | None = None) -> dict:
    """Train with the logit-adjusted cross-entropy. With val_rows, the held-out loss is recorded
    after every epoch and the logits of the best epoch are kept."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(groups)
    base = [g["lr"] for g in optimizer.param_groups]
    step, warmup = 0, 100
    history, best = [], dict(loss=float("inf"), epoch=0, scores=None)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in data.batches(rows, rng):
            for group, lr in zip(optimizer.param_groups, base):
                group["lr"] = lr * min(1.0, (step + 1) / warmup)
            idx = torch.from_numpy(batch)
            loss = nn.functional.cross_entropy(data.logits(model, batch, type_id, rng) + prior[type_id[idx]], y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
        if val_rows is not None:
            scores = predict(model, data, val_rows, type_id)
            idx = torch.from_numpy(val_rows)
            val_loss = float(nn.functional.cross_entropy(torch.from_numpy(scores) + prior[type_id[idx]], y[idx]))
            history.append(val_loss)
            if val_loss < best["loss"]:
                best = dict(loss=val_loss, epoch=epoch, scores=scores)
    return dict(model=model, history=history, best=best)


def combine(scores: np.ndarray, rule: str) -> np.ndarray:
    """Evidence of several routes for each receiver. 'product': add the log-likelihoods.
    'mean': average the per-route probabilities (less swayed by one overconfident route)."""
    if rule == "product":
        return scores.sum(0)
    shifted = scores - scores.max(1, keepdims=True)
    prob = np.exp(shifted)
    return (prob / prob.sum(1, keepdims=True)).mean(0)


def choose_rule(scores: np.ndarray, truth: np.ndarray, rng, n: int = 10, reps: int = 10) -> dict:
    """Accuracy of both ways of combining n held-out routes; the better one is used on the test half."""
    acc = {"product": [], "mean": []}
    for c in np.unique(truth):
        rows = np.flatnonzero(truth == c)
        if len(rows) < n:
            continue
        for _ in range(reps):
            pick = rng.choice(rows, size=n, replace=False)
            for rule in acc:
                acc[rule].append(combine(scores[pick], rule).argmax() == c)
    out = {rule: float(np.mean(v)) if v else float("nan") for rule, v in acc.items()}
    out["rule"] = "mean" if out["mean"] > out["product"] else "product"
    return out


def tuning_split(ctx: dict, train_half: str, seed: int, inner_frac: float):
    """Candidates' routes of the training half, split by game into a part to fit on and a part
    held out for tuning. Returns (all training rows, fit rows, held-out rows, the generator)."""
    pop = ctx["pop"]
    usable = (ctx["cand_idx"] >= 0) & (ctx["type_code"] >= 0) & ctx["keep"]
    train_rows = np.flatnonzero(usable & (pop["half"] == train_half))
    rng = np.random.default_rng(seed + (0 if train_half == "odd" else 1))
    games = rng.permutation(np.unique(pop["game"][train_rows]))
    held = np.isin(pop["game"][train_rows], games[:max(1, int(round(inner_frac * len(games))))])
    return train_rows, train_rows[~held], train_rows[held], rng


def usage_only_loss(ctx: dict, train_half: str, seed: int, inner_frac: float) -> float:
    """Held-out loss of a model that knows only how often each receiver runs each route type
    (flat raw logits). A classifier has learned something about HOW routes are run only as far
    as its held-out loss is below this."""
    _, inner_rows, val_rows, _ = tuning_split(ctx, train_half, seed, inner_frac)
    code, label = ctx["type_code"], ctx["cand_idx"]
    prior = log_prior(code[inner_rows], label[inner_rows], len(ctx["type_list"]), len(ctx["cands"]))
    idx = torch.from_numpy(val_rows)
    return float(nn.functional.cross_entropy(prior[ctx["type_id"][idx]], ctx["y"][idx]))


def fit_fold(name: str, ctx: dict, train_half: str, test_rows: np.ndarray, args) -> dict:
    """Tune on held-out games of the training half, retrain on all of it, score the test routes."""
    started = time.time()
    pop, n_types, n_cand = ctx["pop"], len(ctx["type_list"]), len(ctx["cands"])
    train_rows, inner_rows, val_rows, rng = tuning_split(ctx, train_half, args.seed, args.inner_frac)
    y, type_id = ctx["y"], ctx["type_id"]
    code, label = ctx["type_code"], ctx["cand_idx"]

    if name.startswith("mlp"):
        data = FeatureData(ctx["features"][name], train_rows)
        configs, epochs = MLP_GRID, args.epochs_mlp

        def build(cfg):
            model = FeatureClassifier(data.n_features, n_types, n_cand, cfg["hidden"], cfg["dropout"])
            return model, [dict(params=model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])]
    else:
        data = SequenceData(pop["runner"], pop["rows"])
        configs, epochs = [dict(init="autoencoder" if name == "tf_finetune" else "random")], args.epochs_tf

        def build(cfg):
            pretrained = pop["runner"].model
            backbone = copy.deepcopy(pretrained) if cfg["init"] == "autoencoder" else RouteAutoencoder(**pretrained.config)
            model = RouteClassifier(backbone, n_types, n_cand)
            encoder_lr = 3e-4 if cfg["init"] == "autoencoder" else 1e-3
            head = list(model.type_emb.parameters()) + list(model.head.parameters())
            return model, [dict(params=model.backbone.parameters(), lr=encoder_lr, weight_decay=0.01),
                           dict(params=head, lr=1e-3, weight_decay=0.05)]

    prior = log_prior(code[inner_rows], label[inner_rows], n_types, n_cand)
    tried, chosen = [], None
    for cfg in configs:
        model, groups = build(cfg)
        run = train_model(model, data, y, type_id, prior, inner_rows, epochs, groups, args.seed, val_rows)
        tried.append(dict(config=cfg, best_epoch=run["best"]["epoch"], best_val_loss=run["best"]["loss"],
                          val_loss_by_epoch=[round(v, 4) for v in run["history"]]))
        if chosen is None or run["best"]["loss"] < chosen["best"]["loss"]:
            chosen = dict(run, config=cfg)
        print(f"  [{name} / train {train_half}] {cfg}: best epoch {run['best']['epoch']} of {epochs}, "
              f"held-out loss {run['best']['loss']:.4f}", flush=True)
    val_truth = label[val_rows]
    rule = choose_rule(chosen["best"]["scores"], val_truth, rng)
    held_out_top1 = balanced_top1(chosen["best"]["scores"], val_truth)

    model, groups = build(chosen["config"])
    prior = log_prior(code[train_rows], label[train_rows], n_types, n_cand)
    final = train_model(model, data, y, type_id, prior, train_rows, chosen["best"]["epoch"], groups, args.seed)
    sample = np.random.default_rng(0).choice(train_rows, size=min(5000, len(train_rows)), replace=False)
    seen_top1 = balanced_top1(predict(final["model"], data, sample, type_id), label[sample])
    scores = predict(final["model"], data, test_rows, type_id)
    meta = dict(model=name, train_half=train_half, train_routes=int(len(train_rows)),
                tuning_routes=int(len(val_rows)), config=chosen["config"], epochs=chosen["best"]["epoch"],
                combine=rule, top1_one_route_held_out_games=held_out_top1, top1_one_route_training_routes=seen_top1,
                tried=tried, minutes=(time.time() - started) / 60)
    print(f"  [{name} / train {train_half}] one route: {seen_top1:.1%} on training routes, {held_out_top1:.1%} on "
          f"held-out games of the same half; combining by {rule['rule']}; {meta['minutes']:.1f} min", flush=True)
    return dict(scores=scores, meta=meta)


# ----------------------------------------------------------------------------- evaluation

def evaluate(ctx: dict, supervised: dict, bag_sizes, reps: int, seed: int) -> dict:
    """Every method on the same test routes and the same random sets of n routes.

    supervised: {model: {train_half: dict(scores, rows, meta)}}. Returns per method and n the
    receiver-balanced accuracy among all candidates, his position and his team."""
    pop, cand_idx, n_cand = ctx["pop"], ctx["cand_idx"], len(ctx["cands"])
    steps = list(bag_sizes) + ["all"]
    references = REFERENCES if ctx["keep"].all() else ("re_z",)      # with less data only the main reference
    names = list(references) + list(supervised)
    sums = {m: {n: np.zeros((n_cand, 3)) for n in steps} for m in names}
    cnts = {m: {n: np.zeros((n_cand, 3)) for n in steps} for m in names}
    top5 = {m: [] for m in names}
    rng = np.random.default_rng(seed)
    per_half = []
    for train_half in ("odd", "even"):
        train, test = (pop["half"] == train_half) & ctx["keep"], pop["half"] != train_half
        folds = {ref: Fold(ctx["features"][feat], pop["types"], pop["player"], pop["game"], cand_idx, train, test,
                           ctx["type_list"], n_cand)
                 for ref, feat in (("re_z", "mlp_z"), ("re_all", "mlp_all")) if ref in references}
        if "re_z_shared" in references:
            # One profile per receiver for all route types: centre z on its route type (means from
            # the training half) and fit a single model to every route.
            z, code = ctx["features"]["mlp_z"], ctx["type_code"]
            centred = z.copy()
            for t in range(len(ctx["type_list"])):
                centred[code == t] -= z[train & (code == t)].mean(0)
            folds["re_z_shared"] = Fold(centred, np.where(code >= 0, "ALL", "OTHER"), pop["player"], pop["game"],
                                        cand_idx, train, test, ["ALL"], n_cand)
        rows, truth = folds["re_z"].rows, folds["re_z"].truth
        if not all(np.array_equal(rows, fold.rows) for fold in folds.values()):
            raise SystemExit("The reference models do not cover the same test routes")
        single = {ref: fold.single_scores() for ref, fold in folds.items()}
        bag = {ref: fold.bag for ref, fold in folds.items()}
        for m, by_half in supervised.items():
            if not np.array_equal(by_half[train_half]["rows"], rows):
                raise SystemExit(f"Cached scores of {m} are for other routes; retrain with --force")
            single[m] = by_half[train_half]["scores"]
            rule = by_half[train_half]["meta"]["combine"]["rule"]
            bag[m] = lambda pick, S=single[m], rule=rule: combine(S[pick], rule)

        def record(m, n, c, scores):
            hit = np.array([float(np.argmax(scores) == c), restricted_hit(scores, c, ctx["cand_team"]),
                            restricted_hit(scores, c, ctx["cand_pos"])])
            ok = ~np.isnan(hit)
            sums[m][n][c] += np.nan_to_num(hit)
            cnts[m][n][c] += ok

        for m in names:
            order = np.argsort(-single[m], axis=1, kind="stable")
            rank = (order == truth[:, None]).argmax(1)
            top5[m].append(pd.DataFrame(dict(cand=truth, top5=rank < 5)))
            for i in range(len(rows)):
                record(m, 1, truth[i], single[m][i])
        by_cand = pd.Series(np.arange(len(rows))).groupby(truth).apply(lambda s: s.to_numpy())
        per_half.append(by_cand.map(len))
        for c, local in by_cand.items():
            for n in steps:
                if n == 1:
                    continue
                if n == "all":
                    draws = [local]
                elif len(local) >= n:
                    draws = [rng.choice(local, size=n, replace=False) for _ in range(reps)]
                else:
                    continue
                for pick in draws:
                    for m in names:
                        record(m, n, c, bag[m](pick))

    def summary(values: np.ndarray) -> dict:
        values = values[~np.isnan(values)]
        se = values.std(ddof=1) / np.sqrt(len(values))
        return dict(value=float(values.mean()), ci=[float(values.mean() - 1.96 * se), float(values.mean() + 1.96 * se)],
                    receivers=int(len(values)))

    out = {}
    with np.errstate(invalid="ignore"):
        rate = {m: {n: sums[m][n] / cnts[m][n] for n in steps} for m in names}
    for m in names:
        by_n = {}
        for n in steps:
            entry = {key: summary(rate[m][n][:, j]) for j, key in enumerate(("top1", "team", "position"))}
            if m != "re_z":
                entry["minus_re_z"] = summary(rate[m][n][:, 0] - rate["re_z"][n][:, 0])
            if "mlp_z" in rate and m != "mlp_z":
                entry["minus_mlp_z"] = summary(rate[m][n][:, 0] - rate["mlp_z"][n][:, 0])
            by_n[str(n)] = entry
        five = pd.concat(top5[m]).groupby("cand").top5.mean().mean()
        out[m] = dict(label=LABELS[m], supervised=m in supervised, by_routes=by_n, top5_one_route=float(five))
    team, position = ctx["cand_team"], ctx["cand_pos"]
    chance = dict(top1=1 / n_cand, top5=5 / n_cand,
                  team=float(np.mean([1 / (team == t).sum() for t in team if (team == t).sum() > 1])),
                  position=float(np.mean([1 / (position == t).sum() for t in position if (position == t).sum() > 1])))
    return dict(methods=out, chance=chance, steps=[str(n) for n in steps],
                routes_per_half_season=float(pd.concat(per_half).mean()))


def plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = report["steps"]
    ticks = [s if s != "all" else "half\nseason" for s in steps]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title in ((axs[0], "top1", f"among all {report['candidates']} candidates"),
                           (axs[1], "team", "among his own teammates")):
        for name in [m for m in COLORS if m in report["methods"]]:
            res = report["methods"][name]
            ys = [res["by_routes"][n][key]["value"] for n in steps]
            ax.plot(range(len(steps)), ys, "o-" if res["supervised"] else "s--", color=COLORS[name],
                    label=res["label"], lw=2.4 if name in ("re_z", "mlp_z") else 1.4)
            if name == "re_z":
                ci = np.array([res["by_routes"][n][key]["ci"] for n in steps])
                ax.fill_between(range(len(steps)), ci[:, 0], ci[:, 1], color=COLORS[name], alpha=0.15)
        ax.axhline(report["chance"][key], color="black", ls=":", lw=1, label=f"chance ({report['chance'][key]:.1%})")
        ax.set_xticks(range(len(steps)), ticks)
        ax.set_xlabel("routes from the unknown receiver")
        ax.set_ylabel("receiver named correctly")
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7.5)
    fig.suptitle("Who ran these routes? Classifiers trained on receiver labels (solid) and the random-effects "
                 f"model (dashed); {report['population']} routes", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def setup(args) -> dict:
    cfg = DEFAULTS[args.population]
    pop = load_population(args.run_dir, args.data_dir, args.population, args.condition)
    cands, type_list = choose_sets(pop, cfg["min_per_half"], 20)
    pop["cands"] = cands
    cand_idx = np.where(np.isin(pop["player"], cands), np.searchsorted(cands, pop["player"]), -1)
    lookup = {t: i for i, t in enumerate(type_list)}
    type_code = np.array([lookup.get(t, -1) for t in pop["types"]])
    f = pop["features"]
    keep = np.random.default_rng(args.seed + 7).random(len(cand_idx)) < args.train_frac
    return dict(pop=pop, cands=cands, type_list=type_list, cand_idx=cand_idx, type_code=type_code, keep=keep,
                bag_sizes=cfg["bag_sizes"],
                y=torch.from_numpy(np.maximum(cand_idx, 0)), type_id=torch.from_numpy(np.maximum(type_code, 0)),
                features={"mlp_z": f["z"], "mlp_all": np.hstack([f["z"], f["pca16"], f["positions"], f["speed"]]),
                          "mlp_positions": f["positions"], "mlp_pca16": f["pca16"], "mlp_speed": f["speed"]},
                cand_team=pop["primary_team"].reindex(cands).to_numpy(),
                cand_pos=pop["primary_pos"].reindex(cands).to_numpy())


def test_routes(ctx: dict, train_half: str) -> np.ndarray:
    """The test routes the random-effects model scores for this fold (candidates, usable types)."""
    pop = ctx["pop"]
    train, test = (pop["half"] == train_half) & ctx["keep"], pop["half"] != train_half
    return Fold(ctx["features"]["mlp_z"], pop["types"], pop["player"], pop["game"], ctx["cand_idx"], train, test,
                ctx["type_list"], len(ctx["cands"])).rows


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = args.run_dir / "supervised" / (args.population + ("_alignment" if args.condition != "type" else "")
                                             + (f"_frac{round(100 * args.train_frac)}" if args.train_frac < 1 else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = setup(args)
    print(f"{args.population}: {len(ctx['pop']['rows'])} routes, {len(ctx['cands'])} candidates, "
          f"route types {ctx['type_list']}", flush=True)

    for name in args.models:
        for train_half in args.folds:
            path = out_dir / f"{name}_{train_half}.npz"
            if path.exists() and not args.force:
                continue
            rows = test_routes(ctx, train_half)
            result = fit_fold(name, ctx, train_half, rows, args)
            np.savez_compressed(path, scores=result["scores"], rows=rows, meta=json.dumps(result["meta"]))
    if args.shuffle_control:
        # Break the link between receiver and route in training; test labels stay as they are.
        shuffler = np.random.default_rng(args.seed + 100)
        labels = ctx["cand_idx"].copy()
        usable = (labels >= 0) & (ctx["type_code"] >= 0)
        for t in range(len(ctx["type_list"])):
            sel = np.flatnonzero(usable & (ctx["type_code"] == t))
            labels[sel] = labels[shuffler.permutation(sel)]
        shuffled = {**ctx, "cand_idx": labels, "y": torch.from_numpy(np.maximum(labels, 0)),
                    "features": {**ctx["features"], CONTROL: ctx["features"]["mlp_z"]}}
        for train_half in args.folds:
            path = out_dir / f"{CONTROL}_{train_half}.npz"
            if path.exists() and not args.force:
                continue
            rows = test_routes(ctx, train_half)
            result = fit_fold(CONTROL, shuffled, train_half, rows, args)
            np.savez_compressed(path, scores=result["scores"], rows=rows, meta=json.dumps(result["meta"]))
    if args.no_eval:
        return

    supervised = {}
    for name in MODELS + (CONTROL,):
        paths = {h: out_dir / f"{name}_{h}.npz" for h in ("odd", "even")}
        if all(p.exists() for p in paths.values()):
            supervised[name] = {}
            for h, p in paths.items():
                with np.load(p) as d:
                    supervised[name][h] = dict(scores=d["scores"], rows=d["rows"], meta=json.loads(str(d["meta"])))
    report = dict(population=args.population, condition=args.condition, train_frac=args.train_frac,
                  candidates=int(len(ctx["cands"])),
                  route_types=ctx["type_list"],
                  **evaluate(ctx, supervised, ctx["bag_sizes"], args.reps, args.seed),
                  usage_only_held_out_loss={h: usage_only_loss(ctx, h, args.seed, args.inner_frac)
                                            for h in ("odd", "even")},
                  training={m: {h: v["meta"] for h, v in by_half.items()} for m, by_half in supervised.items()})
    with open(out_dir / "supervised.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if not args.no_plots:
        plot(report, out_dir / "supervised_curve.png")

    steps = report["steps"]
    print(f"\nreceiver named correctly, by routes from the unknown receiver (chance {report['chance']['top1']:.2%})")
    print(f"{'':14s}" + "".join(f"{s:>8s}" for s in steps))
    for name, res in report["methods"].items():
        print(f"{name:14s}" + "".join(f"{res['by_routes'][s]['top1']['value']:8.1%}" for s in steps))
    print("\nsupervised minus z + random-effects, same receivers and route sets (95% interval)")
    for name, res in report["methods"].items():
        if name != "re_z":
            print(f"{name:14s}" + "  ".join(
                f"{s}: {res['by_routes'][s]['minus_re_z']['value']:+.1%} "
                f"[{res['by_routes'][s]['minus_re_z']['ci'][0]:+.1%}, {res['by_routes'][s]['minus_re_z']['ci'][1]:+.1%}]"
                for s in steps))
    for name, by_half in report["training"].items():
        for h, meta in by_half.items():
            best = min(t["best_val_loss"] for t in meta["tried"])
            print(f"{name} (train {h}): {meta['epochs']} epochs, {meta['config']}; held-out loss {best:.3f} "
                  f"(usage only: {report['usage_only_held_out_loss'][h]:.3f}); one route "
                  f"{meta['top1_one_route_training_routes']:.1%} on training routes vs "
                  f"{meta['top1_one_route_held_out_games']:.1%} on held-out games; combined by "
                  f"{meta['combine']['rule']}")


if __name__ == "__main__":
    main()
