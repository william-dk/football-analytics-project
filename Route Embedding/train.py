"""Train the masked-window route autoencoder.

Each training step knocks out one random window per route (8-15 frames by default), encodes the
rest into z and rebuilds the whole route from z alone. A share of the routes (--whole-prob) is
shown whole, so that encoding a complete route, which is what every later analysis does, is
something the model has seen in training. Validation uses one fixed set of windows,
so its loss is comparable across epochs. The best checkpoint (lowest validation objective) is
kept in --out-dir together with history.csv and summary.json.

By default the model trains on every sequence kind in the data: the targeted full routes and the
pre-throw paths of all route runners. --kinds full reproduces the targeted-only setting.

    python train.py --name v2
    python train.py --name targeted_only --kinds full --epochs 60
    python train.py --name smoke --epochs 2 --limit-batches 5   # quick check

The first version's model is --readout cls --decoder transformer --inputs x y s a --lr 1e-3
--vel-weight 0 --whole-prob 0 --kinds full --no-bucket --epochs 80.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from masking import sample_windows, training_knockout, window_mask
from model import (DECODERS, READOUTS, RouteAutoencoder, count_parameters, masked_mse, run_batches,
                   velocity_mse)

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    p.add_argument("--name", default="v2", help="run folder name under --runs-dir")
    p.add_argument("--kinds", nargs="+", default=None,
                   help="sequence kinds to train on: full, pre (default: every kind in the data)")
    p.add_argument("--inputs", nargs="+", default=["x", "y", "vx", "vy", "a"],
                   help="encoder channels. vx, vy are the frame-to-frame change of x, y, so they add no "
                        "information, but they make the encoder far more precise for the same training")
    p.add_argument("--targets", nargs="+", default=["x", "y"], help="reconstructed channels")
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ff", type=int, default=128)
    p.add_argument("--readout", choices=READOUTS, default="frames",
                   help="how z is read from the encoder: [CLS] only, or [CLS] plus every frame state")
    p.add_argument("--decoder", choices=DECODERS, default="mlp",
                   help="how z is turned back into a route")
    p.add_argument("--dec-hidden", type=int, default=256, help="width of the mlp decoder correction")
    p.add_argument("--basis", type=int, default=32,
                   help="eigen-routes the linear/mlp decoder writes with (0 = one free output per frame)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="0 by default: CPU dropout sampling cost ~60%% of each step, and the random "
                        "window knockout already acts as augmentation")
    p.add_argument("--win-min", type=int, default=8)
    p.add_argument("--win-max", type=int, default=15)
    p.add_argument("--max-mask-frac", type=float, default=0.5)
    p.add_argument("--whole-prob", type=float, default=0.2,
                   help="share of training routes shown whole (no knocked-out window)")
    p.add_argument("--visible-weight", type=float, default=0.5)
    p.add_argument("--vel-weight", type=float, default=0.1,
                   help="weight of the frame-to-frame step (velocity) error; 0 = positions only")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--patience", type=int, default=15, help="epochs without validation improvement")
    p.add_argument("--no-bucket", action="store_true",
                   help="draw batches at random and pad every batch to the full length (slower)")
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--limit-batches", type=int, default=0, help="debug: batches per epoch (0 = all)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_data(data_dir: Path) -> dict:
    with np.load(data_dir / "sequences.npz") as d:
        data = {k: d[k] for k in d.files}
    if "kind" not in data:                       # targeted-only data from the first version
        data["kind"] = np.full(len(data["lengths"]), "full")
    return data


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def epoch_batches(rows: np.ndarray, lengths: np.ndarray, batch_size: int, rng: np.random.Generator,
                  bucket: bool = True, chunk: int = 20) -> list[np.ndarray]:
    """Shuffled batches for one epoch. With bucketing, each shuffled chunk of `chunk` batches is
    sorted by route length before it is cut, so a batch holds routes of similar length and can be
    cut to its longest route. Batch order is shuffled again afterwards."""
    order = rng.permutation(rows)
    if not bucket:
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    batches = []
    for i in range(0, len(order), batch_size * chunk):
        part = order[i:i + batch_size * chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    return [batches[k] for k in rng.permutation(len(batches))]


def eigen_routes(target: np.ndarray, lengths: np.ndarray, components: int):
    """Principal components of the training routes (frames after a route ends hold its last
    values). target is (N, T, C) in scaled units. Returns (mean, orthonormal components, the
    standard deviation of each coefficient), flattened time-major to match the decoder output."""
    from sklearn.decomposition import PCA
    idx = np.minimum(np.arange(target.shape[1])[None, :], lengths[:, None] - 1)
    flat = np.take_along_axis(target, idx[:, :, None], axis=1).reshape(len(target), -1)
    pca = PCA(n_components=components).fit(flat)
    return pca.mean_, pca.components_, np.sqrt(pca.explained_variance_)


def split_metrics(model, x, valid, knocked_out, target, pos_scale, xy_idx, visible_weight, vel_weight=0.0):
    """Validation loss terms plus mean Euclidean error (yards) on knocked-out and visible frames."""
    pred, _ = run_batches(model, x, valid, knocked_out)
    objective, hidden_loss, shown_loss = masked_mse(pred, target, valid, knocked_out, visible_weight)
    out = dict(hidden_loss=float(hidden_loss), shown_loss=float(shown_loss))
    if vel_weight:
        vel = velocity_mse(pred, target, valid, knocked_out, visible_weight)
        objective = objective + vel_weight * vel
        out["vel_loss"] = float(vel)
    out = dict(objective=float(objective), **out)
    if xy_idx is not None:
        dist = (pred[..., xy_idx] - target[..., xy_idx]).pow(2).sum(-1).sqrt() * pos_scale
        hidden, shown = knocked_out & valid, valid & ~knocked_out
        out["hidden_err_yd"] = float((dist * hidden).sum() / hidden.sum())
        out["shown_err_yd"] = float((dist * shown).sum() / shown.sum())
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.runs_dir / args.name
    if (out_dir / "model.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has a model; choose another --name or pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_data(args.data_dir)
    channels = [str(c) for c in data["channels"]]
    in_idx = [channels.index(c) for c in args.inputs]
    out_idx = [channels.index(c) for c in args.targets]
    xy_idx = [args.targets.index("x"), args.targets.index("y")] if {"x", "y"} <= set(args.targets) else None
    scales = data["scales"].astype(np.float32)
    pos_scale = float(scales[channels.index("x")])
    X = torch.from_numpy(data["X"] / scales)
    lengths = data["lengths"].astype(np.int64)
    steps = X.shape[1]
    valid = torch.from_numpy(np.arange(steps)[None, :] < lengths[:, None])
    split, kind = data["split"], data["kind"]
    kinds = args.kinds or sorted(set(kind.tolist()))
    if not set(kinds) <= set(kind.tolist()):
        raise SystemExit(f"--kinds {kinds} not all present in the data ({sorted(set(kind.tolist()))})")
    use = np.isin(kind, kinds)
    train_rows = np.flatnonzero(use & (split == "train"))
    val_np = np.flatnonzero(use & (split == "val"))
    val_np = val_np[np.argsort(lengths[val_np], kind="stable")]     # sorted: faster trimmed evaluation
    val_rows = torch.from_numpy(val_np)

    val_starts, val_widths = sample_windows(lengths[val_np], np.random.default_rng(args.seed + 1),
                                            args.win_min, args.win_max, args.max_mask_frac)
    val_mask = torch.from_numpy(window_mask(val_starts, val_widths, steps))
    val_x, val_valid = X[val_rows][..., in_idx], valid[val_rows]
    val_target = X[val_rows][..., out_idx]

    model = RouteAutoencoder(len(in_idx), len(out_idx), max_len=steps, d_model=args.d_model,
                             n_heads=args.heads, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                             z_dim=args.z_dim, dim_ff=args.ff, dropout=args.dropout, readout=args.readout,
                             decoder=args.decoder, dec_hidden=args.dec_hidden, basis=args.basis)
    if model.basis_size:
        model.set_basis(*eigen_routes(X[torch.from_numpy(train_rows)][..., out_idx].numpy(),
                                      lengths[train_rows], model.basis_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches = math.ceil(len(train_rows) / args.batch_size)
    if args.limit_batches:
        batches = min(batches, args.limit_batches)
    total, warmup = args.epochs * batches, max(1, min(args.warmup_steps, args.epochs * batches // 4))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    print(f"kinds={kinds} routes train={len(train_rows)} val={len(val_rows)} | params={count_parameters(model):,} "
          f"| inputs={args.inputs} targets={args.targets} z={args.z_dim} readout={args.readout} "
          f"decoder={args.decoder} basis={model.basis_size} window={args.win_min}-{args.win_max} "
          f"whole_prob={args.whole_prob} vel_weight={args.vel_weight} | {batches} batches/epoch | threads={torch.get_num_threads()}", flush=True)

    def save_history(history: list[dict]) -> None:
        with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    history, best, best_epoch, stale = [], math.inf, 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tick = time.time()
        sums = np.zeros(4)
        plan = epoch_batches(train_rows, lengths, args.batch_size, rng, bucket=not args.no_bucket)[:batches]
        for rows in plan:
            used = steps if args.no_bucket else int(lengths[rows].max())
            knocked_out = torch.from_numpy(training_knockout(lengths[rows], rng, used, args.win_min, args.win_max,
                                                             args.max_mask_frac, args.whole_prob))
            idx = torch.from_numpy(rows)
            x, v = X[idx, :used], valid[idx, :used]
            target = x[..., out_idx]
            pred, _ = model(x[..., in_idx], v, knocked_out)
            loss, hidden_loss, shown_loss = masked_mse(pred, target, v, knocked_out, args.visible_weight)
            vel_loss = torch.zeros(())
            if args.vel_weight:
                vel_loss = velocity_mse(pred, target, v, knocked_out, args.visible_weight)
                loss = loss + args.vel_weight * vel_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sums += [loss.item(), hidden_loss.item(), shown_loss.item(), vel_loss.item()]
        val = split_metrics(model, val_x, val_valid, val_mask, val_target, pos_scale, xy_idx,
                            args.visible_weight, args.vel_weight)
        row = dict(epoch=epoch, lr=scheduler.get_last_lr()[0], train_objective=sums[0] / len(plan),
                   train_hidden_loss=sums[1] / len(plan), train_shown_loss=sums[2] / len(plan),
                   train_vel_loss=sums[3] / len(plan),
                   **{f"val_{k}": v for k, v in val.items()}, seconds=time.time() - tick)
        history.append(row)
        improved = val["objective"] < best - 1e-6
        if improved:
            best, best_epoch, stale = val["objective"], epoch, 0
            torch.save(dict(state_dict=model.state_dict(), model_config=model.config, inputs=args.inputs,
                            targets=args.targets, channels=channels, scales=scales.tolist(), kinds=kinds,
                            train_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                            epoch=epoch, val_metrics=val), out_dir / "model.pt")
        else:
            stale += 1
        save_history(history)
        print(f"epoch {epoch:3d} | train {row['train_objective']:.4f} | val {val['objective']:.4f} "
              f"| window err {val.get('hidden_err_yd', float('nan')):.3f} yd "
              f"| visible err {val.get('shown_err_yd', float('nan')):.3f} yd "
              f"| {row['seconds']:.1f}s{' *' if improved else ''}", flush=True)
        if stale >= args.patience:
            print(f"Stopping: no validation improvement for {args.patience} epochs.", flush=True)
            break

    summary = dict(best_epoch=best_epoch, best_val=history[best_epoch - 1], epochs_run=len(history),
                   minutes=(time.time() - started) / 60, parameters=count_parameters(model),
                   kinds=kinds, train_routes=int(len(train_rows)), val_routes=int(len(val_rows)),
                   gradient_steps=int(len(history) * batches), model_config=model.config,
                   args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   torch=torch.__version__, python=platform.python_version(), git_commit=git_commit())
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Best epoch {best_epoch}: {json.dumps(history[best_epoch - 1])}", flush=True)


if __name__ == "__main__":
    main()
