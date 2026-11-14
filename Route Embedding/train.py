"""Train the masked-window route autoencoder on the targeted full paths.

Each training step knocks out one random window per route (8-15 frames by default), encodes the
rest into z and rebuilds the whole route from z alone. Validation uses one fixed set of windows,
so its loss is comparable across epochs. The best checkpoint (lowest validation objective) is
kept in --out-dir together with history.csv and summary.json.

    python train.py --name v1
    python train.py --name smoke --epochs 2 --limit-batches 5   # quick check
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from masking import sample_windows, window_mask
from model import RouteAutoencoder, count_parameters, masked_mse, run_batches

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    p.add_argument("--name", default="v1", help="run folder name under --runs-dir")
    p.add_argument("--inputs", nargs="+", default=["x", "y", "s", "a"], help="encoder channels")
    p.add_argument("--targets", nargs="+", default=["x", "y"], help="reconstructed channels")
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="0 by default: CPU dropout sampling cost ~60%% of each step, and the random "
                        "window knockout already acts as augmentation")
    p.add_argument("--win-min", type=int, default=8)
    p.add_argument("--win-max", type=int, default=15)
    p.add_argument("--max-mask-frac", type=float, default=0.5)
    p.add_argument("--visible-weight", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=30, help="epochs without validation improvement")
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--limit-batches", type=int, default=0, help="debug: batches per epoch (0 = all)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_data(data_dir: Path) -> dict:
    with np.load(data_dir / "sequences.npz") as d:
        return {k: d[k] for k in d.files}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def split_metrics(model, x, valid, knocked_out, target, pos_scale, xy_idx, visible_weight):
    """Validation loss terms plus mean Euclidean error (yards) on knocked-out and visible frames."""
    pred, _ = run_batches(model, x, valid, knocked_out)
    objective, hidden_loss, shown_loss = masked_mse(pred, target, valid, knocked_out, visible_weight)
    out = dict(objective=float(objective), hidden_loss=float(hidden_loss), shown_loss=float(shown_loss))
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
    split = data["split"]
    train_rows = np.flatnonzero(split == "train")
    val_rows = torch.from_numpy(np.flatnonzero(split == "val"))

    val_starts, val_widths = sample_windows(lengths[val_rows.numpy()], np.random.default_rng(args.seed + 1),
                                            args.win_min, args.win_max, args.max_mask_frac)
    val_mask = torch.from_numpy(window_mask(val_starts, val_widths, steps))
    val_x, val_valid = X[val_rows][..., in_idx], valid[val_rows]
    val_target = X[val_rows][..., out_idx]

    model = RouteAutoencoder(len(in_idx), len(out_idx), max_len=steps, d_model=args.d_model,
                             n_heads=args.heads, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                             z_dim=args.z_dim, dim_ff=args.ff, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches = math.ceil(len(train_rows) / args.batch_size)
    if args.limit_batches:
        batches = min(batches, args.limit_batches)
    total, warmup = args.epochs * batches, max(1, args.warmup_epochs * batches)

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    print(f"routes train={len(train_rows)} val={len(val_rows)} | params={count_parameters(model):,} | "
          f"inputs={args.inputs} targets={args.targets} z={args.z_dim} | threads={torch.get_num_threads()}",
          flush=True)

    history, best, best_epoch, stale = [], math.inf, 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tick = time.time()
        order = rng.permutation(train_rows)
        sums = np.zeros(3)
        for b in range(batches):
            rows = order[b * args.batch_size:(b + 1) * args.batch_size]
            starts, widths = sample_windows(lengths[rows], rng, args.win_min, args.win_max, args.max_mask_frac)
            knocked_out = torch.from_numpy(window_mask(starts, widths, steps))
            idx = torch.from_numpy(rows)
            x, v = X[idx], valid[idx]
            pred, _ = model(x[..., in_idx], v, knocked_out)
            loss, hidden_loss, shown_loss = masked_mse(pred, x[..., out_idx], v, knocked_out, args.visible_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sums += [loss.item(), hidden_loss.item(), shown_loss.item()]
        val = split_metrics(model, val_x, val_valid, val_mask, val_target, pos_scale, xy_idx, args.visible_weight)
        row = dict(epoch=epoch, lr=scheduler.get_last_lr()[0], train_objective=sums[0] / batches,
                   train_hidden_loss=sums[1] / batches, train_shown_loss=sums[2] / batches,
                   **{f"val_{k}": v for k, v in val.items()}, seconds=time.time() - tick)
        history.append(row)
        improved = val["objective"] < best - 1e-6
        if improved:
            best, best_epoch, stale = val["objective"], epoch, 0
            torch.save(dict(state_dict=model.state_dict(), model_config=model.config, inputs=args.inputs,
                            targets=args.targets, channels=channels, scales=scales.tolist(),
                            train_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                            epoch=epoch, val_metrics=val), out_dir / "model.pt")
        else:
            stale += 1
        print(f"epoch {epoch:3d} | train {row['train_objective']:.4f} | val {val['objective']:.4f} "
              f"| window err {val.get('hidden_err_yd', float('nan')):.3f} yd "
              f"| visible err {val.get('shown_err_yd', float('nan')):.3f} yd "
              f"| {row['seconds']:.1f}s{' *' if improved else ''}", flush=True)
        if stale >= args.patience:
            print(f"Stopping: no validation improvement for {args.patience} epochs.", flush=True)
            break

    import csv
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = dict(best_epoch=best_epoch, best_val=history[best_epoch - 1], epochs_run=len(history),
                   minutes=(time.time() - started) / 60, parameters=count_parameters(model),
                   model_config=model.config,
                   args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   torch=torch.__version__, python=platform.python_version(), git_commit=git_commit())
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Best epoch {best_epoch}: {json.dumps(history[best_epoch - 1])}", flush=True)


if __name__ == "__main__":
    main()
