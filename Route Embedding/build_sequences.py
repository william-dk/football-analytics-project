"""Build fixed-length route sequences for the masked-window route autoencoder.

Inputs (read-only):
  - Step 2.0 targeted full paths (snap -> ball arrival) and route table
  - reviewed break features v2 (corrected side, break time)
  - 2023 supplementary plays (possession team, used only for later confound checks)

Outputs in --out-dir (default ./data, ignored by Git):
  sequences.npz      X [N, 77, 6] in CHANNELS order (raw units), lengths, throw index, split,
                     per-channel scales (from the train split) and route keys
  routes_meta.csv    one row per sequence, in the same order as X
  build_report.json  counts, exclusions and validation checks

Player identity, team, position and route label are written only to the metadata; they are
never part of X.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from route_sequences import (CHANNELS, MAX_LEN, ROUTE_POSITIONS, SPLITS, assign_splits,
                             channel_scales, pad_route, route_features)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STEP20 = REPO / "EDA" / "Step 2.0 Route Dataset"
KEY = ["game_id", "play_id", "nfl_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", type=Path, default=STEP20 / "data" / "target_full_paths.csv")
    parser.add_argument("--routes", type=Path, default=STEP20 / "data" / "routes.csv")
    parser.add_argument("--breaks", type=Path,
                        default=STEP20 / "break_features_v2_35deg" / "full_season" / "target_break_features.csv")
    parser.add_argument("--supplementary", type=Path,
                        default=REPO / "EDA" / "Supplementary Data Analysis" / "supplementary_data.csv")
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--seed", type=int, default=20261112, help="game split seed")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [args.out_dir / name for name in ("sequences.npz", "routes_meta.csv", "build_report.json")]
    if any(p.exists() for p in outputs) and not args.overwrite:
        raise SystemExit(f"Outputs already exist in {args.out_dir}; pass --overwrite to replace them.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(args.paths, usecols=KEY + ["frame_id", "phase", "x_norm", "y_norm", "y", "s"])
    routes = pd.read_csv(args.routes, low_memory=False,
                         usecols=KEY + ["player_name", "player_position", "is_targeted", "n_input_frames",
                                        "n_output_frames", "route_label", "week", "side"])
    routes = routes[is_true(routes.is_targeted)].drop(columns="is_targeted")
    breaks = pd.read_csv(args.breaks, usecols=KEY + ["side_corrected", "side_was_flipped", "side_uncertain",
                                                     "has_break", "break_t_s"])
    supp = pd.read_csv(args.supplementary, usecols=["game_id", "play_id", "season", "possession_team"])
    supp = supp.loc[supp.season.eq(2023), ["game_id", "play_id", "possession_team"]]

    meta = (routes.merge(breaks, on=KEY, how="left", validate="one_to_one")
                  .merge(supp, on=["game_id", "play_id"], how="left", validate="many_to_one")
                  .rename(columns={"possession_team": "team", "player_position": "position"}))
    report: dict = {"targeted_routes": int(len(meta)),
                    "missing_break_features": int(meta.side_corrected.isna().sum()),
                    "missing_team": int(meta.team.isna().sum())}
    if report["missing_break_features"]:
        raise SystemExit("Every targeted route needs a corrected side from the break features table.")

    keep = meta.position.isin(ROUTE_POSITIONS)
    report["excluded_positions"] = meta.loc[~keep, "position"].value_counts().to_dict()
    meta = meta[keep].sort_values(KEY).reset_index(drop=True)
    for col in ("side_was_flipped", "side_uncertain", "has_break"):
        meta[col] = is_true(meta[col])
    meta["side_corrected"] = meta.side_corrected.astype(int)

    paths = paths.sort_values(KEY + ["frame_id"], kind="mergesort").reset_index(drop=True)
    keys = paths[KEY].to_numpy()
    starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
    ends = np.r_[starts[1:], len(paths)]
    slices = {tuple(keys[s]): (s, e) for s, e in zip(starts, ends)}

    frame_id = paths.frame_id.to_numpy()
    x_norm, y_norm, y_field = (paths[c].to_numpy(float) for c in ("x_norm", "y_norm", "y"))
    phase_input = paths.phase.eq("input").to_numpy()
    official_s = paths.s.to_numpy(float)

    n = len(meta)
    X = np.zeros((n, args.max_len, len(CHANNELS)), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int16)
    orig_lengths = np.zeros(n, dtype=np.int16)
    truncated = np.zeros(n, dtype=bool)
    missing_paths, gaps, origin_errors = 0, 0, 0
    lateral_vs_step20 = 0.0
    speed_pairs = []
    for i, row in enumerate(meta.itertuples(index=False)):
        span = slices.get((row.game_id, row.play_id, row.nfl_id))
        if span is None:
            missing_paths += 1
            continue
        s, e = span
        gaps += int(np.any(np.diff(frame_id[s:e]) != 1))
        feats = route_features(x_norm[s:e], y_field[s:e], row.side_corrected)
        origin_errors += int(np.abs(feats[0, :2]).max() > 1e-9)
        # Corrected lateral must equal Step 2.0 y_norm times (corrected side / original side).
        expected = y_norm[s:e] * row.side_corrected * row.side
        lateral_vs_step20 = max(lateral_vs_step20, float(np.nanmax(np.abs(feats[:, 1] - expected))))
        inp = phase_input[s:e]
        speed_pairs.append(np.column_stack([feats[inp, 4], official_s[s:e][inp]]))
        X[i], lengths[i], truncated[i] = pad_route(feats, args.max_len)
        orig_lengths[i] = e - s
    if missing_paths or gaps or origin_errors:
        raise SystemExit(f"Validation failed: missing={missing_paths} gaps={gaps} origin={origin_errors}")
    valid = np.arange(args.max_len)[None, :] < lengths[:, None]
    if not np.isfinite(X[valid]).all():
        raise SystemExit("Non-finite feature values in valid frames.")

    meta["length"] = lengths
    meta["orig_length"] = orig_lengths
    meta["was_truncated"] = truncated
    meta["throw_index"] = np.minimum(meta.n_input_frames.to_numpy() - 1, lengths - 1).astype(int)
    meta["split"] = assign_splits(meta.game_id.to_numpy(), args.seed)
    split = meta.split.to_numpy()
    scales = channel_scales(X[split == "train"], lengths[split == "train"])

    pairs = np.concatenate(speed_pairs)
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    frames = X[valid]
    games_by_split = {sp: set(meta.loc[meta.split.eq(sp), "game_id"]) for sp in SPLITS}
    report.update({
        "sequences": n,
        "max_len": args.max_len,
        "channels": list(CHANNELS),
        "scales": dict(zip(CHANNELS, map(float, scales))),
        "truncated_routes": int(truncated.sum()),
        "length_quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0.05, 0.5, 0.95, 1.0)},
        "side_flipped_routes": int(meta.side_was_flipped.sum()),
        "side_uncertain_routes": int(meta.side_uncertain.sum()),
        "labeled_routes": int(meta.route_label.notna().sum()),
        "players": int(meta.nfl_id.nunique()),
        "splits": {sp: {"games": len(games_by_split[sp]), "routes": int((split == sp).sum()),
                        "labeled": int(((split == sp) & meta.route_label.notna().to_numpy()).sum())}
                   for sp in SPLITS},
        "checks": {
            "frame_gaps": gaps,
            "origin_not_zero": origin_errors,
            "max_abs_lateral_minus_step20_y_norm_times_side_change": lateral_vs_step20,
            "games_in_two_splits": int(sum(len(games_by_split[a] & games_by_split[b])
                                           for a, b in (("train", "val"), ("train", "test"), ("val", "test")))),
            "recomputed_vs_official_input_speed": {
                "frames": int(len(pairs)),
                "median_abs_diff": float(np.median(np.abs(pairs[:, 0] - pairs[:, 1]))),
                "pearson_r": float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]),
            },
            "feature_quantiles_valid_frames": {
                ch: {str(q): float(np.quantile(frames[:, j], q)) for q in (0.01, 0.5, 0.99)}
                for j, ch in enumerate(CHANNELS)},
        },
        "sources": {name: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
                    for name, p in (("paths", args.paths), ("routes", args.routes),
                                    ("breaks", args.breaks), ("supplementary", args.supplementary))},
        "seed": args.seed,
    })

    np.savez_compressed(
        args.out_dir / "sequences.npz", X=X, lengths=lengths,
        throw_index=meta.throw_index.to_numpy(np.int16), split=split.astype("<U5"),
        channels=np.array(CHANNELS, dtype="<U2"), scales=scales,
        game_id=meta.game_id.to_numpy(np.int64), play_id=meta.play_id.to_numpy(np.int64),
        nfl_id=meta.nfl_id.to_numpy(np.int64))
    meta.insert(0, "row", np.arange(n))
    meta.to_csv(args.out_dir / "routes_meta.csv", index=False)
    with open(args.out_dir / "build_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources",)}, indent=2))


if __name__ == "__main__":
    main()
