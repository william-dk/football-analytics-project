"""Build fixed-length route sequences for the masked-window route autoencoder.

Two kinds of sequence are written to one file, "full" rows first:
  full  every targeted route, from the snap to ball arrival (Step 2.0 table A)
  pre   every route runner, targeted or not, from the snap to the throw (Step 2.0 table C).
        A targeted route therefore appears twice: once whole and once cut at the throw.

Inputs (read-only):
  - Step 2.0 targeted full paths, all-runner input frames and the route table
  - reviewed break features v2 (corrected side, break time; targeted routes only)
  - 2023 supplementary plays (possession team, used only for later confound checks)

Outputs in --out-dir (default ./data, ignored by Git):
  sequences.npz      X [N, 77, 6] in CHANNELS order (raw units), lengths, throw index, split,
                     kind, per-channel scales (from the train split of the full rows) and keys
  routes_meta.csv    one row per sequence, in the same order as X
  build_report.json  counts, exclusions and validation checks

Player identity, team, position and route label are written only to the metadata; they are
never part of X.

    python build_sequences.py                # both kinds
    python build_sequences.py --kinds full   # targeted routes only (the first version's data)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from route_sequences import (CHANNELS, KINDS, MAX_LEN, ROUTE_POSITIONS, SPLITS, assign_splits,
                             channel_scales, correct_side, pad_route, route_features)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STEP20 = REPO / "EDA" / "Step 2.0 Route Dataset"
KEY = ["game_id", "play_id", "nfl_id"]
CONTEXT = ["alignment", "in_motion_at_snap", "lateral_from_ball_yd", "depth_from_los_yd", "down",
           "yards_to_go", "offense_formation", "receiver_alignment", "play_action", "dropback_type",
           "pass_result", "team_coverage_man_zone", "team_coverage_type"]


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
    parser.add_argument("--frames", type=Path, default=STEP20 / "data" / "route_frames.csv",
                        help="all route runners, input (pre-throw) frames")
    parser.add_argument("--routes", type=Path, default=STEP20 / "data" / "routes.csv")
    parser.add_argument("--breaks", type=Path,
                        default=STEP20 / "break_features_v2_35deg" / "full_season" / "target_break_features.csv")
    parser.add_argument("--supplementary", type=Path,
                        default=REPO / "EDA" / "Supplementary Data Analysis" / "supplementary_data.csv")
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--min-pre-len", type=int, default=10,
                        help="drop pre-throw sequences shorter than this many frames (1.0 s)")
    parser.add_argument("--seed", type=int, default=20261112, help="game split seed")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def route_slices(frames: pd.DataFrame) -> dict:
    """Map (game, play, receiver) to its [start, end) rows; frames must be sorted by KEY, frame."""
    keys = frames[KEY].to_numpy()
    starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
    ends = np.r_[starts[1:], len(frames)]
    return {tuple(keys[s]): (s, e) for s, e in zip(starts, ends)}


def load_frames(path: Path, columns: list[str]) -> pd.DataFrame:
    frames = pd.read_csv(path, usecols=KEY + ["frame_id"] + columns)
    return frames.sort_values(KEY + ["frame_id"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    outputs = [args.out_dir / name for name in ("sequences.npz", "routes_meta.csv", "build_report.json")]
    if any(p.exists() for p in outputs) and not args.overwrite:
        raise SystemExit(f"Outputs already exist in {args.out_dir}; pass --overwrite to replace them.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kinds = [k for k in KINDS if k in args.kinds]

    routes = pd.read_csv(args.routes, low_memory=False,
                         usecols=KEY + ["player_name", "player_position", "is_targeted", "n_input_frames",
                                        "n_output_frames", "route_label", "week", "side", "side_flag"] + CONTEXT)
    routes["is_targeted"] = is_true(routes.is_targeted)
    routes["side_flag"] = is_true(routes.side_flag)
    breaks = pd.read_csv(args.breaks, usecols=KEY + ["side_corrected", "side_was_flipped", "side_uncertain",
                                                     "has_break", "break_t_s"])
    for col in ("side_was_flipped", "side_uncertain", "has_break"):
        breaks[col] = is_true(breaks[col])
    supp = pd.read_csv(args.supplementary, usecols=["game_id", "play_id", "season", "possession_team"])
    supp = supp.loc[supp.season.eq(2023), ["game_id", "play_id", "possession_team"]]

    table = (routes.merge(breaks, on=KEY, how="left", validate="one_to_one")
                   .merge(supp, on=["game_id", "play_id"], how="left", validate="many_to_one")
                   .rename(columns={"possession_team": "team", "player_position": "position"}))
    targeted = table[table.is_targeted]
    report: dict = {"kinds": kinds, "route_runners": int(len(table)), "targeted_routes": int(len(targeted)),
                    "missing_break_features": int(targeted.side_corrected.isna().sum()),
                    "missing_team": int(table.team.isna().sum())}
    if report["missing_break_features"]:
        raise SystemExit("Every targeted route needs a corrected side from the break features table.")
    keep = table.position.isin(ROUTE_POSITIONS)
    report["excluded_positions"] = {
        "targeted": targeted.loc[~keep[targeted.index], "position"].value_counts().to_dict(),
        "all_runners": table.loc[~keep, "position"].value_counts().to_dict()}
    table = table[keep].sort_values(KEY).reset_index(drop=True)

    parts, features, checks = [], [], {}
    speed_pairs: list[np.ndarray] = []

    if "full" in kinds:
        paths = load_frames(args.paths, ["phase", "x_norm", "y_norm", "y", "s"])
        slices = route_slices(paths)
        frame_id = paths.frame_id.to_numpy()
        x_norm, y_norm, y_field = (paths[c].to_numpy(float) for c in ("x_norm", "y_norm", "y"))
        phase_input = paths.phase.eq("input").to_numpy()
        official_s = paths.s.to_numpy(float)
        meta = table[table.is_targeted].reset_index(drop=True).copy()
        meta["side_corrected"] = meta.side_corrected.astype(int)
        missing, gaps, origin, lateral = 0, 0, 0, 0.0
        feats_full = []
        for row in meta.itertuples(index=False):
            span = slices.get((row.game_id, row.play_id, row.nfl_id))
            if span is None:
                missing += 1
                continue
            s, e = span
            gaps += int(np.any(np.diff(frame_id[s:e]) != 1))
            feats = route_features(x_norm[s:e], y_field[s:e], row.side_corrected)
            origin += int(np.abs(feats[0, :2]).max() > 1e-9)
            # Corrected lateral must equal Step 2.0 y_norm times (corrected side / original side).
            expected = y_norm[s:e] * row.side_corrected * row.side
            lateral = max(lateral, float(np.nanmax(np.abs(feats[:, 1] - expected))))
            inp = phase_input[s:e]
            speed_pairs.append(np.column_stack([feats[inp, 4], official_s[s:e][inp]]))
            feats_full.append(feats)
        if missing or gaps or origin:
            raise SystemExit(f"Validation failed (full): missing={missing} gaps={gaps} origin={origin}")
        meta.insert(0, "kind", "full")
        parts.append(meta)
        features += feats_full
        checks["full"] = {"frame_gaps": gaps, "origin_not_zero": origin,
                          "max_abs_lateral_minus_step20_y_norm_times_side_change": lateral}
        del paths

    if "pre" in kinds:
        frames = load_frames(args.frames, ["t", "x_norm", "y"])
        slices = route_slices(frames)
        frame_id = frames.frame_id.to_numpy()
        t_all, x_norm, y_field = (frames[c].to_numpy(float) for c in ("t", "x_norm", "y"))
        meta = table.copy()
        missing, gaps, origin, too_short, side_mismatch = 0, 0, 0, 0, 0
        feats_pre, keep_rows = [], []
        side_corr = np.zeros(len(meta), dtype=int)
        flipped = np.zeros(len(meta), dtype=bool)
        uncertain = np.zeros(len(meta), dtype=bool)
        for i, row in enumerate(meta.itertuples(index=False)):
            span = slices.get((row.game_id, row.play_id, row.nfl_id))
            if span is None:
                missing += 1
                continue
            s, e = span
            gaps += int(np.any(np.diff(frame_id[s:e]) != 1))
            rule = correct_side(t_all[s:e], y_field[s:e], int(row.side), bool(row.side_flag))
            if row.is_targeted:
                # Targeted routes keep the reviewed side so both of their sequences agree.
                side_mismatch += int(rule[0] != int(row.side_corrected))
                side_corr[i], flipped[i], uncertain[i] = int(row.side_corrected), row.side_was_flipped, row.side_uncertain
            else:
                side_corr[i], flipped[i], uncertain[i] = rule
            if e - s < args.min_pre_len:
                too_short += 1
                continue
            feats = route_features(x_norm[s:e], y_field[s:e], side_corr[i])
            origin += int(np.abs(feats[0, :2]).max() > 1e-9)
            feats_pre.append(feats)
            keep_rows.append(i)
        if missing or gaps or origin:
            raise SystemExit(f"Validation failed (pre): missing={missing} gaps={gaps} origin={origin}")
        meta["side_corrected"], meta["side_was_flipped"], meta["side_uncertain"] = side_corr, flipped, uncertain
        meta = meta.iloc[keep_rows].reset_index(drop=True)
        meta.insert(0, "kind", "pre")
        parts.append(meta)
        features += feats_pre
        checks["pre"] = {"frame_gaps": gaps, "origin_not_zero": origin, "dropped_shorter_than_min": too_short,
                         "min_pre_len": args.min_pre_len,
                         "targeted_routes_where_side_rule_differs_from_reviewed_side": side_mismatch}
        del frames

    meta = pd.concat(parts, ignore_index=True)
    # Break features exist only for targeted routes (measured on the full path).
    meta["has_break"] = meta.has_break.map(lambda v: v is True or v == "True")
    n = len(meta)
    X = np.zeros((n, args.max_len, len(CHANNELS)), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int16)
    orig_lengths = np.zeros(n, dtype=np.int16)
    truncated = np.zeros(n, dtype=bool)
    for i, feats in enumerate(features):
        X[i], lengths[i], truncated[i] = pad_route(feats, args.max_len)
        orig_lengths[i] = len(feats)
    valid = np.arange(args.max_len)[None, :] < lengths[:, None]
    if not np.isfinite(X[valid]).all():
        raise SystemExit("Non-finite feature values in valid frames.")

    kind = meta.kind.to_numpy()
    meta["length"] = lengths
    meta["orig_length"] = orig_lengths
    meta["was_truncated"] = truncated
    # Last frame at or before the throw. A pre-throw sequence ends at the throw.
    meta["throw_index"] = np.minimum(meta.n_input_frames.to_numpy() - 1, lengths - 1).astype(int)
    meta["split"] = assign_splits(meta.game_id.to_numpy(), args.seed)
    split = meta.split.to_numpy()
    scale_rows = (split == "train") & (kind == kinds[0])
    scales = channel_scales(X[scale_rows], lengths[scale_rows])

    games_by_split = {sp: set(meta.loc[meta.split.eq(sp), "game_id"]) for sp in SPLITS}
    report.update({
        "sequences": n,
        "max_len": args.max_len,
        "channels": list(CHANNELS),
        "scales": dict(zip(CHANNELS, map(float, scales))),
        "scales_from": f"train split, kind {kinds[0]}",
        "players": int(meta.nfl_id.nunique()),
        "by_kind": {},
        "splits": {sp: {"games": len(games_by_split[sp])} for sp in SPLITS},
        "checks": {**checks, "games_in_two_splits": int(sum(
            len(games_by_split[a] & games_by_split[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))))},
        "seed": args.seed,
    })
    for k in kinds:
        sel = kind == k
        sub = meta[sel]
        frames_k = X[sel][valid[sel]]
        report["by_kind"][k] = {
            "sequences": int(sel.sum()),
            "players": int(sub.nfl_id.nunique()),
            "targeted": int(sub.is_targeted.sum()),
            "labeled_routes": int(sub.route_label.notna().sum()),
            "positions": sub.position.value_counts().to_dict(),
            "truncated_routes": int(sub.was_truncated.sum()),
            "length_quantiles": {str(q): float(np.quantile(lengths[sel], q)) for q in (0.05, 0.5, 0.95, 1.0)},
            "side_flipped_routes": int(sub.side_was_flipped.sum()),
            "side_uncertain_routes": int(sub.side_uncertain.sum()),
            "splits": {sp: int((sub.split == sp).sum()) for sp in SPLITS},
            "feature_quantiles_valid_frames": {
                ch: {str(q): float(np.quantile(frames_k[:, j], q)) for q in (0.01, 0.5, 0.99)}
                for j, ch in enumerate(CHANNELS)},
        }
    if speed_pairs:
        pairs = np.concatenate(speed_pairs)
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        report["checks"]["recomputed_vs_official_input_speed"] = {
            "frames": int(len(pairs)),
            "median_abs_diff": float(np.median(np.abs(pairs[:, 0] - pairs[:, 1]))),
            "pearson_r": float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])}
    sources = [("routes", args.routes), ("breaks", args.breaks), ("supplementary", args.supplementary)]
    sources += [("paths", args.paths)] if "full" in kinds else []
    sources += [("frames", args.frames)] if "pre" in kinds else []
    report["sources"] = {name: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
                         for name, p in sources}

    np.savez_compressed(
        args.out_dir / "sequences.npz", X=X, lengths=lengths,
        throw_index=meta.throw_index.to_numpy(np.int16), split=split.astype("<U5"), kind=kind.astype("<U4"),
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
