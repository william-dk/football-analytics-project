"""Observed defender frames and separation, anchored to existing Step 2.0 tables."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_route_dataset import PLAY, ROUTE, KEY, FPS, TableWriter, flip_field, require

DB = {"CB", "DB", "S", "FS", "SS"}
RKEY = PLAY + ["receiver_nfl_id"]
FKEY = PLAY + ["frame_id", "receiver_nfl_id"]
DKEY = FKEY + ["defender_nfl_id"]
DCOLUMNS = DKEY + ["phase", "t", "defender_name", "defender_position", "is_db",
    "x", "y", "s", "a", "vx_field", "vy_field", "dist_to_receiver", "rel_x", "rel_y",
    "along_track", "cross_track", "closing_speed", "in_output_set"]
SCOLUMNS = FKEY + ["phase", "t", "n_defenders_observed", "n_db_observed", "nearest_def_id",
    "nearest_def_position", "sep_nearest", "nearest_db_id", "sep_nearest_db",
    "trailing_component", "leverage_component", "cushion_flag", "nearest_switch"]


def locate_table(directory, name):
    # Honor actual files, including the user's decompressed CSVs; reject ambiguity.
    files = [directory / f"{name}.{ext}" for ext in ["csv", "parquet", "csv.gz"]]
    found = [p for p in files if p.is_file()]
    require(len(found) == 1, f"Expected exactly one {name} table, found {found}")
    return found[0]


def read_table(path, **kwargs):
    return pd.read_parquet(path, **kwargs) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False, **kwargs)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def defender_frames(raw_input, raw_output, targets, routes):
    """Use existing target frames as the only observation/time/side authority."""
    meta = routes.loc[routes.is_targeted].copy()
    require(not meta.duplicated(ROUTE).any(), "Duplicate targeted route keys")
    require(not targets.duplicated(KEY).any(), "Duplicate target path keys")
    require(not raw_input.duplicated(KEY).any(), "Duplicate raw input keys")
    require(not raw_output.duplicated(KEY).any(), "Duplicate raw output keys")
    player_meta = raw_input.sort_values(KEY).drop_duplicates(ROUTE)
    metadata = player_meta[ROUTE + ["player_name", "player_position", "player_role", "player_side", "play_direction"]]
    require(raw_input.loc[raw_input.player_side.eq("Defense"), "player_role"].eq("Defensive Coverage").all(),
            "Unexpected defensive input role")
    starts = raw_output.groupby(ROUTE).frame_id.min()
    require(starts.eq(1).all(), "Official output frame_id must restart at 1")
    out = raw_output.merge(metadata, on=ROUTE, how="left", validate="many_to_one", indicator=True)
    require(out._merge.eq("both").all(), "Output player absent from input metadata")
    out = out.drop(columns="_merge")
    throws = meta[PLAY + ["n_input_frames"]].drop_duplicates()
    require(not throws.duplicated(PLAY).any(), "Targets disagree on throw frame")
    actual_throws = raw_input.groupby(PLAY).frame_id.max().rename("actual_throw")
    checked = throws.merge(actual_throws, on=PLAY, validate="one_to_one")
    require(checked.n_input_frames.eq(checked.actual_throw).all(), "Existing route/raw throw frame mismatch")
    require(len(checked) == len(throws), "Target play absent from raw input")
    out = out.merge(throws, on=PLAY, how="inner", validate="many_to_one")
    out["frame_id"] += out.n_input_frames.astype(int)
    di = raw_input.loc[raw_input.player_side.eq("Defense")].copy()
    do = out.loc[out.player_side.eq("Defense")].copy()
    output_set = do[ROUTE].drop_duplicates().assign(in_output_set=True)
    di = flip_field(di).assign(phase="input")
    do = flip_field(do).assign(phase="output", s=np.nan, a=np.nan)
    columns = KEY + ["player_name", "player_position", "x", "y", "s", "a", "phase"]
    defenders = pd.concat([di[columns], do[columns]], ignore_index=True).sort_values(KEY)
    require(not defenders.duplicated(KEY).any(), "Duplicate stitched defender keys")
    require(np.isfinite(defenders[["x", "y"]]).all().all(), "Nonfinite defender coordinates")
    groups = defenders.groupby(ROUTE, sort=False)
    consecutive = groups.frame_id.diff().eq(1)
    defenders["vx_field"] = (groups.x.diff()*FPS).where(consecutive)
    defenders["vy_field"] = (groups.y.diff()*FPS).where(consecutive)
    speed_xy = np.hypot(defenders.vx_field, defenders.vy_field)
    acceleration = (speed_xy.groupby([defenders[k] for k in ROUTE]).diff()*FPS).where(consecutive)
    is_output = defenders.phase.eq("output")
    defenders.loc[is_output, "s"] = speed_xy.loc[is_output]
    defenders.loc[is_output, "a"] = acceleration.loc[is_output]
    defenders = defenders.merge(output_set, on=ROUTE, how="left", validate="many_to_one")
    defenders["in_output_set"] = defenders.in_output_set.eq(True)
    defenders = defenders.rename(columns={"nfl_id": "defender_nfl_id", "player_name": "defender_name", "player_position": "defender_position"})
    receiver = targets.rename(columns={"nfl_id": "receiver_nfl_id", "x": "receiver_x", "y": "receiver_y", "s": "receiver_s"})
    check_side = receiver.merge(meta[ROUTE + ["side"]].rename(columns={"nfl_id": "receiver_nfl_id", "side": "route_side"}),
                               on=RKEY, validate="many_to_one")
    require(check_side.side.eq(check_side.route_side).all(), "Existing frame and route sides disagree")
    require(receiver.phase.isin(["input", "output"]).all(), "Unknown receiver phase")
    timings = receiver.merge(meta[ROUTE + ["n_input_frames"]].rename(columns={"nfl_id": "receiver_nfl_id"}), on=RKEY, validate="many_to_one")
    require(timings.phase.eq("input").eq(timings.frame_id.le(timings.n_input_frames)).all(), "Existing target phase mismatch")
    require(np.allclose(receiver.t, (receiver.frame_id-1)/FPS), "Existing target time mismatch")
    d = defenders.merge(receiver[FKEY + ["t", "phase", "receiver_x", "receiver_y", "receiver_s", "vx", "vy", "side"]],
                        on=PLAY + ["frame_id", "phase"], how="inner", validate="many_to_many")
    d["rel_x"] = d.x-d.receiver_x
    d["rel_y"] = (d.y-d.receiver_y)*d.side
    d["dist_to_receiver"] = np.hypot(d.rel_x, d.rel_y)
    velocity_norm = np.hypot(d.vx, d.vy)
    usable = d.receiver_s.ge(1) & velocity_norm.gt(0) & velocity_norm.notna()
    ux, uy = d.vx/velocity_norm, d.vy/velocity_norm
    d["along_track"] = (d.rel_x*ux + d.rel_y*uy).where(usable)
    # Choose the perpendicular whose route-frame y component is nonnegative.
    # Exactly lateral motion: choose canonical (-uy, ux), since neither normal points laterally.
    normal_sign = np.where(ux.ge(0), 1, -1)
    d["cross_track"] = ((-uy*d.rel_x + ux*d.rel_y)*normal_sign).where(usable)
    d = d.sort_values(RKEY + ["defender_nfl_id", "frame_id"])
    dg = d.groupby(RKEY + ["defender_nfl_id"], sort=False)
    d["closing_speed"] = (dg.dist_to_receiver.diff()*FPS).where(dg.frame_id.diff().eq(1))
    d["is_db"] = d.defender_position.isin(DB)
    d = d[DCOLUMNS].sort_values(DKEY).reset_index(drop=True)
    require(not d.duplicated(DKEY).any(), "Duplicate defender-frame keys")
    return d


def separation_frames(d, targets):
    grid = targets.rename(columns={"nfl_id": "receiver_nfl_id"})[FKEY + ["phase", "t"]].copy()
    counts = d.groupby(FKEY).agg(n_defenders_observed=("defender_nfl_id", "size"), n_db_observed=("is_db", "sum"))
    grid = grid.merge(counts, on=FKEY, how="left", validate="one_to_one")
    for col in ["n_defenders_observed", "n_db_observed"]:
        grid[col] = grid[col].fillna(0).astype(int)
    ordered = d.sort_values(["dist_to_receiver", "defender_nfl_id"], kind="stable")
    nearest = ordered.drop_duplicates(FKEY)[FKEY + ["defender_nfl_id", "defender_position", "dist_to_receiver", "along_track", "cross_track"]]
    nearest = nearest.rename(columns={"defender_nfl_id": "nearest_def_id", "defender_position": "nearest_def_position", "dist_to_receiver": "sep_nearest"})
    grid = grid.merge(nearest, on=FKEY, how="left", validate="one_to_one")
    nearest_db = ordered.loc[ordered.is_db].drop_duplicates(FKEY)[FKEY + ["defender_nfl_id", "dist_to_receiver"]]
    nearest_db = nearest_db.rename(columns={"defender_nfl_id": "nearest_db_id", "dist_to_receiver": "sep_nearest_db"})
    grid = grid.merge(nearest_db, on=FKEY, how="left", validate="one_to_one")
    grid["nearest_def_id"] = grid.nearest_def_id.astype("Int64")
    grid["nearest_db_id"] = grid.nearest_db_id.astype("Int64")
    grid["trailing_component"] = (-grid.along_track).clip(lower=0)
    grid["leverage_component"] = abs(grid.cross_track)
    grid["cushion_flag"] = grid.along_track.gt(0).astype("boolean").where(grid.along_track.notna())
    grid = grid.sort_values(RKEY + ["frame_id"])
    groups = grid.groupby(RKEY, sort=False)
    previous = groups.nearest_def_id.shift()
    grid["nearest_switch"] = grid.nearest_def_id.ne(previous).where(
        previous.notna() & grid.nearest_def_id.notna() & groups.frame_id.diff().eq(1)).astype("boolean")
    grid = grid[SCOLUMNS].sort_values(FKEY).reset_index(drop=True)
    require(not grid.duplicated(FKEY).any(), "Duplicate separation-frame keys")
    both = grid.sep_nearest.notna() & grid.sep_nearest_db.notna()
    require(grid.loc[both, "sep_nearest"].le(grid.loc[both, "sep_nearest_db"]+1e-12).all(), "All-defender distance exceeds DB minimum")
    require(grid.loc[grid.n_defenders_observed.eq(0), "sep_nearest"].isna().all(), "No-defender separation must be NaN")
    return grid


def separation_summary(d, s, routes):
    summary = routes.loc[routes.is_targeted].copy().rename(columns={"nfl_id": "receiver_nfl_id"})
    require(not summary.duplicated(RKEY).any(), "Duplicate targeted summary key")
    inp = s.loc[s.phase.eq("input")].sort_values(FKEY).groupby(RKEY).tail(1)
    out = s.loc[s.phase.eq("output")].sort_values(FKEY).groupby(RKEY).tail(1)
    for phase, frame, suffix in [("input", inp, "throw"), ("output", out, "arrival")]:
        fields = ["sep_nearest", "sep_nearest_db", "n_defenders_observed"]
        if phase == "input":
            fields += ["nearest_def_id"]
        renamed = {"sep_nearest": f"sep_at_{suffix}", "sep_nearest_db": f"sep_db_at_{suffix}",
                   "n_defenders_observed": f"n_defenders_at_{suffix}", "nearest_def_id": "_throw_defender"}
        summary = summary.merge(frame[RKEY + fields].rename(columns=renamed), on=RKEY, how="left", validate="one_to_one")
        switches = s.loc[s.phase.eq(phase)].groupby(RKEY).nearest_switch.sum(min_count=1).rename(f"nearest_switches_{phase}")
        summary = summary.merge(switches, on=RKEY, how="left", validate="one_to_one")
    output_set = d.loc[d.in_output_set, RKEY + ["defender_nfl_id"]].drop_duplicates().rename(columns={"defender_nfl_id": "_throw_defender"})
    output_set["_in_set"] = True
    summary = summary.merge(output_set, on=RKEY + ["_throw_defender"], how="left", validate="many_to_one")
    summary["nearest_at_throw_in_output"] = summary._in_set.eq(True).astype("boolean").where(summary._throw_defender.notna())
    arrival = out[RKEY + ["frame_id"]]
    fixed = summary[RKEY + ["_throw_defender"]].merge(arrival, on=RKEY, how="inner", validate="one_to_one")
    fixed = fixed.merge(d[RKEY + ["frame_id", "defender_nfl_id", "dist_to_receiver"]],
        left_on=RKEY + ["frame_id", "_throw_defender"], right_on=RKEY + ["frame_id", "defender_nfl_id"], how="left", validate="one_to_one")
    summary = summary.merge(fixed[RKEY + ["dist_to_receiver"]].rename(columns={"dist_to_receiver": "sep_fixed_throw_def_at_arrival"}), on=RKEY, how="left", validate="one_to_one")
    extrema = s.loc[s.phase.eq("input")].groupby(RKEY).sep_nearest.agg(max_sep_input="max", min_sep_input="min")
    summary = summary.merge(extrema, on=RKEY, how="left", validate="one_to_one")
    break_frame = summary.loc[summary.break_detected.eq(True), RKEY + ["break_time_s"]].copy()
    break_frame["frame_id"] = (break_frame.break_time_s*FPS).round().astype(int)+1
    break_frame = break_frame.merge(s[FKEY + ["sep_nearest"]], on=RKEY + ["frame_id"], how="left", validate="one_to_one")
    summary = summary.merge(break_frame[RKEY + ["sep_nearest"]].rename(columns={"sep_nearest": "sep_at_break"}), on=RKEY, how="left", validate="one_to_one")
    summary["sep_change_flight"] = summary.sep_at_arrival-summary.sep_at_throw
    # Preserve original targeted-route columns and provide the frame tables' explicit receiver alias.
    summary["nfl_id"] = summary.receiver_nfl_id
    return summary.drop(columns=["_throw_defender", "_in_set"]).sort_values(RKEY).reset_index(drop=True)


def raw_sample_errors(raw_input, raw_output, sample, routes):
    """Independent distances using official UNFLIPPED receiver and defender x/y."""
    meta = raw_input.sort_values(KEY).drop_duplicates(ROUTE)[ROUTE + ["player_side"]]
    ends = routes.loc[routes.is_targeted, PLAY + ["n_input_frames"]].drop_duplicates()
    errors = []
    for phase, raw in [("input", raw_input), ("output", raw_output.merge(meta, on=ROUTE, validate="many_to_one"))]:
        chosen = sample.loc[sample.phase.eq(phase), FKEY + ["sep_nearest"]].merge(ends, on=PLAY, validate="many_to_one")
        chosen["source_frame"] = chosen.frame_id - (chosen.n_input_frames if phase == "output" else 0)
        receiver = raw[KEY + ["x", "y"]].rename(columns={"nfl_id": "receiver_nfl_id", "frame_id": "source_frame", "x": "rx", "y": "ry"})
        chosen = chosen.merge(receiver, on=RKEY + ["source_frame"], how="left", validate="one_to_one")
        require(chosen.rx.notna().all(), "Sample target missing from raw coordinates")
        defenders = raw.loc[raw.player_side.eq("Defense"), KEY + ["x", "y"]].rename(columns={"frame_id": "source_frame"})
        pairs = chosen.merge(defenders, on=PLAY + ["source_frame"], how="left", validate="many_to_many")
        pairs["raw_distance"] = np.hypot(pairs.x-pairs.rx, pairs.y-pairs.ry)
        recomputed = pairs.groupby(FKEY).raw_distance.min()
        checked = chosen.set_index(FKEY).sep_nearest.to_frame().join(recomputed)
        require(checked.sep_nearest.isna().eq(checked.raw_distance.isna()).all(), "Raw sample missingness mismatch")
        errors.extend(abs(checked.sep_nearest-checked.raw_distance).fillna(0).tolist())
    return errors


def run_defenders(args):
    data_dir = args.existing_data_dir or args.output_dir
    inputs = {name: locate_table(data_dir, name) for name in ["routes", "target_full_paths"]}
    if any((data_dir / f"route_frames.{ext}").is_file() for ext in ["csv", "parquet", "csv.gz"]):
        inputs["route_frames"] = locate_table(data_dir, "route_frames")
    hashes = {name: digest(path) for name, path in inputs.items()}
    routes = read_table(inputs["routes"])
    targets = read_table(inputs["target_full_paths"])
    require(not routes.duplicated(ROUTE).any(), "Duplicate existing route keys")
    require(not targets.duplicated(KEY).any(), "Duplicate existing target frame keys")
    routes = routes.loc[routes.week.isin(args.weeks)]
    if args.max_games is not None:
        require(args.max_games > 0, "--max-games must be positive")
        games = sorted(routes.game_id.unique())[:args.max_games]
        routes = routes.loc[routes.game_id.isin(games)]
    target_routes = routes.loc[routes.is_targeted]
    require(len(target_routes) > 0, "No targeted routes in selected scope")
    targets = targets.merge(target_routes[ROUTE], on=ROUTE, how="inner", validate="many_to_one")
    expected = (target_routes.set_index(ROUTE).n_input_frames + target_routes.set_index(ROUTE).n_output_frames).sort_index()
    counts = targets.groupby(ROUTE).size().sort_index()
    require(counts.index.equals(expected.index) and np.array_equal(counts, expected), "Existing target paths do not match route frame lengths")
    baseline_path = args.baseline or Path(__file__).resolve().parents[1] / "separation_baseline/outputs/separation_baseline/receiver_play_metrics.csv"
    require(baseline_path.is_file(), f"Missing validation baseline: {baseline_path}")
    baseline = pd.read_csv(baseline_path, low_memory=False)
    baseline = baseline.loc[baseline.is_targeted].rename(columns={"receiver_id": "receiver_nfl_id"})
    require(not baseline.duplicated(RKEY).any(), "Duplicate baseline targeted keys")
    sample_keys = targets.sample(n=min(500, len(targets)), random_state=20261025)[KEY].rename(columns={"nfl_id": "receiver_nfl_id"})
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "defender_validation.json"
    validation_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    writer = TableWriter(output_dir, args.format != "csv.gz" and importlib.util.find_spec("pyarrow") is not None)
    require(args.format != "parquet" or writer.parquet, "Install PyArrow for Parquet")
    summaries, samples, distributions, source_files = [], [], {"input": Counter(), "output": Counter()}, []
    scope_seen = []
    try:
        for week in sorted(target_routes.week.unique()):
            selected_routes = routes.loc[routes.week.eq(week)]
            plays = selected_routes[PLAY].drop_duplicates()
            selected_targets = targets.merge(plays, on=PLAY, how="inner", validate="many_to_one")
            raw = {}
            for phase in ["input", "output"]:
                path = args.train_dir / f"{phase}_2023_w{int(week):02}.csv"
                cols = KEY + ["x", "y"] + (["player_name", "player_position", "player_role", "player_side", "play_direction", "s", "a"] if phase == "input" else [])
                raw[phase] = pd.read_csv(path, usecols=cols).merge(plays, on=PLAY, how="inner", validate="many_to_one")
                source_files.append({"path": str(path.resolve()), "bytes": path.stat().st_size})
            d = defender_frames(raw["input"], raw["output"], selected_targets, selected_routes)
            s = separation_frames(d, selected_targets)
            summary = separation_summary(d, s, selected_routes)
            require(len(s) == len(selected_targets), "Separation frame grid lost target observations")
            require(len(summary) == int(selected_routes.is_targeted.sum()), "Missing targeted summary rows")
            scoped_sample = s.merge(sample_keys, on=FKEY, how="inner", validate="one_to_one")
            samples.extend(raw_sample_errors(raw["input"], raw["output"], scoped_sample, selected_routes))
            for phase in distributions:
                distributions[phase].update(s.loc[s.phase.eq(phase), "n_defenders_observed"].value_counts().to_dict())
            writer.write("target_defender_frames", d)
            writer.write("target_separation_frames", s)
            writer.write("target_separation_summary", summary)
            summaries.append(summary)
            scope_seen.append(s[FKEY])
            print(f"Defenders week {int(week):02}: {len(d):,} defender frames; {len(s):,} separation frames; {len(summary):,} routes", flush=True)
    finally:
        writer.close()
    # Read saved keys back, checking row counts and global key uniqueness after serialization.
    readback = {}
    for name, keys in [("target_defender_frames", DKEY), ("target_separation_frames", FKEY), ("target_separation_summary", RKEY)]:
        path = output_dir / f"{name}.{'parquet' if writer.parquet else 'csv.gz'}"
        chunks = [pd.read_parquet(path, columns=keys)] if writer.parquet else pd.read_csv(path, usecols=keys, chunksize=250_000)
        hashes_saved, rows_saved = [], 0
        for chunk in chunks:
            require(not chunk.duplicated(keys).any(), f"Duplicate serialized {name} keys")
            hashes_saved.append(pd.util.hash_pandas_object(chunk[keys], index=False).to_numpy())
            rows_saved += len(chunk)
        hashed = np.concatenate(hashes_saved)
        require(len(np.unique(hashed)) == len(hashed), f"Repeated serialized {name} keys across chunks")
        require(rows_saved == writer.counts[name], f"Serialized {name} row count mismatch")
        readback[name] = rows_saved
    summary = pd.concat(summaries, ignore_index=True)
    require(not summary.duplicated(RKEY).any(), "Summary route repeated across weeks")
    require(not pd.concat(scope_seen, ignore_index=True).duplicated(FKEY).any(), "Separation frame repeated across weeks")
    compare = summary.merge(baseline[RKEY + ["input_end_db_yards"]], on=RKEY, how="left", validate="one_to_one", indicator=True)
    missing_baseline = compare._merge.ne("both")
    error = abs(compare.sep_db_at_throw-compare.input_end_db_yards)
    mismatch = missing_baseline | compare.sep_db_at_throw.isna().ne(compare.input_end_db_yards.isna()) | error.ge(1e-6)
    mismatch_rows = compare.loc[mismatch, RKEY + ["sep_db_at_throw", "input_end_db_yards"]]
    mismatch_rows.to_csv(output_dir / "defender_baseline_mismatches.csv", index=False)
    distribution_report = {}
    for phase, counter in distributions.items():
        total = sum(counter.values())
        distribution_report[phase] = {"frames": int(total), "mean": sum(k*v for k,v in counter.items())/total,
            "min": int(min(counter)), "max": int(max(counter)), "counts": {str(int(k)): int(v) for k,v in sorted(counter.items())}}
    known = summary.nearest_at_throw_in_output.notna()
    false_count = int(summary.nearest_at_throw_in_output.eq(False).sum())
    medians = summary.groupby("route_label").sep_at_throw.median().sort_values()
    same_inputs = {name: digest(path) == hashes[name] for name, path in inputs.items()}
    report = {"status": "passed", "games": int(summary.game_id.nunique()), "targeted_routes": len(summary),
        "rows": writer.counts, "format": "parquet" if writer.parquet else "csv.gz", "columns": writer.columns,
        "raw_random_frames_checked": len(samples), "raw_distance_max_abs_error_yd": max(samples, default=0),
        "baseline_rows_compared": len(compare), "baseline_mismatch_count": int(mismatch.sum()),
        "baseline_both_missing": int((compare.sep_db_at_throw.isna() & compare.input_end_db_yards.isna()).sum()),
        "baseline_max_abs_error_yd": float(error.max()), "defender_count_distribution_by_phase": distribution_report,
        "nearest_at_throw_absent_from_output_plays": false_count,
        "nearest_at_throw_absent_from_output_share": false_count/int(known.sum()) if known.any() else None,
        "nearest_at_throw_output_flag_unknown": int((~known).sum()),
        "median_sep_at_throw_by_route_yd": medians.to_dict(),
        "arrival_frames_without_defenders": int(summary.n_defenders_at_arrival.eq(0).sum()),
        "serialized_tables_readback_rows": readback,
        "route_order_sanity_enforced": False,
        "existing_tables_unchanged_sha256": same_inputs,
        "existing_table_sha256": hashes, "existing_table_paths": {k: str(v) for k,v in inputs.items()},
        "source_files": source_files, "baseline_path": str(baseline_path),
        "unique_keys_checked": True, "complete_target_frame_grid_checked": True,
        "sep_nearest_le_db_checked": True,
        "note": "Observed coverage defenders only; output is a selected subset. First output switch includes the phase boundary."}
    failures = []
    if len(samples) != min(500, len(targets)) or max(samples, default=0) >= 1e-6:
        failures.append("Independent raw-coordinate sample failed")
    if mismatch.any():
        failures.append("Baseline distance mismatch; see defender_baseline_mismatches.csv")
    if not all(same_inputs.values()):
        failures.append("Existing primary tables changed")
    report["status"] = "failed" if failures else "passed"
    report["failures"] = failures
    validation_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    require(not failures, "; ".join(failures))
    print(json.dumps({"status": "passed", "rows": writer.counts, "raw_max_error": max(samples), "baseline_mismatches": int(mismatch.sum())}), flush=True)
    return report
