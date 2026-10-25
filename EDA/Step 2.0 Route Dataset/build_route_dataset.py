"""Build Step 2.0 from official weekly input/output files (never combined extracts).

Run with --train-dir PATH. See README.md for units, proxies and feature definitions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

PLAY = ["game_id", "play_id"]
ROUTE = PLAY + ["nfl_id"]
KEY = ROUTE + ["frame_id"]
ROLES = ["Targeted Receiver", "Other Route Runner"]
FPS = 10
SUPP = [
    "season", "week", "down", "yards_to_go", "quarter", "offense_formation",
    "receiver_alignment", "play_action", "dropback_type", "pass_result", "pass_length",
    "team_coverage_man_zone", "team_coverage_type", "defenders_in_the_box",
    "play_nullified_by_penalty", "yards_gained", "expected_points_added",
]
FRAME_COLUMNS = KEY + ["t", "phase", "x_norm", "y_norm", "vx", "vy", "s", "a",
    "heading_sin", "heading_cos", "turn_rate", "x", "y", "side", "side_flag"]
INPUT_COLUMNS = KEY + ["player_name", "player_position", "player_role", "play_direction",
    "absolute_yardline_number", "x", "y", "s", "a", "num_frames_output"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flip_field(frame):
    """Apply identical 180-degree rotation to EVERY player, including defenders."""
    frame = frame.copy()
    require(frame.play_direction.isin(["left", "right"]).all(), "Unknown direction")
    left = frame.play_direction.eq("left")
    frame.loc[left, "x"] = 120 - frame.loc[left, "x"]
    frame.loc[left, "y"] = 53.3 - frame.loc[left, "y"]
    return frame


def check_frames(frame, name):
    require(not frame.duplicated(KEY).any(), f"{name}: duplicate frame keys")
    require(np.isfinite(frame[["x", "y"]]).all().all(), f"{name}: invalid coordinates")
    ordered = frame.sort_values(KEY)
    delta = ordered.groupby(ROUTE, sort=False).frame_id.diff().dropna()
    require(delta.eq(1).all(), f"{name}: nonconsecutive frames")
    require(ordered.groupby(ROUTE).frame_id.min().eq(1).all(), f"{name}: missing frame 1")


def normalize_paths(frame):
    frame = frame.sort_values(KEY).reset_index(drop=True).copy()
    frame["x_norm"] = frame.x - frame.x0
    frame["y_norm"] = (frame.y - frame.y0) * frame.side
    frame["t"] = (frame.frame_id - 1) / FPS
    groups = frame.groupby(ROUTE, sort=False)
    frame["vx"] = groups.x_norm.diff() * FPS
    frame["vy"] = groups.y_norm.diff() * FPS
    output = frame.phase.eq("output")
    speed_xy = np.hypot(frame.vx, frame.vy)
    frame.loc[output, "s"] = speed_xy.loc[output]
    # Output acceleration is derived only from positions, including the throw seam.
    accel_xy = speed_xy.groupby([frame[k] for k in ROUTE], sort=False).diff() * FPS
    frame.loc[output, "a"] = accel_xy.loc[output]
    heading = np.arctan2(frame.vy, frame.vx).where(frame.s.ge(1) & speed_xy.gt(0))
    frame["heading_sin"] = np.sin(heading)
    frame["heading_cos"] = np.cos(heading)
    change = heading.groupby([frame[k] for k in ROUTE], sort=False).diff()
    frame["turn_rate"] = np.degrees(np.arctan2(np.sin(change), np.cos(change))) * FPS
    return frame


def wrapped_degrees(angle):
    return (angle + 180) % 360 - 180


def detect_break(x, y, t, speed, throw_time):
    """Strongest >=35 degree turn between adjoining 0.3s displacement windows."""
    result = dict(break_detected=False, break_depth_yd=np.nan, break_time_s=np.nan,
                  break_angle_deg=np.nan, speed_before_break=np.nan,
                  speed_after_break=np.nan, break_before_throw=pd.NA)
    if len(x) < 7:
        return result
    kernel = np.array([1, 4, 6, 4, 1]) / 16
    smooth = lambda a: np.convolve(np.pad(a, (2, 2), mode="edge"), kernel, mode="valid")
    sx, sy = smooth(x), smooth(y)
    k = np.arange(3, len(x) - 3)
    ux, uy = sx[k] - sx[k-3], sy[k] - sy[k-3]
    vx, vy = sx[k+3] - sx[k], sy[k+3] - sy[k]
    angles = np.degrees(np.arctan2(ux * vy - uy * vx, ux * vx + uy * vy))
    valid = (np.hypot(ux, uy) >= .6) & (np.hypot(vx, vy) >= .6) & (abs(angles) >= 35)
    if valid.any():
        j = np.flatnonzero(valid)[np.argmax(abs(angles[valid]))]
        i = k[j]
        result.update(break_detected=True, break_depth_yd=float(x[i]),
            break_time_s=float(t[i]), break_angle_deg=float(angles[j]),
            speed_before_break=float(np.mean(speed[i-2:i+1])),
            speed_after_break=float(np.mean(speed[i+1:i+4])),
            break_before_throw=bool(t[i] < throw_time))
    return result


def summarize_paths(paths, metadata):
    rows = []
    lookup = metadata.set_index(ROUTE)
    for key, path in paths.groupby(ROUTE, sort=False):
        meta = lookup.loc[key]
        x, y, t, speed = (path[c].to_numpy() for c in ["x_norm", "y_norm", "t", "s"])
        heading = np.degrees(np.arctan2(path.heading_sin.to_numpy(), path.heading_cos.to_numpy()))
        throw = int(meta.n_input_frames) - 1
        row = dict(zip(ROUTE, key))
        for frame in [6, 11, 16, 21, 26, 31]:
            suffix = f"{(frame-1)/10:.1f}s".replace(".", "_")
            i = frame - 1
            for name, values in [("x_norm", x), ("y_norm", y), ("heading", heading), ("speed", speed)]:
                row[f"{name}_{suffix}"] = float(values[i]) if i < len(path) else np.nan
        for name, values in [("x_norm", x), ("y_norm", y), ("heading", heading), ("speed", speed)]:
            row[f"{name}_at_throw"] = float(values[throw])
        row["heading_change_last_0_3s_deg"] = (float(wrapped_degrees(heading[throw] - heading[throw-3]))
            if throw >= 3 else np.nan)
        row["speed_change_last_0_3s"] = float(speed[throw] - speed[throw-3]) if throw >= 3 else np.nan
        if meta.is_targeted:
            row.update(detect_break(x, y, t, speed, meta.throw_time_s))
        rows.append(row)
    result = metadata.merge(pd.DataFrame(rows), on=ROUTE, validate="one_to_one")
    for column in ["break_detected", "break_before_throw"]:
        result[column] = result[column].astype("boolean")
    return result


def build_week(raw_input, raw_output, supplementary):
    check_frames(raw_input, "input")
    check_frames(raw_output, "output")
    scope = supplementary[PLAY]
    inp = raw_input.merge(scope, on=PLAY, how="inner", validate="many_to_one")
    out = raw_output.merge(scope, on=PLAY, how="inner", validate="many_to_one")
    require(len(inp) == len(raw_input), "Tracking plays missing 2023 supplementary context")
    inp = flip_field(inp)
    players = inp.sort_values(KEY).groupby(ROUTE, sort=False).head(1).copy()
    end = inp.groupby(PLAY).frame_id.max().rename("input_end")
    player_ends = inp.groupby(ROUTE).frame_id.max().rename("player_end")
    ends = player_ends.reset_index().merge(end, on=PLAY, validate="many_to_one")
    require(ends.player_end.eq(ends.input_end).all(), "Input players have unequal throw frames")
    qb = players.loc[players.player_role.eq("Passer"), PLAY + ["x", "y"]]
    require(not qb.duplicated(PLAY).any(), "Multiple passers at frame 1")
    qb = qb.rename(columns={"x": "qb_x0", "y": "qb_y0"})
    meta = players.loc[players.player_role.isin(ROLES)].merge(qb, on=PLAY, how="left", validate="many_to_one")
    meta = meta.merge(end, on=PLAY, validate="many_to_one")
    meta["is_targeted"] = meta.player_role.eq("Targeted Receiver")
    meta["x0"], meta["y0"] = meta.x, meta.y
    meta["side_fallback"] = meta.qb_y0.isna() | meta.y0.eq(meta.qb_y0)
    meta["side"] = np.sign(meta.y0 - meta.qb_y0).fillna(1).replace(0, 1).astype("int8")
    meta["lateral_from_ball_yd"] = abs(meta.y0 - meta.qb_y0)
    meta["side_flag"] = meta.side_fallback | meta.player_position.isin(["RB", "FB"]) | meta.lateral_from_ball_yd.le(2)
    meta["in_motion_at_snap"] = meta.s.gt(1)
    meta["sideline_dist_yd"] = np.minimum(meta.y0, 53.3 - meta.y0)
    los = np.where(meta.play_direction.eq("left"), 120 - meta.absolute_yardline_number, meta.absolute_yardline_number)
    meta["depth_from_los_yd"] = meta.x0 - los
    widest = meta.groupby(PLAY + ["side"]).lateral_from_ball_yd.transform("max")
    meta["alignment"] = np.where(meta.lateral_from_ball_yd.eq(widest), "outermost", "slot")
    meta.loc[meta.lateral_from_ball_yd.le(4) & meta.depth_from_los_yd.abs().le(1), "alignment"] = "inline"
    meta.loc[meta.x0.le(meta.qb_x0), "alignment"] = "backfield"
    meta.loc[meta.qb_y0.isna(), "alignment"] = pd.NA
    meta["n_input_frames"] = meta.input_end.astype(int)
    meta["throw_time_s"] = (meta.input_end - 1) / FPS

    # Recover output roles/direction from input on all three player identity keys.
    out = out.merge(players[ROUTE + ["player_role", "play_direction"]], on=ROUTE,
                    how="left", validate="many_to_one", indicator=True)
    require(out._merge.eq("both").all(), "Output player absent from input metadata")
    out = flip_field(out.drop(columns="_merge"))
    output_min = out.groupby(ROUTE).frame_id.min()
    require(output_min.eq(1).all(), "Output does not restart at 1; inspect source before remapping")
    out["source_frame_id"] = out.frame_id
    out = out.merge(end, on=PLAY, validate="many_to_one")
    out["frame_id"] += out.input_end
    out_count = out.groupby(ROUTE).size().rename("observed_output_frames")
    meta = meta.merge(out_count, on=ROUTE, how="left", validate="one_to_one")
    meta["n_output_frames"] = meta.observed_output_frames.fillna(0).where(meta.is_targeted).astype("Int64")
    require(meta.loc[meta.is_targeted, "n_output_frames"].eq(
        meta.loc[meta.is_targeted, "num_frames_output"]).all(), "Target output length does not match declared flight length")
    meta["ball_flight_s"] = meta.n_output_frames / FPS
    origin_columns = ROUTE + ["x0", "y0", "side", "side_flag"]
    rr = inp.loc[inp.player_role.isin(ROLES), KEY + ["x", "y", "s", "a"]].copy()
    rr["phase"] = "input"
    target_keys = meta.loc[meta.is_targeted, ROUTE]
    target_out = out.merge(target_keys, on=ROUTE, how="inner", validate="many_to_one")
    target_out = target_out[KEY + ["x", "y"]].assign(s=np.nan, a=np.nan, phase="output")
    paths = pd.concat([rr, target_out], ignore_index=True).merge(meta[origin_columns], on=ROUTE, validate="many_to_one")
    paths = normalize_paths(paths)
    require(not paths.duplicated(KEY).any(), "Duplicated frames at phase seam")
    metadata_columns = ROUTE + ["player_name", "player_position", "is_targeted", "n_input_frames",
        "throw_time_s", "n_output_frames", "ball_flight_s", "in_motion_at_snap", "lateral_from_ball_yd",
        "sideline_dist_yd", "depth_from_los_yd", "alignment", "side_flag", "side_fallback", "side"]
    routes = summarize_paths(paths, meta[metadata_columns])
    routes = routes.merge(supplementary[PLAY + SUPP + ["route_of_targeted_receiver"]], on=PLAY,
                          how="left", validate="many_to_one")
    routes["route_label"] = routes.route_of_targeted_receiver.where(routes.is_targeted)
    routes = routes.drop(columns="route_of_targeted_receiver")
    targets = paths.merge(target_keys, on=ROUTE, how="inner", validate="many_to_one")[FRAME_COLUMNS]
    prethrow = paths.loc[paths.phase.eq("input"), FRAME_COLUMNS].drop(columns="phase")
    input_plays = inp[PLAY].drop_duplicates()
    target_counts = meta.loc[meta.is_targeted].groupby(PLAY).size().rename("n_targeted")
    exceptions = input_plays.merge(target_counts, on=PLAY, how="left").fillna({"n_targeted": 0})
    exceptions = exceptions.loc[exceptions.n_targeted.ne(1)]
    audit = dict(input_rows=len(inp), output_rows=len(out), plays=len(input_plays),
        route_runners=len(routes), targeted_routes=int(routes.is_targeted.sum()),
        target_full_path_rows=len(targets), route_frame_rows=len(prethrow),
        labeled_target_routes=int((routes.is_targeted & routes.route_label.notna()).sum()),
        unlabeled_target_routes=int((routes.is_targeted & routes.route_label.isna()).sum()),
        output_frame_id_restarts_at_1=True, output_players=len(output_min),
        output_coverage_players=int(out.loc[out.player_role.eq("Defensive Coverage"), ROUTE].drop_duplicates().shape[0]),
        missing_passer_routes=int(meta.qb_y0.isna().sum()), side_flag_routes=int(meta.side_flag.sum()),
        detected_target_breaks=int(routes.break_detected.fillna(False).sum()),
        target_breaks_after_throw=int((routes.break_detected.fillna(False) & routes.break_before_throw.eq(False)).sum()),
        input_left_rows=int(inp.play_direction.eq("left").sum()))
    return targets, routes, prethrow, audit, exceptions, input_plays


class TableWriter:
    """Stream weeks to single reusable tables without retaining all frames in RAM."""
    def __init__(self, directory, parquet):
        self.directory, self.parquet = directory, parquet
        self.writers, self.counts, self.columns = {}, {}, {}

    def write(self, name, frame):
        self.columns[name] = list(frame.columns)
        if self.parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if name not in self.writers:
                self.writers[name] = pq.ParquetWriter(self.directory / f"{name}.parquet", table.schema, compression="zstd")
            table = table.cast(self.writers[name].schema)
            self.writers[name].write_table(table)
        else:
            import gzip
            if name not in self.writers:
                self.writers[name] = gzip.open(self.directory / f"{name}.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=3)
            frame.to_csv(self.writers[name], index=False, header=name not in self.counts)
        self.counts[name] = self.counts.get(name, 0) + len(frame)

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path, default=Path(__file__).resolve().parents[1] / "Supplementary Data Analysis" / "supplementary_data.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--weeks", nargs="+", type=int, default=list(range(1, 19)))
    parser.add_argument("--skip-route-frames", action="store_true")
    parser.add_argument("--skip-defenders", action="store_true", help="Skip defender/separation extension")
    parser.add_argument("--defenders-only", action="store_true", help="Read existing route tables; write only new defender/separation files")
    parser.add_argument("--existing-data-dir", type=Path, help="Existing route tables for --defenders-only (defaults to output directory)")
    parser.add_argument("--max-games", type=int, help="Defender-only validation subset: first N games in selected weeks")
    parser.add_argument("--baseline", type=Path, help="Existing receiver_play_metrics.csv for separation validation")
    parser.add_argument("--format", choices=["auto", "csv.gz", "parquet"], default="auto")
    args = parser.parse_args()
    require(len(set(args.weeks)) == len(args.weeks) and all(1 <= w <= 18 for w in args.weeks), "Invalid or duplicate week")
    require(not (args.defenders_only and args.skip_defenders), "Cannot combine --defenders-only and --skip-defenders")
    require(args.max_games is None or args.defenders_only, "--max-games requires --defenders-only")
    if args.defenders_only:
        from defender_dataset import run_defenders
        run_defenders(args)
        return
    for week in args.weeks:
        for phase in ["input", "output"]:
            require((args.train_dir / f"{phase}_2023_w{week:02}.csv").is_file(), f"Missing {phase} week {week}")
    supplementary = pd.read_csv(args.supplementary, low_memory=False)
    season_counts = supplementary.groupby("season").size().to_dict()
    supplementary = supplementary.loc[supplementary.season.eq(2023)].copy()
    require(not supplementary.duplicated(PLAY).any(), "Duplicate supplementary play key")
    has_arrow = importlib.util.find_spec("pyarrow") is not None
    require(args.format != "parquet" or has_arrow, "Install pyarrow for Parquet or use --format csv.gz")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A failed build never carries a prior successful completion marker.
    for marker in ["manifest.json", "validation.json"]:
        (args.output_dir / marker).unlink(missing_ok=True)
    writer = TableWriter(args.output_dir, has_arrow and args.format != "csv.gz")
    audits, exceptions, play_sets, source_files, unlabeled = [], [], [], [], []
    start = time.monotonic()
    try:
        for week in sorted(args.weeks):
            print(f"Week {week:02}: reading official input/output", flush=True)
            input_path = args.train_dir / f"input_2023_w{week:02}.csv"
            output_path = args.train_dir / f"output_2023_w{week:02}.csv"
            inp = pd.read_csv(input_path, usecols=INPUT_COLUMNS)
            out = pd.read_csv(output_path)
            target, routes, frames, audit, missing, plays = build_week(inp, out, supplementary)
            require(routes.week.eq(week).all(), "Source week and supplementary week disagree")
            writer.write("target_full_paths", target)
            writer.write("routes", routes)
            if not args.skip_route_frames:
                writer.write("route_frames", frames)
            audits.append(dict(week=week, **audit))
            exceptions.append(missing.assign(week=week))
            unlabeled.append(routes.loc[routes.is_targeted & routes.route_label.isna(), ROUTE + ["week", "pass_result"]])
            play_sets.append(plays)
            source_files.extend([dict(path=str(p.resolve()), bytes=p.stat().st_size) for p in [input_path, output_path]])
            print(f"Week {week:02}: {len(routes):,} routes; {audit['targeted_routes']:,} targets; {len(target):,} target frames", flush=True)
    finally:
        writer.close()
    seen = pd.concat(play_sets, ignore_index=True)
    require(not seen.duplicated(PLAY).any(), "Play repeated across weekly files")
    expected = supplementary.loc[supplementary.week.isin(args.weeks), PLAY]
    coverage = expected.merge(seen, on=PLAY, how="outer", indicator=True, validate="one_to_one")
    require(coverage._merge.eq("both").all(), "Tracking/supplementary play coverage mismatch")
    pd.DataFrame(audits).to_csv(args.output_dir / "weekly_audit.csv", index=False)
    pd.concat(exceptions, ignore_index=True).to_csv(args.output_dir / "target_count_exceptions.csv", index=False)
    pd.concat(unlabeled, ignore_index=True).to_csv(args.output_dir / "target_label_missing.csv", index=False)
    manifest = dict(status="complete", source="official 2023 weekly input/output", weeks=sorted(args.weeks),
        supplementary=str(args.supplementary.resolve()), supplementary_season_counts=season_counts,
        tracking_plays=len(seen), format="parquet" if writer.parquet else "csv.gz", rows=writer.counts,
        columns=writer.columns, source_files=source_files, elapsed_seconds=round(time.monotonic()-start, 1),
        output_frame_mapping="input_end_frame + source_output_frame_id (output starts at 1)",
        fps=FPS, field_width_yd=53.3, heading_units="degrees from +x in route frame; signed toward own sideline")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": writer.counts, "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    if not args.skip_defenders:
        from defender_dataset import run_defenders
        run_defenders(args)


if __name__ == "__main__":
    main()
