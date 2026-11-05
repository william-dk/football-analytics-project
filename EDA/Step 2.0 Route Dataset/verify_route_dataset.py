"""Read delivered tables back and validate geometry, timing and cohort integrity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_route_dataset import ROUTE, PLAY, KEY, FRAME_COLUMNS


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    extension = manifest["format"]
    read = pd.read_parquet if extension == "parquet" else lambda p: pd.read_csv(p, low_memory=False)
    routes = read(directory / f"routes.{extension}")
    paths = read(directory / f"target_full_paths.{extension}")
    assert not routes.duplicated(ROUTE).any()
    assert not paths.duplicated(KEY).any()
    assert set(FRAME_COLUMNS) <= set(paths.columns)
    assert routes.season.eq(2023).all()
    assert routes.loc[~routes.is_targeted, "route_label"].isna().all()
    assert routes.loc[~routes.is_targeted, ["n_output_frames", "ball_flight_s", "break_detected"]].isna().all().all()
    supplementary = pd.read_csv(manifest["supplementary"], low_memory=False)
    labels = supplementary.loc[supplementary.season.eq(2023), PLAY + ["route_of_targeted_receiver"]]
    label_check = routes.merge(labels, on=PLAY, validate="many_to_one")
    expected_labels = label_check.route_of_targeted_receiver.where(label_check.is_targeted)
    assert label_check.route_label.fillna("<missing>").eq(expected_labels.fillna("<missing>")).all()
    assert len(routes) == manifest["rows"]["routes"]
    assert len(paths) == manifest["rows"]["target_full_paths"]
    assert len(routes[PLAY].drop_duplicates()) == manifest["tracking_plays"]
    targets = routes.loc[routes.is_targeted].set_index(ROUTE)
    counts = paths.groupby(ROUTE).size().sort_index()
    expected = (targets.n_input_frames + targets.n_output_frames).sort_index()
    pd.testing.assert_index_equal(counts.index, expected.index)
    np.testing.assert_array_equal(counts, expected)
    joined = paths.merge(targets[["n_input_frames", "throw_time_s", "ball_flight_s"]], on=ROUTE, validate="many_to_one")
    assert joined.phase.eq("input").eq(joined.frame_id.le(joined.n_input_frames)).all()
    np.testing.assert_allclose(joined.t, (joined.frame_id-1)/10)
    np.testing.assert_allclose(targets.throw_time_s, (targets.n_input_frames-1)/10)
    np.testing.assert_allclose(targets.ball_flight_s, targets.n_output_frames/10)
    origin = paths.loc[paths.frame_id.eq(1)]
    np.testing.assert_allclose(origin[["x_norm", "y_norm"]], 0, atol=1e-12)
    paths = paths.sort_values(KEY)
    groups = paths.groupby(ROUTE)
    assert groups.frame_id.diff().dropna().eq(1).all()
    vx, vy = groups.x_norm.diff()*10, groups.y_norm.diff()*10
    np.testing.assert_allclose(paths.vx, vx, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(paths.vy, vy, atol=1e-10, equal_nan=True)
    output = paths.phase.eq("output")
    np.testing.assert_allclose(paths.loc[output, "s"], np.hypot(vx[output], vy[output]), atol=1e-10)
    speed_xy = np.hypot(vx, vy)
    accel = speed_xy.groupby([paths[k] for k in ROUTE]).diff()*10
    np.testing.assert_allclose(paths.loc[output, "a"], accel[output], atol=1e-9, equal_nan=True)
    assert paths.loc[paths.s.lt(1) | speed_xy.eq(0), ["heading_sin", "heading_cos"]].isna().all().all()
    headings = paths[["heading_sin", "heading_cos"]].dropna()
    np.testing.assert_allclose((headings**2).sum(axis=1), 1, atol=1e-12)
    assert paths.turn_rate.dropna().abs().le(1800+1e-8).all()
    # Independently sample every weekly source and recover the written route geometry.
    sampled = 0
    for file in manifest["source_files"]:
        source = Path(file["path"])
        if not source.name.startswith("input_"):
            continue
        raw = pd.read_csv(source, usecols=KEY + ["player_role", "play_direction", "x", "y", "s", "a"])
        target_rows = raw.loc[raw.player_role.eq("Targeted Receiver")]
        if target_rows.empty:
            continue
        key = tuple(target_rows.iloc[0][ROUTE].astype(int))
        g = raw.set_index(ROUTE).loc[[key]].sort_values("frame_id")
        written = paths.set_index(ROUTE).loc[[key]].sort_values("frame_id")
        written = written.loc[written.phase.eq("input")]
        x = g.x.to_numpy() if g.play_direction.iloc[0] == "right" else 120-g.x.to_numpy()
        y = g.y.to_numpy() if g.play_direction.iloc[0] == "right" else 53.3-g.y.to_numpy()
        np.testing.assert_allclose(written.x, x, atol=1e-10)
        np.testing.assert_allclose(written.y, y, atol=1e-10)
        np.testing.assert_allclose(written.x_norm, x-x[0], atol=1e-10)
        np.testing.assert_allclose(written.y_norm, (y-y[0])*written.side.to_numpy(), atol=1e-10)
        np.testing.assert_allclose(written.s, g.s, atol=1e-10)
        np.testing.assert_allclose(written.a, g.a, atol=1e-10)
        output_path = source.with_name(source.name.replace("input_", "output_"))
        raw_output = pd.read_csv(output_path).set_index(ROUTE).loc[[key]].sort_values("frame_id")
        written_output = paths.set_index(ROUTE).loc[[key]].sort_values("frame_id")
        written_output = written_output.loc[written_output.phase.eq("output")]
        np.testing.assert_array_equal(written_output.frame_id, raw_output.frame_id.to_numpy()+len(g))
        ox = raw_output.x if g.play_direction.iloc[0] == "right" else 120-raw_output.x
        oy = raw_output.y if g.play_direction.iloc[0] == "right" else 53.3-raw_output.y
        np.testing.assert_allclose(written_output.x, ox, atol=1e-10)
        np.testing.assert_allclose(written_output.y, oy, atol=1e-10)
        sampled += 1
    # Fixed-clock features use actual observations; no tail extrapolation.
    for second in [.5, 1., 1.5, 2., 2.5, 3.]:
        suffix = f"{second:.1f}s".replace(".", "_")
        at_time = paths.loc[paths.frame_id.eq(round(second*10)+1)].set_index(ROUTE)
        values = targets[f"x_norm_{suffix}"]
        np.testing.assert_allclose(values, at_time.x_norm.reindex(values.index), atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(targets[f"y_norm_{suffix}"], at_time.y_norm.reindex(values.index), atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(targets[f"speed_{suffix}"], at_time.s.reindex(values.index), atol=1e-10, equal_nan=True)
        expected_heading = np.degrees(np.arctan2(at_time.heading_sin, at_time.heading_cos))
        np.testing.assert_allclose(targets[f"heading_{suffix}"], expected_heading.reindex(values.index), atol=1e-10, equal_nan=True)
    throw_frames = joined.loc[joined.phase.eq("input") & joined.frame_id.eq(joined.n_input_frames)].set_index(ROUTE)
    for feature, source_column in [("x_norm", "x_norm"), ("y_norm", "y_norm"), ("speed", "s")]:
        np.testing.assert_allclose(targets[f"{feature}_at_throw"], throw_frames[source_column].reindex(targets.index), atol=1e-10, equal_nan=True)
    detected = targets.loc[targets.break_detected.eq(True)]
    assert detected.break_angle_deg.abs().ge(35).all()
    assert detected.break_before_throw.eq(detected.break_time_s.lt(detected.throw_time_s)).all()
    assert targets.loc[targets.break_detected.eq(False), "break_before_throw"].isna().all()
    if "route_frames" in manifest["rows"]:
        path = directory / f"route_frames.{extension}"
        chunks = [pd.read_parquet(path)] if extension == "parquet" else pd.read_csv(path, chunksize=250_000)
        observed = []
        for chunk in chunks:
            assert not chunk.duplicated(KEY).any()
            observed.append(chunk.groupby(ROUTE).frame_id.agg(["size", "min", "max"]))
        agg = pd.concat(observed).groupby(ROUTE).agg({"size": "sum", "min": "min", "max": "max"}).sort_index()
        route_index = routes.set_index(ROUTE).sort_index()
        pd.testing.assert_index_equal(agg.index, route_index.index)
        np.testing.assert_array_equal(agg["size"], route_index.n_input_frames)
        np.testing.assert_array_equal(agg["max"], route_index.n_input_frames)
        assert agg["min"].eq(1).all()
        assert agg["size"].sum() == manifest["rows"]["route_frames"]
    report = {"status": "passed", "routes": len(routes), "targeted_routes": len(targets),
              "labeled_targets": int(targets.route_label.notna().sum()),
              "unlabeled_targets_preserved": int(targets.route_label.isna().sum()),
              "target_full_path_rows": len(paths), "raw_source_routes_checked": sampled,
              "checks": ["unique keys", "2023 cohort", "target-only labels", "phase seam", "frame continuity",
                         "origin geometry", "velocity and acceleration", "heading mask and turn wrapping",
                         "fixed-clock features", "weekly raw input/output reconstruction", "route frame coverage"]}
    (directory / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    verify(parser.parse_args().data_dir)
