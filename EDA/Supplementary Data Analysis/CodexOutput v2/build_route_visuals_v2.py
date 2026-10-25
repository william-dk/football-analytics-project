"""Build 2023 target route figures from Step 2.0 and official weekly tracking.

Dependencies: pandas, numpy, matplotlib (plus Python standard library).
All writes are confined to --out. Source CSVs are never changed.
"""
from pathlib import Path
import argparse
import html
import json
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

LABELS = ["HITCH", "OUT", "FLAT", "CROSS", "GO", "IN", "SLANT", "POST",
          "ANGLE", "CORNER", "SCREEN", "WHEEL"]
KEY = ["game_id", "play_id", "nfl_id"]
SEED = 20261025
BLUE, ORANGE, GREEN, RED, GOLD = "#165b9a", "#ef8428", "#21915b", "#c74742", "#c5930a"
SOURCE = "Source: official 2023 tracking, Weeks 1–18 + supplementary route labels + Step 2.0"
WARNING = "arrival = last tracked frame, not a verified catch event"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "savefig.dpi": 160, "figure.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def read_csv(path, **kwargs):
    return pd.read_csv(path, low_memory=False, **kwargs)


def bool_col(series):
    return series.astype(str).str.lower().eq("true")


def key_of(row):
    return tuple(int(row[c]) for c in KEY)


def progress(message):
    print(message, flush=True)


def scan_metadata(train_dir, wanted):
    """Read only play-level fields, retaining one frame-1 row per play."""
    pieces = []
    columns = ["game_id", "play_id", "frame_id", "play_direction",
               "absolute_yardline_number", "ball_land_x", "ball_land_y"]
    for week in range(1, 19):
        path = train_dir / f"input_2023_w{week:02d}.csv"
        week_parts = []
        for chunk in read_csv(path, usecols=columns, chunksize=250_000):
            first = chunk.loc[chunk.frame_id.eq(1)].drop_duplicates(["game_id", "play_id"])
            mask = pd.MultiIndex.from_frame(first[["game_id", "play_id"]]).isin(wanted)
            week_parts.append(first.loc[mask])
        pieces.append(pd.concat(week_parts).drop_duplicates(["game_id", "play_id"]))
        progress(f"Loaded play geometry: week {week:02d}")
    meta = pd.concat(pieces, ignore_index=True).drop(columns="frame_id")
    if meta.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate official play metadata")
    left = meta.play_direction.eq("left")
    if not meta.play_direction.isin(["left", "right"]).all():
        raise ValueError("Unknown play direction")
    meta["los_x"] = np.where(left, 120 - meta.absolute_yardline_number,
                             meta.absolute_yardline_number)
    meta["ball_x"] = np.where(left, 120 - meta.ball_land_x, meta.ball_land_x)
    meta["ball_y"] = np.where(left, 53.3 - meta.ball_land_y, meta.ball_land_y)
    return meta


def load_routes(data_dir, meta):
    routes = read_csv(data_dir / "routes.csv")
    routes = routes.loc[bool_col(routes.is_targeted) & routes.route_label.notna()].copy()
    if routes.duplicated(KEY).any():
        raise ValueError("Duplicate targeted route key")
    summary = read_csv(data_dir / "target_separation_summary.csv", usecols=[
        "game_id", "play_id", "receiver_nfl_id", "sep_at_throw", "sep_at_arrival",
        "n_defenders_at_arrival"])
    summary = summary.rename(columns={"receiver_nfl_id": "nfl_id"})
    routes = routes.merge(summary, on=KEY, how="left", validate="one_to_one")
    routes = routes.merge(meta, on=["game_id", "play_id"], how="left", validate="many_to_one")
    if routes.los_x.isna().any():
        raise ValueError("Missing official field geometry")
    paths_df = read_csv(data_dir / "target_full_paths.csv", usecols=KEY + [
        "frame_id", "t", "phase", "x", "y"])
    paths_df = paths_df.merge(routes[KEY], on=KEY, validate="many_to_one")
    if paths_df.duplicated(KEY + ["frame_id"]).any():
        raise ValueError("Duplicate target frame key")
    paths = {}
    derived = []
    lookup = routes.set_index(KEY)
    for key, frame in paths_df.groupby(KEY, sort=False):
        frame = frame.sort_values("t")
        row = lookup.loc[key]
        t, x, y = (frame[c].to_numpy(dtype=float) for c in ["t", "x", "y"])
        phase = frame.phase.to_numpy()
        if np.any(np.diff(t) <= 0) or not (phase == "input").any() or not (phase == "output").any():
            raise ValueError(f"Invalid target path phases or time: {key}")
        side = int(row.side)
        if side not in [-1, 1]:
            raise ValueError(f"Invalid side: {key}")
        lateral, depth = (y - y[0]) * side, x - row.los_x
        throw_index = np.flatnonzero(phase == "input")[-1]
        if abs(t[throw_index] - row.throw_time_s) > 1e-6:
            raise ValueError(f"Throw proxy disagrees with source: {key}")
        paths[key] = dict(t=t, x=x, y=y, lateral=lateral, depth=depth,
                          phase=phase, frame_id=frame.frame_id.to_numpy(), throw_index=throw_index)
        derived.append(dict(zip(KEY, key), x0=x[0], y0=y[0], arrival_x=x[-1], arrival_y=y[-1],
                            arrival_lateral=lateral[-1], arrival_depth_yd=depth[-1],
                            arrival_time_s=t[-1],
                            own_sideline_yd=53.3-y[0] if side == 1 else y[0],
                            ball_lateral=(row.ball_y-y[0])*side,
                            ball_depth_yd=row.ball_x-row.los_x,
                            landing_to_arrival_yd=math.hypot(row.ball_x-x[-1], row.ball_y-y[-1])))
    routes = routes.merge(pd.DataFrame(derived), on=KEY, validate="one_to_one")
    if not np.allclose(routes.x0-routes.los_x, routes.depth_from_los_yd, atol=1e-6):
        raise ValueError("LOS depth does not match Step 2.0 start depth")
    return routes, paths


def median_path(group, paths):
    keys = [key_of(row) for _, row in group.iterrows()]
    grid = np.arange(int(round(max(paths[k]["t"][-1] for k in keys)*10))+1)/10
    stack = np.full((len(keys), len(grid), 2), np.nan)
    for i, key in enumerate(keys):
        p = paths[key]
        inside = (grid >= p["t"][0]-1e-8) & (grid <= p["t"][-1]+1e-8)
        for axis, column in enumerate(["lateral", "depth"]):
            stack[i, inside, axis] = np.interp(grid[inside], p["t"], p[column])
    support = np.isfinite(stack[:, :, 0]).sum(axis=0)
    minimum = max(5, math.ceil(.25*len(keys)))
    keep = np.logical_and.accumulate(support >= minimum)
    med = np.nanmedian(stack, axis=0)
    return pd.DataFrame({"time_since_snap_s": grid, "median_lateral_yd": med[:, 0],
                         "median_depth_from_los_yd": med[:, 1], "n_paths_observed": support,
                         "minimum_support_n": minimum, "included_in_median": keep})


def stratified_sample(group):
    """Largest-remainder allocation across complete / not complete, up to 200."""
    rng = np.random.default_rng(SEED)
    size = min(200, len(group))
    strata = [group.loc[group.pass_result.eq("C")], group.loc[~group.pass_result.eq("C")]]
    exact = np.array([len(g) for g in strata])*size/len(group)
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact-counts), kind="stable")[:size-counts.sum()]:
        counts[i] += 1
    indices = np.concatenate([rng.choice(g.index.to_numpy(), n, replace=False)
                              for g, n in zip(strata, counts) if n])
    return group.loc[np.sort(indices)]


def summarize(label, group, sampled):
    detected = group.loc[bool_col(group.break_detected), "break_depth_yd"]
    return dict(route_label=label, n_routes=len(group),
                n_complete=int(group.pass_result.eq("C").sum()),
                completion_rate=float(group.pass_result.eq("C").mean()),
                median_break_depth_yd=detected.median(), n_break_depth=int(detected.notna().sum()),
                median_arrival_depth_yd=group.arrival_depth_yd.median(),
                median_throw_time_s=group.throw_time_s.median(),
                median_sep_at_throw_yd=group.sep_at_throw.median(),
                n_sep_at_throw=int(group.sep_at_throw.notna().sum()),
                median_sep_at_arrival_yd=group.sep_at_arrival.median(),
                n_sep_at_arrival=int(group.sep_at_arrival.notna().sum()),
                median_own_sideline_yd=group.own_sideline_yd.median(),
                p10_own_sideline_yd=group.own_sideline_yd.quantile(.1),
                p90_own_sideline_yd=group.own_sideline_yd.quantile(.9),
                n_sampled=len(sampled),
                sampled_completion_rate=sampled.pass_result.eq("C").mean(),
                n_landing_excluded_sampled=int(sampled.landing_to_arrival_yd.gt(40).sum()),
                n_landing_excluded_all=int(group.landing_to_arrival_yd.gt(40).sum()))


def axes_route(ax):
    ax.axhline(0, color="#354358", lw=1.2, zorder=2)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(alpha=.13, linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Lateral movement (yd; + toward own sideline)")
    ax.set_ylabel("Depth from line of scrimmage (yd)")


def result_marker(ax, x, y, complete, size=18, alpha=.7, zorder=4):
    if complete:
        ax.scatter(x, y, s=size, c=GREEN, marker="o", alpha=alpha, edgecolors="none", zorder=zorder)
    else:
        ax.scatter(x, y, s=size*1.2, c=RED, marker="x", alpha=alpha, linewidths=.9, zorder=zorder)


def spaghetti(label, sampled, paths, median, stat, out):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([.085, .20, .63, .68])
    axes_route(ax)
    ax.axvspan(stat["p10_own_sideline_yd"], stat["p90_own_sideline_yd"], color="#76879a", alpha=.10)
    ax.axvline(stat["median_own_sideline_yd"], color="#64758b", ls=":", lw=1.5)
    for _, row in sampled.iterrows():
        p = paths[key_of(row)]
        i = p["throw_index"]
        ax.plot(p["lateral"][:i+1], p["depth"][:i+1], c=BLUE, alpha=.16, lw=.65)
        ax.plot(p["lateral"][i:], p["depth"][i:], c=BLUE, alpha=.23, lw=.65, ls="--")
        ax.scatter(p["lateral"][i], p["depth"][i], c=ORANGE, s=7, alpha=.48, zorder=3)
        result_marker(ax, p["lateral"][-1], p["depth"][-1], row.pass_result == "C", size=12, alpha=.5)
        if row.landing_to_arrival_yd <= 40:
            ax.plot([p["lateral"][-1], row.ball_lateral], [p["depth"][-1], row.ball_depth_yd],
                    color=GOLD, lw=.5, ls=":", alpha=.4)
            ax.scatter(row.ball_lateral, row.ball_depth_yd, c=GOLD, marker="*", s=24, alpha=.65, zorder=4)
    m = median.loc[median.included_in_median]
    ax.plot(m.median_lateral_yd, m.median_depth_from_los_yd, color="white", lw=4.7, zorder=6)
    ax.plot(m.median_lateral_yd, m.median_depth_from_los_yd, color="#122e4c", lw=2.5, zorder=7)
    ax.margins(.04)
    fig.text(.085, .945, f"{label}  |  {stat['n_routes']:,} targeted routes", fontsize=21, weight="bold")
    fig.text(.085, .91, f"2023 Weeks 1–18  •  {len(sampled)} sampled paths  •  statistics use every route in this label", fontsize=11, color="#576475")
    panel = (f"ALL {stat['n_routes']:,} ROUTES\n\n"
             f"Completion rate\n{stat['completion_rate']:.1%}\n\n"
             f"Median detected break depth\n{stat['median_break_depth_yd']:.1f} yd  (n={stat['n_break_depth']:,})\n\n"
             f"Median arrival depth\n{stat['median_arrival_depth_yd']:.1f} yd\n\n"
             f"Median throw time\n{stat['median_throw_time_s']:.1f} s\n\n"
             f"Median separation at throw\n{stat['median_sep_at_throw_yd']:.2f} yd  (n={stat['n_sep_at_throw']:,})\n\n"
             f"Median separation at arrival\n{stat['median_sep_at_arrival_yd']:.2f} yd  (n={stat['n_sep_at_arrival']:,})")
    fig.text(.75, .86, panel, va="top", fontsize=11.5, linespacing=1.25,
             bbox=dict(boxstyle="round,pad=.8", facecolor="#f3f5f7", edgecolor="none"))
    handles = [Line2D([], [], c=BLUE, lw=1.2, label="Before throw"),
               Line2D([], [], c=BLUE, ls="--", label="After throw"),
               Line2D([], [], c=ORANGE, marker="o", ls="", markersize=5, label="Throw proxy"),
               Line2D([], [], c=GREEN, marker="o", ls="", label="Complete: arrival"),
               Line2D([], [], c=RED, marker="x", ls="", label="Incomplete / intercepted: arrival"),
               Line2D([], [], c=GOLD, marker="*", ls=":", label="Ball landing + connector"),
               Line2D([], [], c="#122e4c", lw=2.5, label="Same-time median (all routes)"),
               Line2D([], [], c="#64758b", ls=":", label="Median own sideline"),
               Patch(facecolor="#76879a", alpha=.16, label="Own sideline: 10th–90th percentile")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.075, .065), ncol=3,
               frameon=False, fontsize=9, columnspacing=1.6)
    fig.text(.085, .045, f"Landing stars omitted (>40 yd from arrival): {stat['n_landing_excluded_sampled']} of {len(sampled)} sampled routes. "
             "Sideline is a distribution across plays.", fontsize=9, color="#555f6b")
    fig.text(.085, .024, SOURCE, fontsize=8.5, color="#555f6b")
    fig.text(.085, .006, WARNING, fontsize=8.5, color="#555f6b")
    path = out / "spaghetti" / f"route_{label}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)


def overview(routes, medians, out):
    all_lateral = list(routes.arrival_lateral)
    all_depth = list(routes.arrival_depth_yd)
    for m in medians.values():
        keep = m.loc[m.included_in_median]
        all_lateral.extend(keep.median_lateral_yd)
        all_depth.extend(keep.median_depth_from_los_yd)
    xlim = (5*math.floor(min(all_lateral)/5)-5, 5*math.ceil(max(all_lateral)/5)+5)
    ylim = (min(-5, 5*math.floor(min(all_depth)/5)-5), 5*math.ceil(max(all_depth)/5)+5)
    fig, axs = plt.subplots(3, 4, figsize=(19, 15), sharex=True, sharey=True)
    for ax, label in zip(axs.flat, LABELS):
        g = routes.loc[routes.route_label.eq(label)]
        axes_route(ax)
        for complete in [True, False]:
            subset = g.loc[g.pass_result.eq("C").eq(complete)]
            result_marker(ax, subset.arrival_lateral, subset.arrival_depth_yd, complete, size=8, alpha=.22)
        m = medians[label].loc[medians[label].included_in_median]
        ax.plot(m.median_lateral_yd, m.median_depth_from_los_yd, color="#122e4c", lw=2.2, zorder=6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{label}  |  n={len(g):,}", weight="bold", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MultipleLocator(10))
    fig.suptitle("Route atlas: same-time median paths and all receiver arrival points", fontsize=23, weight="bold", y=.975)
    fig.text(.5, .945, "Identical axis limits and equal yard scales in every panel  •  All labeled targets, 2023 Weeks 1–18", ha="center", fontsize=13)
    fig.supxlabel("Lateral movement (yd; + toward own sideline)", y=.075, fontsize=14)
    fig.supylabel("Depth from line of scrimmage (yd)", x=.025, fontsize=14)
    fig.legend(handles=[Line2D([], [], c="#122e4c", lw=2.2, label="Same-time median path"),
                        Line2D([], [], c=GREEN, marker="o", ls="", label="Complete: arrival"),
                        Line2D([], [], c=RED, marker="x", ls="", label="Incomplete / intercepted: arrival")],
               loc="lower center", bbox_to_anchor=(.5, .035), frameon=False, ncol=3)
    fig.text(.06, .025, SOURCE, fontsize=9)
    fig.text(.06, .01, WARNING, fontsize=9)
    fig.subplots_adjust(left=.06, right=.98, top=.91, bottom=.12, hspace=.23, wspace=.1)
    fig.savefig(out / "spaghetti" / "all_routes_overview.png", dpi=160)
    plt.close(fig)
    return {"lateral_yd": xlim, "depth_from_los_yd": ylim}


def select_representatives(routes, paths, medians):
    chosen, audits = [], []
    for label in LABELS:
        g = routes.loc[routes.route_label.eq(label)].copy()
        base = (g.play_nullified_by_penalty.eq("N") & ~bool_col(g.side_flag)
                & g.landing_to_arrival_yd.le(5))
        throw_q = g.throw_time_s.quantile([.25, .75]).to_numpy()
        depth_q = g.arrival_depth_yd.quantile([.25, .75]).to_numpy()
        iqr = g.throw_time_s.between(*throw_q) & g.arrival_depth_yd.between(*depth_q)
        defended = g.n_defenders_at_arrival.ge(1)
        stages = [("none", base & iqr & defended),
                  ("IQR", base & defended), ("IQR;defender", base)]
        counts = {name: int(mask.sum()) for name, mask in stages}
        for relaxed, mask in stages:
            candidates = g.loc[mask].copy()
            if not candidates.empty:
                break
        if candidates.empty:
            raise ValueError(f"No representative for {label}, even after allowed relaxations")
        median = medians[label].loc[medians[label].included_in_median]
        distances = []
        for _, row in candidates.iterrows():
            p = paths[key_of(row)]
            m = median.loc[median.time_since_snap_s.between(p["t"][0]-1e-8, p["t"][-1]+1e-8)]
            if m.empty:
                distances.append(np.inf)
                continue
            lateral = np.interp(m.time_since_snap_s, p["t"], p["lateral"])
            depth = np.interp(m.time_since_snap_s, p["t"], p["depth"])
            distances.append(float(np.hypot(lateral-m.median_lateral_yd, depth-m.median_depth_from_los_yd).mean()))
        candidates["selection_distance_yd"] = distances
        selected = candidates.sort_values(["selection_distance_yd", "game_id", "play_id", "nfl_id"]).iloc[0].copy()
        selected["relaxed_conditions"] = relaxed
        chosen.append(selected)
        audits.append(dict(route_label=label, candidate_counts=counts,
                           throw_time_iqr_s=throw_q.tolist(), arrival_depth_iqr_yd=depth_q.tolist(),
                           selected_game_id=int(selected.game_id), selected_play_id=int(selected.play_id),
                           relaxed_conditions=relaxed))
    return pd.DataFrame(chosen), audits


def load_selected_context(selected, data_dir, train_dir):
    wanted = pd.MultiIndex.from_frame(selected[["game_id", "play_id"]])
    defenders = []
    cols = ["game_id", "play_id", "frame_id", "receiver_nfl_id", "defender_nfl_id",
            "x", "y", "phase", "t", "defender_name"]
    for chunk in read_csv(data_dir / "target_defender_frames.csv", usecols=cols, chunksize=250_000):
        mask = pd.MultiIndex.from_frame(chunk[["game_id", "play_id"]]).isin(wanted)
        defenders.append(chunk.loc[mask])
    defenders = pd.concat(defenders, ignore_index=True)
    input_rows, output_rows = [], []
    cols_in = ["game_id", "play_id", "nfl_id", "frame_id", "play_direction", "player_name", "player_role", "x", "y"]
    for week in sorted(selected.week.unique()):
        for phase, usecols, holder in [("input", cols_in, input_rows),
                                       ("output", ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"], output_rows)]:
            for chunk in read_csv(train_dir / f"{phase}_2023_w{int(week):02d}.csv", usecols=usecols, chunksize=250_000):
                mask = pd.MultiIndex.from_frame(chunk[["game_id", "play_id"]]).isin(wanted)
                holder.append(chunk.loc[mask])
    inputs = pd.concat(input_rows, ignore_index=True)
    left = inputs.play_direction.eq("left")
    inputs["x"] = np.where(left, 120-inputs.x, inputs.x)
    inputs["y"] = np.where(left, 53.3-inputs.y, inputs.y)
    outputs = pd.concat(output_rows, ignore_index=True)
    outputs = outputs.merge(selected[["game_id", "play_id", "play_direction"]], on=["game_id", "play_id"], validate="many_to_one")
    left = outputs.play_direction.eq("left")
    outputs["x"] = np.where(left, 120-outputs.x, outputs.x)
    outputs["y"] = np.where(left, 53.3-outputs.y, outputs.y)
    return defenders, inputs, outputs


def draw_field(ax, row):
    xmin = row.los_x - 12
    xmax = max(row.arrival_x, row.ball_x)+8
    if xmax-xmin < 20:
        xmax = xmin+20
    ax.set_facecolor("#e6ece9")
    ax.axvspan(0, 120, color="#387757", zorder=0)
    for lo, hi in [(0, 10), (110, 120)]:
        ax.axvspan(lo, hi, color="#28583f", zorder=.1)
    for x in range(0, 121, 5):
        ax.plot([x, x], [0, 53.3], color="white", alpha=.7 if x % 10 == 0 else .28,
                lw=1.2 if x % 10 == 0 else .6, zorder=.3)
    for x in range(10, 111):
        for y in [23.58, 29.72]:
            ax.plot([x, x], [y-.25, y+.25], color="white", lw=.6, alpha=.65, zorder=.3)
        for lo, hi in [(0, .55), (52.75, 53.3)]:
            ax.plot([x, x], [lo, hi], color="white", lw=.55, alpha=.6, zorder=.3)
    for x in range(10, 111, 10):
        if xmin <= x <= xmax:
            number = min(x-10, 110-x)
            for y, rotation in [(5.3, 0), (48, 180)]:
                ax.text(x, y, str(number), color="white", alpha=.48, ha="center", va="center",
                        fontsize=14, weight="bold", rotation=rotation, zorder=.4)
    ax.axhline(0, color="white", lw=1.5)
    ax.axhline(53.3, color="white", lw=1.5)
    ax.axvline(row.los_x, color="#142b59", lw=2.6, zorder=1)
    ax.text(row.los_x+.35, 51.2, "LOS", color="#101f3e", fontsize=10, weight="bold", zorder=8)
    first = row.los_x+row.yards_to_go
    if xmin <= first <= xmax:
        ax.axvline(first, color="#ffe457", lw=2.2, zorder=1)
        ax.text(first+.35, 1.4, "1st down", color="#fff1a6", fontsize=9, zorder=8)
    ax.set(xlim=(xmin, xmax), ylim=(0, 53.3), xlabel="Field x (yd; offense moves right →)", ylabel="Field y (yd)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.xaxis.set_major_locator(MultipleLocator(5))


def phase_path(ax, p, color, lw, alpha=1, zorder=3):
    p = p.sort_values("t")
    pre, post = p.loc[p.phase.eq("input")], p.loc[p.phase.eq("output")]
    if not pre.empty:
        ax.plot(pre.x, pre.y, color=color, lw=lw, alpha=alpha, zorder=zorder)
    if not post.empty:
        joined = pd.concat([pre.tail(1), post])
        ax.plot(joined.x, joined.y, color=color, lw=lw, ls="--", alpha=alpha, zorder=zorder)


def representative(row, paths, defenders, inputs, description, out):
    p = paths[key_of(row)]
    ds = defenders.loc[defenders.game_id.eq(row.game_id) & defenders.play_id.eq(row.play_id)
                       & defenders.receiver_nfl_id.eq(row.nfl_id)].copy()
    ins = inputs.loc[inputs.game_id.eq(row.game_id) & inputs.play_id.eq(row.play_id)].copy()
    i = p["throw_index"]
    nearest = {}
    for moment, idx in [("throw", i), ("arrival", len(p["t"])-1)]:
        at = ds.loc[ds.frame_id.eq(p["frame_id"][idx]) & ds.phase.eq(p["phase"][idx])].copy()
        if not at.empty:
            at["distance"] = np.hypot(at.x-p["x"][idx], at.y-p["y"][idx])
            nearest[moment] = at.sort_values(["distance", "defender_nfl_id"]).iloc[0]
            expected = row[f"sep_at_{moment}"]
            if pd.notna(expected) and abs(nearest[moment].distance-expected) > 1e-5:
                raise ValueError(f"Separation marker disagrees with Table F: {key_of(row)}, {moment}")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([.075, .20, .53, .62])
    draw_field(ax, row)
    nearest_throw_id = nearest["throw"].defender_nfl_id if "throw" in nearest else None
    for defender_id, d in ds.groupby("defender_nfl_id"):
        highlight = defender_id == nearest_throw_id
        phase_path(ax, d, "#ff9c99" if highlight else "#ed7475", 2.1 if highlight else .95,
                   .95 if highlight else .7, zorder=4 if highlight else 2)
        start = d.sort_values("t").iloc[0]
        ax.scatter(start.x, start.y, marker="x", c="#ffaaa5", s=22, lw=1.1, zorder=5)
    if "throw" in nearest:
        d = nearest["throw"]
        ax.annotate(f"NEAREST DEFENDER AT THROW\n{d.defender_name}\nHighlighted red path", (d.x, d.y),
                    xytext=(.65, .72), textcoords=fig.transFigure, color="#843b39", fontsize=11,
                    linespacing=1.6, va="top",
                    bbox=dict(facecolor="#fff3f1", edgecolor="none", pad=9),
                    arrowprops=dict(arrowstyle="-", color="#cc8b84", lw=.9), zorder=9)
    passer = ins.loc[ins.player_role.eq("Passer")]
    for _, qb in passer.groupby("nfl_id"):
        qb = qb.sort_values("frame_id")
        ax.plot(qb.x, qb.y, c="#111820", lw=1.8, zorder=4)
    others = ins.loc[ins.player_role.eq("Other Route Runner") & ~ins.nfl_id.eq(row.nfl_id)]
    for _, runner in others.groupby("nfl_id"):
        runner = runner.sort_values("frame_id")
        ax.plot(runner.x, runner.y, c="#d3d6d7", lw=1, alpha=.8, zorder=2)
    ax.plot(p["x"][:i+1], p["y"][:i+1], c="white", lw=5.4, zorder=5)
    ax.plot(p["x"][:i+1], p["y"][:i+1], c=BLUE, lw=3.4, zorder=6)
    ax.plot(p["x"][i:], p["y"][i:], c="white", lw=5.4, ls="--", zorder=5)
    ax.plot(p["x"][i:], p["y"][i:], c=BLUE, lw=3.4, ls="--", zorder=6)
    ax.scatter(p["x"][0], p["y"][0], c=BLUE, s=65, edgecolors="white", zorder=7)
    ax.scatter(p["x"][i], p["y"][i], c=ORANGE, s=65, edgecolors="white", zorder=8)
    # A large arrival circle remains visible when ball and receiver nearly coincide.
    result_marker(ax, p["x"][-1], p["y"][-1], row.pass_result == "C", size=310, alpha=1, zorder=9)
    ax.scatter(row.ball_x, row.ball_y, marker="*", c="#ffd44d", edgecolors="#8b6510", lw=.7, s=135, zorder=10)
    qb_at_throw = passer.loc[passer.frame_id.eq(p["frame_id"][i])].sort_values("nfl_id")
    if not qb_at_throw.empty:
        qb = qb_at_throw.iloc[0]
        ax.plot([qb.x, row.ball_x], [qb.y, row.ball_y], color="#ffdc6a", lw=1.3, ls=":", zorder=5)
        arrival_key = "Green circle" if row.pass_result == "C" else "Red cross"
        fig.text(.65, .275, f"Gold star: ball landing\nDotted gold line: approximate flight\n{arrival_key}: receiver arrival", fontsize=10,
                 linespacing=1.6, color="#655123")
    else:
        ax.text(.015, .015, "Passer unavailable at throw: ball flight line omitted", transform=ax.transAxes,
                color="white", fontsize=9, zorder=10)
    for moment, idx, text_y in [("throw", i, .56), ("arrival", len(p["t"])-1, .41)]:
        if moment in nearest:
            d = nearest[moment]
            ax.plot([p["x"][idx], d.x], [p["y"][idx], d.y], ls="--", c="#c9cdcd", lw=1.5, zorder=8)
            ax.annotate(f"{moment.upper()} SEPARATION\n{row[f'sep_at_{moment}']:.2f} yd\nNearest observed defender",
                        ((p["x"][idx]+d.x)/2, (p["y"][idx]+d.y)/2), xytext=(.65, text_y),
                        textcoords=fig.transFigure, fontsize=11, color="#233343", linespacing=1.6, va="top",
                        bbox=dict(facecolor="#f1f4f6", edgecolor="none", pad=9),
                        arrowprops=dict(arrowstyle="-", color="#929da5", lw=.9), zorder=11)
    result_name = {"C": "Complete", "I": "Incomplete", "IN": "Intercepted"}.get(row.pass_result, str(row.pass_result))
    fig.text(.075, .955, f"{row.route_label}  |  {row.player_name}  |  {result_name}", fontsize=21, weight="bold")
    fig.text(.075, .92, f"Game {int(row.game_id)} • Play {int(row.play_id)}  |  Down {int(row.down)} & {int(row.yards_to_go)}  |  "
             f"{row.team_coverage_man_zone} / {row.team_coverage_type}", fontsize=11.5)
    fig.text(.075, .892, f"Throw proxy {row.throw_time_s:.1f} s  |  Flight {row.ball_flight_s:.1f} s  |  Pass length {row.pass_length:g} yd  |  "
             f"Selection distance {row.selection_distance_yd:.2f} yd  |  Relaxed: {row.relaxed_conditions}", fontsize=11)
    fig.text(.075, .862, textwrap.shorten(str(description), width=120, placeholder="…"), fontsize=10.5, color="#4c5968")
    handles = [Line2D([], [], c=BLUE, lw=3, label="Target before throw"),
               Line2D([], [], c=BLUE, lw=3, ls="--", label="Target after throw"),
               Line2D([], [], marker="o", c=BLUE, markeredgecolor="white", ls="", label="Target start"),
               Line2D([], [], marker="o", c=ORANGE, ls="", label="Throw proxy"),
               Line2D([], [], marker="o", c=GREEN, ls="", label="Complete: arrival"),
               Line2D([], [], marker="x", c=RED, ls="", label="Incomplete / intercepted"),
               Line2D([], [], c="#ed7475", marker="x", label="Coverage defender (start ×)"),
               Line2D([], [], c="#ff9c99", lw=2.2, label="Nearest defender at throw"),
               Line2D([], [], c="#111820", label="QB (input only)"),
               Line2D([], [], c="#a6acae", label="Other runners (input only)"),
               Line2D([], [], c=GOLD, marker="*", ls=":", label="Ball landing / approximate flight"),
               Line2D([], [], c="#91999f", ls="--", label="Nearest-defender separation")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .065), ncol=4,
               fontsize=9, frameon=False, columnspacing=1.5)
    fig.text(.075, .05, "Defender paths: solid before throw, dashed after throw. Nearest defender may change between moments.", fontsize=9)
    fig.text(.075, .028, SOURCE, fontsize=8.5)
    fig.text(.075, .009, WARNING, fontsize=8.5)
    filename = f"route_{row.route_label}_{int(row.game_id)}_{int(row.play_id)}.png"
    fig.savefig(out / "representative" / filename, dpi=160)
    plt.close(fig)
    return dict(game_id=int(row.game_id), play_id=int(row.play_id),
                missing_passer=passer.empty, missing_passer_at_throw=qb_at_throw.empty,
                nearest_throw_defender_nfl_id=int(nearest_throw_id) if nearest_throw_id is not None else None,
                separation_markers_verified=list(nearest))


def write_gallery(out, selected):
    sections = []
    for _, row in selected.iterrows():
        label = row.route_label
        filename = f"representative/route_{label}_{int(row.game_id)}_{int(row.play_id)}.png"
        sections.append(f'<section id="{label}"><h2>{label}</h2><div class="pair">'
                        f'<a href="spaghetti/route_{label}.png"><img src="spaghetti/route_{label}.png" alt="{label} route distribution"></a>'
                        f'<a href="{filename}"><img src="{filename}" alt="{label} representative play"></a></div>'
                        f'<p>Representative: {html.escape(row.player_name)}; game {int(row.game_id)}, play {int(row.play_id)}; '
                        f'result {row.pass_result}. Selection distance {row.selection_distance_yd:.2f} yd.</p></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route Visualizations v2 — 2023</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#243343;background:#f5f6f8;margin:0 auto;padding:32px;max-width:1600px}
h1{font-size:30px;margin-bottom:8px}h2{border-top:1px solid #cfd5db;padding-top:24px}
a{color:#165b9a}nav a{display:inline-block;margin:0 16px 8px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
img{width:100%;background:white}.overview{max-width:1100px;display:block;margin:auto}section{margin-top:32px}
@media(max-width:900px){.pair{grid-template-columns:1fr}body{padding:16px}}
</style><h1>Route Visualizations v2</h1>
<p>2023 Weeks 1–18 · 14,105 labeled targeted receiver routes · 12 route labels</p>
<p><a href="VISUALIZATION_READING_GUIDE.md">Full English reading guide</a> · <a href="README.md">Methods and rebuild instructions</a> ·
<a href="label_summary.csv">All-route statistics</a> · <a href="representative_plays.csv">Selected plays</a></p>
<p>Start with the atlas, then compare each route distribution with its selected real-field play. Click any figure for its full-resolution PNG.</p>
<p><strong>Arrival is the last tracked frame, not a verified catch event.</strong> Gold stars show ball landing, and the blue path changes from solid to dashed at the throw proxy.</p><nav>'''
    page += " ".join(f'<a href="#{label}">{label}</a>' for label in LABELS)
    page += '</nav><a href="spaghetti/all_routes_overview.png"><img class="overview" src="spaghetti/all_routes_overview.png" alt="Common-scale route atlas"></a>'
    page += "".join(sections) + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    for directory in [out / "spaghetti", out / "representative"]:
        directory.mkdir(parents=True, exist_ok=True)
    initial = read_csv(args.data_dir / "routes.csv", usecols=KEY+["is_targeted", "route_label"])
    initial = initial.loc[bool_col(initial.is_targeted) & initial.route_label.notna()]
    wanted = pd.MultiIndex.from_frame(initial[["game_id", "play_id"]].drop_duplicates())
    meta = scan_metadata(args.train_dir, wanted)
    routes, paths = load_routes(args.data_dir, meta)
    checks = dict(total_labeled_target_routes=len(routes), expected_total=14105,
                  route_counts_per_label={str(k): int(v) for k, v in routes.route_label.value_counts().items()},
                  seed=SEED, ball_landing_flip_validation={},
                  inputs={"data_dir": str(args.data_dir.resolve()), "train_dir": str(args.train_dir.resolve()),
                          "supplementary": str(args.supplementary.resolve())})
    if len(routes) != 14105 or set(routes.route_label) != set(LABELS):
        checks["status"] = "failed_cohort_check"
        (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        raise ValueError("Expected 14,105 labeled target routes and 12 route labels")
    for direction in ["left", "right"]:
        complete = routes.loc[routes.pass_result.eq("C") & routes.play_direction.eq(direction)]
        checks["ball_landing_flip_validation"][direction] = {
            "n_completions": len(complete),
            "median_landing_to_arrival_yd": float(complete.landing_to_arrival_yd.median()),
            "p90_landing_to_arrival_yd": float(complete.landing_to_arrival_yd.quantile(.9))}
    valid = all(v["n_completions"] > 0 and v["median_landing_to_arrival_yd"] < 2
                for v in checks["ball_landing_flip_validation"].values())
    checks["ball_landing_flip_passed"] = valid
    if not valid:
        checks["status"] = "stopped_ball_landing_flip_validation"
        (out / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        raise ValueError("Ball landing flip check failed; stopped before drawing")
    progress(f"Ball landing validation passed: {checks['ball_landing_flip_validation']}")
    medians, stats, samples, median_rows = {}, [], [], []
    for label in LABELS:
        group = routes.loc[routes.route_label.eq(label)]
        sampled = stratified_sample(group)
        median = median_path(group, paths)
        medians[label] = median
        median_rows.append(median.assign(route_label=label))
        stat = summarize(label, group, sampled)
        stats.append(stat)
        samples.append(sampled[KEY+["route_label", "pass_result", "landing_to_arrival_yd"]])
        spaghetti(label, sampled, paths, median, stat, out)
        progress(f"Built distribution: {label}")
    pd.DataFrame(stats).to_csv(out / "label_summary.csv", index=False)
    pd.concat(samples).to_csv(out / "sampled_routes.csv", index=False)
    pd.concat(median_rows).to_csv(out / "median_path_support.csv", index=False)
    checks["overview_axis_limits"] = overview(routes, medians, out)
    selected, audits = select_representatives(routes, paths, medians)
    checks["representative_selection"] = audits
    progress("Loading selected-play coverage defenders and official input/output context")
    defenders, inputs, outputs = load_selected_context(selected, args.data_dir, args.train_dir)
    supplementary = read_csv(args.supplementary)
    supplementary = supplementary.loc[supplementary.season.eq(2023)]
    descriptions = supplementary.set_index(["game_id", "play_id"]).play_description
    play_checks = []
    for _, row in selected.iterrows():
        official_target = outputs.loc[outputs.game_id.eq(row.game_id) & outputs.play_id.eq(row.play_id)
                                      & outputs.nfl_id.eq(row.nfl_id)].sort_values("frame_id")
        p = paths[key_of(row)]
        if len(official_target) != int((p["phase"] == "output").sum()):
            raise ValueError(f"Official output frame count mismatch: {key_of(row)}")
        observed = np.column_stack([p["x"][p["phase"] == "output"], p["y"][p["phase"] == "output"]])
        if not np.allclose(official_target[["x", "y"]].to_numpy(), observed, atol=1e-6):
            raise ValueError(f"Official output coordinates mismatch: {key_of(row)}")
        desc = descriptions.loc[(int(row.game_id), int(row.play_id))]
        play_checks.append(representative(row, paths, defenders, inputs, desc, out))
        progress(f"Built field example: {row.route_label}, {int(row.game_id)}, {int(row.play_id)}")
    columns = ["route_label", "game_id", "play_id", "nfl_id", "player_name", "selection_distance_yd",
               "relaxed_conditions", "pass_result", "sep_at_throw", "sep_at_arrival", "throw_time_s", "ball_flight_s",
               "landing_to_arrival_yd", "arrival_depth_yd", "play_direction", "week", "down", "yards_to_go",
               "n_defenders_at_arrival", "side_flag", "play_nullified_by_penalty"]
    selected[columns].rename(columns={"nfl_id": "receiver_nfl_id"}).to_csv(out / "representative_plays.csv", index=False)
    checks["landing_points_excluded_from_drawing_per_label"] = {
        s["route_label"]: {"sampled_excluded": s["n_landing_excluded_sampled"],
                           "all_routes_over_40yd": s["n_landing_excluded_all"]} for s in stats}
    checks["selected_play_checks"] = play_checks
    checks["number_of_plays_missing_passer"] = sum(p["missing_passer"] for p in play_checks)
    checks["number_of_plays_missing_passer_at_throw"] = sum(p["missing_passer_at_throw"] for p in play_checks)
    checks["selected_official_output_paths_verified"] = len(selected)
    checks["start_depth_matches_step2_los"] = True
    checks["status"] = "complete"
    checks["png_dpi"] = 160
    checks["expected_png_count"] = 25
    checks["actual_png_count"] = len(list((out / "spaghetti").glob("*.png")))+len(list((out / "representative").glob("*.png")))
    (out / "checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False), encoding="utf-8")
    write_gallery(out, selected)
    progress(f"Complete: {checks['actual_png_count']} PNGs saved in {out}")


if __name__ == "__main__":
    main()
