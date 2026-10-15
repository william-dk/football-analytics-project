"""Execute professor-feedback steps 1-4 using local official weekly inputs.

Run from the repository root. No clustering or outcome prediction is fitted.
Input-end is an observed boundary, not a verified release timestamp.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_network.separation import DB_POSITIONS, PLAY_KEY, ROW_KEY, nearest_defenders

ROOT = Path("data/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final")
COLS = ROW_KEY + ["player_name", "player_position", "player_role", "player_side", "x", "y",
                  "play_direction", "absolute_yardline_number"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.item()), encoding="utf-8")


def summarize_receiver_play(frames):
    keys = PLAY_KEY + ["receiver_id"]
    frames = frames.sort_values(keys + ["frame_id"])
    summary = frames.groupby(keys, sort=False).agg(
        player_name=("player_name", "first"), player_position=("player_position", "first"),
        is_targeted=("is_targeted", "first"), input_end_frame=("input_end_frame", "first"),
        frames_observed=("frame_id", "size"), first_frame=("frame_id", "min"),
        last_frame=("frame_id", "max"), db_frames=("separation_db_yards", "count"),
        mean_input_db_yards=("separation_db_yards", "mean"),
        min_input_db_yards=("separation_db_yards", "min"),
        max_input_db_yards=("separation_db_yards", "max"),
        mean_input_coverage_yards=("separation_coverage_yards", "mean"),
    ).reset_index()
    endpoints = frames.loc[frames.frame_id.eq(frames.input_end_frame),
                           keys + ["separation_db_yards", "nearest_db_id", "nearest_db_position",
                                   "separation_coverage_yards", "nearest_coverage_id", "nearest_coverage_position"]]
    endpoints = endpoints.rename(columns={"separation_db_yards": "input_end_db_yards",
                                          "separation_coverage_yards": "input_end_coverage_yards"})
    summary = summary.merge(endpoints, on=keys, how="left", validate="one_to_one")
    switches = frames.groupby(keys).nearest_db_id.apply(lambda s: int((s.dropna().diff().dropna() != 0).sum()))
    summary = summary.merge(switches.rename("nearest_db_switches"), on=keys, validate="one_to_one")
    summary["complete_input_track"] = (summary.first_frame.eq(1)
        & summary.last_frame.eq(summary.input_end_frame)
        & summary.frames_observed.eq(summary.input_end_frame))
    summary["complete_db_window"] = summary.complete_input_track & summary.db_frames.eq(summary.frames_observed)
    summary["input_span_seconds"] = (summary.input_end_frame - 1) / 10
    return summary


def player_summary(rows):
    return rows.groupby(["receiver_id", "player_name", "player_position"], sort=False).agg(
        receiver_plays=("play_id", "size"), games=("game_id", "nunique"),
        targets=("is_targeted", "sum"),
        mean_input_end_db_yards=("input_end_db_yards", "mean"),
        median_input_end_db_yards=("input_end_db_yards", "median"),
        p25_input_end_db_yards=("input_end_db_yards", lambda s: s.quantile(.25)),
        p75_input_end_db_yards=("input_end_db_yards", lambda s: s.quantile(.75)),
        mean_of_play_mean_input_db_yards=("mean_input_db_yards", "mean"),
        mean_input_span_seconds=("input_span_seconds", "mean"),
    ).reset_index().assign(at_least_30_receiver_plays=lambda x: x.receiver_plays.ge(30)).sort_values(
        ["mean_input_end_db_yards", "receiver_id"], ascending=[False, True])


def verify_week(raw, frames):
    """Independently recompute a deterministic spread of observations by loops."""
    errors = []
    sample = frames.iloc[np.unique(np.linspace(0, len(frames)-1, min(24, len(frames))).astype(int))]
    for row in sample.itertuples():
        defense = raw.loc[raw.game_id.eq(row.game_id) & raw.play_id.eq(row.play_id)
            & raw.frame_id.eq(row.frame_id) & raw.player_side.eq("Defense")
            & raw.player_role.eq("Defensive Coverage")]
        for group in ["db", "coverage"]:
            candidates = defense.loc[defense.player_position.isin(DB_POSITIONS)] if group == "db" else defense
            choices = sorted((float(np.hypot(row.x-d.x, row.y-d.y)), int(d.nfl_id)) for d in candidates.itertuples())
            actual = getattr(row, f"separation_{group}_yards")
            if choices:
                assert int(getattr(row, f"nearest_{group}_id")) == choices[0][1]
                errors.append(abs(actual - choices[0][0]))
            else:
                assert pd.isna(actual)
    comparable = frames.dropna(subset=["separation_db_yards", "separation_coverage_yards"])
    assert (comparable.separation_db_yards + 1e-10 >= comparable.separation_coverage_yards).all()
    # A rigid 180-degree rotation must preserve distance and nearest-player ID.
    game, play = raw[PLAY_KEY].iloc[0]
    one = raw.loc[raw.game_id.eq(game) & raw.play_id.eq(play)].copy()
    before = nearest_defenders(one)
    one.x, one.y = 120-one.x, 53.3-one.y
    after = nearest_defenders(one)
    assert np.allclose(before.separation_db_yards, after.separation_db_yards, equal_nan=True)
    return len(sample), max(errors, default=0)


def svg_case(raw, frame, receiver, meta):
    """Self-contained SVG: field trajectories and distance-versus-time curve."""
    raw = raw.copy()
    if raw.play_direction.iloc[0] == "left":
        raw.x, raw.y = 120-raw.x, 53.3-raw.y
    target = raw.loc[raw.nfl_id.eq(receiver)].sort_values("frame_id")
    traces = raw.loc[raw.player_role.eq("Defensive Coverage") | raw.nfl_id.eq(receiver)]
    xmin, xmax = max(0, traces.x.min()-3), min(120, traces.x.max()+3)
    ymin, ymax = max(0, traces.y.min()-3), min(53.3, traces.y.max()+3)
    # Preserve spatial aspect ratio so apparent separation is not distorted.
    scale = min(490/max(xmax-xmin, 1), 270/max(ymax-ymin, 1))
    fx = lambda x: 65+(x-xmin)*scale
    fy = lambda y: 340-(y-ymin)*scale
    esc = lambda s: html.escape(str(s))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="470" viewBox="0 0 1100 470">',
        '<rect width="1100" height="470" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202a34;font-size:12px}.title{font-size:16px;font-weight:bold}</style>',
        f'<text x="25" y="27" class="title">{esc(meta["player_name"])} | {int(meta["game_id"])}/{int(meta["play_id"])} | {esc(meta["route_of_targeted_receiver"])} | {esc(meta["pass_result"])}</text>',
        '<text x="65" y="62">Observed input trajectories; offense normalized toward +x</text>',
        '<text x="650" y="62">Nearest distance over the observed input window</text>']
    for x in np.arange(np.ceil(xmin/5)*5, xmax+0.01, 5):
        parts.append(f'<line x1="{fx(x):.2f}" y1="{fy(ymin):.2f}" x2="{fx(x):.2f}" y2="{fy(ymax):.2f}" stroke="#dde2e7"/><text x="{fx(x):.2f}" y="360">{x:.0f}</text>')
    for y in np.arange(np.ceil(ymin/10)*10, ymax+0.01, 10):
        parts.append(f'<text x="30" y="{fy(y):.2f}">{y:.0f}</text>')
    label_boxes = []
    for pid, path in traces.groupby("nfl_id", sort=True):
        path = path.sort_values("frame_id")
        is_receiver = pid == receiver
        is_db = path.player_position.iloc[0] in DB_POSITIONS
        color = "#176b91" if is_receiver else "#8c5144" if is_db else "#707b85"
        points = " ".join(f'{fx(x):.2f},{fy(y):.2f}' for x,y in path[["x","y"]].itertuples(index=False, name=None))
        dash = '' if is_db or is_receiver else ' stroke-dasharray="4 3"'
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{3 if is_receiver else 1.3}"{dash}/>')
        last = path.iloc[-1]
        parts.append(f'<circle cx="{fx(last.x):.2f}" cy="{fy(last.y):.2f}" r="3" fill="{color}"/>')
        label = f'{int(pid)} {last.player_position}'
        lx, ly = fx(last.x)+4, fy(last.y)-4
        width = len(label)*7
        for _ in range(15):
            if not any(lx < bx+bw and lx+width > bx and abs(ly-by) < 14 for bx,by,bw in label_boxes):
                break
            ly -= 15
        label_boxes.append((lx,ly,width))
        if abs(ly-(fy(last.y)-4)) > 1:
            parts.append(f'<line x1="{fx(last.x):.2f}" y1="{fy(last.y):.2f}" x2="{lx:.2f}" y2="{ly+2:.2f}" stroke="#aab2b9"/>')
        parts.append(f'<text x="{lx:.2f}" y="{ly:.2f}">{esc(label)}</text>')
    t = frame.seconds_to_input_end.to_numpy()
    span = max(-t.min(), .1)
    top = max(1, np.ceil(frame.separation_db_yards.max()/2)*2)
    gx = lambda v: 650+(v+span)/span*400
    gy = lambda v: 340-v/top*250
    for v in np.linspace(0, top, 6):
        parts.append(f'<line x1="650" x2="1050" y1="{gy(v)}" y2="{gy(v)}" stroke="#e4e7eb"/><text x="615" y="{gy(v)+4}">{v:.1f}</text>')
    for col,color,dash in [("separation_db_yards","#176b91",""),("separation_coverage_yards","#8c5144",' stroke-dasharray="5 4"')]:
        points = " ".join(f'{gx(tx):.2f},{gy(d):.2f}' for tx,d in zip(t,frame[col]) if np.isfinite(d))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    for tx in np.linspace(-span,0,5):
        parts.append(f'<text x="{gx(tx)-10:.2f}" y="360">{tx:.1f}</text>')
    parts += ['<text x="230" y="385">Normalized longitudinal x (yards)</text>',
              '<text transform="translate(16,270) rotate(-90)">Normalized lateral y (yards)</text>',
              '<text x="730" y="385">Seconds relative to input end (not verified release)</text>',
              '<text transform="translate(595,290) rotate(-90)">Distance (yards)</text>',
              '<text x="650" y="410" fill="#176b91">Solid: nearest DB</text><text x="830" y="410">Dashed: nearest tracked coverage player</text>',
              '<text x="25" y="438">Field: blue = receiver; brown = DB; gray dashed = other coverage. Dots mark final input positions.</text>',
              '<text x="25" y="458">Source: local BDB 2026 Analytics inputs, 2023 season. 10 Hz; no snap/release/arrival events. Displayed paths are observed, not predicted.</text></svg>']
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/separation_baseline"))
    args = parser.parse_args()
    out = args.output
    (out/"frames").mkdir(parents=True, exist_ok=True)
    (out/"cases").mkdir(exist_ok=True)
    sources = sorted((args.source/"train").glob("input_2023_w*.csv"))
    if len(sources) != 18:
        raise ValueError(f"Expected 18 weekly inputs; got {len(sources)}")
    context = pd.read_csv(args.source/"supplementary_data.csv", low_memory=False)
    assert not context.duplicated(PLAY_KEY).any()
    audits, summaries, seen = [], [], set()
    sample_preview = None
    for source in sources:
        raw = pd.read_csv(source, usecols=COLS)
        week = int(source.stem[-2:])
        duplicate = int(raw.duplicated(ROW_KEY).sum())
        null_keys = int(raw[ROW_KEY].isna().any(axis=1).sum())
        if duplicate or null_keys:
            raise ValueError(f"{source}: invalid keys; duplicate={duplicate}, null={null_keys}")
        keys = set(raw[PLAY_KEY].itertuples(index=False, name=None))
        assert not (keys & seen), "Plays overlap weekly partitions"
        seen |= keys
        invalid = ~np.isfinite(raw[["x","y"]].to_numpy()).all(axis=1)
        invalid_keys = set(raw.loc[invalid, PLAY_KEY].itertuples(index=False, name=None))
        # Exclude entire affected plays rather than produce artificial large gaps.
        keep = np.array([k not in invalid_keys for k in raw[PLAY_KEY].itertuples(index=False, name=None)])
        clean = raw.loc[keep].copy()
        ends = clean.groupby(PLAY_KEY).frame_id.max().rename("input_end_frame")
        frames = nearest_defenders(clean).merge(ends, on=PLAY_KEY, validate="many_to_one")
        frames["seconds_to_input_end"] = (frames.frame_id-frames.input_end_frame)/10
        nverify, error = verify_week(clean, frames)
        frames["week"] = week
        frames.to_csv(out/"frames"/f"week_{week:02d}.csv.gz", index=False, float_format="%.9f")
        if sample_preview is None:
            sample_preview = frames.head(10)
            sample_preview.to_csv(out/"source_preview.csv", index=False)
        summaries.append(summarize_receiver_play(frames).assign(week=week))
        audit = {"week":week,"source":str(source),"rows":len(raw),"plays":len(keys),
            "games":int(raw.game_id.nunique()),"duplicate_keys":duplicate,"null_keys":null_keys,
            "invalid_xy_rows":int(invalid.sum()),"excluded_invalid_xy_plays":len(invalid_keys),
            "out_of_bounds_xy_rows":int((~raw.x.between(0,120)|~raw.y.between(0,53.3)).sum()),
            "receiver_frames":len(frames),"missing_db_frames":int(frames.separation_db_yards.isna().sum()),
            "independently_verified_frames":nverify,"max_distance_error_yards":error,
            "player_positions":raw.player_position.value_counts().to_dict(),
            "roles":raw.player_role.value_counts().to_dict()}
        audits.append(audit)
        print(f"Week {week:02}: {len(keys):,} plays; {len(frames):,} receiver frames; validation passed", flush=True)
    all_rows = pd.concat(summaries, ignore_index=True)
    context_cols = PLAY_KEY + ["game_date", "route_of_targeted_receiver", "team_coverage_man_zone",
                              "pass_result", "play_nullified_by_penalty", "play_description"]
    all_rows = all_rows.merge(context[context_cols], on=PLAY_KEY, how="left", validate="many_to_one", indicator=True)
    all_rows["target_route_label"] = all_rows.route_of_targeted_receiver.where(all_rows.is_targeted)
    all_rows["eligible_summary"] = (all_rows.complete_db_window & all_rows.input_end_db_yards.notna()
        & all_rows.play_nullified_by_penalty.eq("N") & all_rows["_merge"].eq("both"))
    all_rows.to_csv(out/"receiver_play_metrics.csv", index=False)
    eligible = all_rows.loc[all_rows.eligible_summary].copy()
    wr = eligible.loc[eligible.player_position.eq("WR")]
    targeted_wr = wr.loc[wr.is_targeted]
    for name, rows in [("all_receivers",eligible),("wr_all_routes",wr),("wr_targets",targeted_wr)]:
        player_summary(rows).to_csv(out/f"players_{name}.csv", index=False)
    target = eligible.loc[eligible.is_targeted]
    groups = target.groupby(["player_position","target_route_label","team_coverage_man_zone","pass_result"], dropna=False).agg(
        targets=("play_id","size"),mean_input_end_db_yards=("input_end_db_yards","mean"),
        median_input_end_db_yards=("input_end_db_yards","median"),
        mean_of_play_mean_input_db_yards=("mean_input_db_yards","mean"))
    groups.to_csv(out/"target_context_summary.csv")
    pd.DataFrame(audits).drop(columns=["player_positions","roles"]).to_csv(out/"weekly_quality.csv",index=False)
    selected = []
    for route, outcome in [("HITCH","C"),("GO","C"),("OUT","I"),("POST","I")]:
        candidates = target.loc[target.target_route_label.eq(route) & target.pass_result.eq(outcome)]
        if not candidates.empty:
            median = candidates.input_end_db_yards.median()
            selected.append(candidates.assign(_difference=(candidates.input_end_db_yards-median).abs()).sort_values(["_difference"]+PLAY_KEY).iloc[0])
    candidates = target.loc[~target.nearest_coverage_position.isin(DB_POSITIONS)]
    if not candidates.empty:
        selected.append(candidates.assign(_gap=candidates.input_end_db_yards-candidates.input_end_coverage_yards).sort_values("_gap",ascending=False).iloc[0])
    candidates = target.loc[target.nearest_db_switches.gt(0)]
    if not candidates.empty:
        selected.append(candidates.sort_values(["nearest_db_switches"]+PLAY_KEY,ascending=[False,True,True]).iloc[0])
    case_records = []
    for i, row in enumerate(selected):
        raw = pd.read_csv(sources[int(row.week)-1], usecols=COLS)
        raw = raw.loc[raw.game_id.eq(row.game_id)&raw.play_id.eq(row.play_id)]
        frame = pd.read_csv(out/"frames"/f"week_{int(row.week):02d}.csv.gz")
        frame = frame.loc[frame.game_id.eq(row.game_id)&frame.play_id.eq(row.play_id)&frame.receiver_id.eq(row.receiver_id)].sort_values("frame_id")
        stem=f"case_{i+1}_{int(row.game_id)}_{int(row.play_id)}"
        (out/"cases"/f"{stem}.svg").write_text(svg_case(raw,frame,int(row.receiver_id),row),encoding="utf-8")
        raw.to_csv(out/"cases"/f"{stem}_tracking.csv",index=False)
        frame.to_csv(out/"cases"/f"{stem}_distances.csv",index=False)
        case_records.append({"svg":f"cases/{stem}.svg","name":row.player_name,"game_id":int(row.game_id),
            "play_id":int(row.play_id),"route":row.target_route_label,"result":row.pass_result,
            "description":row.play_description,"db_switches":int(row.nearest_db_switches)})
    complete = target.loc[target.pass_result.eq("C")]
    incomplete = target.loc[target.pass_result.eq("I")]
    stats = {"source":"Local official BDB 2026 Analytics input_2023_w01..w18.csv + supplementary_data.csv",
        "season":2023,"weeks":18,"games":len({k[0] for k in seen}),"plays":len(seen),
        "tracking_rows":sum(a["rows"] for a in audits),"receiver_frames":sum(a["receiver_frames"] for a in audits),
        "receiver_plays":len(all_rows),"eligible_receiver_plays":len(eligible),"eligible_wr_receiver_plays":len(wr),
        "eligible_targets":len(target),"eligible_wr_targets":len(targeted_wr),
        "unmatched_receiver_plays":int(all_rows["_merge"].eq("left_only").sum()),
        "incomplete_receiver_tracks":int((~all_rows.complete_input_track).sum()),
        "incomplete_db_windows":int((~all_rows.complete_db_window).sum()),
        "nullified_receiver_plays":int(all_rows.play_nullified_by_penalty.eq("Y").sum()),
        "missing_db_frames":sum(a["missing_db_frames"] for a in audits),
        "invalid_xy_rows":sum(a["invalid_xy_rows"] for a in audits),
        "out_of_bounds_xy_rows":sum(a["out_of_bounds_xy_rows"] for a in audits),
        "independently_verified_frames":sum(a["independently_verified_frames"] for a in audits),
        "max_distance_error_yards":max(a["max_distance_error_yards"] for a in audits),
        "target_results":target.pass_result.value_counts(dropna=False).to_dict(),
        "tracked_non_db_nearest_at_input_end_targets":int((~target.nearest_coverage_position.isin(DB_POSITIONS)).sum()),
        "mean_wr_all_routes_input_end_db_yards":float(wr.input_end_db_yards.mean()),
        "mean_wr_targets_input_end_db_yards":float(targeted_wr.input_end_db_yards.mean()),
        "complete_target_mean_input_end_db_yards":float(complete.input_end_db_yards.mean()),
        "incomplete_target_mean_input_end_db_yards":float(incomplete.input_end_db_yards.mean()),
        "quality_by_week":audits,"cases":case_records,
        "timing":"10 Hz; t=0 is last supplied input frame, not an event-verified release. No verified snap or arrival.",
        "population":"All supplied route runners (WR/TE/RB/etc), not all NFL routes. Defensive comparison restricted to supplied coverage players.",
        "summary_policy":"Matched non-nullified plays, continuous full input receiver track, DB observed each frame. Equal weight per receiver-play.",
        "coordinate_policy":"Official raw coordinates; rigid direction normalization used only in case plots. No combined/output tracking used."}
    dump(out/"audit_and_results.json",stats)
    write_report(out,stats,wr,targeted_wr,sample_preview)
    print(json.dumps({k:v for k,v in stats.items() if k not in ["quality_by_week","cases"]},indent=2),flush=True)


def write_report(out, stats, wr, targeted_wr, preview):
    esc=lambda s: html.escape(str(s))
    raw_leader = player_summary(targeted_wr).iloc[0]
    tables=[]
    for title,rows in [("WR: all observed routes",wr),("WR: targets only",targeted_wr)]:
        ranked=player_summary(rows)
        columns=["player_name","receiver_plays","games","targets","mean_input_end_db_yards","median_input_end_db_yards","p25_input_end_db_yards","p75_input_end_db_yards"]
        tables.append(f'<h3>{title}: descriptive means, minimum 30 observations</h3>'+ranked.loc[ranked.receiver_plays.ge(30),columns].head(20).round(3).to_html(index=False))
    cases="".join(f'<section><h3>Case {i+1}: {esc(c["name"])} — {esc(c["route"])} / {esc(c["result"])}</h3><p>{esc(c["description"])}</p><img src="{c["svg"]}" alt="Observed receiver and defense trajectories plus nearest-distance curve"/><p>Nearest DB changed {c["db_switches"]} times. <a href="{c["svg"].replace(".svg","_distances.csv")}">Frame distances</a></p></section>' for i,c in enumerate(stats["cases"]))
    result_text=f'''# Separation baseline: professor-feedback steps 1–4

Source: {stats['source']}. This run uses all 18 available 2023-season weekly input partitions.

## Data coverage and timing

- {stats['tracking_rows']:,} tracking rows; {stats['plays']:,} plays; {stats['games']} games.
- {stats['receiver_plays']:,} receiver-play observations; {stats['eligible_receiver_plays']:,} satisfy summary eligibility.
- {stats['eligible_wr_receiver_plays']:,} eligible WR routes; {stats['eligible_wr_targets']:,} eligible WR targets.
- Input data has no snap, release, arrival, football trajectory, or complete all-22 event tracking. The last input frame is only a release-adjacent boundary proxy. A true snap-to-throw mean and arrival separation cannot be reported from these files.
- Existing supplementary route labels describe the targeted receiver only. Never transfer them to other route runners.

## Definitions

Nearest DB = minimum Euclidean x/y distance to supplied defensive-coverage players with position CB/DB/S/FS/SS. Secondary comparison = nearest supplied defensive-coverage player, including linebackers. Neither identifies a coverage assignment or guarantees all 11 defenders are present.

Main metrics: input-end distance and arithmetic mean across all supplied input frames (10 Hz). Time averages may mix different observed window lengths. Player means give equal weight to eligible receiver-plays. Target and all-route summaries remain separate.

Eligibility: supplementary match, penalty flag N, receiver frames 1..input-end with no gaps, DB observed every frame, finite coordinates. Missing comparisons remain missing. Entire plays with nonfinite coordinates are excluded. Out-of-field positions are flagged and retained without clipping, as players may run out of bounds. Raw records and eligibility flags remain inspectable.

## Quality and independent validation

- Duplicate/null key checks passed weekly; weekly play partitions do not overlap.
- {stats['unmatched_receiver_plays']:,} unmatched receiver-plays; {stats['incomplete_receiver_tracks']:,} incomplete input tracks; {stats['incomplete_db_windows']:,} incomplete DB windows.
- {stats['invalid_xy_rows']} nonfinite-coordinate rows; {stats['out_of_bounds_xy_rows']} out-of-field rows; {stats['missing_db_frames']} receiver frames without a DB.
- Independently recomputed nearest IDs and distances for {stats['independently_verified_frames']} spread-out frame observations; maximum error {stats['max_distance_error_yards']:.3g} yards. Rotation invariance checked on one play per week; nearest-coverage distance <= nearest-DB distance checked for every comparable frame.

## Descriptive results, not route-running ability rankings

- WR all-route mean input-end DB distance: {stats['mean_wr_all_routes_input_end_db_yards']:.3f} yards.
- WR targets-only mean input-end DB distance: {stats['mean_wr_targets_input_end_db_yards']:.3f} yards.
- All-position targets: completed {stats['complete_target_mean_input_end_db_yards']:.3f} vs incomplete {stats['incomplete_target_mean_input_end_db_yards']:.3f} yards. These are unadjusted descriptive means, not causal effects or a prediction model.
- In {stats['tracked_non_db_nearest_at_input_end_targets']:,} eligible target plays, the nearest tracked coverage player at input end was not a DB.
- Pass-result counts: {stats['target_results']}. Incomplete passes remain in the analysis; interceptions remain a separate source label. No invented drop/throw-error labels.
- Player summary files include count, games, mean, median and quartiles. A 30-observation display threshold is an exploratory convention, not a reliability guarantee.
- Small-sample illustration: the largest unfiltered WR target mean belongs to {raw_leader.player_name}, {raw_leader.mean_input_end_db_yards:.3f} yards across just {int(raw_leader.receiver_plays)} eligible targets. This is not evidence of the best route runner.

## Scope and next check-in

The dataset selects plays and players; do not generalize to all 2023 NFL routes. Untargeted receivers have no individual completion outcome. Each example is deterministically chosen for illustration, not representative proof. Four cases are nearest to their route/outcome median input-end separation; two highlight the largest DB-vs-coverage endpoint gap and most DB switches.

Next: discuss input-window timing with the professor, review the six trajectory examples, and decide whether existing target route labels suffice before attempting clustering. No clustering or success classifier was added.

## Reproduce

Run `python scripts/run_separation_baseline.py` from the project root with NumPy and pandas installed. See `frames/week_*.csv.gz`, `receiver_play_metrics.csv`, `players_*.csv`, `target_context_summary.csv`, `weekly_quality.csv`, `cases/`, and `audit_and_results.json`. Existing source data and prior analyses are unchanged.
'''
    (out/"README.md").write_text(result_text,encoding="utf-8")
    narrative=[]
    for line in result_text.splitlines():
        if line.startswith("# ") or not line.strip():
            continue
        if line.startswith("## "):
            narrative.append(f'<h2>{esc(line[3:])}</h2>')
        elif line.startswith("- "):
            narrative.append(f'<p style="margin:6px 0 6px 20px">• {esc(line[2:])}</p>')
        else:
            narrative.append(f'<p>{esc(line)}</p>')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Receiver separation baseline</title><style>body{{font:15px/1.6 system-ui,sans-serif;color:#202a34;max-width:1180px;margin:35px auto;padding:0 24px}}h1{{font-size:28px}}h2{{margin-top:38px}}section{{padding:16px 0;border-top:1px solid #ddd}}img{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{padding:7px;text-align:left;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}.note{{background:#f1f4f6;padding:16px}}</style><h1>Receiver separation baseline</h1><p>2023 season · local BDB 2026 Analytics · professor-feedback steps 1–4</p><p class="note">Input end is a timing proxy, not verified release. Distances describe supplied coverage players. These are descriptive comparisons, not receiver-skill rankings.</p>{''.join(narrative)}<h2>Receiver comparisons with sample sizes</h2><div class="scroll">{''.join(tables)}</div><h2>Validated play examples</h2><p>C = complete; I = incomplete. Cases retain source route labels for targets only.</p>{cases}<h2>Deterministic source preview</h2><p>First 10 computed receiver-frame records; raw source paths are listed above.</p><div class="scroll">{preview.round(4).to_html(index=False)}</div></html>'''
    (out/"report.html").write_text(page,encoding="utf-8")


if __name__ == "__main__":
    main()
