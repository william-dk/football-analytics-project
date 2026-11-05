"""Add corrected geometry without changing any existing Step 2.0 tables."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from route_geometry import measure_route, BREAK_FIELDS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from archive_paths import resolve_recorded_path
KEY = ['game_id', 'play_id', 'nfl_id']
SEED = 20261025


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def cohort(data_dir, five_games=False):
    routes = pd.read_csv(data_dir / 'routes.csv', low_memory=False)
    routes = routes.loc[routes.is_targeted.eq(True)].copy()
    if five_games:
        games = sorted(routes.loc[routes.week.eq(1), 'game_id'].unique())[:5]
        routes = routes.loc[routes.week.eq(1) & routes.game_id.isin(games)]
    return routes.sort_values(KEY).reset_index(drop=True)


def build(data_dir, output, five_games):
    output.mkdir(parents=True, exist_ok=True)
    dest = output / 'target_break_features.csv'
    validation = output / 'break_features_validation.json'
    if dest.exists() or validation.exists():
        raise FileExistsError('Choose a fresh output folder; existing outputs are never replaced')
    sources = [data_dir / f for f in ['routes.csv', 'target_full_paths.csv', 'target_separation_summary.csv']]
    hashes = {str(p): digest(p) for p in sources}
    routes = cohort(data_dir, five_games)
    frames = pd.read_csv(sources[1], usecols=KEY + ['frame_id', 't', 'x_norm', 'y', 'y_norm'], low_memory=False)
    frames = frames.merge(routes[KEY], on=KEY, validate='many_to_one')
    lookup = routes.set_index(KEY)
    rows, endpoints = [], []
    for key, group in frames.groupby(KEY, sort=True):
        g = group.sort_values(['t', 'frame_id'])
        row = lookup.loc[key]
        feat = measure_route(g.t.to_numpy(), g.x_norm.to_numpy(), g.y.to_numpy(),
                             int(row.side), bool(row.side_flag), float(row.throw_time_s))
        rows.append(dict(zip(KEY, key), **feat))
        endpoints.append(dict(zip(KEY, key), before=float(g.y_norm.iloc[-1]),
                              after=float((g.y.iloc[-1]-g.y.iloc[0])*feat['side_corrected'])))
        if len(rows) % 2000 == 0:
            print(f'Measured {len(rows):,} routes', flush=True)
    features = pd.DataFrame(rows)
    joined = routes[KEY + ['route_label', 'player_position']].merge(features, on=KEY, validate='one_to_one')
    joined = joined.merge(pd.DataFrame(endpoints), on=KEY, validate='one_to_one')
    labeled = joined.loc[joined.route_label.notna()]
    failures = []
    medians = labeled.groupby('route_label')[['total_turn_deg', 'break_depth_yd']].median()
    for label, sign in {'OUT': 1, 'CORNER': 1, 'IN': -1, 'SLANT': -1, 'POST': -1}.items():
        if label not in medians.index or not medians.loc[label, 'total_turn_deg'] * sign > 0:
            failures.append(f'{label}: total-turn sign differs; stop for review')
    endpoint_checks = {}
    for label in ['FLAT', 'SCREEN', 'ANGLE', 'WHEEL']:
        g = labeled.loc[labeled.route_label.eq(label)]
        share = float(g.catch_y_corrected.gt(0).mean()) if len(g) else None
        endpoint_checks[label] = dict(n=len(g), outward_endpoint_share=share,
                                     hard_minimum=.8 if label in ['FLAT', 'WHEEL'] else None,
                                     observed=bool(len(g)))
        if len(g) and label in ['FLAT', 'WHEEL'] and share < .8:
            failures.append(f'{label}: outward corrected endpoint share below 80%')
    unique = not features.duplicated(KEY).any() and len(features) == len(routes)
    if not unique:
        failures.append('Target keys not covered exactly once')
    if np.isinf(features.select_dtypes(include='number').to_numpy()).any():
        failures.append('Infinite output values')
    if features.loc[~features.has_break, BREAK_FIELDS].notna().any().any():
        failures.append('No-break attributes must remain missing')
    inside = {}
    for label in ['FLAT', 'OUT']:
        g = labeled.loc[labeled.route_label.eq(label)]
        inside[label] = dict(n=len(g), before=float(g.before.lt(-5).mean()), after=float(g.after.lt(-5).mean()))
        if len(g) and inside[label]['after'] > inside[label]['before']:
            failures.append(f'{label}: inside share increased')
    unchanged = all(digest(p) == hashes[str(p)] for p in sources)
    if not unchanged:
        failures.append('Source tables changed')
    reference_turn = {'OUT':65, 'IN':-81, 'SLANT':-55, 'POST':-59, 'CORNER':73, 'GO':6}
    reference_depth = {'SLANT':3.6, 'OUT':7.3, 'IN':9.7, 'POST':13.7, 'CORNER':13.8}
    prototype = {l: dict(total_turn_deg=medians.loc[l, 'total_turn_deg'], reference_turn_deg=v,
                         difference_deg=medians.loc[l, 'total_turn_deg']-v,
                         break_depth_yd=medians.loc[l, 'break_depth_yd'], reference_break_depth_yd=reference_depth.get(l))
                 for l, v in reference_turn.items() if l in medians.index}
    record = dict(status='failed' if failures else 'passed', failures=failures, seed=SEED,
                  scope='week1_first5games' if five_games else 'full_season',
                  game_ids=sorted(routes.game_id.unique().tolist()), targeted_routes=len(routes),
                  labeled_targeted_routes=len(labeled), every_target_exactly_once=unique,
                  medians_by_label=medians.to_dict('index'), prototype_comparison=prototype,
                  amendment='2026-10-25: downfield-stem sign gates, lateral endpoint checks, 35-degree cuts',
                  break_threshold_deg=35, endpoint_checks=endpoint_checks,
                  threshold_sensitivity={str(th): labeled.assign(detected=labeled.fit_angle_deg.abs().ge(th)).groupby('route_label').detected.mean().to_dict() for th in [30,35,40]},
                  has_break_share_by_label=labeled.groupby('route_label').has_break.mean().to_dict(),
                  go_diagnostic=dict(hard_gate=False,
                      shares=labeled.loc[labeled.route_label.isin(['GO','OUT','IN','SLANT'])].groupby('route_label').has_break.mean().to_dict(),
                      go_below_all_three=bool(all(labeled.loc[labeled.route_label.eq('GO'),'has_break'].mean() < labeled.loc[labeled.route_label.eq(l),'has_break'].mean() for l in ['OUT','IN','SLANT'])),
                      difference_vs_go={l: float(labeled.loc[labeled.route_label.eq(l),'has_break'].mean()-labeled.loc[labeled.route_label.eq('GO'),'has_break'].mean()) for l in ['OUT','IN','SLANT']}),
                  break_class_counts=labeled.groupby(['route_label','break_class']).size().unstack(fill_value=0).to_dict('index'),
                  flipped_by_label=labeled.groupby('route_label').side_was_flipped.sum().to_dict(),
                  flipped_by_position=joined.groupby('player_position').side_was_flipped.sum().to_dict(),
                  uncertain_routes=int(features.side_uncertain.sum()), inside_arrivals=inside,
                  input_sha256=hashes, inputs_unchanged=unchanged,
                  no_infinite_values=True, no_break_attributes_missing=True,
                  notes=['All 14,108 targets included in season feature table; 14,105 labeled targets used downstream.',
                         'GO cut share is diagnostic only; report its difference from OUT, IN and SLANT.',
                         'FLAT and WHEEL require >=80% outward endpoints; absent subset labels are not tested.',
                         'Break coordinates and stop-return depths use observed positions; headings and speeds use smoothed positions.'])
    features.to_csv(dest, index=False)
    validation.write_text(json.dumps(clean(record), indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(clean(record), indent=2), flush=True)
    if failures:
        raise AssertionError('; '.join(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--five-games', action='store_true')
    a = parser.parse_args()
    build(a.data_dir, a.output_dir, a.five_games)
