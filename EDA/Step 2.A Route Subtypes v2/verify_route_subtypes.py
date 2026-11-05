"""Audit persisted subtype counts, gates, short-route assignment and comparisons."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from subtype_helpers import KEY, SEED, digest, clean, resolve_recorded_path

ROOT=Path(__file__).resolve().parent


def verify(output, report=None):
    dest=report or output/'verification.json'
    if dest.exists():raise FileExistsError(dest)
    gate=json.loads((output/'validation.json').read_text())
    diagnostics=json.loads((output/'clustering_diagnostics.json').read_text())
    assigned=pd.read_csv(output/'route_subtypes.csv').fillna({'flags':''})
    descriptors=pd.read_csv(output/'route_descriptors.csv')
    routes=assigned.merge(descriptors,on=KEY+['route_label'],validate='one_to_one')
    comparison=pd.read_csv(output/'comparison_v1_v2.csv').set_index('label')
    sensitivity=pd.read_csv(output/'sensitivity_comparison.csv').set_index('label')
    sep=pd.read_csv(output/'separation_comparisons.csv')
    failures=[]
    if gate['status']!='passed' or gate['failures']:failures.append('Build validation not passed')
    if len(routes)!=gate['labeled_targeted_routes'] or assigned.duplicated(KEY).any():failures.append('Route-key coverage differs')
    shorts_verified=0
    for label,g in routes.groupby('route_label'):
        info=diagnostics[label]
        if g.cluster_id.nunique()!=info['chosen_k']:failures.append(f'{label}: actual k differs')
        if info['chosen_k']>1 and (g.groupby('cluster_id').size().min()<100 or g.groupby('cluster_id').size().min()<.1*len(g)):
            failures.append(f'{label}: subtype size rule fails')
        for stable in [info,info['sensitivity']['diagnostics']]:
            if len(stable['aris'])!=50 or stable['resamples']!=50 or not np.isclose(np.mean(stable['aris']),stable['mean_ari']):
                failures.append(f'{label}: bootstrap ARIs/counts differ')
        features=info['geometry_columns']
        if features[:5]!=['catch_x','catch_y','max_depth_yd','total_turn_deg','retreat_yd']:
            failures.append(f'{label}: unexpected geometry inputs')
        if ('break_depth_yd' in features)!=(g.has_break.mean()>=.5):failures.append(f'{label}: break-depth eligibility differs')
        if any(c in ['no_break','break_detected','break_angle_deg'] for c in features):failures.append(f'{label}: legacy feature included')
        sensitive_features=ast.literal_eval(sensitivity.loc[label,'geometry_columns'])
        if any(c in ['catch_x','catch_y','max_depth_yd'] for c in sensitive_features):failures.append(f'{label}: sensitivity retains catch features')
        decision=gate['decisions'][label]['decision']
        if decision=='split':
            if info['mean_ari']<.6 or not sep.loc[sep.label.eq(label),'meets_separation_rule'].any():
                failures.append(f'{label}: accepted split does not satisfy decision rule')
            if not g.route_subtype.eq(g.provisional_subtype).all():failures.append(f'{label}: final subtype differs')
        elif not g.route_subtype.eq(label).all():failures.append(f'{label}: retained labels changed')
        if comparison.loc[label,'v2_k']!=info['chosen_k'] or comparison.loc[label,'v2_decision']!=decision:
            failures.append(f'{label}: v1/v2 comparison differs')
        if info['chosen_k']>1:
            eligible=g.loc[g.path_length_yd.ge(1)]
            centers=eligible.groupby('cluster_id')[['catch_x','catch_y']].median()
            for r in g.loc[g.path_length_yd.lt(1)].itertuples():
                distance=((centers-np.array([r.catch_x,r.catch_y]))**2).sum(axis=1)
                if r.cluster_id!=distance.idxmin() or 'nearest_median_catch' not in r.flags:
                    failures.append(f'{label}: incorrect short-route assignment')
                shorts_verified+=1
    if diagnostics.get('WHEEL',{}).get('chosen_k',1)!=1:failures.append('WHEEL must be k=1')
    if not np.isclose(sep.difference_b_minus_a_yd.abs().ge(.5).where((sep.ci_low_yd>0)|(sep.ci_high_yd<0),False).where(sep.valid_bootstraps.eq(1000),False).astype(int),sep.meets_separation_rule.astype(int)).all():
        failures.append('Separation thresholds differ')
    def recorded_path(p):
        path=Path(p)
        return resolve_recorded_path(path if path.is_absolute() else ROOT.parents[1]/path)
    unchanged=all(digest(recorded_path(p))==h for p,h in gate['input_sha256'].items())
    if not unchanged:failures.append('Source hashes changed')
    record=dict(status='failed' if failures else 'passed',failures=failures,seed=SEED,scope=gate['scope'],
                labeled_targeted_routes=len(routes),labels_verified=len(diagnostics),short_nearest_catch_assignments_verified=shorts_verified,
                stability_refits_per_analysis=50,accepted_split_labels=[l for l,d in gate['decisions'].items() if d['decision']=='split'],
                all_provisional_splits_surviving_sensitivity=sensitivity.loc[sensitivity.main_k.gt(1)&sensitivity.survives].index.tolist(),
                source_hashes_unchanged=unchanged)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(clean(record),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(clean(record),indent=2))
    if failures:raise AssertionError('; '.join(failures))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    verify(args.output_dir,args.report)
