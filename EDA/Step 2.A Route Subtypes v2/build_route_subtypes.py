"""Version 2 orchestration; numerical and plotting logic copied from v1."""
from subtype_helpers import *
from subtype_helpers import clean, cohort


def load_geometry(input_dir, break_dir, five_games):
    gate = json.loads((break_dir / 'break_features_validation.json').read_text())
    if gate['status'] != 'passed' or gate['failures'] or gate.get('break_threshold_deg') != 35:
        raise ValueError('Part 1 validation must pass before Part 2')
    all_routes = cohort(input_dir, five_games)
    cols = KEY + ['route_label', 'player_position', 'team_coverage_man_zone', 'throw_time_s', 'ball_flight_s']
    routes = all_routes.loc[all_routes.route_label.notna(), cols].copy()
    features = pd.read_csv(break_dir / 'target_break_features.csv')
    routes = routes.merge(features, on=KEY, validate='one_to_one', how='left')
    if routes.side_corrected.isna().any():
        raise ValueError('Missing corrected geometry')
    routes = routes.sort_values(KEY).reset_index(drop=True)
    frames = pd.read_csv(input_dir / 'target_full_paths.csv', usecols=KEY + ['t','frame_id','x_norm','y'], low_memory=False)
    frames = frames.merge(routes[KEY + ['side_corrected']], on=KEY, validate='many_to_one')
    if frames.duplicated(KEY + ['frame_id']).any():
        raise ValueError('Duplicate path-frame keys')
    paths, shapes, lengths, catches = {}, [], [], []
    for key, g in frames.groupby(KEY, sort=True):
        g = g.sort_values(['t','frame_id'])
        if (np.diff(g.t.to_numpy()) <= 0).any():
            raise ValueError(f'Non-increasing path time: {key}')
        y = (g.y.to_numpy() - g.y.iloc[0])*g.side_corrected.iloc[0]
        paths[key] = np.c_[g.t.to_numpy(), g.x_norm.to_numpy(), y]
    for key in routes[KEY].itertuples(index=False, name=None):
        p = paths[key]
        shape, length = arc_shape(p[:,1:])
        shapes.append(shape)
        lengths.append(length)
        catches.append(p[-1,1:])
    routes['path_length_yd'] = lengths
    routes['catch_x'], routes['catch_y'] = np.array(catches).T
    routes['too_short'] = routes.path_length_yd.lt(1)
    return routes, paths, np.array(shapes)


def fit_label(raw, games, label, total_n, seed):
    x, prep = encode(raw)
    models, bics, rows = {}, {}, []
    for k in ([1] if label == 'WHEEL' else range(1, min(4, len(raw)) + 1)):
        model, caught = mixture(x, k, seed)
        pred = model.predict(x)
        sizes = np.bincount(pred, minlength=k)
        valid = k == 1 or (sizes.min() >= 100 and sizes.min() >= .1*total_n)
        models[k], bics[k] = model, float(model.bic(x))
        rows.append(dict(label=label, k=k, bic=bics[k], cluster_sizes=json.dumps(sizes.tolist()),
                         size_rule_pass=bool(valid), converged=bool(model.converged_), warnings=caught))
    best = min(bics, key=bics.get)
    selected = best
    while selected > 1 and not next(r['size_rule_pass'] for r in rows if r['k'] == selected):
        selected -= 1
    model = models[selected]
    assignments = model.predict(x)
    stable = stability(raw, games, assignments, selected, 50, np.random.default_rng(seed))
    info = dict(chosen_k=selected, bic_winner_k=best, bic=bics, size_reduction=selected != best,
                shape_pca_components=int(prep[2].n_components_), shape_pca_variance=float(prep[2].explained_variance_ratio_.sum()),
                selected_model_converged=bool(model.converged_), **stable)
    info['silhouette'] = None
    if selected > 1:
        km = KMeans(selected,n_init=20,random_state=SEED).fit(x)
        info['silhouette'] = float(silhouette_score(x,km.labels_,sample_size=min(2000,len(x)),random_state=SEED))
    return assignments, model, x, info, rows


def football_name(label, depth):
    return {'OUT': 'quick out' if depth < 10 else 'deep out',
            'CROSS': 'drag' if depth < 10 else 'deep cross',
            'HITCH': 'hitch' if depth < 10 else 'curl',
            'IN': 'short in' if depth < 10 else 'dig'}.get(label, f'{label.lower()} (review)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT.parent/'Step 2.0 Route Dataset'/'data')
    parser.add_argument('--break-dir', type=Path, default=ROOT.parent/'Step 2.0 Route Dataset'/'break_features_v2_35deg'/'full_season')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--supplementary', type=Path, default=ROOT.parent/'Supplementary Data Analysis'/'supplementary_data.csv')
    parser.add_argument('--five-games', action='store_true')
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.supplementary = args.supplementary.resolve()
    args.break_dir = args.break_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.five_games:
        subset_gate = json.loads((ROOT/'validation_5games'/'validation.json').read_text())
        if subset_gate['status'] != 'passed' or subset_gate['failures']:
            raise ValueError('Five-game subtype validation must pass before full season')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output/'route_subtypes.csv').exists() or (output/'validation.json').exists():
        raise FileExistsError('Use a fresh output folder')
    break_dir = args.break_dir or args.input_dir
    sources = [args.input_dir/f for f in ['routes.csv','target_full_paths.csv','target_separation_summary.csv']] + [break_dir/'target_break_features.csv', args.supplementary]
    hashes = {str(p):digest(p) for p in sources}
    routes, paths, shapes = load_geometry(args.input_dir, break_dir, args.five_games)
    routes['cluster_id'], routes['provisional_subtype'] = 1, routes.route_label
    routes['gmm_probability'], routes['gmm_log_likelihood'] = np.nan, np.nan
    routes['outlier'], routes['short_nearest_catch_assigned'] = False, False
    diagnostics, bic_rows, sensitivity_rows, sensitivity_assignments = {}, [], [], []
    for label, whole in routes.groupby('route_label', sort=True):
        index = whole.index[~whole.too_short].to_numpy()
        fit = routes.loc[index]
        geom = ['catch_x','catch_y','max_depth_yd','total_turn_deg','retreat_yd']
        if whole.has_break.mean() >= .5:
            geom.append('break_depth_yd')
        raw = np.c_[shapes[index], fit[geom].to_numpy(float)]
        log(f'Fitting {label}: {len(index)} eligible; geometry={geom}')
        assignments, model, x, info, rows = fit_label(raw, fit.game_id.to_numpy(), label, len(whole), SEED)
        info.update(geometry_columns=geom, has_break_share=float(whole.has_break.mean()), fitted_routes=len(index), label_routes=len(whole))
        bic_rows.extend(rows)
        mapping, names = name_clusters(fit, assignments)
        routes.loc[index,'cluster_id'] = [mapping[int(c)] for c in assignments]
        routes.loc[index,'provisional_subtype'] = [names[int(c)] for c in assignments] if info['chosen_k'] > 1 else label
        routes.loc[index,'gmm_probability'] = model.predict_proba(x).max(axis=1)
        ll = model.score_samples(x)
        routes.loc[index,'gmm_log_likelihood'] = ll
        routes.loc[index[np.argsort(ll,kind='stable')[:math.ceil(.01*len(index))]],'outlier'] = True
        if info['chosen_k'] > 1:
            centers = routes.loc[index].groupby('cluster_id')[['catch_x','catch_y']].median()
            for idx in whole.index[whole.too_short]:
                nearest = ((centers-routes.loc[idx,['catch_x','catch_y']].astype(float))**2).sum(axis=1).idxmin()
                routes.loc[idx,'cluster_id'] = nearest
                routes.loc[idx,'provisional_subtype'] = routes.loc[index].loc[routes.loc[index,'cluster_id'].eq(nearest),'provisional_subtype'].iloc[0]
                routes.loc[idx,'short_nearest_catch_assigned'] = True
        sensitive_geom = [g for g in geom if g not in ['catch_x','catch_y','max_depth_yd']]
        sr = np.c_[shapes[index], fit[sensitive_geom].to_numpy(float)]
        sa, _, _, si, _ = fit_label(sr, fit.game_id.to_numpy(), label, len(whole), SEED)
        sensitivity_assignments.append(fit[KEY + ['route_label']].assign(sensitivity_cluster_id=sa))
        ari = float(adjusted_rand_score(assignments, sa))
        survived = si['chosen_k'] == info['chosen_k'] and ari >= .6 and si['mean_ari'] >= .6
        sens = dict(label=label, main_k=info['chosen_k'], sensitivity_k=si['chosen_k'], assignment_ari=ari,
                    sensitivity_stability_ari=si['mean_ari'], survives=bool(survived),
                    mainly_pass_depth=bool(info['chosen_k'] > 1 and si['chosen_k'] == 1),
                    geometry_columns=sensitive_geom)
        sensitivity_rows.append(sens)
        info['sensitivity'] = dict(**sens, diagnostics=si)
        diagnostics[label] = info
        (output/'clustering_progress.json').write_text(json.dumps(clean(diagnostics),indent=2,allow_nan=False),encoding='utf-8')
        log(f"Selected {label} k={info['chosen_k']}, ARI={info['mean_ari']:.3f}; sensitivity k={si['chosen_k']}, agreement={ari:.3f}")
    sep = pd.read_csv(args.input_dir/'target_separation_summary.csv', usecols=['game_id','play_id','receiver_nfl_id']+METRICS).rename(columns={'receiver_nfl_id':'nfl_id'})
    routes = routes.merge(sep, on=KEY, how='left', validate='one_to_one', indicator=True)
    if not routes._merge.eq('both').all():
        raise ValueError('Missing separation-summary route keys')
    routes = routes.drop(columns='_merge')
    comparisons, decisions = [], {}
    for label, whole in routes.groupby('route_label', sort=True):
        info = diagnostics[label]
        eligible = whole.loc[~whole.too_short]
        diffs = separation_bootstrap(eligible,1000,np.random.default_rng(SEED)) if info['chosen_k'] > 1 else []
        comparisons.extend(diffs)
        if len(eligible) < 200:
            decision, reason = 'keep label','insufficient sample (<200 eligible routes)'
        elif info['chosen_k'] == 1:
            decision, reason = 'keep label','minimum size rule reduced to k=1' if info['size_reduction'] else 'BIC favors one cluster'
        elif info['mean_ari'] < .6:
            decision, reason = 'keep label','unstable: mean game-bootstrap ARI <0.6'
        elif any(d['meets_separation_rule'] for d in diffs):
            decision, reason = 'split','stable and separation difference >=0.5 yd with 95% CI excluding zero'
        else:
            decision, reason = 'keep label','no separation comparison meets the rule'
        decisions[label] = dict(decision=decision, reason=reason)
    routes['route_subtype'] = [r.provisional_subtype if decisions[r.route_label]['decision']=='split' else r.route_label for r in routes.itertuples()]
    routes['flags'] = [';'.join(f for f,yes in [('too_short',r.too_short),('nearest_median_catch',r.short_nearest_catch_assigned),('outlier',r.outlier)] if yes) for r in routes.itertuples()]
    outcols = KEY + ['route_label','cluster_id','provisional_subtype','gmm_probability','route_subtype','flags']
    routes[outcols].to_csv(output/'route_subtypes.csv', index=False)
    summaries = []
    for (label,subtype),g in routes.groupby(['route_label','provisional_subtype']):
        row = dict(label=label, subtype=subtype, n=len(g), share_of_label=len(g)/sum(routes.route_label.eq(label)),
                   chosen_k=diagnostics[label]['chosen_k'], stability_ari=diagnostics[label]['mean_ari'],
                   silhouette=diagnostics[label]['silhouette'],
                   bic_table=json.dumps(diagnostics[label]['bic']),unstable=diagnostics[label]['mean_ari']<.6,
                   too_short_n=int(g.too_short.sum()), outlier_n=int(g.outlier.sum()), **decisions[label])
        for col in GEOM + TIMING + ['retreat_yd']:
            row[f'median_{col}'] = g[col].median()
            row[f'n_{col}'] = int(g[col].notna().sum())
        row['suggested_football_name'] = football_name(label, g.break_depth_yd.median() if g.has_break.mean()>=.5 else g.catch_x.median())
        for pos in ['WR','TE','RB']:
            row[f'share_{pos}'] = float(g.player_position.eq(pos).mean())
        row['share_other_position'] = float((~g.player_position.isin(['WR','TE','RB'])).mean())
        for coverage in ['MAN_COVERAGE','ZONE_COVERAGE']:
            row[f'share_{coverage}'] = float(g.team_coverage_man_zone.eq(coverage).mean())
        row['share_missing_coverage'] = float(g.team_coverage_man_zone.isna().mean())
        for metric in METRICS:
            v = g[metric].dropna()
            row.update({f'{metric}_n':len(v), f'{metric}_median':v.median(), f'{metric}_q25':v.quantile(.25), f'{metric}_q75':v.quantile(.75), f'{metric}_iqr':v.quantile(.75)-v.quantile(.25)})
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output/'subtype_summary.csv',index=False)
    pd.DataFrame(bic_rows).to_csv(output/'bic_diagnostics.csv',index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output/'sensitivity_comparison.csv',index=False)
    pd.concat(sensitivity_assignments,ignore_index=True).to_csv(output/'route_subtypes_sensitivity.csv',index=False)
    comparison_cols = ['label','subtype_a','subtype_b','metric','difference_b_minus_a_yd','ci_low_yd','ci_high_yd','valid_bootstraps','bootstrap_resamples','meets_separation_rule']
    pd.DataFrame(comparisons,columns=comparison_cols).to_csv(output/'separation_comparisons.csv',index=False)
    supp = pd.read_csv(args.supplementary,usecols=['game_id','play_id','play_description']).drop_duplicates()
    review = routes.loc[routes.outlier,KEY+['route_label','provisional_subtype','gmm_log_likelihood']].rename(columns={'route_label':'label','provisional_subtype':'nearest_subtype','gmm_log_likelihood':'log_likelihood'})
    review.merge(supp,on=['game_id','play_id'],how='left',validate='many_to_one').to_csv(output/'outlier_review.csv',index=False)
    routes[KEY+['route_label','path_length_yd']+GEOM+TIMING+['side_corrected','side_was_flipped','side_uncertain','has_break','stem_heading_deg','fit_angle_deg','break_class','catch_y_corrected']].to_csv(output/'route_descriptors.csv',index=False)
    v1 = resolve_recorded_path(ROOT.parent/'Step 2.A Route Subtypes')
    old = pd.read_csv(v1/'route_subtypes.csv')
    oldval = json.loads((v1/'validation.json').read_text())
    comparison = []
    for label,info in diagnostics.items():
        match = routes.loc[routes.route_label.eq(label)].merge(old[KEY+['provisional_subtype']],on=KEY,suffixes=('_v2','_v1'),validate='one_to_one')
        oldk = oldval['chosen_k_per_label'][label]
        comparison.append(dict(label=label,v1_k=oldk,v1_decision=oldval['decisions'][label]['decision'],v2_k=info['chosen_k'],v2_decision=decisions[label]['decision'],
                               assignment_ari=float(adjusted_rand_score(match.provisional_subtype_v1,match.provisional_subtype_v2)) if oldk>1 and info['chosen_k']>1 else np.nan))
    pd.DataFrame(comparison).to_csv(output/'comparison_v1_v2.csv',index=False)
    cards(routes,paths,output,np.random.default_rng(SEED))
    failures = []
    expected = len(cohort(args.input_dir,args.five_games).dropna(subset=['route_label']))
    if routes.duplicated(KEY).any() or len(routes)!=expected or routes.route_subtype.isna().any():
        failures.append('Every labeled route must be assigned exactly once')
    for label,g in routes.groupby('route_label'):
        if diagnostics[label]['chosen_k']>1 and (g.groupby('cluster_id').size().min()<100 or g.groupby('cluster_id').size().min()<.1*len(g)):
            failures.append(f'{label}: undersized subtype')
    if np.isinf(routes.select_dtypes(include='number').to_numpy()).any():
        failures.append('Infinite values')
    for label,info in diagnostics.items():
        if not info['selected_model_converged'] or not info['sensitivity']['diagnostics']['selected_model_converged']:
            failures.append(f'{label}: selected mixture did not converge')
        if 'break_depth_yd' in info['geometry_columns'] and info['has_break_share'] < .5:
            failures.append(f'{label}: break depth ineligible')
        if decisions[label]['decision']=='split' and info['mean_ari'] < .6:
            failures.append(f'{label}: unstable accepted split')
    unchanged = all(digest(p)==hashes[str(p)] for p in sources)
    if not unchanged: failures.append('Source files changed')
    validation = dict(status='failed' if failures else 'passed',failures=failures,seed=SEED,
                      scope='week1_first5games' if args.five_games else 'full_season',labeled_targeted_routes=len(routes),
                      every_route_assigned_exactly_once=True, too_short_total=int(routes.too_short.sum()),
                      short_nearest_catch_assigned=int(routes.short_nearest_catch_assigned.sum()),
                      chosen_k_per_label={l:d['chosen_k'] for l,d in diagnostics.items()}, decisions=decisions,
                      stability_resamples=50,separation_bootstrap_resamples=1000,covariance_type='diag',n_init=10,
                      missing_arrival_separation=int(routes.sep_at_arrival.isna().sum()),inputs_unchanged=unchanged,input_sha256=hashes)
    validation['break_threshold_deg'] = 35
    validation['geometry_columns_per_label'] = {l:d['geometry_columns'] for l,d in diagnostics.items()}
    validation['versions'] = dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__)
    (output/'validation.json').write_text(json.dumps(clean(validation),indent=2,allow_nan=False),encoding='utf-8')
    (output/'clustering_diagnostics.json').write_text(json.dumps(clean(diagnostics),indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# Step 2.A route subtypes v2','',f'{len(routes):,} labeled targets; {routes.game_id.nunique()} games. Seed {SEED}.','',
             '## Changes from v1','',
             'Corrected lateral coordinates throughout. Replaced every legacy break feature and the no_break indicator with total turn and retreat; fitted break depth is included only where at least half of a label has a break. Missing values are median-imputed only inside each fit. Shape retains 12 arc-length samples and PCA to 95% variance. GMM uses diagonal covariance, k=1..4, ten initializations (including all 50 game-bootstrap refits); WHEEL is k=1. The v1 minimum-size reduction and separation decision rules are retained. Short routes join the nearest median catch point of an existing subtype. Names add turn direction, with football names for manual review.','',
             'The sensitivity fit removes catch_x, catch_y, and max_depth_yd, repeats model selection and stability, and reports assignment ARI. Shape still encodes depth, so a surviving split is not proof of depth independence. A split reduced to k=1 is marked mainly pass depth.','',
             '## Per-label decisions and v1 comparison','']
    for c in comparison:
        label=c['label']; d=diagnostics[label]; s=d['sensitivity']
        lines.append(f"- **{label}**: v1 k={c['v1_k']} ({c['v1_decision']}) → v2 k={c['v2_k']} ({c['v2_decision']}); stability ARI {d['mean_ari']:.3f}. {decisions[label]['reason']}. Sensitivity k={s['sensitivity_k']}, agreement ARI={s['assignment_ari']:.3f}, survives={s['survives']}, mainly pass depth={s['mainly_pass_depth']}.")
    lines += ['', '## Interpretation','',
              'Separation is loaded only after both geometry fits. Intervals use 1,000 paired game resamples, 95% percentile intervals, and the unchanged >=0.5 yd rule. They are exploratory, unadjusted, and not causal. Geometry clusters with stability below 0.6 remain provisional and do not change the retained route label. All missing break attributes remain missing in descriptors. Football suggestions in subtype_summary.csv need review. Outliers remain in the analysis.','',
              'Part 1 uses the reviewed 35-degree cut threshold, with 20–35 degrees labeled bend. Continuous fitted angles remain available but are not extra clustering inputs. Cut depth eligibility is evaluated at 35 degrees, not the superseded 20-degree definition.','',
              f'Validation: {validation["status"]}; inputs unchanged={unchanged}; missing arrival separation={validation["missing_arrival_separation"]}.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if failures: raise AssertionError('; '.join(failures))
    log(f'Complete: {len(routes)} routes; validation passed')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
