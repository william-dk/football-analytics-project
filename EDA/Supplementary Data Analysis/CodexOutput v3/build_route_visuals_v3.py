"""Fresh v3 outputs using reviewed 35-degree geometry and corrected sides."""
from visual_helpers import *
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]/'Step 2.0 Route Dataset'))
from build_break_features import digest, clean, cohort, resolve_recorded_path


def require_passed(path):
    gate = json.loads(path.read_text(encoding='utf-8'))
    if gate.get('status') != 'passed' or gate.get('failures') != []:
        raise ValueError(f'Previous validation has not passed: {path}')
    return gate


def select_representatives(routes, paths, medians, five_games=False):
    """Rank C and I separately against the same full-label median as v2."""
    chosen, audits = [], []
    for label in LABELS:
        g = routes.loc[routes.route_label.eq(label)].copy()
        if g.empty:
            audits.append(dict(route_label=label,status='label_absent_from_subset'))
            continue
        base = (g.play_nullified_by_penalty.eq('N') & ~bool_col(g.side_flag)
                & g.landing_to_arrival_yd.le(10))
        throw_q = g.throw_time_s.quantile([.25,.75]).to_numpy()
        depth_q = g.arrival_depth_yd.quantile([.25,.75]).to_numpy()
        iqr = g.throw_time_s.between(*throw_q) & g.arrival_depth_yd.between(*depth_q)
        defended = g.n_defenders_at_arrival.ge(1)
        for result, result_name in [('C','completion'),('I','incompletion')]:
            result_mask = g.pass_result.eq(result)
            stages = [('none',base & iqr & defended & result_mask),
                      ('IQR',base & defended & result_mask),
                      ('IQR;defender',base & result_mask)]
            counts = {name:int(mask.sum()) for name,mask in stages}
            candidates = pd.DataFrame()
            for relaxed,mask in stages:
                candidates = g.loc[mask].copy()
                if len(candidates): break
            audit = dict(route_label=label,result_group=result_name,pass_result=result,
                         candidate_counts=counts,throw_time_iqr_s=throw_q.tolist(),arrival_depth_iqr_yd=depth_q.tolist())
            if candidates.empty:
                audit['status'] = 'no_eligible_'+result_name
                audits.append(audit)
                if result == 'C' and not five_games:
                    raise ValueError(f'No eligible completion for {label}')
                continue
            median = medians[label].loc[medians[label].included_in_median]
            distances = []
            for _,row in candidates.iterrows():
                p=paths[key_of(row)]
                m=median.loc[median.time_since_snap_s.between(p['t'][0]-1e-8,p['t'][-1]+1e-8)]
                if m.empty:
                    distances.append(np.inf)
                    continue
                lateral=np.interp(m.time_since_snap_s,p['t'],p['lateral'])
                depth=np.interp(m.time_since_snap_s,p['t'],p['depth'])
                distances.append(float(np.hypot(lateral-m.median_lateral_yd,depth-m.median_depth_from_los_yd).mean()))
            candidates['selection_distance_yd']=distances
            selected=candidates.sort_values(['selection_distance_yd']+KEY).iloc[0].copy()
            if not np.isfinite(selected.selection_distance_yd):
                raise ValueError('No observed median overlap')
            selected['relaxed_conditions']=relaxed
            selected['result_group']=result_name
            chosen.append(selected)
            audit.update(status='selected',selected_game_id=int(selected.game_id),selected_play_id=int(selected.play_id),
                         receiver_nfl_id=int(selected.nfl_id),relaxed_conditions=relaxed,
                         selection_distance_yd=float(selected.selection_distance_yd))
            audits.append(audit)
    return pd.DataFrame(chosen),audits


def write_gallery(out, selected, routes, audits):
    sections=[]
    for label in LABELS:
        if not routes.route_label.eq(label).any(): continue
        examples=[]
        for _,r in selected.loc[selected.route_label.eq(label)].iterrows():
            filename=f'representative/route_{label}_{int(r.game_id)}_{int(r.play_id)}.png'
            examples.append(f'<figure><a href="{filename}"><img src="{filename}" alt="{label} {r.result_group}"></a>'
                            f'<figcaption>{html.escape(r.result_group.title())}: {html.escape(r.player_name)}; '
                            f'game {int(r.game_id)}, play {int(r.play_id)}; result {r.pass_result}; '
                            f'distance to median {r.selection_distance_yd:.2f} yd.</figcaption></figure>')
        notes=' '.join(html.escape(a['status'].replace('_',' '))+'.' for a in audits if a['route_label']==label and a.get('status','').startswith('no_eligible'))
        sections.append(f'<section id="{label}"><h2>{label}</h2><a href="spaghetti/route_{label}.png">'
                        f'<img class="distribution" src="spaghetti/route_{label}.png" alt="{label} corrected route distribution"></a>'
                        f'<div class="pair">{"".join(examples)}</div><p>{notes}</p></section>')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route Visualizations v3 — 2023</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#243343;background:#f5f6f8;margin:0 auto;padding:32px;max-width:1600px}}
h1{{font-size:30px;margin-bottom:8px}}h2{{border-top:1px solid #cfd5db;padding-top:24px}}a{{color:#165b9a}}
nav a{{display:inline-block;margin:0 16px 8px 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
figure{{margin:0}}img{{width:100%;background:white}}.overview{{max-width:1100px;display:block;margin:auto}}
.distribution{{max-width:1000px;display:block;margin:auto}}section{{margin-top:32px}}
@media(max-width:900px){{.pair{{grid-template-columns:1fr}}body{{padding:16px}}}}
</style><h1>Route Visualizations v3</h1>
<p>2023 · {len(routes):,} labeled targeted routes · corrected release side · 35° cut threshold</p>
<p><a href="README.md">Methods and rebuild</a> · <a href="label_summary.csv">All-route statistics</a> ·
<a href="representative_plays.csv">Selected plays</a> · <a href="checks.json">Validation</a></p>
<p>Each label shows its corrected distribution, a representative completion, and an eligible incompletion where available.
Colored dash-dot paths show accepted Step 2.A v2 subtype medians for split labels.</p>
<p><strong>Arrival is the last tracked frame, not a verified catch event.</strong> Blue is the target route;
solid is before throw and dashed is after throw. Gold stars show ball landing.</p><nav>'''
    page+=' '.join(f'<a href="#{label}">{label}</a>' for label in LABELS if routes.route_label.eq(label).any())
    page+='</nav><a href="spaghetti/all_routes_overview.png"><img class="overview" src="spaghetti/all_routes_overview.png" alt="Common-scale corrected route atlas"></a>'
    (out/'index.html').write_text(page+''.join(sections)+'</html>',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT.parents[1]/'Step 2.0 Route Dataset'/'data')
    parser.add_argument('--break-dir',type=Path,default=ROOT.parents[1]/'Step 2.0 Route Dataset'/'break_features_v2_35deg'/'full_season')
    parser.add_argument('--subtype-dir',type=Path,default=ROOT.parents[1]/'Step 2.A Route Subtypes v2')
    parser.add_argument('--train-dir',type=Path,default=Path('C:/Local/Projects/Datalab Football Project/data/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'))
    parser.add_argument('--supplementary',type=Path,default=ROOT.parent/'supplementary_data.csv')
    parser.add_argument('--out',type=Path,default=ROOT/'full_season')
    parser.add_argument('--subset-validation-dir',type=Path,default=ROOT/'validation_5games_layout_v2')
    parser.add_argument('--five-games',action='store_true')
    args=parser.parse_args()
    break_gate=require_passed(args.break_dir/'break_features_validation.json')
    require_passed(args.subtype_dir/'validation.json')
    if break_gate.get('break_threshold_deg') != 35:
        raise ValueError('Only reviewed 35-degree break features may be used')
    if not args.five_games: require_passed(args.subset_validation_dir/'checks.json')
    out=args.out.resolve()
    if any((out/p).exists() for p in ['checks.json','label_summary.csv','index.html','spaghetti','representative']):
        raise FileExistsError('Choose a fresh output folder')
    for folder in ['spaghetti','representative']:(out/folder).mkdir(parents=True,exist_ok=False)
    source_files=[args.data_dir/f for f in ['routes.csv','target_full_paths.csv','target_separation_summary.csv','target_defender_frames.csv']]+[args.break_dir/'target_break_features.csv',args.subtype_dir/'route_subtypes.csv',args.supplementary]
    for folder in [ROOT.parent/'CodexOutput v2',ROOT.parents[1]/'Step 2.A Route Subtypes']:
        folder=resolve_recorded_path(folder)
        source_files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    hashes={str(p.resolve()):digest(p) for p in source_files}
    initial=cohort(args.data_dir,args.five_games).dropna(subset=['route_label'])
    wanted=pd.MultiIndex.from_frame(initial[['game_id','play_id']].drop_duplicates())
    meta=scan_metadata(args.train_dir,wanted,sorted(initial.week.unique()))
    routes,paths=load_routes(args.data_dir,meta,args.break_dir)
    subtype=read_csv(args.subtype_dir/'route_subtypes.csv',usecols=KEY+['route_subtype'])
    routes=routes.merge(subtype,on=KEY,how='left',validate='one_to_one')
    if routes.route_subtype.isna().any():raise ValueError('Missing subtype assignments')
    checks=dict(total_labeled_target_routes=len(routes),expected_total=len(initial),seed=SEED,
                scope='week1_first5games' if args.five_games else 'full_season',break_threshold_deg=35,
                route_counts_per_label=routes.route_label.value_counts().to_dict(),
                ball_landing_flip_validation={},every_route_exactly_once=not routes.duplicated(KEY).any(),
                side_corrected_routes_per_label=routes.groupby('route_label').side_was_flipped.sum().to_dict(),
                side_uncertain_routes_per_label=routes.groupby('route_label').side_uncertain.sum().to_dict(),
                inside_arrivals={},subtype_overlays={})
    for label in ['FLAT','OUT']:
        g=routes.loc[routes.route_label.eq(label)]
        original=g.arrival_lateral*g.side/g.side_corrected
        checks['inside_arrivals'][label]=dict(n=len(g),before=float(original.lt(-5).mean()),after=float(g.arrival_lateral.lt(-5).mean()))
    for direction,g in routes.groupby('play_direction'):
        complete=g.loc[g.pass_result.eq('C')]
        checks['ball_landing_flip_validation'][direction]=dict(n_completions=len(complete),
            median_landing_to_arrival_yd=float(complete.landing_to_arrival_yd.median()),
            p90_landing_to_arrival_yd=float(complete.landing_to_arrival_yd.quantile(.9)))
    failures=[]
    valid=all(v['n_completions']>0 and v['median_landing_to_arrival_yd']<2 for v in checks['ball_landing_flip_validation'].values())
    checks['ball_landing_flip_passed']=valid
    if not valid:
        checks.update(status='failed',failures=['Ball landing direction validation failed'])
        (out/'checks.json').write_text(json.dumps(clean(checks),indent=2,allow_nan=False),encoding='utf-8')
        raise ValueError('Ball landing direction validation failed')
    medians,stats,samples,median_rows={ },[],[],[]
    for label in LABELS:
        group=routes.loc[routes.route_label.eq(label)]
        if group.empty:continue
        sampled=stratified_sample(group)
        median=median_path(group,paths)
        medians[label]=median
        median_rows.append(median.assign(route_label=label))
        stat=summarize(label,group,sampled)
        stats.append(stat)
        samples.append(sampled[KEY+['route_label','pass_result','landing_to_arrival_yd']])
        overlays=[]
        if group.route_subtype.nunique()>1:
            for name,g in group.groupby('route_subtype'):
                submedian=median_path(g,paths)
                submedian['minimum_support_n']=math.ceil(.5*len(g))
                submedian['included_in_median']=np.logical_and.accumulate(
                    submedian.n_paths_observed.ge(math.ceil(.5*len(g))).to_numpy())
                overlays.append((name,submedian))
        checks['subtype_overlays'][label]=[name for name,_ in overlays]
        spaghetti(label,sampled,paths,median,stat,out,overlays)
        progress(f'Built corrected distribution: {label}')
    pd.DataFrame(stats).to_csv(out/'label_summary.csv',index=False)
    pd.concat(samples).to_csv(out/'sampled_routes.csv',index=False)
    pd.concat(median_rows).to_csv(out/'median_path_support.csv',index=False)
    checks['overview_axis_limits']=overview(routes,medians,out)
    selected,audits=select_representatives(routes,paths,medians,args.five_games)
    checks['representative_selection']=audits
    if selected.empty:raise ValueError('No eligible representative plays')
    progress('Loading selected-play defenders and official tracking')
    defenders,inputs,outputs=load_selected_context(selected,args.data_dir,args.train_dir)
    supplementary=read_csv(args.supplementary)
    supplementary=supplementary.loc[supplementary.season.eq(2023)]
    descriptions=supplementary.set_index(['game_id','play_id']).play_description
    play_checks=[]
    for _,row in selected.iterrows():
        official_target=outputs.loc[outputs.game_id.eq(row.game_id)&outputs.play_id.eq(row.play_id)&outputs.nfl_id.eq(row.nfl_id)].sort_values('frame_id')
        p=paths[key_of(row)]
        observed=np.c_[p['x'][p['phase']=='output'],p['y'][p['phase']=='output']]
        if len(official_target)!=len(observed) or not np.allclose(official_target[['x','y']].to_numpy(),observed,atol=1e-6):
            raise ValueError(f'Official output coordinates mismatch: {key_of(row)}')
        desc=descriptions.loc[(int(row.game_id),int(row.play_id))]
        play_checks.append(representative(row,paths,defenders,inputs,desc,out))
        progress(f'Built field example: {row.route_label}, {row.pass_result}, {int(row.game_id)}, {int(row.play_id)}')
    columns=['route_label','game_id','play_id','nfl_id','player_name','result_group','selection_distance_yd',
             'relaxed_conditions','pass_result','sep_at_throw','sep_at_arrival','throw_time_s','ball_flight_s',
             'landing_to_arrival_yd','arrival_depth_yd','play_direction','week','down','yards_to_go',
             'n_defenders_at_arrival','side_flag','side_corrected','side_was_flipped','play_nullified_by_penalty']
    selected[columns].rename(columns={'nfl_id':'receiver_nfl_id'}).to_csv(out/'representative_plays.csv',index=False)
    checks['selected_play_checks']=play_checks
    checks['selected_pass_results']=selected[['route_label','game_id','play_id','pass_result','result_group']].to_dict('records')
    checks['number_of_plays_missing_passer']=sum(p['missing_passer'] for p in play_checks)
    checks['number_of_plays_missing_passer_at_throw']=sum(p['missing_passer_at_throw'] for p in play_checks)
    checks['selected_official_output_paths_verified']=len(selected)
    checks['start_depth_matches_step2_los']=True
    checks['landing_points_excluded_from_drawing_per_label']={s['route_label']:dict(sampled_excluded=s['n_landing_excluded_sampled'],all_routes_over_40yd=s['n_landing_excluded_all']) for s in stats}
    checks['png_dpi']=160
    checks['expected_png_count']=routes.route_label.nunique()+1+len(selected)
    checks['actual_png_count']=len(list((out/'spaghetti').glob('*.png')))+len(list((out/'representative').glob('*.png')))
    checks['input_sha256']=hashes
    checks['inputs_unchanged']=all(digest(p)==hashes[str(p.resolve())] for p in source_files)
    if not checks['inputs_unchanged']:failures.append('Source or prior-version files changed')
    if len(routes)!=len(initial) or routes.duplicated(KEY).any():failures.append('Cohort coverage differs')
    if checks['expected_png_count']!=checks['actual_png_count']:failures.append('Unexpected PNG count')
    if selected.duplicated(['route_label','pass_result']).any():failures.append('Duplicate representative result group')
    if selected.landing_to_arrival_yd.gt(10).any():failures.append('Representative landing beyond 10 yards')
    checks.update(status='failed' if failures else 'passed',failures=failures)
    (out/'checks.json').write_text(json.dumps(clean(checks),indent=2,allow_nan=False),encoding='utf-8')
    if not (out/'README.md').exists():
        (out/'README.md').write_text((ROOT/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
    write_gallery(out,selected,routes,audits)
    if failures:raise AssertionError('; '.join(failures))
    progress(f"Complete: {checks['actual_png_count']} PNGs; validation passed")


if __name__ == '__main__':main()
