"""Read-back audit of persisted v3 medians, representative ranking and images."""
from build_route_visuals_v3 import *
from PIL import Image
from html.parser import HTMLParser


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]

    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ['href','src'] and not value.startswith('#'):
                self.links.append(value)


def verify(out,data_dir,break_dir,train_dir,report=None):
    dest=report or out/'verification.json'
    if dest.exists():raise FileExistsError(dest)
    gate=require_passed(out/'checks.json')
    initial=cohort(data_dir,gate['scope']=='week1_first5games').dropna(subset=['route_label'])
    meta=scan_metadata(train_dir,pd.MultiIndex.from_frame(initial[['game_id','play_id']].drop_duplicates()),sorted(initial.week.unique()))
    routes,paths=load_routes(data_dir,meta,break_dir)
    medians=read_csv(out/'median_path_support.csv')
    sampled=read_csv(out/'sampled_routes.csv')
    selected=read_csv(out/'representative_plays.csv').rename(columns={'receiver_nfl_id':'nfl_id'})
    summaries=read_csv(out/'label_summary.csv').set_index('route_label')
    failures=[]
    checked_points,max_median_error,max_support_error=0,0.,0
    for label,g in routes.groupby('route_label'):
        m=medians.loc[medians.route_label.eq(label)].sort_values('time_since_snap_s')
        grid=m.time_since_snap_s.to_numpy()
        values=np.full((len(g),len(grid),2),np.nan)
        for i,key in enumerate(g[KEY].itertuples(index=False,name=None)):
            p=paths[key]
            inside=(grid>=p['t'][0]-1e-8)&(grid<=p['t'][-1]+1e-8)
            values[i,inside,0]=np.interp(grid[inside],p['t'],p['lateral'])
            values[i,inside,1]=np.interp(grid[inside],p['t'],p['depth'])
        support=np.isfinite(values[:,:,0]).sum(axis=0)
        with np.errstate(all='ignore'):
            calculated=np.nanmedian(values,axis=0)
        error=float(np.nanmax(np.abs(calculated-m[['median_lateral_yd','median_depth_from_los_yd']].to_numpy())))
        max_median_error=max(max_median_error,error)
        max_support_error=max(max_support_error,int(np.max(np.abs(support-m.n_paths_observed))))
        minimum=min(len(g),max(5,math.ceil(.25*len(g))))
        if not np.array_equal(np.logical_and.accumulate(support>=minimum),m.included_in_median.to_numpy()):
            failures.append(f'{label}: retained median support differs')
        chosen=stratified_sample(g)
        actual=sampled.loc[sampled.route_label.eq(label)]
        if set(chosen[KEY].itertuples(index=False,name=None))!=set(actual[KEY].itertuples(index=False,name=None)):
            failures.append(f'{label}: sample differs from fixed seed')
        stat=summarize(label,g,chosen)
        for column,expected in stat.items():
            if column=='route_label':continue
            observed=summaries.loc[label,column]
            if not np.isclose(expected,observed,atol=1e-9,equal_nan=True):
                failures.append(f'{label}: summary {column} differs')
        checked_points+=len(grid)
    if max_median_error>1e-9 or max_support_error:
        failures.append('Saved median coordinates/support differ')
    # Independently enforce the earliest allowable eligibility relaxation and
    # minimize distances using the delivered median, rather than call selector.
    rankings_verified=0
    for audit in gate['representative_selection']:
        if audit.get('status')=='label_absent_from_subset':continue
        label,result=audit['route_label'],audit['pass_result']
        g=routes.loc[routes.route_label.eq(label)]
        base=(g.play_nullified_by_penalty.eq('N')&~bool_col(g.side_flag)&g.landing_to_arrival_yd.le(10)&g.pass_result.eq(result))
        qtime=g.throw_time_s.quantile([.25,.75]).to_numpy()
        qdepth=g.arrival_depth_yd.quantile([.25,.75]).to_numpy()
        iqr=g.throw_time_s.between(*qtime)&g.arrival_depth_yd.between(*qdepth)
        stages=[('none',base&iqr&g.n_defenders_at_arrival.ge(1)),('IQR',base&g.n_defenders_at_arrival.ge(1)),('IQR;defender',base)]
        candidates=pd.DataFrame()
        for stage,mask in stages:
            candidates=g.loc[mask].copy()
            if len(candidates):break
        found=selected.loc[selected.route_label.eq(label)&selected.pass_result.eq(result)]
        if candidates.empty:
            if len(found):failures.append(f'{label} {result}: selected ineligible result')
            continue
        if len(found)!=1:
            failures.append(f'{label} {result}: missing representative')
            continue
        m=medians.loc[medians.route_label.eq(label)&medians.included_in_median]
        scores=[]
        for key in candidates[KEY].itertuples(index=False,name=None):
            p=paths[key]
            inside=m.time_since_snap_s.between(p['t'][0]-1e-8,p['t'][-1]+1e-8)
            t=m.loc[inside,'time_since_snap_s'].to_numpy()
            xy=np.c_[np.interp(t,p['t'],p['lateral']),np.interp(t,p['t'],p['depth'])]
            med=m.loc[inside,['median_lateral_yd','median_depth_from_los_yd']].to_numpy()
            distance=float(np.sqrt(((xy-med)**2).sum(axis=1)).mean())
            scores.append((distance,)+tuple(key))
        score=min(scores)
        r=found.iloc[0]
        if tuple(r[c] for c in KEY)!=score[1:] or r.relaxed_conditions!=stage or not np.isclose(r.selection_distance_yd,score[0],atol=1e-9):
            failures.append(f'{label} {result}: representative is not nearest eligible path')
        rankings_verified+=1
    pngs=list((out/'spaghetti').glob('*.png'))+list((out/'representative').glob('*.png'))
    for path in pngs:
        with Image.open(path) as picture:
            picture.verify()
        with Image.open(path) as picture:
            expected=(3040,2400) if path.name=='all_routes_overview.png' else ((1920,1440) if path.parent.name=='spaghetti' else (2240,1600))
            if picture.size!=expected:failures.append(f'{path.name}: incorrect dimensions')
            if any(abs(dpi-160)>1 for dpi in picture.info.get('dpi',(0,0))):failures.append(f'{path.name}: incorrect DPI')
    parser=Links()
    parser.feed((out/'index.html').read_text(encoding='utf-8'))
    missing=[link for link in parser.links if not (out/link).exists()]
    if missing:failures.append(f'Missing gallery targets: {missing}')
    unchanged=all(digest(resolve_recorded_path(p))==h for p,h in gate['input_sha256'].items())
    if not unchanged:failures.append('Source/prior-output preservation hash differs')
    record=dict(status='failed' if failures else 'passed',failures=failures,seed=SEED,
                scope=gate['scope'],labeled_targeted_routes=len(routes),median_points_verified=checked_points,
                maximum_median_coordinate_error_yd=max_median_error,maximum_support_count_error=max_support_error,
                representative_rankings_verified=rankings_verified,images_verified=len(pngs),gallery_links_verified=len(parser.links),
                source_and_prior_version_hashes_unchanged=unchanged,
                notes=['Coordinates rebuilt from source paths and corrected sides. All-label medians and supports independently recalculated.',
                       'Representative eligibility and distance ranking independently recalculated for both result groups.',
                       'Builder separately verifies selected official output paths and nearest-defender distances.'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(clean(record),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(clean(record),indent=2))
    if failures:raise AssertionError('; '.join(failures))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'full_season')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ROOT.parents[1]/'Step 2.0 Route Dataset'/'data')
    parser.add_argument('--break-dir',type=Path,default=ROOT.parents[1]/'Step 2.0 Route Dataset'/'break_features_v2_35deg'/'full_season')
    parser.add_argument('--train-dir',type=Path,default=Path('C:/Local/Projects/Datalab Football Project/data/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'))
    args=parser.parse_args()
    verify(args.out,args.data_dir,args.break_dir,args.train_dir,args.report)
