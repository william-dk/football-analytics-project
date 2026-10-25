"""Check the two EDA corrections and same-clock alignment invariants."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from build_route_analysis import fixed_time_median, ROUTES

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'CodexOuput_route_visualizations'

def check_time_alignment():
    # Same physical 2 yd/s movement, observed for 1.5s vs 3.5s.
    # Relative-duration alignment would incorrectly pair different seconds.
    paths=[]
    for duration in [1.5]*5+[3.5]*5:
        t=np.arange(round(duration*10)+1)/10
        paths.append(np.column_stack([t*0,t*2,t]))
    d=fixed_time_median(paths)
    assert np.allclose(d.downfield_median_yd,d.time_since_snap_proxy_s*2)
    assert d.loc[np.isclose(d.time_since_snap_proxy_s,1),'downfield_median_yd'].iloc[0]==2
    assert d.loc[np.isclose(d.time_since_snap_proxy_s,1.6),'n_observed'].iloc[0]==5
    assert d.n_observed.iloc[-1]==5  # ended tracks are not padded to 3.5s
    # A single long survivor must not extend a group median below its cutoff.
    d=fixed_time_median(paths[:5]+[paths[-1]])
    assert d.loc[d.included_in_median_plot,'time_since_snap_proxy_s'].max()==1.5
    assert d.loc[np.isclose(d.time_since_snap_proxy_s,3.5),'n_observed'].iloc[0]==1
    # Explicit observed timestamps, rather than array index or duration fraction.
    t=np.arange(1.,2.1,.1)
    d=fixed_time_median([np.column_stack([t*0,t*3,t])]*5)
    assert d.loc[d.time_since_snap_proxy_s.lt(.9),'n_observed'].eq(0).all()
    print('PASS: real-time alignment, no endpoint extension, support cutoff, observed timestamps')

def check_outputs():
    raw=pd.read_csv(ROOT/'supplementary_data.csv',low_memory=False)
    counts=pd.read_csv(OUT/'route_counts.csv',index_col='route')
    expected=raw.loc[raw.season.eq(2023),'route_of_targeted_receiver'].value_counts().reindex(ROUTES)
    assert counts.supplementary_count.eq(expected).all()
    assert counts.unmatched_count.eq(0).all()
    summary=pd.read_csv(OUT/'descriptive_statistics.csv')
    assert np.array_equal(summary.source_n,summary.tracking_n)
    assert int(summary.source_n.sum())==14105
    matches=pd.read_csv(OUT/'supplementary_play_match_audit.csv')
    assert len(matches)==14108 and matches.season.eq(2023).all() and matches.has_tracking.all()
    scope=pd.read_csv(OUT/'season_scope_audit.csv')
    excluded=scope.loc[~scope.in_analysis]
    assert excluded.season.eq(2024).all() and set(excluded.week)==set(range(14,19))
    assert excluded.plays.sum()==3901 and excluded.labeled_routes.sum()==3900
    df=pd.read_csv(OUT/'route_features.csv')
    assert df.start_frame.eq(1).all()
    assert np.allclose(df.snap_to_throw_proxy_s,(df.end_frame-1)/10)
    assert np.allclose(df.break_time_since_snap_proxy_s,df.break_time_from_window_start_s,equal_nan=True)
    assert summary.verified_snap_to_throw_s.isna().all()
    support=pd.read_csv(OUT/'median_time_support.csv')
    for _,g in support.groupby(['route','context','context_level']):
        assert np.all(np.diff(g.n_observed)<=0)
        assert g.n_observed.le(g.group_n).all()
        assert np.allclose(g.observed_share,g.n_observed/g.group_n)
        assert g.loc[g.included_in_median_plot,'n_observed'].ge(g.minimum_support_n.iloc[0]).all()
        assert np.allclose(np.diff(g.time_since_snap_proxy_s),.1)
    starts=pd.read_csv(OUT/'snap_proxy_first_frame_audit.csv')
    runners=starts[starts.player_role.isin(['Targeted Receiver','Other Route Runner'])]
    evidence=json.loads((OUT/'snap_proxy_validation.json').read_text())
    assert np.isclose(runners.s.median(),evidence['runner_speed_median_yd_s'])
    assert np.isclose(runners.s.lt(1).mean(),evidence['runner_speed_below_1_share'])
    assert evidence['runner_speed_median_yd_s']<.1 and evidence['runner_speed_below_1_share']>.9
    assert 4<evidence['qb_behind_los_median_yd']<6
    assert len(list(OUT.glob('animation_*.gif')))==6
    assert all(Image.open(p).n_frames>1 for p in OUT.glob('animation_*.gif'))
    report=(OUT/'route_analysis.html').read_text(encoding='utf-8')
    assert report.count('data:image/')==16
    assert 'same elapsed seconds' in report and '25%' in report
    assert all((OUT/('route_'+r.lower()+'.png')).exists() for r in ROUTES)
    print('PASS: season filter, all-play match coverage, timing proxies, support exports, evidence, report, animations')

if __name__=='__main__':
    check_time_alignment()
    check_outputs()
