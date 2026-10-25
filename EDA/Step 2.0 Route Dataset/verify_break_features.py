"""Verify current 35-degree features from source paths, without legacy tables."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_break_features import KEY, clean, cohort, digest, resolve_recorded_path
from route_geometry import BREAK_FIELDS

ROOT=Path(__file__).resolve().parent


def verify(data_dir,feature_dir,report):
    if report.exists():raise FileExistsError(report)
    gate=json.loads((feature_dir/'break_features_validation.json').read_text())
    assert gate['status']=='passed' and gate['failures']==[] and gate['break_threshold_deg']==35
    five_games=gate['scope']=='week1_first5games'
    routes=cohort(data_dir,five_games)
    features=pd.read_csv(feature_dir/'target_break_features.csv').sort_values(KEY).reset_index(drop=True)
    assert features[KEY].equals(routes[KEY]) and not features.duplicated(KEY).any()
    cuts=features.fit_angle_deg.abs().ge(35)
    np.testing.assert_array_equal(features.has_break.to_numpy(),cuts.to_numpy())
    np.testing.assert_array_equal(features.break_class.to_numpy(),np.where(cuts,'cut',np.where(features.fit_angle_deg.abs().ge(20),'bend','straight')))
    assert features.loc[~cuts,BREAK_FIELDS].isna().all().all()
    assert not np.isinf(features.select_dtypes(include='number').to_numpy()).any()
    lookup=features.set_index(KEY)
    route_lookup=routes.set_index(KEY)
    frames=pd.read_csv(data_dir/'target_full_paths.csv',usecols=KEY+['frame_id','t','x_norm','y'])
    frames=frames.merge(routes[KEY],on=KEY,validate='many_to_one')
    assert not frames.duplicated(KEY+['frame_id']).any()
    errors={c:0. for c in ['side_corrected','stem_heading_deg','total_turn_deg','catch_y_corrected',
                          'max_depth_yd','retreat_yd','time_at_max_depth_s','break_depth_yd','break_lateral_yd',
                          'speed_before','speed_min','speed_after','speed_retention']}
    def compare(column,actual,expected):
        assert pd.isna(actual)==pd.isna(expected),column
        if np.isfinite(expected):
            error=abs(float(actual)-float(expected))
            errors[column]=max(errors[column],error)
            assert error<1e-8,(column,actual,expected)
    for key,g in frames.groupby(KEY):
        g=g.sort_values('t')
        r=lookup.loc[key]
        source=route_lookup.loc[key]
        t,x,y=g.t.to_numpy(),g.x_norm.to_numpy(),g.y.to_numpy()
        release=(y[np.flatnonzero(t<=1+1e-8)[-1]]-y[0])*source.side
        side=source.side*np.sign(release) if source.side_flag and abs(release)>=1 else source.side
        compare('side_corrected',r.side_corrected,side)
        lateral=(y-y[0])*side
        xy=np.c_[x,lateral]
        kernel=np.array([1,4,6,4,1])/16
        smooth=np.column_stack([np.convolve(np.pad(xy[:,j],2,'edge'),kernel,'valid') for j in range(2)])
        stem=np.flatnonzero((t>=.3-1e-8)&(t<=1+1e-8))
        final=np.flatnonzero(t>=t[-1]-.5-1e-8)
        def heading(indices):
            if len(indices)<2:return np.nan
            delta=smooth[indices[-1]]-smooth[indices[0]]
            return np.degrees(np.arctan2(delta[1],delta[0])) if np.linalg.norm(delta)>1e-12 else np.nan
        a0=heading(stem)
        turn=(heading(final)-a0+180)%360-180 if len(stem)>=2 and len(final)>=2 and stem[-1]<final[0] else np.nan
        compare('stem_heading_deg',r.stem_heading_deg,a0)
        compare('total_turn_deg',r.total_turn_deg,turn)
        for column,value in [('catch_y_corrected',lateral[-1]),('max_depth_yd',x.max()),
                             ('retreat_yd',x.max()-x[-1]),('time_at_max_depth_s',t[np.argmax(x)])]:
            compare(column,r[column],value)
        if r.has_break:
            indices=np.flatnonzero(np.abs(t-r.break_t_s)<1e-8)
            assert len(indices)==1
            k=indices[0]
            compare('break_depth_yd',r.break_depth_yd,x[k])
            compare('break_lateral_yd',r.break_lateral_yd,lateral[k])
            speed=np.r_[np.nan,np.linalg.norm(np.diff(smooth,axis=0),axis=1)*10]
            before=np.nanmean(speed[(t>=t[k]-.5-1e-8)&(t<t[k]-1e-8)])
            after=np.nanmean(speed[(t>t[k]+1e-8)&(t<=t[k]+.5+1e-8)])
            compare('speed_before',r.speed_before,before)
            compare('speed_after',r.speed_after,after)
            compare('speed_min',r.speed_min,np.nanmin(speed[np.abs(t-t[k])<=.3+1e-8]))
            compare('speed_retention',r.speed_retention,after/before if before>=1 else np.nan)
            assert (str(r.break_after_throw).lower()=='true')==bool(t[k]>source.throw_time_s)
    unchanged=all(digest(resolve_recorded_path(p))==h for p,h in gate['input_sha256'].items())
    assert unchanged
    record=dict(status='passed',failures=[],seed=20261025,scope=gate['scope'],targeted_routes=len(features),
                legacy_tables_required=False,classification_and_missingness_verified=True,
                maximum_absolute_errors=errors,source_hashes_unchanged=unchanged)
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(clean(record),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(clean(record),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--feature-dir',type=Path,default=ROOT/'break_features_v2_35deg'/'full_season')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    verify(args.data_dir,args.feature_dir,args.report or args.feature_dir/'current_verification.json')
