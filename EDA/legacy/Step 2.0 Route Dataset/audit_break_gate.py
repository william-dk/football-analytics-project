"""Independent source reconstruction of the failed FLAT sign gate and GO fit.

Does not call route_geometry or modify any prior output. Writes a new review
record beside this script, suitable for inspecting before downstream work.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEY = ['game_id', 'play_id', 'nfl_id']


def main():
    dest = ROOT / 'break_features_review.json'
    if dest.exists():
        raise FileExistsError(dest)
    data = ROOT / 'data'
    routes = pd.read_csv(data/'routes.csv', usecols=KEY+['is_targeted','route_label','side','side_flag','player_position'])
    routes = routes.loc[routes.is_targeted & routes.route_label.isin(['FLAT','GO'])]
    features = pd.read_csv(data/'target_break_features.csv')
    rows = routes.merge(features,on=KEY,validate='one_to_one').set_index(KEY)
    paths = pd.read_csv(data/'target_full_paths.csv',usecols=KEY+['frame_id','t','x_norm','y'])
    paths = paths.merge(routes[KEY],on=KEY,validate='many_to_one')
    observations, max_error, max_angle_error = [], 0., 0.
    for key,g in paths.groupby(KEY):
        g = g.sort_values('t')
        row = rows.loc[key]
        t = g.t.to_numpy()
        y = g.y.to_numpy()
        release = (y[np.flatnonzero(t<=1+1e-8)[-1]]-y[0])*row.side
        side = row.side*np.sign(release) if row.side_flag and abs(release)>=1 else row.side
        assert side == row.side_corrected
        xy = np.c_[g.x_norm.to_numpy(), (y-y[0])*side]
        kernel = np.array([1,4,6,4,1])/16
        xy = np.column_stack([np.convolve(np.pad(xy[:,j],(2,2),'edge'),kernel,'valid') for j in range(2)])
        stem = np.flatnonzero((t>=.3-1e-8)&(t<=1+1e-8))
        final = np.flatnonzero(t>=t[-1]-.5-1e-8)
        angle = np.nan
        a0 = a1 = np.nan
        if len(stem)>=2 and len(final)>=2 and stem[-1]<final[0]:
            d0,d1 = xy[stem[-1]]-xy[stem[0]],xy[final[-1]]-xy[final[0]]
            if min(np.linalg.norm(d0),np.linalg.norm(d1))>1e-12:
                a0,a1 = np.degrees(np.arctan2(d0[1],d0[0])),np.degrees(np.arctan2(d1[1],d1[0]))
                angle = (a1-a0+180)%360-180
        assert np.isnan(angle) == pd.isna(row.total_turn_deg)
        if np.isfinite(angle): max_error=max(max_error,abs(angle-row.total_turn_deg))
        # Independently calculate TLS residuals using covariance eigenvalues,
        # rather than the production SVD implementation.
        best = None
        for k in range(3,len(xy)-3):
            residual,headings = 0.,[]
            for segment in [xy[:k+1],xy[k:]]:
                centered=segment-segment.mean(axis=0)
                vals,vectors=np.linalg.eigh(centered.T@centered)
                residual+=max(0.,vals[0])
                vector=vectors[:,-1]
                travel=segment[-1]-segment[0]
                if np.dot(vector,travel)<0: vector=-vector
                headings.append(np.degrees(np.arctan2(vector[1],vector[0])) if np.linalg.norm(travel)>1e-12 else np.nan)
            if best is None or residual<best[0]-1e-12:
                best=(residual,(headings[1]-headings[0]+180)%360-180,k)
        if best is not None and np.isfinite(best[1]) and np.isfinite(row.fit_angle_deg):
            max_angle_error=max(max_angle_error,abs((best[1]-row.fit_angle_deg+180)%360-180))
        observations.append(dict(label=row.route_label,turn=angle,stem_heading=a0,final_heading=a1,
                                 side_flag=bool(row.side_flag),side_flipped=bool(row.side_was_flipped),
                                 position=row.player_position,fit_angle=row.fit_angle_deg,has_break=bool(row.has_break)))
    obs=pd.DataFrame(observations)
    groups={}
    for label,g in obs.groupby('label'):
        groups[label]=dict(n=len(g),valid_turn_n=int(g.turn.notna().sum()),median_total_turn_deg=g.turn.median(),
                           median_stem_heading_deg=g.stem_heading.median(),median_final_heading_deg=g.final_heading.median(),
                           positive_turn_share_of_valid=float(g.turn.dropna().gt(0).mean()),
                           has_break_share=float(g.has_break.mean()),median_abs_fit_angle_deg=g.fit_angle.abs().median(),
                           turn_by_position=g.groupby('position').turn.median().to_dict(),
                           turn_by_side_flag={str(k):v for k,v in g.groupby('side_flag').turn.median().to_dict().items()})
    assert max_error < 1e-8 and max_angle_error < 1e-7
    record=dict(status='review_required',failures=['Full-season FLAT median total_turn_deg must be >0; observed -1.3875235819598402'],
                independent_reconstruction_passed=True,max_total_turn_error_deg=max_error,max_tls_angle_error_deg=max_angle_error,
                seed=20261025,groups=groups,
                conclusion='The independent reconstruction agrees with the feature table. The required FLAT sign gate fails under the prescribed definition. GO detection is 52.68%, above the expected 15%. No downstream results have been produced.',
                next_action='Review the sign requirement and TLS break definition before authorizing any change or downstream run.')
    dest.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
