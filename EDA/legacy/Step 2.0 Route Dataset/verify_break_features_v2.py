"""Verify amended outputs and unchanged geometry against the preserved first run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from build_break_features import KEY, clean, digest

ROOT=Path(__file__).resolve().parent


def main():
    new_dir=ROOT/'break_features_v2_35deg'/'full_season'
    dest=new_dir/'verification.json'
    if dest.exists():raise FileExistsError(dest)
    gate=json.loads((new_dir/'break_features_validation.json').read_text())
    old=pd.read_csv(ROOT/'data'/'target_break_features.csv').sort_values(KEY).reset_index(drop=True)
    new=pd.read_csv(new_dir/'target_break_features.csv').sort_values(KEY).reset_index(drop=True)
    assert gate['status']=='passed' and gate['failures']==[]
    assert new[KEY].equals(old[KEY]) and not new.duplicated(KEY).any()
    invariant=['total_turn_deg','fit_angle_deg','fit_residual_ss','max_depth_yd','retreat_yd','time_at_max_depth_s','side_corrected','side_was_flipped','side_uncertain']
    for col in invariant:
        np.testing.assert_allclose(new[col].to_numpy(float),old[col].to_numpy(float),equal_nan=True,atol=1e-10)
    has=new.fit_angle_deg.abs().ge(35)
    np.testing.assert_array_equal(new.has_break.to_numpy(),has.to_numpy())
    classes=np.where(has,'cut',np.where(new.fit_angle_deg.abs().ge(20),'bend','straight'))
    np.testing.assert_array_equal(new.break_class.to_numpy(),classes)
    strengths=['break_t_s','break_depth_yd','break_lateral_yd','speed_before','speed_min','speed_after','speed_retention','sharpness_s','break_after_throw']
    assert new.loc[~has,strengths].isna().all().all()
    for col in strengths:
        np.testing.assert_allclose(new.loc[has,col].to_numpy(float),old.loc[has,col].to_numpy(float),equal_nan=True,atol=1e-10)
    lookup=new.set_index(KEY)
    paths=pd.read_csv(ROOT/'data'/'target_full_paths.csv',usecols=KEY+['t','x_norm','y'])
    max_stem_error,max_endpoint_error=0.,0.
    for key,g in paths.groupby(KEY):
        g=g.sort_values('t')
        r=lookup.loc[key]
        xy=np.c_[g.x_norm.to_numpy(),(g.y.to_numpy()-g.y.iloc[0])*r.side_corrected]
        max_endpoint_error=max(max_endpoint_error,abs(xy[-1,1]-r.catch_y_corrected))
        kernel=np.array([1,4,6,4,1])/16
        smooth=np.column_stack([np.convolve(np.pad(xy[:,j],2,'edge'),kernel,'valid') for j in range(2)])
        i=np.flatnonzero(g.t.between(.3-1e-8,1+1e-8).to_numpy())
        heading=np.nan
        if len(i)>=2:
            delta=smooth[i[-1]]-smooth[i[0]]
            if np.linalg.norm(delta)>1e-12:heading=np.degrees(np.arctan2(delta[1],delta[0]))
        assert np.isnan(heading)==pd.isna(r.stem_heading_deg)
        if np.isfinite(heading):max_stem_error=max(max_stem_error,abs(heading-r.stem_heading_deg))
    assert max_stem_error<1e-9 and max_endpoint_error<1e-9
    unchanged=all(digest(Path(p))==h for p,h in gate['input_sha256'].items())
    assert unchanged
    record=dict(status='passed',failures=[],seed=20261025,targeted_routes=len(new),
                invariant_columns_against_preserved_run=invariant,cut_strength_columns_unchanged=strengths,
                classification_and_missingness_verified=True,maximum_stem_heading_error_deg=max_stem_error,
                maximum_corrected_endpoint_error_yd=max_endpoint_error,source_hashes_unchanged=unchanged)
    dest.write_text(json.dumps(clean(record),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(clean(record),indent=2))


if __name__=='__main__':main()
