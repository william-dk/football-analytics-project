import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from build_route_visuals_v3 import select_representatives
from visual_helpers import load_routes, median_path


class VisualChecks(unittest.TestCase):
    def test_separate_completion_incompletion_and_ten_yard_landing(self):
        rows=[]
        paths={}
        for i,(result,lateral,landing) in enumerate([('C',1,8),('C',8,1),('I',2,9),('I',.1,11)],1):
            rows.append(dict(game_id=1,play_id=i,nfl_id=1,route_label='OUT',pass_result=result,
                             play_nullified_by_penalty='N',side_flag=False,landing_to_arrival_yd=landing,
                             throw_time_s=1,arrival_depth_yd=10,n_defenders_at_arrival=1))
            paths[1,i,1]=dict(t=np.array([0.,1.,2.]),lateral=np.array([0,lateral,lateral]),depth=np.array([0,5,10]))
        frame=pd.DataFrame(rows)
        median=pd.DataFrame(dict(time_since_snap_s=[0.,1.,2.],median_lateral_yd=[0,0,0],median_depth_from_los_yd=[0,5,10],included_in_median=[True]*3))
        chosen,audits=select_representatives(frame,paths,{'OUT':median},True)
        self.assertEqual(chosen.play_id.tolist(),[1,3])
        self.assertEqual(chosen.pass_result.tolist(),['C','I'])
        frame=frame.loc[frame.pass_result.eq('C')]
        chosen,audits=select_representatives(frame,paths,{'OUT':median},True)
        self.assertEqual(len(chosen),1)
        self.assertTrue(any(a.get('status')=='no_eligible_incompletion' for a in audits))

    def test_corrected_receiver_ball_and_sideline_share_one_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            row=dict(game_id=1,play_id=2,nfl_id=3,is_targeted=True,route_label='FLAT',side=1,
                     throw_time_s=.1,depth_from_los_yd=-1)
            row.update({c:np.nan for c in ['break_detected','break_depth_yd','break_time_s','break_angle_deg','speed_before_break','speed_after_break','break_before_throw']})
            pd.DataFrame([row]).to_csv(root/'routes.csv',index=False)
            pd.DataFrame([dict(game_id=1,play_id=2,nfl_id=3,side_corrected=-1,has_break=False,break_depth_yd=np.nan)]).to_csv(root/'target_break_features.csv',index=False)
            pd.DataFrame([dict(game_id=1,play_id=2,receiver_nfl_id=3,sep_at_throw=2,sep_at_arrival=3,n_defenders_at_arrival=1)]).to_csv(root/'target_separation_summary.csv',index=False)
            pd.DataFrame(dict(game_id=[1]*3,play_id=[2]*3,nfl_id=[3]*3,frame_id=[1,2,3],t=[0,.1,.2],phase=['input','input','output'],x=[50,51,52],y=[40,39,38])).to_csv(root/'target_full_paths.csv',index=False)
            meta=pd.DataFrame([dict(game_id=1,play_id=2,los_x=51,ball_x=53,ball_y=37)])
            routes,paths=load_routes(root,meta,root)
            r=routes.iloc[0]
            self.assertEqual(r.arrival_lateral,2)
            self.assertEqual(r.ball_lateral,3)
            self.assertEqual(r.own_sideline_yd,40)
            np.testing.assert_equal(paths[1,2,3]['lateral'],[0,1,2])


if __name__ == '__main__':unittest.main()
