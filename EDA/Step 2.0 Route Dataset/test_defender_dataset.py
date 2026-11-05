"""Defender extension: geometry, ties, missing frames, and selection sensitivity."""
import unittest

import numpy as np
import pandas as pd

from build_route_dataset import build_week
from defender_dataset import defender_frames, separation_frames, separation_summary
from test_route_dataset import fixture


def setup_fixture():
    inp, out, supp = fixture()
    inp["player_side"] = np.where(inp.player_role.eq("Defensive Coverage"), "Defense", "Offense")
    inp.loc[inp.nfl_id.eq(13), "player_position"] = "CB"
    targets, routes, *_ = build_week(inp, out, supp)
    return inp, out, targets, routes


class DefenderDatasetTests(unittest.TestCase):
    def test_flip_projection_and_boundary_kinematics(self):
        inp, out, targets, routes = setup_fixture()
        d = defender_frames(inp, out, targets, routes)
        a = d.loc[d.game_id.eq(1)].reset_index(drop=True)
        b = d.loc[d.game_id.eq(2)].reset_index(drop=True)
        for c in ["dist_to_receiver", "rel_x", "rel_y", "along_track", "cross_track", "closing_speed"]:
            np.testing.assert_allclose(a[c], b[c], atol=1e-10, equal_nan=True)
        self.assertAlmostEqual(a.loc[9, "along_track"], 2.)
        self.assertAlmostEqual(a.loc[9, "cross_track"], 0.)
        self.assertAlmostEqual(a.loc[10, "vx_field"], 3.)
        self.assertAlmostEqual(a.loc[10, "s"], 3.)
        self.assertAlmostEqual(a.loc[10, "a"], 0.)
        s = separation_frames(d, targets)
        self.assertTrue(s.nearest_def_id.eq(13).all())
        self.assertTrue(s.sep_nearest_db.eq(s.sep_nearest).all())
        summary = separation_summary(d, s, routes)
        self.assertTrue(summary.nearest_at_throw_in_output.all())
        np.testing.assert_allclose(summary.sep_fixed_throw_def_at_arrival, summary.sep_at_arrival)

    def test_tie_selection_and_missing_throw_defender(self):
        inp, out, targets, routes = setup_fixture()
        extra = inp.loc[inp.nfl_id.eq(13)].copy()
        extra["nfl_id"], extra["player_position"] = 9, "LB"
        inp = pd.concat([inp, extra], ignore_index=True)
        d = defender_frames(inp, out, targets, routes)
        s = separation_frames(d, targets)
        self.assertTrue(s.loc[s.phase.eq("input"), "nearest_def_id"].eq(9).all())
        self.assertTrue(s.loc[s.phase.eq("input"), "nearest_db_id"].eq(13).all())
        self.assertTrue(d.loc[d.defender_nfl_id.eq(9), "in_output_set"].eq(False).all())
        summary = separation_summary(d, s, routes)
        self.assertTrue(summary.nearest_at_throw_in_output.eq(False).all())
        self.assertTrue(summary.sep_fixed_throw_def_at_arrival.isna().all())
        self.assertTrue(summary.nearest_switches_output.eq(1).all())

    def test_missing_frame_is_not_imputed(self):
        inp, out, targets, routes = setup_fixture()
        inp = inp.loc[~(inp.nfl_id.eq(13) & inp.frame_id.eq(5))]
        d = defender_frames(inp, out, targets, routes)
        self.assertFalse(d.frame_id.eq(5).any())
        self.assertTrue(d.loc[d.frame_id.eq(6), ["vx_field", "vy_field", "closing_speed"]].isna().all().all())
        s = separation_frames(d, targets)
        missing = s.loc[s.frame_id.eq(5)]
        self.assertTrue(missing.n_defenders_observed.eq(0).all())
        self.assertTrue(missing.sep_nearest.isna().all())
        self.assertTrue(s.loc[s.frame_id.isin([5, 6]), "nearest_switch"].isna().all())
        self.assertEqual(len(s), len(targets))

    def test_slow_receiver_and_no_db(self):
        inp, out, targets, routes = setup_fixture()
        inp.loc[inp.nfl_id.eq(13), "player_position"] = "LB"
        targets.loc[targets.phase.eq("input"), "s"] = .5
        d = defender_frames(inp, out, targets, routes)
        self.assertTrue(d.loc[d.phase.eq("input"), ["along_track", "cross_track"]].isna().all().all())
        s = separation_frames(d, targets)
        self.assertTrue(s.sep_nearest_db.isna().all())
        self.assertTrue(s.n_db_observed.eq(0).all())
        self.assertTrue(s.loc[s.phase.eq("input"), "cushion_flag"].isna().all())


if __name__ == "__main__":
    unittest.main()
