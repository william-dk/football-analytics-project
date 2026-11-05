"""Small regression fixtures for geometry, joins and input/output timing."""
import unittest

import numpy as np
import pandas as pd

from build_route_dataset import build_week, detect_break, flip_field, wrapped_degrees, SUPP


def fixture():
    rows, outputs = [], []
    for game, direction in [(1, "right"), (2, "left")]:
        for player, role, x0, y0 in [(10, "Passer", 27, 25),
                (11, "Targeted Receiver", 30, 35), (12, "Other Route Runner", 30, 30),
                (13, "Defensive Coverage", 32, 35)]:
            for f in range(1, 11):
                x, y = x0 + (f-1)*.3, y0
                if direction == "left":
                    x, y = 120-x, 53.3-y
                rows.append(dict(game_id=game, play_id=7, nfl_id=player, frame_id=f,
                    player_name=str(player), player_position="QB" if player == 10 else "WR",
                    player_role=role, play_direction=direction,
                    absolute_yardline_number=30 if direction == "right" else 90,
                    x=x, y=y, s=3., a=0., num_frames_output=30))
            if player in [11, 13]:
                for f in range(1, 31):
                    # Target turns only after the throw.
                    x = x0 + min(9 + f, 18)*.3
                    y = y0 + max(9 + f - 18, 0)*.3
                    if direction == "left":
                        x, y = 120-x, 53.3-y
                    outputs.append(dict(game_id=game, play_id=7, nfl_id=player, frame_id=f, x=x, y=y))
    supplementary = pd.DataFrame([{**{c: 0 for c in SUPP}, "game_id": g,
        "play_id": 7, "season": 2023, "week": 1, "route_of_targeted_receiver": label}
        for g, label in [(1, "OUT"), (2, "IN")]])
    return pd.DataFrame(rows), pd.DataFrame(outputs), supplementary


class RouteDatasetTests(unittest.TestCase):
    def test_full_build_geometry_labels_and_seam(self):
        inp, out, supp = fixture()
        targets, routes, frames, audit, exceptions, plays = build_week(inp, out, supp)
        self.assertEqual(len(targets), 80)
        self.assertEqual(len(frames), 40)
        self.assertTrue(exceptions.empty)
        self.assertEqual(routes.loc[routes.is_targeted, "route_label"].tolist(), ["OUT", "IN"])
        self.assertTrue(routes.loc[~routes.is_targeted, "route_label"].isna().all())
        a = targets.loc[targets.game_id.eq(1)].reset_index(drop=True)
        b = targets.loc[targets.game_id.eq(2)].reset_index(drop=True)
        np.testing.assert_allclose(a[["x_norm", "y_norm"]], b[["x_norm", "y_norm"]], atol=1e-12)
        self.assertEqual(a.loc[9, "phase"], "input")
        self.assertEqual(a.loc[10, "phase"], "output")
        self.assertEqual(a.loc[10, "frame_id"], 11)
        self.assertAlmostEqual(a.loc[10, "vx"], 3.)
        self.assertAlmostEqual(a.loc[10, "s"], 3.)
        self.assertAlmostEqual(a.loc[10, "a"], 0.)
        self.assertTrue(np.isnan(a.loc[0, "heading_sin"]))
        target_routes = routes.loc[routes.is_targeted]
        self.assertTrue(target_routes.break_detected.all())
        self.assertTrue(target_routes.break_before_throw.eq(False).all())
        self.assertTrue(target_routes.ball_flight_s.eq(3).all())
        self.assertTrue(routes.loc[~routes.is_targeted, "x_norm_1_0s"].isna().all())
        self.assertTrue(routes.loc[~routes.is_targeted, "break_detected"].isna().all())
        self.assertEqual(routes.alignment.tolist(), ["outermost", "slot", "outermost", "slot"])
        self.assertTrue(routes.depth_from_los_yd.eq(0).all())
        defenders = flip_field(inp.loc[inp.nfl_id.eq(13) & inp.frame_id.eq(1)])
        np.testing.assert_allclose(defenders.x, [32, 32])
        np.testing.assert_allclose(defenders.y, [35, 35])

    def test_side_fallback_and_rb_flag(self):
        inp, out, supp = fixture()
        inp.loc[inp.nfl_id.eq(11), "player_position"] = "RB"
        inp = inp.loc[~(inp.game_id.eq(1) & inp.player_role.eq("Passer"))]
        targets, routes, *_ = build_week(inp, out, supp)
        self.assertTrue(routes.loc[routes.game_id.eq(1), "side_fallback"].all())
        self.assertTrue(routes.loc[routes.is_targeted, "side_flag"].all())
        self.assertTrue(targets.loc[targets.game_id.eq(1), "side"].eq(1).all())
        self.assertTrue(routes.loc[routes.game_id.eq(1), "alignment"].isna().all())

    def test_duplicate_and_gap_rejected(self):
        inp, out, supp = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_week(pd.concat([inp, inp.iloc[:1]]), out, supp)
        with self.assertRaisesRegex(ValueError, "nonconsecutive"):
            build_week(inp.drop(index=2), out, supp)

    def test_slow_noise_is_not_a_break(self):
        t = np.arange(20)/10
        x, y = np.sin(t)*.1, np.cos(t)*.1
        self.assertFalse(detect_break(x, y, t, np.ones(20)*.1, 1)["break_detected"])

    def test_missing_label_preserved_and_low_speed_heading_masked(self):
        inp, out, supp = fixture()
        supp.loc[supp.game_id.eq(1), "route_of_targeted_receiver"] = pd.NA
        inp.loc[inp.nfl_id.eq(11), "s"] = .5
        targets, routes, _, audit, *_ = build_week(inp, out, supp)
        self.assertEqual(len(routes.loc[routes.is_targeted]), 2)
        self.assertEqual(audit["unlabeled_target_routes"], 1)
        self.assertTrue(targets.loc[targets.phase.eq("input"), "heading_sin"].isna().all())
        self.assertEqual(wrapped_degrees(-340), 20)
        self.assertEqual(wrapped_degrees(340), -20)


if __name__ == "__main__":
    unittest.main()
