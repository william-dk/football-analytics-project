"""Focused checks of resampling, missing values and non-stretched path medians."""
import unittest
import numpy as np
import pandas as pd
from build_route_subtypes import arc_shape, encode, pointwise_median, separation_bootstrap, fit_label
from subtype_helpers import mixture, threadpool_limits


class NumericalChecks(unittest.TestCase):
    def test_arc_length_ignores_frame_spacing_and_handles_stationary_frames(self):
        a, length = arc_shape(np.array([[0, 0], [0, 0], [1, 0], [10, 0], [10, 10]], float))
        b, other_length = arc_shape(np.array([[0, 0], [10, 0], [10, 2], [10, 10]], float))
        np.testing.assert_allclose(a, b)
        self.assertEqual(length, 20)
        self.assertEqual(other_length, 20)
        stationary, length = arc_shape(np.zeros((4, 2)))
        np.testing.assert_equal(stationary, np.zeros(24))
        self.assertEqual(length, 0)

    def test_fit_imputation_does_not_modify_raw_and_uses_training_median(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(30, 30))
        raw[:, 27] = np.nan
        raw[0, 28] = np.nan
        raw[:, 29] = np.arange(30)
        copy = raw.copy()
        x, prep = encode(raw)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(prep[0][27], 0)
        self.assertEqual(prep[0][28], np.median(raw[1:, 28]))
        np.testing.assert_equal(raw, copy)
        self.assertGreater(len(np.unique(x[:,-1])), 2)
        np.testing.assert_allclose(encode(raw, prep)[0], x)
        shifted = raw.copy()
        shifted[0,28] = np.nan
        shifted[1:,28] = 10000
        old_median = prep[0][28]
        encode(shifted,prep)
        self.assertEqual(prep[0][28],old_median)

    def test_diagonal_model_and_wheel_forced_single_component(self):
        rng=np.random.default_rng(20261025)
        raw=rng.normal(size=(30,29))
        with threadpool_limits(limits=1):
            _,model,_,info,rows=fit_label(raw,np.repeat(np.arange(5),6),'WHEEL',30,20261025)
        self.assertEqual(model.covariance_type,'diag')
        self.assertEqual(model.n_init,10)
        self.assertEqual(info['chosen_k'],1)
        self.assertEqual(len(rows),1)
        self.assertEqual(info['resamples'],50)

    def test_median_uses_actual_time_and_does_not_extend_short_path(self):
        short = np.array([[0, 0, 0], [1, 1, 0]], float)
        long = np.array([[0, 0, 0], [2, 4, 0]], float)
        grid, median = pointwise_median([short, short, long])
        self.assertAlmostEqual(grid[-1], 1)
        self.assertAlmostEqual(median[-1, 0], 1)
        self.assertAlmostEqual(median[5, 0], .5)

    def test_game_bootstrap_preserves_known_pair_difference_and_missingness(self):
        frame = pd.DataFrame({"game_id": [1, 1, 2, 2, 3, 3], "route_label": ["OUT"] * 6,
                              "cluster_id": [1, 2] * 3, "provisional_subtype": ["OUT_short", "OUT_deep"] * 3,
                              "sep_at_throw": [1, 3] * 3, "sep_at_arrival": [np.nan, np.nan, 2, 5, 2, 5]})
        result = separation_bootstrap(frame, 1000, np.random.default_rng(7))
        self.assertEqual(result[0]["difference_b_minus_a_yd"], 2)
        self.assertEqual(result[0]["ci_low_yd"], 2)
        self.assertEqual(result[0]["ci_high_yd"], 2)
        self.assertTrue(result[0]["meets_separation_rule"])
        self.assertEqual(result[1]["difference_b_minus_a_yd"], 3)
        self.assertLess(result[1]["valid_bootstraps"], 1000)
        self.assertFalse(result[1]["meets_separation_rule"])


if __name__ == "__main__":
    unittest.main()
