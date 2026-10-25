import unittest
import numpy as np
from route_geometry import (measure_route, correct_side, smooth_positions,
                            total_turn, two_segment_fit, BREAK_FIELDS, wrap_angle,
                            classify_turn, stem_heading)


class GeometryTests(unittest.TestCase):
    def test_amended_threshold_boundaries_and_lateral_stem(self):
        for angle, expected in [(19.99,'straight'),(20,'bend'),(-34.99,'bend'),(35,'cut'),(-35,'cut')]:
            self.assertEqual(classify_turn(angle), expected)
        t = np.arange(41)/10
        xy = np.c_[np.maximum(t-2,0)*5,np.minimum(t,2)*5]
        r = measure_route(t, xy[:,0], xy[:,1], 1, False, 1.)
        self.assertAlmostEqual(r['stem_heading_deg'], 90)
        self.assertAlmostEqual(r['total_turn_deg'], -90)
        self.assertGreater(r['catch_y_corrected'], 0)
        self.assertEqual(r['break_class'], 'cut')
        bend = np.c_[np.minimum(t,2)*5 + np.maximum(t-2,0)*5*np.cos(np.radians(27)),
                     np.maximum(t-2,0)*5*np.sin(np.radians(27))]
        b = measure_route(t,bend[:,0],bend[:,1],1,False,1.)
        self.assertEqual(b['break_class'],'bend')
        self.assertFalse(b['has_break'])
        self.assertTrue(np.isfinite(b['fit_angle_deg']))
        self.assertTrue(all(np.isnan(b[c]) for c in BREAK_FIELDS))

    def test_release_correction_and_uncertainty(self):
        t = np.arange(16) / 10
        self.assertEqual(correct_side(t, -2*t, 1, True)['side_corrected'], -1)
        self.assertFalse(correct_side(t, -2*t, 1, False)['side_was_flipped'])
        self.assertTrue(correct_side(t, .2*t, -1, True)['side_uncertain'])
        self.assertEqual(correct_side(t[:8], -2*t[:8], 1, True)['side_corrected'], -1)

    def test_smoothing_edges(self):
        xy = np.c_[np.arange(7.), np.zeros(7)]
        self.assertAlmostEqual(smooth_positions(xy)[0, 0], 6/16)
        self.assertAlmostEqual(smooth_positions(xy)[-1, 0], 6-6/16)

    def test_straight_route_no_break_and_earliest_tie(self):
        t = np.arange(41) / 10
        r = measure_route(t, t*5, t*0, 1, False, 2.)
        self.assertFalse(r['has_break'])
        self.assertTrue(all(np.isnan(r[c]) for c in BREAK_FIELDS))
        self.assertEqual(two_segment_fit(np.c_[t, t*0])['split_index'], 3)
        self.assertAlmostEqual(r['total_turn_deg'], 0.)

    def test_vertical_tls_and_turn_sign(self):
        t = np.arange(41) / 10
        x = np.minimum(t, 2)*5
        y = np.maximum(t-2, 0)*5
        a = measure_route(t, x, y, 1, False, 1.)
        b = measure_route(t, x, y, -1, False, 1.)
        self.assertTrue(a['has_break'])
        self.assertAlmostEqual(a['break_t_s'], 2., delta=.1)
        self.assertAlmostEqual(a['fit_angle_deg'], 90, delta=2)
        self.assertAlmostEqual(a['total_turn_deg'], 90, delta=1)
        self.assertAlmostEqual(b['fit_angle_deg'], -a['fit_angle_deg'])
        self.assertTrue(a['break_after_throw'])
        self.assertGreater(a['speed_before'], 1)

    def test_overlap_short_stationary_and_retreat(self):
        t = np.arange(11) / 10
        self.assertTrue(np.isnan(total_turn(t, np.c_[t, t])))
        r = measure_route(t, t*0, t*0, 1, False, .5)
        self.assertFalse(r['has_break'])
        self.assertTrue(np.isnan(r['total_turn_deg']))
        r = measure_route(t, np.r_[np.arange(6), np.arange(4,-1,-1)], t*0, 1, False, .5)
        self.assertEqual(r['max_depth_yd'], 5)
        self.assertEqual(r['retreat_yd'], 5)
        self.assertEqual(r['time_at_max_depth_s'], .5)
        self.assertEqual(wrap_angle(181), -179)


if __name__ == '__main__':
    unittest.main()
