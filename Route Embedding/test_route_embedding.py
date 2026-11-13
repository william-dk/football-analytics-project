"""Unit tests. Run inside this folder:  python -m unittest discover -s . -p "test_*.py" -v
Model tests are skipped where PyTorch cannot be imported (for example Windows with Smart App Control)."""
import unittest

import numpy as np

from masking import (centered_windows, interior_windows, mean_distance, sample_windows,
                     straight_line_fill, window_frames, window_mask)
from route_sequences import (MAX_LEN, assign_splits, channel_scales, kinematics, pad_route,
                             route_features, route_xy, smooth_positions)

try:
    import torch
    from identify_receivers import balanced, multi_route, topk_hits, within_team
    from model import RouteAutoencoder, masked_mse
    HAVE_TORCH = True
except Exception:  # noqa: BLE001 - ImportError, or OSError when a DLL is blocked
    HAVE_TORCH = False


class SequenceTests(unittest.TestCase):
    def test_smoothing_keeps_straight_lines(self):
        xy = np.c_[np.arange(10.0), 2 * np.arange(10.0)]
        np.testing.assert_allclose(smooth_positions(xy)[2:-2], xy[2:-2])

    def test_constant_velocity_kinematics(self):
        t = np.arange(30) / 10
        k = kinematics(np.c_[5 * t, 2 * t])           # columns vx, vy, s, a
        np.testing.assert_allclose(k[3:-3, 2], np.hypot(5, 2), atol=1e-9)
        np.testing.assert_allclose(k[4:-4, 3], 0, atol=1e-9)

    def test_braking_gives_negative_acceleration(self):
        t = np.arange(40) / 10
        x = 8 * t - 1.0 * t ** 2                        # slowing from 8 yd/s at 2 yd/s^2
        k = kinematics(np.c_[x, 0 * t])
        np.testing.assert_allclose(k[5:-5, 3], -2.0, atol=1e-6)

    def test_route_xy_uses_corrected_side(self):
        xy = route_xy([0.0, 1.0, 2.0], np.array([10.0, 10.5, 11.0]), -1)
        np.testing.assert_allclose(xy, [[0, 0], [1, -0.5], [2, -1.0]])

    def test_pad_and_truncate(self):
        padded, length, truncated = pad_route(np.ones((MAX_LEN + 5, 6)))
        self.assertEqual((padded.shape, length, truncated), ((MAX_LEN, 6), MAX_LEN, True))
        padded, length, truncated = pad_route(np.ones((30, 6)))
        self.assertEqual((length, truncated), (30, False))
        self.assertTrue(np.all(padded[30:] == 0))

    def test_features_use_frames_beyond_the_cut(self):
        t = np.arange(90) / 10
        padded, length, _ = pad_route(route_features(4 * t, 20 + 0 * t, 1))
        self.assertAlmostEqual(float(padded[length - 1, 4]), 4.0, places=6)

    def test_splits_keep_games_together(self):
        games = np.repeat(np.arange(50), 7)
        split = assign_splits(games, seed=1)
        for g in np.unique(games):
            self.assertEqual(len(set(split[games == g])), 1)
        self.assertEqual(set(split), {"train", "val", "test"})

    def test_position_channels_share_a_scale(self):
        X = np.random.default_rng(0).normal(size=(20, MAX_LEN, 6)).astype(np.float32)
        scales = channel_scales(X, np.full(20, 40))
        self.assertEqual(scales[0], scales[1])
        self.assertEqual(scales[2], scales[3])


class MaskingTests(unittest.TestCase):
    def test_training_windows_stay_inside_the_route(self):
        rng = np.random.default_rng(0)
        lengths = rng.integers(22, MAX_LEN + 1, size=5000)
        starts, widths = sample_windows(lengths, rng, 8, 15, 0.5)
        self.assertTrue(np.all(starts >= 1))
        self.assertTrue(np.all(starts + widths <= lengths))
        self.assertTrue(np.all((widths >= 8) & (widths <= 15)))
        mask = window_mask(starts, widths, MAX_LEN)
        np.testing.assert_array_equal(mask.sum(1), widths)
        self.assertFalse(np.any(mask & (np.arange(MAX_LEN)[None, :] >= lengths[:, None])))

    def test_width_is_capped_for_short_routes(self):
        _, widths = sample_windows(np.full(1000, 12), np.random.default_rng(1), 8, 15, 0.5)
        self.assertTrue(np.all(widths == 6))

    def test_windows_cover_route_start_and_end(self):
        starts, widths = sample_windows(np.full(20000, 30), np.random.default_rng(2), 8, 8, 0.5)
        self.assertTrue(np.any(starts == 1))
        self.assertTrue(np.any(starts + widths == 30))

    def test_interior_and_centred_windows(self):
        starts = interior_windows(np.array([10, 12, 40]), 10, np.random.default_rng(3))
        self.assertEqual(starts[0], -1)
        self.assertEqual(starts[1], 1)
        self.assertTrue(1 <= starts[2] <= 29)
        np.testing.assert_array_equal(centered_windows(np.array([40, 40]), np.array([2, 38]), 10), [1, 29])
        self.assertFalse(window_mask(np.array([-1]), 10, MAX_LEN).any())

    def test_straight_line_fill_is_exact_on_a_line(self):
        xy = np.c_[np.arange(40.0), 0.5 * np.arange(40.0)][None]
        starts = np.array([7])
        err = mean_distance(straight_line_fill(xy, starts, 10), window_frames(xy, starts, 10))
        np.testing.assert_allclose(err, [0.0], atol=1e-12)


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not importable in this environment")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = RouteAutoencoder(n_in=4, n_out=2, max_len=20, d_model=32, n_heads=4, enc_layers=2,
                                      dec_layers=2, z_dim=16, dim_ff=64, dropout=0.0).eval()
        self.x = torch.randn(3, 20, 4)
        self.valid = torch.arange(20)[None, :] < torch.tensor([20, 15, 10])[:, None]
        self.ko = torch.zeros(3, 20, dtype=torch.bool)
        self.ko[:, 3:7] = True

    def test_shapes(self):
        pred, z = self.model(self.x, self.valid, self.ko)
        self.assertEqual(tuple(pred.shape), (3, 20, 2))
        self.assertEqual(tuple(z.shape), (3, 16))

    def test_knocked_out_and_padded_frames_do_not_reach_z(self):
        changed = self.x.clone()
        changed[:, 3:7] += 100.0
        changed[1, 15:] += 100.0
        changed[2, 10:] += 100.0
        with torch.no_grad():
            torch.testing.assert_close(self.model.encode(self.x, self.valid, self.ko),
                                       self.model.encode(changed, self.valid, self.ko))

    def test_visible_frames_reach_z(self):
        changed = self.x.clone()
        changed[:, 8] += 1.0
        with torch.no_grad():
            diff = (self.model.encode(self.x, self.valid, self.ko)
                    - self.model.encode(changed, self.valid, self.ko)).abs().sum(1)
        self.assertTrue(torch.all(diff > 1e-6))

    def test_forward_has_no_path_around_z(self):
        with torch.no_grad():
            pred, z = self.model(self.x, self.valid, self.ko)
            torch.testing.assert_close(pred, self.model.decode(z, self.valid))

    def test_loss_ignores_padding(self):
        target = torch.zeros(1, 5, 2)
        target[0, 4] = 100.0
        valid = torch.tensor([[True, True, True, True, False]])
        ko = torch.tensor([[False, True, True, False, False]])
        total, _, _ = masked_mse(torch.zeros(1, 5, 2), target, valid, ko, 0.5)
        self.assertEqual(float(total), 0.0)

    def test_learns_to_fill_a_window_on_a_tiny_batch(self):
        torch.manual_seed(1)
        model = RouteAutoencoder(n_in=4, n_out=2, max_len=20, d_model=32, n_heads=4, enc_layers=2,
                                 dec_layers=2, z_dim=16, dim_ff=64, dropout=0.0)
        t = torch.linspace(0, 1.9, 20)
        routes = []
        for k in range(8):
            depth, side = 0.6 + 0.12 * k, (1.0 if k % 2 else -1.0)
            x = torch.minimum(t, torch.tensor(depth))
            y = side * torch.clamp(t - depth, min=0)
            routes.append(torch.stack([x, y, torch.ones_like(t), torch.zeros_like(t)], dim=1))
        x = torch.stack(routes)
        valid = torch.ones(8, 20, dtype=torch.bool)
        ko = torch.zeros(8, 20, dtype=torch.bool)
        ko[:, 8:13] = True
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = None
        for _ in range(200):
            pred, _ = model(x, valid, ko)
            loss, _, _ = masked_mse(pred, x[..., :2], valid, ko)
            first = first if first is not None else loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.2 * first)


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not importable in this environment")
class IdentificationTests(unittest.TestCase):
    def test_constant_guess_is_chance_under_balanced_accuracy(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        always_first = y == 0
        self.assertAlmostEqual(always_first.mean(), 0.5)              # plain accuracy is inflated
        self.assertAlmostEqual(balanced(always_first, y, 3), 1 / 3)   # balanced accuracy is chance

    def test_topk_hits(self):
        probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
        np.testing.assert_array_equal(topk_hits(probs, np.array([2, 1]), 1), [False, False])
        np.testing.assert_array_equal(topk_hits(probs, np.array([2, 1]), 2), [True, True])

    def test_within_team_ranks_only_teammates(self):
        probs = np.array([[0.1, 0.6, 0.3]])                            # best overall is on team B
        res = within_team(probs, np.array([2]), np.array(["A"]), np.array(["A", "B", "A"]))
        self.assertEqual((res["accuracy"], res["chance"]), (1.0, 0.5))

    def test_combining_routes_accumulates_weak_evidence(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1], 40)
        right = rng.random(len(y)) < 0.6                                # 60% of routes favour the truth
        p_true = np.where(right, 0.6, 0.4)
        probs = np.where(np.eye(2)[y] == 1, p_true[:, None], 1 - p_true[:, None])
        self.assertGreater(multi_route(probs, y, 15, rng, reps=100), multi_route(probs, y, 1, rng, reps=100))


if __name__ == "__main__":
    unittest.main()
