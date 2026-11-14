"""Unit tests. Run inside this folder:  python -m unittest discover -s . -p "test_*.py" -v
Model tests are skipped where PyTorch cannot be imported (for example Windows with Smart App Control)."""
import unittest

import numpy as np

from masking import (centered_windows, interior_windows, mean_distance, sample_windows,
                     straight_line_fill, training_knockout, window_frames, window_mask)
from player_model import (bag_noise, bag_scores, fit_type_model, player_posteriors, project, route_scores,
                          variance_components)
from route_sequences import (MAX_LEN, assign_splits, channel_scales, correct_side, kinematics, pad_route,
                             route_features, route_xy, smooth_positions)

try:
    import torch
    from evaluate import RoutePCA, assign_route_types
    from identify_receivers import balanced, multi_route, topk_hits, within_team
    from model import RouteAutoencoder, masked_mse, run_batches, velocity_mse
    from player_style import half_season_products
    from train import eigen_routes, epoch_batches
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

    def test_corrected_side_follows_the_release_of_flagged_runners(self):
        t = np.arange(20) / 10
        inside = 20 - 1.5 * t                                      # releases 1.5 yd toward -y in 1 s
        self.assertEqual(correct_side(t, inside, 1, True), (-1, True, False))
        self.assertEqual(correct_side(t, inside, 1, False), (1, False, False))     # not flagged: keep
        self.assertEqual(correct_side(t, 20 - 0.3 * t, 1, True), (1, False, True))  # small release: unsure

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

    def test_a_share_of_training_routes_is_shown_whole(self):
        rng = np.random.default_rng(0)
        lengths = rng.integers(22, MAX_LEN + 1, size=4000)
        mask = training_knockout(lengths, rng, MAX_LEN, 8, 15, 0.5, whole_prob=0.2)
        whole = ~mask.any(axis=1)
        self.assertAlmostEqual(whole.mean(), 0.2, delta=0.03)
        self.assertTrue(np.all((mask[~whole].sum(1) >= 8) & (mask[~whole].sum(1) <= 15)))
        self.assertFalse(training_knockout(lengths, rng, MAX_LEN, whole_prob=0.0).any(axis=1).min() == 0)

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
class ModelVariantTests(unittest.TestCase):
    """The readout and decoder options keep the bottleneck guarantees of the first version."""

    def build(self, **kwargs):
        torch.manual_seed(0)
        model = RouteAutoencoder(n_in=4, n_out=2, max_len=20, d_model=32, n_heads=4, enc_layers=2,
                                 dec_layers=2, z_dim=8, dim_ff=64, dropout=0.0, **kwargs).eval()
        if model.basis_size:
            rng = np.random.default_rng(0)
            routes = np.cumsum(rng.normal(size=(200, 20, 2)), axis=1).astype(np.float32)
            model.set_basis(*eigen_routes(routes, np.full(200, 20), model.basis_size))
        return model

    def setUp(self):
        torch.manual_seed(1)
        self.x = torch.randn(3, 20, 4)
        self.valid = torch.arange(20)[None, :] < torch.tensor([20, 15, 10])[:, None]
        self.ko = torch.zeros(3, 20, dtype=torch.bool)
        self.ko[:, 3:7] = True

    def test_every_variant_decodes_from_z_alone(self):
        for kwargs in (dict(readout="frames", decoder="linear"), dict(readout="frames", decoder="mlp", basis=6),
                       dict(readout="cls", decoder="linear", basis=6), dict(readout="frames")):
            model = self.build(**kwargs)
            with torch.no_grad():
                pred, z = model(self.x, self.valid, self.ko)
                self.assertEqual(tuple(pred.shape), (3, 20, 2))
                torch.testing.assert_close(pred, model.decode(z, self.valid))

    def test_frame_readout_ignores_knocked_out_and_padded_frames(self):
        model = self.build(readout="frames", decoder="mlp", basis=6)
        changed = self.x.clone()
        changed[:, 3:7] += 100.0
        changed[1, 15:] += 100.0
        changed[2, 10:] += 100.0
        with torch.no_grad():
            torch.testing.assert_close(model.encode(self.x, self.valid, self.ko),
                                       model.encode(changed, self.valid, self.ko))
            moved = self.x.clone()
            moved[:, 8] += 1.0
            diff = (model.encode(self.x, self.valid, self.ko) - model.encode(moved, self.valid, self.ko)).abs().sum(1)
        self.assertTrue(torch.all(diff > 1e-6))

    def test_cutting_a_batch_to_its_longest_route_changes_nothing(self):
        model = self.build(readout="frames", decoder="mlp", basis=6)
        short = self.valid[1:]                                      # routes of 15 and 10 frames
        with torch.no_grad():
            full_pred, full_z = model(self.x[1:], short, self.ko[1:])
        cut_pred, cut_z = run_batches(model, self.x[1:], short, self.ko[1:])
        torch.testing.assert_close(cut_z, full_z, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(cut_pred[:, :15], full_pred[:, :15], atol=1e-5, rtol=1e-4)
        self.assertTrue(torch.all(cut_pred[:, 15:] == 0))

    def test_linear_decoder_decodes_the_mean_z_to_the_mean_route(self):
        model = self.build(readout="frames", decoder="linear", basis=6)
        z = torch.randn(5, 8)
        valid = torch.ones(5, 20, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(model.decode(z.mean(0, keepdim=True), valid[:1])[0],
                                       model.decode(z, valid).mean(0), atol=1e-5, rtol=1e-4)

    def test_velocity_error_sees_jitter_that_position_error_barely_sees(self):
        target = torch.zeros(1, 30, 2)
        target[0, :, 0] = torch.linspace(0, 2.9, 30)
        valid = torch.ones(1, 30, dtype=torch.bool)
        ko = torch.zeros(1, 30, dtype=torch.bool)
        ko[0, 10:20] = True
        self.assertEqual(float(velocity_mse(target, target, valid, ko)), 0.0)
        jitter = target.clone()
        jitter[0, ::2, 1] += 0.02
        position, _, _ = masked_mse(jitter, target, valid, ko)
        self.assertGreater(float(velocity_mse(jitter, target, valid, ko)), 50 * float(position))

    def test_bucketed_batches_cover_every_route_once(self):
        rng = np.random.default_rng(0)
        lengths = rng.integers(10, 78, size=1000)
        rows = np.flatnonzero(lengths > 12)
        batches = epoch_batches(rows, lengths, 64, rng, bucket=True, chunk=4)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), rows)
        spread = np.mean([np.ptp(lengths[b]) for b in batches])
        loose = np.mean([np.ptp(lengths[b]) for b in epoch_batches(rows, lengths, 64, rng, bucket=False)])
        self.assertLess(spread, 0.5 * loose)


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not importable in this environment")
class BaselineTests(unittest.TestCase):
    def test_pca_fill_recovers_a_route_that_lies_in_its_span(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 1, MAX_LEN)
        shapes = np.stack([np.c_[t, 0 * t], np.c_[t ** 2, t], np.c_[0 * t, np.sin(3 * t)]])    # 3 basis routes
        routes = np.einsum("nk,ktc->ntc", rng.normal(size=(300, 3)), shapes)
        lengths = np.full(300, MAX_LEN)
        pca = RoutePCA(routes, lengths, np.arange(250), components=3)
        visible = np.ones((50, MAX_LEN), dtype=bool)
        visible[:, 30:45] = False                                   # a 15-frame knocked-out window
        _, rebuilt = pca.fit(routes[250:], visible, noise_yd=1e-4)
        np.testing.assert_allclose(rebuilt, routes[250:], atol=1e-3)

    def test_assigned_route_types_cover_labeled_and_unlabeled_rows(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(3, 4)) * 6
        kind = rng.integers(0, 3, size=600)
        z = centers[kind] + rng.normal(size=(600, 4))
        labels = np.array(["GO", "HITCH", "OUT"], dtype=object)[kind]
        labels[400:] = None                                         # untargeted: no label
        assigned, prob, accuracy = assign_route_types(z, labels, np.arange(600) % 20, folds=5, seed=0)
        self.assertGreater(accuracy, 0.95)
        self.assertGreater(np.mean(assigned[400:] == np.array(["GO", "HITCH", "OUT"])[kind[400:]]), 0.95)
        self.assertTrue(np.all((prob > 0) & (prob <= 1)))


class PlayerModelTests(unittest.TestCase):
    """Simulated receivers with known receiver, game and play variances."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.n_players, d, games = 150, 6, 16
        cls.lam = np.array([1.0, 0.3, 0.1, 0, 0, 0])
        cls.game_var = np.array([0.3, 0.0, 0.2, 0.1, 0, 0])
        player, game = [], []
        for p in range(cls.n_players):
            for g in range(games):
                k = rng.poisson(3)
                player += [p] * k
                game += [g] * k
        cls.player, cls.game = np.array(player), np.array(game)
        effect = rng.normal(size=(cls.n_players, d)) * np.sqrt(cls.lam)
        game_effect = rng.normal(size=(cls.n_players, games, d)) * np.sqrt(cls.game_var)
        cls.V = (effect[cls.player] + game_effect[cls.player, cls.game]
                 + rng.normal(size=(len(cls.player), d)) * np.sqrt(1 - cls.game_var))
        cls.F = cls.V @ rng.normal(size=(d, d)) + 3.0              # mixed, shifted features
        cls.train = cls.game % 2 == 0
        cls.model = fit_type_model(cls.F[cls.train], cls.player[cls.train], cls.game[cls.train])
        cls.rng = rng

    def test_variances_are_recovered(self):
        # 150 receivers: a variance of 1.0 is estimated to about +-0.12 (one standard error).
        np.testing.assert_allclose(self.model.lam[:3], self.lam[:3], atol=0.35)
        self.assertLess(self.model.lam[3:].max(), 0.05)
        between, game, play = variance_components(self.V, self.player, self.game)
        np.testing.assert_allclose(between, self.lam, atol=0.35)
        np.testing.assert_allclose(game, self.game_var, atol=0.08)
        np.testing.assert_allclose(play, 1 - self.game_var, atol=0.08)

    def test_shuffled_receivers_carry_no_signal(self):
        shuffled = self.rng.permutation(self.player[self.train])
        self.assertLess(fit_type_model(self.F[self.train], shuffled, self.game[self.train]).lam.max(), 0.06)

    def test_more_routes_identify_better_and_scores_agree(self):
        m, test = self.model, ~self.train
        mean, var, _ = player_posteriors(m, project(m, self.F[self.train]), self.player[self.train],
                                         self.game[self.train], self.n_players)
        v, truth, games = project(m, self.F[test]), self.player[test], self.game[test]
        single = route_scores(m, mean, var, v)
        np.testing.assert_allclose(bag_scores(mean, var, v[0], bag_noise(m, games[:1])), single[0])
        one = np.mean(single.argmax(1) == truth)
        hits = []
        for p in range(self.n_players):
            rows = np.flatnonzero(truth == p)
            hits.append(bag_scores(mean, var, v[rows].mean(0), bag_noise(m, games[rows])).argmax() == p)
        self.assertGreater(one, 3 / self.n_players)
        self.assertGreater(np.mean(hits), 3 * one)

    def test_receivers_without_routes_keep_the_prior(self):
        m = self.model
        mean, var, counts = player_posteriors(m, project(m, self.F[:10]), np.zeros(10, dtype=int),
                                              self.game[:10], 3)
        self.assertEqual(counts.tolist(), [10, 0, 0])
        np.testing.assert_allclose(mean[1:], 0)
        np.testing.assert_allclose(var[1:], np.tile(m.lam, (2, 1)))


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not importable in this environment")
class VarianceSplitTests(unittest.TestCase):
    def test_team_and_individual_parts_are_separated(self):
        rng = np.random.default_rng(1)
        teams, per_team, d, games = 32, 8, 4, 16
        n_players = teams * per_team
        team = np.repeat(np.arange(teams), per_team)
        effect = rng.normal(size=(teams, d))[team] * np.sqrt(0.3) + rng.normal(size=(n_players, d)) * np.sqrt(0.2)
        player = np.repeat(np.arange(n_players), games * 3)
        game = np.tile(np.repeat(np.arange(games), 3), n_players)
        shared = rng.normal(size=(teams, games, d)) * np.sqrt(0.5)   # team-game effect: same game, same team
        f = effect[player] + shared[team[player], game] + rng.normal(size=(len(player), d)) * 2.0
        half = np.where(game % 2 == 1, "odd", "even")
        same, mate = half_season_products(f - f.mean(0), player, team, half, n_players)
        self.assertAlmostEqual(mate / d, effect.reshape(teams, per_team, d).mean(1).var(0).mean(), delta=0.08)
        self.assertAlmostEqual((same - mate) / d, 0.2, delta=0.06)


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
