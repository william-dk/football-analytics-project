"""Pure helpers that turn Step 2.0 targeted full paths into fixed-length route sequences.

Coordinates are yards in the receiver's route frame: origin at his snap position, +x downfield,
+y toward his own sideline using the corrected side from the reviewed break features (v2).
Time steps are 0.1 s. Speed and acceleration are recomputed from smoothed positions for every
frame, because the official `a` exists only before the throw (unsigned magnitude) and the
post-throw values are noisy signed differences.
"""
from __future__ import annotations

import numpy as np

MAX_LEN = 77                      # frames; covers 99.2% of the 14,108 targeted full paths
FPS = 10.0
KERNEL = np.array([1, 4, 6, 4, 1], dtype=float) / 16   # same smoother as route_geometry.py
CHANNELS = ("x", "y", "vx", "vy", "s", "a")
ROUTE_POSITIONS = ("WR", "TE", "RB", "FB")
SPLITS = ("train", "val", "test")


def smooth_positions(xy: np.ndarray) -> np.ndarray:
    """[1,4,6,4,1]/16 smoothing with edge padding, identical to route_geometry.smooth_positions."""
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([np.convolve(np.pad(xy[:, j], 2, mode="edge"), KERNEL, mode="valid")
                            for j in range(xy.shape[1])])


def route_xy(x_norm: np.ndarray, y_field: np.ndarray, side_corrected: int) -> np.ndarray:
    """Route-frame positions: downfield from the snap spot, lateral toward the corrected sideline."""
    y_field = np.asarray(y_field, dtype=float)
    return np.column_stack([np.asarray(x_norm, dtype=float), (y_field - y_field[0]) * side_corrected])


def kinematics(xy: np.ndarray) -> np.ndarray:
    """Columns vx, vy (yd/s), speed s (yd/s) and signed tangential acceleration a (yd/s^2).

    Derivatives use central differences of the smoothed positions (one-sided at the ends).
    Positive a means speeding up, negative means braking.
    """
    smoothed = smooth_positions(xy)
    velocity = np.gradient(smoothed, 1 / FPS, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    accel = np.gradient(speed, 1 / FPS)
    return np.column_stack([velocity, speed, accel])


def route_features(x_norm, y_field, side_corrected) -> np.ndarray:
    """(L, 6) array in CHANNELS order, computed on the full route before any truncation."""
    xy = route_xy(x_norm, y_field, side_corrected)
    return np.column_stack([xy, kinematics(xy)])


def pad_route(features: np.ndarray, max_len: int = MAX_LEN) -> tuple[np.ndarray, int, bool]:
    """Truncate to max_len frames and zero-pad. Returns (padded, kept_length, was_truncated)."""
    length = min(len(features), max_len)
    padded = np.zeros((max_len, features.shape[1]), dtype=np.float32)
    padded[:length] = features[:length]
    return padded, length, len(features) > max_len


def assign_splits(game_ids, seed: int, fractions=(0.8, 0.1, 0.1)) -> np.ndarray:
    """Assign whole games to train/val/test so no game contributes to two splits."""
    game_ids = np.asarray(game_ids)
    games = np.unique(game_ids)
    order = np.random.default_rng(seed).permutation(games)
    n_train = int(round(fractions[0] * len(games)))
    n_val = int(round(fractions[1] * len(games)))
    split_of = {}
    for i, game in enumerate(order):
        split_of[game] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return np.array([split_of[g] for g in game_ids])


def channel_scales(X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """One scale per channel from valid frames. x and y share one scale (and vx, vy another) so a
    squared error in scaled units stays proportional to the Euclidean error in yards."""
    valid = np.arange(X.shape[1])[None, :] < np.asarray(lengths)[:, None]
    frames = X[valid]
    pos = float(np.sqrt(np.mean(frames[:, 0:2] ** 2)))
    vel = float(np.sqrt(np.mean(frames[:, 2:4] ** 2)))
    s = float(np.sqrt(np.mean(frames[:, 4] ** 2)))
    a = float(np.sqrt(np.mean(frames[:, 5] ** 2)))
    return np.array([pos, pos, vel, vel, s, a], dtype=np.float32)
