"""Window knockout and the straight-line baseline (NumPy).

Training knocks out one contiguous window per route. Its width is uniform on [win_min, win_max]
frames, capped at max_frac of the route. Its first frame is uniform over the route's real
frames, so a window never falls into padding. Frame 0 is the snap position (always the origin)
and stays visible.
"""
from __future__ import annotations

import numpy as np


def sample_windows(lengths, rng: np.random.Generator, win_min: int = 8, win_max: int = 15,
                   max_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (starts, widths), one window per route; the window may reach the last frame."""
    lengths = np.asarray(lengths, dtype=np.int64)
    cap = np.maximum(1, np.floor(max_frac * lengths).astype(np.int64))
    hi = np.minimum(win_max, cap)
    lo = np.minimum(win_min, hi)
    widths = rng.integers(lo, hi + 1)
    starts = rng.integers(1, lengths - widths + 1)
    return starts, widths


def interior_windows(lengths, width: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-width windows with a visible frame on each side, as the straight-line fill needs.
    Routes shorter than width + 2 frames get start -1 and are skipped."""
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = np.full(len(lengths), -1, dtype=np.int64)
    ok = lengths >= width + 2
    starts[ok] = rng.integers(1, lengths[ok] - width)
    return starts


def centered_windows(lengths, centers, width: int) -> np.ndarray:
    """Fixed-width interior windows centred on given frames (for example the break)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.int64)
    starts = np.clip(centers - width // 2, 1, np.maximum(1, lengths - width - 1))
    starts[lengths < width + 2] = -1
    return starts


def window_mask(starts, widths, max_len: int) -> np.ndarray:
    """Boolean (N, max_len) mask, True inside each window; start -1 gives an empty row."""
    starts = np.asarray(starts, dtype=np.int64)
    widths = np.broadcast_to(np.asarray(widths, dtype=np.int64), starts.shape)
    t = np.arange(max_len)[None, :]
    return (starts[:, None] >= 0) & (t >= starts[:, None]) & (t < (starts + widths)[:, None])


def window_frames(values: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Gather (N, width, C) frames inside each window; skipped rows are NaN."""
    starts = np.asarray(starts, dtype=np.int64)
    out = np.full((len(values), width, values.shape[-1]), np.nan)
    ok = np.flatnonzero(starts >= 0)
    frames = starts[ok][:, None] + np.arange(width)[None, :]
    out[ok] = values[ok[:, None], frames]
    return out


def straight_line_fill(xy: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Fill each window by joining the frames just before and after it with a straight line at
    constant speed. Returns (N, width, 2); skipped rows are NaN."""
    starts = np.asarray(starts, dtype=np.int64)
    out = np.full((len(xy), width, 2), np.nan)
    ok = np.flatnonzero(starts >= 0)
    left = xy[ok, starts[ok] - 1]
    right = xy[ok, starts[ok] + width]
    frac = (np.arange(1, width + 1) / (width + 1))[None, :, None]
    out[ok] = left[:, None, :] + (right - left)[:, None, :] * frac
    return out


def mean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean Euclidean distance per route over the window frames (yards); NaN rows stay NaN."""
    return np.linalg.norm(a - b, axis=-1).mean(axis=1)
