"""Pure route geometry. Positions are yards, time seconds, angles degrees.

All headings use the corrected receiver frame (+x downfield, +y own side).
Sharpness is the elapsed time from the last <=0% heading before the first
80% heading after the stem window. Headings use smoothed position differences;
the search ends at the path end and never extrapolates an unfinished turn.
"""
import numpy as np

KERNEL = np.array([1, 4, 6, 4, 1], dtype=float) / 16
BREAK_FIELDS = ['break_t_s', 'break_depth_yd', 'break_lateral_yd',
                'speed_before', 'speed_min', 'speed_after', 'speed_retention',
                'sharpness_s', 'break_after_throw']


def smooth_positions(xy):
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([np.convolve(np.pad(xy[:, j], 2, mode='edge'),
                                       KERNEL, mode='valid') for j in range(2)])


def wrap_angle(degrees):
    return (degrees + 180) % 360 - 180


def correct_side(t, y, side, side_flag):
    """Release is measured in the original, unsmoothed route frame."""
    indices = np.flatnonzero(np.asarray(t) <= 1.0 + 1e-8)
    release = (y[indices[-1]] - y[0]) * side if len(indices) else 0.
    corrected = int(side * np.sign(release)) if side_flag and abs(release) >= 1 else int(side)
    return dict(side_corrected=corrected, side_was_flipped=corrected != side,
                side_uncertain=bool(side_flag and abs(release) < 1))


def direction(xy):
    delta = xy[-1] - xy[0]
    return np.degrees(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-12 else np.nan


def total_turn(t, xy):
    stem = np.flatnonzero((t >= .3 - 1e-8) & (t <= 1 + 1e-8))
    final = np.flatnonzero(t >= t[-1] - .5 - 1e-8)
    if len(stem) < 2 or len(final) < 2 or stem[-1] >= final[0]:
        return np.nan
    return float(wrap_angle(direction(xy[final]) - direction(xy[stem])))


def stem_heading(t, xy):
    stem = np.flatnonzero((t >= .3 - 1e-8) & (t <= 1 + 1e-8))
    return float(direction(xy[stem])) if len(stem) >= 2 else np.nan


def classify_turn(angle):
    return 'cut' if abs(angle) >= 35 else ('bend' if abs(angle) >= 20 else 'straight')


def tls_line(xy):
    centered = xy - xy.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    vector = vh[0].copy()
    travel = xy[-1] - xy[0]
    if np.dot(vector, travel) < 0:
        vector *= -1
    # For a perfectly stationary segment a travel heading is undefined.
    angle = np.degrees(np.arctan2(vector[1], vector[0])) if np.linalg.norm(travel) > 1e-12 else np.nan
    return float(singular[1] ** 2), float(angle)


def two_segment_fit(xy):
    best = None
    # Split point belongs to both segments: four observations on each side.
    for k in range(3, len(xy) - 3):
        e0, a0 = tls_line(xy[:k + 1])
        e1, a1 = tls_line(xy[k:])
        error = e0 + e1
        if best is None or error < best['fit_residual_ss'] - 1e-12:
            best = dict(split_index=k, fit_residual_ss=error,
                        fit_stem_heading_deg=a0, fit_second_heading_deg=a1,
                        fit_angle_deg=float(wrap_angle(a1 - a0)))
    return best


def heading_sharpness(t, xy, stem, second):
    change = wrap_angle(second - stem)
    if not np.isfinite(change) or abs(change) < 1e-8:
        return np.nan
    delta = np.diff(xy, axis=0)
    heading = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    progress = wrap_angle(heading - stem) / change
    valid = (t[1:] >= 1.0 - 1e-8) & (np.linalg.norm(delta, axis=1) > 1e-10)
    reached = np.flatnonzero(valid & (progress >= .8))
    if not len(reached):
        return np.nan
    end = reached[0]
    starts = np.flatnonzero(valid & (np.arange(len(delta)) <= end) & (progress <= 0))
    start = starts[-1] if len(starts) else np.flatnonzero(valid)[0]
    return float(t[end + 1] - t[start + 1])


def measure_route(t, x_norm, y, side, side_flag, throw_time_s):
    t, x_norm, y = (np.asarray(v, dtype=float) for v in (t, x_norm, y))
    if not len(t) or np.any(np.diff(t) <= 0) or not np.isfinite(np.c_[t, x_norm, y]).all():
        raise ValueError('Route needs finite positions and strictly increasing times')
    result = correct_side(t, y, side, side_flag)
    lateral = (y - y[0]) * result['side_corrected']
    raw = np.c_[x_norm, lateral]
    smoothed = smooth_positions(raw)
    speed = np.r_[np.nan, np.linalg.norm(np.diff(smoothed, axis=0), axis=1) * 10]
    result.update(total_turn_deg=total_turn(t, smoothed), stem_heading_deg=stem_heading(t, smoothed),
                  catch_y_corrected=float(lateral[-1]), has_break=False, break_class='straight',
                  fit_angle_deg=np.nan, fit_residual_ss=np.nan,
                  max_depth_yd=float(x_norm.max()), retreat_yd=float(x_norm.max() - x_norm[-1]),
                  time_at_max_depth_s=float(t[np.argmax(x_norm)]))
    result.update({c: np.nan for c in BREAK_FIELDS})
    fit = two_segment_fit(smoothed)
    if fit is None:
        return result
    result.update({c: fit[c] for c in ['fit_angle_deg', 'fit_residual_ss']})
    result['break_class'] = classify_turn(fit['fit_angle_deg'])
    result['has_break'] = result['break_class'] == 'cut'
    if not result['has_break']:
        return result
    k = fit['split_index']
    before = speed[(t >= t[k] - .5 - 1e-8) & (t < t[k] - 1e-8)]
    after = speed[(t > t[k] + 1e-8) & (t <= t[k] + .5 + 1e-8)]
    near = speed[np.abs(t - t[k]) <= .3 + 1e-8]
    sb = float(np.nanmean(before))
    sa = float(np.nanmean(after))
    result.update(break_t_s=float(t[k]), break_depth_yd=float(x_norm[k]),
                  break_lateral_yd=float(lateral[k]), speed_before=sb,
                  speed_min=float(np.nanmin(near)), speed_after=sa,
                  speed_retention=sa / sb if sb >= 1 else np.nan,
                  sharpness_s=heading_sharpness(t, smoothed, fit['fit_stem_heading_deg'], fit['fit_second_heading_deg']),
                  break_after_throw=bool(t[k] > throw_time_s))
    return result
