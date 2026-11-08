"""Small, robust feature extraction for one-dimensional telemetry windows."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "missing_ratio",
    "unique_ratio",
    "flatline_ratio",
    "longest_flatline_ratio",
    "max_robust_z",
    "spike_fraction",
    "trend_score",
    "diff_mad_ratio",
    "diff_std_ratio",
    "noise_to_level_ratio",
    "level_shift_score",
)


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive True run without Python-side loops."""
    if mask.size == 0 or not np.any(mask):
        return 0
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.max(ends - starts))


def extract_features(values: np.ndarray | list[float]) -> dict[str, float]:
    """Extract scale-aware signal quality features from a possibly sparse window.

    Missing and infinite values are treated as absent samples. The function is
    defensive by design: empty, all-missing and constant windows return finite
    feature values rather than raising numerical errors.
    """
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    size = int(raw.size)
    if size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    finite_mask = np.isfinite(raw)
    finite = raw[finite_mask]
    missing_ratio = 1.0 - (finite.size / size)

    if finite.size == 0:
        result = {name: 0.0 for name in FEATURE_NAMES}
        result["missing_ratio"] = 1.0
        return result

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    std = float(np.std(finite))
    scale = max(1.4826 * mad, 0.25 * std, abs(median) * 1e-9, 1e-9)

    unique_ratio = float(np.unique(np.round(finite, decimals=8)).size / finite.size)

    # A spike is a discontinuity in the derivative, not merely an extreme
    # value. This avoids flagging the peaks of legitimate speed/throttle waves.
    # A flatline is measured over adjacent valid pairs only. Missing samples do
    # not create artificial jumps or count as flat readings.
    pair_mask = finite_mask[:-1] & finite_mask[1:]
    raw_diffs = np.diff(raw)
    valid_diffs = raw_diffs[pair_mask]
    tolerance = max(scale * 1e-4, 1e-9)
    flat_pairs = np.abs(valid_diffs) <= tolerance
    flatline_ratio = float(np.mean(flat_pairs)) if flat_pairs.size else 0.0
    longest_flatline_ratio = _longest_true_run(flat_pairs) / max(size - 1, 1)

    if valid_diffs.size:
        diff_center = float(np.median(valid_diffs))
        diff_mad_raw = float(np.median(np.abs(valid_diffs - diff_center)))
        diff_scale = max(1.4826 * diff_mad_raw, 0.25 * float(np.std(valid_diffs)), scale * 1e-4, 1e-9)
        diff_robust_z = np.abs(valid_diffs - diff_center) / diff_scale
        spike_fraction = float(np.mean(diff_robust_z > 6.0))
        max_robust_z = float(np.max(diff_robust_z))
    else:
        spike_fraction = 0.0
        max_robust_z = 0.0

    # The trend is normalized by robust signal scale and window length.
    positions = np.flatnonzero(finite_mask).astype(np.float64)
    if finite.size >= 2 and np.ptp(positions) > 0:
        centered_x = positions - positions.mean()
        slope = float(np.dot(centered_x, finite - finite.mean()) / max(np.dot(centered_x, centered_x), 1e-12))
        trend_score = abs(slope) * max(size - 1, 1) / scale
    else:
        trend_score = 0.0

    if valid_diffs.size:
        diff_median = float(np.median(valid_diffs))
        diff_mad = float(np.median(np.abs(valid_diffs - diff_median)))
        diff_mad = 1.4826 * diff_mad
        diff_mad_ratio = diff_mad / scale
        diff_std_ratio = float(np.std(valid_diffs) / scale)
        noise_to_level_ratio = diff_mad / max(abs(median), scale, 1e-9)
    else:
        diff_mad_ratio = 0.0
        diff_std_ratio = 0.0
        noise_to_level_ratio = 0.0

    quarter = max(1, finite.size // 4)
    level_shift = abs(float(np.median(finite[-quarter:]) - np.median(finite[:quarter]))) / scale

    return {
        "missing_ratio": float(np.clip(missing_ratio, 0.0, 1.0)),
        "unique_ratio": float(np.clip(unique_ratio, 0.0, 1.0)),
        "flatline_ratio": float(np.clip(flatline_ratio, 0.0, 1.0)),
        "longest_flatline_ratio": float(np.clip(longest_flatline_ratio, 0.0, 1.0)),
        "max_robust_z": float(np.clip(max_robust_z, 0.0, 100.0)),
        "spike_fraction": float(np.clip(spike_fraction, 0.0, 1.0)),
        "trend_score": float(np.clip(trend_score, 0.0, 100.0)),
        "diff_mad_ratio": float(np.clip(diff_mad_ratio, 0.0, 100.0)),
        "diff_std_ratio": float(np.clip(diff_std_ratio, 0.0, 100.0)),
        "noise_to_level_ratio": float(np.clip(noise_to_level_ratio, 0.0, 100.0)),
        "level_shift_score": float(np.clip(level_shift, 0.0, 100.0)),
    }
