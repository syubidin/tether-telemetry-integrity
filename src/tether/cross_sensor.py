"""Cross-sensor residual checks that separate signal trends from inconsistency."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tether.baseline import TrustedBaseline, relationship_baseline_for
from tether.native import speed_accel_residuals
from tether.profiles import TelemetryProfile

WINDOW_SIZE = 64


def _window_slices(length: int, size: int = WINDOW_SIZE) -> list[tuple[int, int]]:
    if length < 8:
        return []
    result = [(start, min(start + size, length)) for start in range(0, length, size)]
    if len(result) > 1 and result[-1][1] - result[-1][0] < 8:
        result[-2] = (result[-2][0], result[-1][1])
        result.pop()
    return result


def _robust_z(residuals: np.ndarray, center: float, scale: float) -> tuple[float, float]:
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return 0.0, 0.0
    shift = abs(float(np.median(finite)) - center) / max(scale, 1e-9)
    outlier_fraction = float(np.mean(np.abs(finite - center) > 4.5 * scale))
    return shift, outlier_fraction


def _time_values(frame: pd.DataFrame, profile: TelemetryProfile) -> np.ndarray:
    if profile.timestamp_column in frame:
        return pd.to_numeric(frame[profile.timestamp_column], errors="coerce").to_numpy(dtype=float)
    return np.arange(len(frame), dtype=float) / profile.sample_rate_hz


def analyze_cross_sensor(
    frame: pd.DataFrame,
    profile: TelemetryProfile,
    baseline: TrustedBaseline,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Return relationship residual findings; ambiguous mismatches are REVIEW."""
    if baseline.profile_id != profile.profile_id or baseline.profile_version != profile.version:
        raise ValueError("Trusted baseline profile ID/version does not match the active profile.")
    records: list[dict[str, Any]] = []
    slices = _window_slices(len(frame), window_size)
    times = _time_values(frame, profile)

    for relationship in profile.relationships:
        if relationship.x not in frame or relationship.y not in frame:
            continue
        x_all = pd.to_numeric(frame[relationship.x], errors="coerce").to_numpy(dtype=float)
        y_all = pd.to_numeric(frame[relationship.y], errors="coerce").to_numpy(dtype=float)
        condition = (
            frame[relationship.condition_by].to_numpy()
            if relationship.condition_by and relationship.condition_by in frame.columns
            else None
        )

        for window_index, (start, stop) in enumerate(slices, start=1):
            x, y = x_all[start:stop], y_all[start:stop]
            score_values: list[float] = []
            outlier_values: list[float] = []
            used_samples = 0

            if relationship.kind == "kinematic":
                fitted = relationship_baseline_for(baseline, relationship)
                if fitted is None:
                    continue
                residuals = kinematic_speed_accel_residuals(x, times[start:stop], y, int(fitted.intercept))
                z, outlier_fraction = _robust_z(
                    residuals, fitted.residual_center, fitted.residual_scale
                )
                score_values.append(z)
                outlier_values.append(outlier_fraction)
                used_samples = int(np.isfinite(residuals).sum())
            else:
                group_values = condition[start:stop] if condition is not None else None
                if group_values is None:
                    groups = [("all", np.ones(stop - start, dtype=bool))]
                else:
                    groups = [(_group_key(key), group_values == key) for key in np.unique(group_values)]

                for group, mask in groups:
                    fitted = relationship_baseline_for(
                        baseline, relationship, None if group == "all" else group
                    )
                    if fitted is None:
                        continue
                    valid = mask & np.isfinite(x) & np.isfinite(y)
                    if valid.sum() < max(8, relationship.min_samples // 3):
                        continue
                    residuals = y[valid] - (fitted.intercept + fitted.slope * x[valid])
                    z, outlier_fraction = _robust_z(
                        residuals, fitted.residual_center, fitted.residual_scale
                    )
                    score_values.append(z)
                    outlier_values.append(outlier_fraction)
                    used_samples += int(valid.sum())

            if not score_values:
                continue
            residual_z = float(np.average(score_values))
            outlier_fraction = float(np.average(outlier_values))
            flagged = residual_z >= relationship.residual_z_threshold or outlier_fraction >= 0.25
            records.append(
                {
                    "window": window_index,
                    "start_row": start,
                    "end_row": stop - 1,
                    "relationship": relationship.name,
                    "channels": f"{relationship.x} ↔ {relationship.y}",
                    "residual_z": round(residual_z, 3),
                    "outlier_fraction_pct": round(outlier_fraction * 100.0, 1),
                    "samples": used_samples,
                    "status": "REVIEW" if flagged else "HEALTHY",
                    "candidate_fault": "DRIFT" if flagged else "NONE",
                    "interpretation": (
                        "Relationship differs from the trusted baseline; inspect the candidate channel and operating context."
                        if flagged
                        else "Cross-sensor relationship is consistent with the trusted baseline."
                    ),
                }
            )

    return pd.DataFrame(records)


def _group_key(value: Any) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except (TypeError, ValueError):
        return str(value)


def kinematic_speed_accel_residuals(
    speed_kph: np.ndarray,
    timestamps_s: np.ndarray,
    measured_accel_mps2: np.ndarray,
    lag_samples: int = 0,
) -> np.ndarray:
    """Compute time-aligned derived minus measured acceleration."""
    raw = speed_accel_residuals(speed_kph, timestamps_s, measured_accel_mps2)
    lag = max(-8, min(8, int(lag_samples)))
    if lag == 0:
        return raw
    derived = raw + measured_accel_mps2
    if lag > 0:
        return derived[:-lag] - measured_accel_mps2[lag:]
    return derived[-lag:] - measured_accel_mps2[:lag]
