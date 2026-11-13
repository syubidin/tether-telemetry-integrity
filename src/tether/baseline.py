"""Fit robust cross-channel expectations from an explicitly trusted baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tether.native import speed_accel_residuals
from tether.profiles import RelationshipProfile, TelemetryProfile


@dataclass(frozen=True)
class LinearBaseline:
    intercept: float
    slope: float
    residual_center: float
    residual_scale: float
    samples: int


@dataclass(frozen=True)
class TrustedBaseline:
    profile_id: str
    profile_version: str
    relationships: dict[str, dict[str, LinearBaseline]]
    provenance: str = "user-supplied trusted baseline"


def _robust_linear_fit(x: np.ndarray, y: np.ndarray) -> LinearBaseline | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 8 or np.ptp(x) < 1e-9:
        return None
    keep = np.ones(x.size, dtype=bool)
    for _ in range(3):
        if np.count_nonzero(keep) < 8 or np.ptp(x[keep]) < 1e-9:
            break
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residuals = y - (intercept + slope * x)
        center = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - center)))
        scale = max(1.4826 * mad, float(np.std(residuals[keep]) * 0.2), 1e-9)
        updated = np.abs(residuals - center) <= 4.5 * scale
        if np.array_equal(updated, keep) or np.count_nonzero(updated) < 8:
            break
        keep = updated
    if np.count_nonzero(keep) < 8:
        return None
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    residuals = y[keep] - (intercept + slope * x[keep])
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    scale = max(1.4826 * mad, float(np.std(residuals) * 0.2), 1e-6)
    return LinearBaseline(float(intercept), float(slope), center, scale, int(keep.sum()))


def _fit_kinematic(frame: pd.DataFrame, profile: TelemetryProfile, relationship: RelationshipProfile) -> LinearBaseline | None:
    speed = pd.to_numeric(frame[relationship.x], errors="coerce").to_numpy(dtype=float)
    accel = pd.to_numeric(frame[relationship.y], errors="coerce").to_numpy(dtype=float)
    if profile.timestamp_column in frame:
        timestamps = pd.to_numeric(frame[profile.timestamp_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
            return None
    else:
        timestamps = np.arange(len(frame), dtype=float) / profile.sample_rate_hz
    residuals = speed_accel_residuals(speed, timestamps, accel)
    derived = residuals + accel
    best: tuple[float, int, np.ndarray] | None = None
    for lag in range(-relationship.max_lag_samples, relationship.max_lag_samples + 1):
        if lag > 0:
            candidate = derived[:-lag] - accel[lag:]
        elif lag < 0:
            candidate = derived[-lag:] - accel[:lag]
        else:
            candidate = residuals
        finite = candidate[np.isfinite(candidate)]
        if finite.size < relationship.min_samples:
            continue
        center = float(np.median(finite))
        mad = float(np.median(np.abs(finite - center)))
        scale = max(1.4826 * mad, float(np.std(finite) * 0.2), 0.05)
        score = scale + abs(center) * 0.05
        if best is None or score < best[0]:
            best = (score, lag, finite)
    if best is None:
        return None
    _, selected_lag, finite = best
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = max(1.4826 * mad, float(np.std(finite) * 0.2), 0.05)
    return LinearBaseline(float(selected_lag), 0.0, center, scale, int(finite.size))


def fit_trusted_baseline(
    frame: pd.DataFrame,
    profile: TelemetryProfile,
    provenance: str = "user-supplied trusted baseline",
) -> TrustedBaseline:
    """Fit expected relationships only from a separately declared clean baseline."""
    fitted: dict[str, dict[str, LinearBaseline]] = {}
    for relationship in profile.relationships:
        if relationship.x not in frame.columns or relationship.y not in frame.columns:
            continue
        groups: dict[str, LinearBaseline] = {}
        if relationship.kind == "kinematic":
            model = _fit_kinematic(frame, profile, relationship)
            if model:
                groups["all"] = model
        else:
            x = pd.to_numeric(frame[relationship.x], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame[relationship.y], errors="coerce").to_numpy(dtype=float)
            if relationship.condition_by and relationship.condition_by in frame.columns:
                condition = pd.to_numeric(frame[relationship.condition_by], errors="coerce").to_numpy(dtype=float)
                for key in np.unique(condition[np.isfinite(condition)]):
                    mask = condition == key
                    model = _robust_linear_fit(x[mask], y[mask])
                    if model and model.samples >= relationship.min_samples:
                        groups[_group_key(key)] = model
            else:
                model = _robust_linear_fit(x, y)
                if model and model.samples >= relationship.min_samples:
                    groups["all"] = model
        if groups:
            fitted[relationship.name] = groups
    return TrustedBaseline(profile.profile_id, profile.version, fitted, provenance[:200])


def _group_key(value: Any) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except (TypeError, ValueError):
        return str(value)


def relationship_baseline_for(
    baseline: TrustedBaseline,
    relationship: RelationshipProfile,
    group: Any = None,
) -> LinearBaseline | None:
    groups = baseline.relationships.get(relationship.name, {})
    return groups.get(_group_key(group)) if relationship.condition_by is not None else groups.get("all")
