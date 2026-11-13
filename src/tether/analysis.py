"""Windowed sensor analysis and human-readable findings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tether.features import FEATURE_NAMES, extract_features
from tether.validation import validate_sensor_selection

WINDOW_SIZE = 64
MIN_WINDOW_SIZE = 32

KNOWN_RANGES: dict[str, tuple[float, float, str]] = {
    "speed_kph": (0.0, 380.0, "km/h"),
    "throttle_pct": (0.0, 100.0, "%"),
    "brake_pct": (0.0, 100.0, "%"),
    "rpm": (0.0, 20_000.0, "rpm"),
    "gear": (0.0, 10.0, "gear"),
    "coolant_temp_c": (-20.0, 160.0, "°C"),
    "oil_pressure_bar": (0.0, 20.0, "bar"),
    "battery_voltage_v": (0.0, 40.0, "V"),
}

FAULT_ACTIONS = {
    "NORMAL": "No strong integrity issue detected in this window.",
    "DROPOUT": "Check packet continuity, connector health, and sensor power.",
    "STUCK": "Check whether the sensor, channel mapping, or acquisition task is frozen.",
    "DRIFT": "Compare against a trusted reference and inspect calibration or thermal effects.",
    "SPIKE": "Inspect isolated samples, electrical interference, and timestamp alignment.",
    "NOISE": "Review shielding, grounding, filtering, and acquisition configuration.",
}


def _window_slices(length: int, window_size: int = WINDOW_SIZE) -> list[tuple[int, int]]:
    """Create fixed windows and merge a short tail into the preceding window."""
    if length < MIN_WINDOW_SIZE:
        return []
    slices = [(start, min(start + window_size, length)) for start in range(0, length, window_size)]
    if len(slices) > 1 and slices[-1][1] - slices[-1][0] < MIN_WINDOW_SIZE:
        slices[-2] = (slices[-2][0], slices[-1][1])
        slices.pop()
    return slices


def _rule_classification(features: dict[str, float]) -> str | None:
    """Return a fault class only when a high-signal integrity rule is met."""
    if features["missing_ratio"] >= 0.20:
        return "DROPOUT"
    if (
        features["longest_flatline_ratio"] >= 0.72
        and features["unique_ratio"] <= 0.30
        and features["missing_ratio"] < 0.20
    ):
        return "STUCK"
    return None


def _severity(fault_type: str) -> str:
    if fault_type == "NORMAL":
        return "INFO"
    if fault_type == "DRIFT":
        return "REVIEW"
    if fault_type in {"DROPOUT", "STUCK"}:
        return "HIGH"
    return "MEDIUM"


def _range_check(sensor: str, values: np.ndarray) -> tuple[int, str]:
    limits = KNOWN_RANGES.get(sensor.casefold())
    finite = values[np.isfinite(values)]
    if limits is None or finite.size == 0:
        return 0, ""
    lower, upper, unit = limits
    count = int(np.count_nonzero((finite < lower) | (finite > upper)))
    return count, f"Expected {lower:g}–{upper:g} {unit} for this known channel."


def analyze_dataframe(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    model: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """Analyze selected sensors in bounded windows using batched ML inference.

    Returns one row per sensor-window plus non-fatal data-quality warnings.
    No uploaded data is persisted by this function.
    """
    validate_sensor_selection(frame, sensor_columns)
    records: list[dict[str, Any]] = []
    pending_features: list[list[float]] = []
    pending_indices: list[int] = []
    warnings: list[str] = []

    if frame.empty:
        raise ValueError("The dataset has no rows to analyze.")

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows:,} duplicate rows detected; row order was preserved.")

    for sensor in sensor_columns:
        numeric = pd.to_numeric(frame[sensor], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        numeric[~np.isfinite(numeric)] = np.nan
        if numeric.size < MIN_WINDOW_SIZE:
            warnings.append(f"{sensor}: fewer than {MIN_WINDOW_SIZE} samples; skipped as insufficient data.")
            continue

        range_count, range_note = _range_check(sensor, numeric)
        if range_count:
            warnings.append(f"{sensor}: {range_count:,} sample(s) outside a known reference range. {range_note}")

        for window_number, (start, stop) in enumerate(_window_slices(len(numeric)), start=1):
            features = extract_features(numeric[start:stop])
            rule_label = _rule_classification(features)
            record = {
                "sensor": sensor,
                "window": window_number,
                "start_row": start + 1,
                "end_row": stop,
                "fault_type": rule_label or "",
                "status": "ALERT" if rule_label else "",
                "severity": _severity(rule_label) if rule_label else "",
                "model_score_pct": None,
                "method": "RULE" if rule_label else "ML",
                "missing_pct": round(features["missing_ratio"] * 100.0, 1),
                "range_violations": range_count if window_number == 1 else 0,
                "recommended_check": FAULT_ACTIONS.get(rule_label, ""),
            }
            records.append(record)

            if rule_label is None:
                pending_features.append([features[name] for name in FEATURE_NAMES])
                pending_indices.append(len(records) - 1)

    # One vectorized predict call avoids thousands of per-window Python calls
    # on larger CSV uploads and keeps CPU overhead predictable.
    if pending_features:
        probabilities = model.predict_proba(np.asarray(pending_features, dtype=np.float32))
        classes = model.classes_
        for probability_row, record_index in zip(probabilities, pending_indices):
            winner_index = int(np.argmax(probability_row))
            fault_type = str(classes[winner_index])
            score = float(probability_row[winner_index])
            record = records[record_index]
            record["fault_type"] = fault_type
            record["model_score_pct"] = round(score * 100.0, 1)
            record["severity"] = _severity(fault_type)
            record["recommended_check"] = FAULT_ACTIONS[fault_type]

            # Class scores are not calibrated probabilities and synthetic
            # training cannot prove a vehicle fault. ML candidates remain
            # review-only; only deterministic dropout / stuck rules escalate.
            if fault_type == "NORMAL" and score >= 0.55:
                record["status"] = "HEALTHY"
            else:
                record["status"] = "REVIEW"

    if not records:
        warnings.append("No windows could be analyzed. Each selected sensor needs at least 32 rows.")
    return pd.DataFrame(records), warnings


def analyze_with_cross_sensor(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    model: Any,
    profile: Any,
    trusted_baseline: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Run univariate ML plus cross-sensor checks and de-escalate corroborated false alarms.

    A non-normal ML candidate can be de-escalated to HEALTHY only when a
    configured cross-sensor relationship is healthy in the same window.
    Missing or failed correlation evidence leaves it in REVIEW. ML scores
    never create a hard ALERT; deterministic dropout / stuck rules do.
    """
    from tether.cross_sensor import analyze_cross_sensor

    results, warnings = analyze_dataframe(frame, sensor_columns, model)
    relationships = analyze_cross_sensor(frame, profile, trusted_baseline)
    if results.empty or relationships.empty:
        return results, relationships, warnings

    healthy_by_start: dict[int, set[str]] = {}
    for item in relationships[relationships["status"] == "HEALTHY"].itertuples(index=False):
        channels = {part.strip() for part in str(item.channels).split("↔")}
        healthy_by_start.setdefault(int(item.start_row), set()).update(channels)

    for index, item in results.iterrows():
        if item["method"] != "ML" or item["fault_type"] == "NORMAL":
            continue
        start_zero_based = int(item["start_row"]) - 1
        if str(item["sensor"]) in healthy_by_start.get(start_zero_based, set()):
            results.at[index, "status"] = "HEALTHY"
            results.at[index, "correlation_guard"] = "DE-ESCALATED"
            results.at[index, "recommended_check"] = (
                "The configured cross-sensor relationship remains healthy; this ML-only candidate was de-escalated."
            )
        else:
            results.at[index, "status"] = "REVIEW"
            results.at[index, "correlation_guard"] = "NOT-CORROBORATED"

    if "correlation_guard" not in results.columns:
        results["correlation_guard"] = "NOT-APPLICABLE"
    else:
        results["correlation_guard"] = results["correlation_guard"].fillna("NOT-APPLICABLE")
    return results, relationships, warnings
