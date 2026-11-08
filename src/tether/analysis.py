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

            # A univariate trend is not enough to prove sensor drift: speed,
            # throttle and temperatures can legitimately change during a lap.
            # Keep DRIFT as a human-review candidate pending reference evidence.
            if fault_type == "DRIFT" or score < 0.55:
                record["status"] = "REVIEW"
            else:
                record["status"] = "HEALTHY" if fault_type == "NORMAL" else "ALERT"

    if not records:
        warnings.append("No windows could be analyzed. Each selected sensor needs at least 32 rows.")
    return pd.DataFrame(records), warnings
