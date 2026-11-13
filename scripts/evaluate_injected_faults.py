"""Evaluate univariate and cross-sensor findings with controlled fault injection.

Metrics are window-level and use synthetic ground truth created by this script.
They are not measurements of naturally occurring faults in the source recording.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tether.analysis import analyze_dataframe, analyze_with_cross_sensor
from tether.baseline import fit_trusted_baseline
from tether.cross_sensor import analyze_cross_sensor
from tether.model import train_model
from tether.profiles import load_profile
from tether.synthetic import make_demo_telemetry
from tether.validation import numeric_sensor_columns, read_telemetry_csv

WINDOW = 64
INJECTION_START = WINDOW * 2


def binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tp = int(np.count_nonzero(actual & predicted))
    fp = int(np.count_nonzero(~actual & predicted))
    fn = int(np.count_nonzero(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _fault_channels(profile_id: str, available: list[str]) -> dict[str, str]:
    preferred = (
        {
            "DROPOUT": "rpm",
            "STUCK": "oil_pressure_bar",
            "DRIFT": "coolant_temp_c",
            "SPIKE": "throttle_pct",
            "NOISE": "speed_kph",
        }
        if profile_id == "generic-track-car-20hz"
        else {
            "DROPOUT": "wheel_speed_fl_kph",
            "STUCK": "yaw_rate_dps",
            "DRIFT": "longitudinal_accel_mps2",
            "SPIKE": "wheel_speed_fl_kph",
            "NOISE": "lateral_accel_mps2",
        }
    )
    fallback = available[0] if available else None
    return {fault: channel if channel in available else fallback for fault, channel in preferred.items()}


def _inject_fault(frame: pd.DataFrame, channel: str, fault: str, random_state: int) -> pd.DataFrame:
    scenario = frame.copy()
    values = pd.to_numeric(scenario[channel], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < WINDOW:
        raise ValueError(f"Not enough finite values to inject {fault} into {channel}.")
    scale = max(float(np.std(finite)), 0.2)
    start, stop = INJECTION_START, INJECTION_START + WINDOW
    if fault == "DROPOUT":
        values[start : start + 24] = np.nan
    elif fault == "STUCK":
        values[start:stop] = values[start - 1]
    elif fault == "DRIFT":
        values[start:stop] += np.linspace(0.0, 3.0 * scale, WINDOW)
    elif fault == "SPIKE":
        rng = np.random.default_rng(random_state)
        spike_indices = (start + 9, start + 31, start + 52)
        values[list(spike_indices)] += rng.choice([-1.0, 1.0], size=3) * 10.0 * scale
    elif fault == "NOISE":
        rng = np.random.default_rng(random_state)
        values[start:stop] += rng.normal(0.0, 0.9 * scale, WINDOW)
    else:
        raise ValueError(f"Unsupported univariate fault: {fault}")
    scenario[channel] = values
    return scenario


def _evaluate_univariate_faults(
    evaluation_frame: pd.DataFrame,
    profile_id: str,
    sensor_channels: list[str],
    model,
    sample_rate_hz: float,
    random_state: int,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    channels = _fault_channels(profile_id, sensor_channels)
    if not sensor_channels:
        return results

    for fault, channel in channels.items():
        if channel is None:
            continue
        scenario = _inject_fault(evaluation_frame, channel, fault, random_state)
        findings, _ = analyze_dataframe(scenario, [channel], model)
        actual = findings["start_row"].to_numpy() == INJECTION_START + 1
        predicted = findings["fault_type"].to_numpy() == fault
        metrics = binary_metrics(actual, predicted)
        target = findings[actual]
        detected = not target.empty and bool(predicted[actual][0])
        metrics.update(
            {
                "injected_channel": channel,
                "injection_start_row": INJECTION_START,
                "injection_end_row": INJECTION_START + WINDOW - 1,
                "detection_delay_s": round(WINDOW / sample_rate_hz, 4) if detected else None,
            }
        )
        results[fault] = metrics
    return results


def _evaluate_cross_sensor_faults(
    evaluation_frame: pd.DataFrame,
    profile,
    trusted,
    sample_rate_hz: float,
) -> dict[str, dict[str, object]]:
    per_fault: dict[str, dict[str, object]] = {}
    for relationship in profile.relationships:
        if relationship.name not in trusted.relationships or relationship.y not in evaluation_frame:
            continue
        scenario = evaluation_frame.copy()
        channel = relationship.y
        current = pd.to_numeric(scenario[channel], errors="coerce").to_numpy(dtype=float)
        finite = current[np.isfinite(current)]
        if finite.size < WINDOW:
            continue
        if relationship.kind == "kinematic":
            offset = 1.8
            fault_name = "ACCEL_BIAS"
        else:
            offset = max(float(np.std(finite)) * 0.25, 500.0 if channel.casefold() == "rpm" else 10.0)
            fault_name = f"{channel.upper()}_DRIFT"
        start, stop = INJECTION_START, INJECTION_START + WINDOW
        scenario.loc[start : stop - 1, channel] = current[start:stop] + offset
        findings = analyze_cross_sensor(scenario, profile, trusted)
        relation_findings = findings[findings["relationship"] == relationship.name].copy()
        actual = relation_findings["window"].to_numpy() == (start // WINDOW + 1)
        predicted = relation_findings["status"].to_numpy() == "REVIEW"
        metrics = binary_metrics(actual, predicted)
        target = relation_findings[actual]
        detected = not target.empty and bool(predicted[actual][0])
        metrics.update(
            {
                "injected_channel": channel,
                "injected_offset": round(float(offset), 5),
                "injection_start_row": start,
                "injection_end_row": stop - 1,
                "relationship": relationship.name,
                "detection_delay_s": round(WINDOW / sample_rate_hz, 4) if detected else None,
            }
        )
        per_fault[fault_name] = metrics
    return per_fault


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Clean baseline CSV (e.g. mapped public vehicle data); omitted means a 30-minute synthetic session",
    )
    parser.add_argument("--profile", default=str(ROOT / "config" / "channel_profiles.json"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "injected_fault_metrics.json"))
    args = parser.parse_args()

    if args.baseline:
        frame = read_telemetry_csv(Path(args.baseline).read_bytes())
        baseline_name = Path(args.baseline).name
    else:
        frame = make_demo_telemetry(rows=36_000, random_state=71, inject_faults=False)
        baseline_name = "generated_synthetic_30min"

    profile = load_profile(args.profile)
    split = max(WINDOW, int(len(frame) * 0.6))
    if len(frame) - split < 128:
        parser.error("Baseline needs at least 320 rows to create separate fit and evaluation portions.")

    baseline_frame = frame.iloc[:split].copy()
    evaluation_frame = frame.iloc[split:].copy().reset_index(drop=True)
    trusted = fit_trusted_baseline(baseline_frame, profile, provenance=f"{baseline_name}: first 60%")
    if not trusted.relationships:
        parser.error("No configured cross-sensor relationship could be fitted. Check profile fields and column mapping.")

    sample_rate = profile.sample_rate_hz
    duration_hours = max(len(evaluation_frame) / sample_rate / 3600.0, 1e-9)
    cross_clean = analyze_cross_sensor(evaluation_frame, profile, trusted)
    model = train_model()
    sensor_channels = numeric_sensor_columns(evaluation_frame)
    if sensor_channels:
        guarded_clean, _, _ = analyze_with_cross_sensor(
            evaluation_frame, sensor_channels, model, profile, trusted
        )
        clean_hard_alerts = int((guarded_clean["status"] == "ALERT").sum())
        clean_review_candidates = int((guarded_clean["status"] == "REVIEW").sum())
    else:
        clean_hard_alerts = 0
        clean_review_candidates = 0

    report: dict[str, object] = {
        "evaluation_kind": "controlled_fault_injection_on_clean_baseline",
        "baseline_file": baseline_name,
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "baseline_rows": len(baseline_frame),
        "evaluation_rows": len(evaluation_frame),
        "evaluation_duration_s": round(len(evaluation_frame) / sample_rate, 3),
        "clean_hard_alerts_per_hour": round(clean_hard_alerts / duration_hours, 3),
        "clean_review_candidates_per_hour": round(clean_review_candidates / duration_hours, 3),
        "clean_cross_sensor_reviews_per_hour": round(
            float((cross_clean["status"] == "REVIEW").sum()) / duration_hours, 3
        ),
        "univariate_per_fault": {},
        "cross_sensor_per_fault": {},
        "limitations": [
            "Fault labels are generated by controlled injections, not naturally occurring failure labels.",
            "Precision and recall are window-level; the experiment injects one positive 64-sample window per scenario.",
            "The clean evaluation segment is a held-out portion of a deterministic synthetic or user-supplied baseline recording.",
            "The public vehicle CAN source itself does not provide sensor-fault ground truth.",
            "Detection delay is measured at completed 64-sample non-overlapping windows.",
        ],
    }

    report["univariate_per_fault"] = _evaluate_univariate_faults(
        evaluation_frame, profile.profile_id, sensor_channels, model, sample_rate, random_state=71
    )
    report["cross_sensor_per_fault"] = _evaluate_cross_sensor_faults(
        evaluation_frame, profile, trusted, sample_rate
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
