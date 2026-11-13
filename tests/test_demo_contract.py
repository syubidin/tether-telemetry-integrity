from pathlib import Path

import pytest

from tether.analysis import analyze_with_cross_sensor
from tether.baseline import fit_trusted_baseline
from tether.model import train_model
from tether.profiles import load_profile
from tether.synthetic import make_demo_telemetry
from tether.validation import numeric_sensor_columns


@pytest.fixture(scope="module")
def model():
    return train_model()


def analyze(frame, model):
    root = Path(__file__).resolve().parents[1]
    profile = load_profile(root / "config" / "channel_profiles.json")
    baseline_frame = make_demo_telemetry(inject_faults=False)
    baseline = fit_trusted_baseline(baseline_frame, profile, "bundled synthetic demo baseline")
    return analyze_with_cross_sensor(frame, numeric_sensor_columns(frame), model, profile, baseline)


def test_clean_synthetic_run_has_no_escalated_alerts(model):
    frame = make_demo_telemetry(inject_faults=False)
    results, _cross_sensor, warnings = analyze(frame, model)
    assert not warnings
    assert not (results["status"] == "ALERT").any()
    assert (results["status"] == "HEALTHY").any()


def test_fault_injected_run_exposes_expected_integrity_signatures(model):
    frame = make_demo_telemetry(inject_faults=True)
    results, cross_sensor, _ = analyze(frame, model)
    alerts = results[results["status"] == "ALERT"]
    reviews = results[results["status"] == "REVIEW"]
    assert {"DROPOUT", "STUCK"}.issubset(set(alerts["fault_type"]))
    assert {"SPIKE", "NOISE"}.issubset(set(reviews["fault_type"]))
    # ML-only candidates, including drift, are review-only until corroborated.
    assert ((results["fault_type"] == "DRIFT") & (results["status"] == "REVIEW")).any()
    assert (cross_sensor["status"] == "REVIEW").any()
