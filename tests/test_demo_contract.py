import pytest

from tether.analysis import analyze_dataframe
from tether.model import train_model
from tether.synthetic import make_demo_telemetry
from tether.validation import numeric_sensor_columns


@pytest.fixture(scope="module")
def model():
    return train_model()


def test_clean_synthetic_run_has_no_escalated_alerts(model):
    frame = make_demo_telemetry(inject_faults=False)
    results, warnings = analyze_dataframe(frame, numeric_sensor_columns(frame), model)
    assert not warnings
    assert not (results["status"] == "ALERT").any()
    assert (results["status"] == "HEALTHY").any()


def test_fault_injected_run_exposes_expected_integrity_signatures(model):
    frame = make_demo_telemetry(inject_faults=True)
    results, _ = analyze_dataframe(frame, numeric_sensor_columns(frame), model)
    alerts = results[results["status"] == "ALERT"]
    assert {"DROPOUT", "STUCK", "SPIKE", "NOISE"}.issubset(set(alerts["fault_type"]))
    # Drift is intentionally a review candidate, not an automatic hard alert.
    assert ((results["fault_type"] == "DRIFT") & (results["status"] == "REVIEW")).any()
