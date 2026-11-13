import json
from pathlib import Path

import pandas as pd
import pytest

from tether.baseline import fit_trusted_baseline
from tether.cross_sensor import analyze_cross_sensor
from tether.profiles import ProfileError, load_profile
from tether.quality import check_operating_envelope, diagnose_frame_timing

ROOT = Path(__file__).resolve().parents[1]


def load_clean():
    return pd.read_csv(ROOT / "data" / "demo_clean.csv")


def test_profile_fits_baseline_and_clean_data_has_no_cross_sensor_review():
    profile = load_profile(ROOT / "config" / "channel_profiles.json")
    frame = load_clean()
    baseline = fit_trusted_baseline(frame.iloc[:640], profile, "synthetic calibration baseline")
    evaluated = analyze_cross_sensor(frame.iloc[640:].reset_index(drop=True), profile, baseline)
    assert set(baseline.relationships) == {
        "rpm_vs_speed_conditioned_on_gear",
        "speed_derivative_vs_accelerometer",
    }
    assert not (evaluated["status"] == "REVIEW").any()


def test_rpm_drift_is_detected_by_gear_conditioned_residual():
    profile = load_profile(ROOT / "config" / "channel_profiles.json")
    frame = load_clean()
    baseline = fit_trusted_baseline(frame.iloc[:640], profile)
    evaluated = frame.iloc[640:].reset_index(drop=True).copy()
    evaluated.loc[128:191, "rpm"] += 1100
    findings = analyze_cross_sensor(evaluated, profile, baseline)
    target = findings[
        (findings["relationship"] == "rpm_vs_speed_conditioned_on_gear")
        & (findings["start_row"] == 128)
    ]
    assert len(target) == 1
    assert target.iloc[0]["status"] == "REVIEW"
    assert target.iloc[0]["candidate_fault"] == "DRIFT"


def test_accelerometer_bias_is_detected_without_calling_it_a_vehicle_fault():
    profile = load_profile(ROOT / "config" / "channel_profiles.json")
    frame = load_clean()
    baseline = fit_trusted_baseline(frame.iloc[:640], profile)
    evaluated = frame.iloc[640:].reset_index(drop=True).copy()
    evaluated.loc[128:191, "longitudinal_accel_mps2"] += 2.0
    findings = analyze_cross_sensor(evaluated, profile, baseline)
    target = findings[
        (findings["relationship"] == "speed_derivative_vs_accelerometer")
        & (findings["start_row"] == 128)
    ]
    assert len(target) == 1
    assert target.iloc[0]["status"] == "REVIEW"


def test_profile_rejects_duplicate_or_unknown_relationship_channel(tmp_path):
    payload = json.loads((ROOT / "config" / "channel_profiles.json").read_text())
    payload["relationships"][0]["x"] = "missing_channel"
    path = tmp_path / "bad-profile.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ProfileError, match="configured channels"):
        load_profile(path)


def test_operating_envelope_and_timestamp_gap_diagnostics():
    profile = load_profile(ROOT / "config" / "channel_profiles.json")
    frame = load_clean().iloc[:20].copy()
    frame.loc[3, "speed_kph"] = 500
    frame.loc[10, "time_s"] += 0.5
    frame.loc[10:, "time_s"] += 0.5
    frame.loc[10, "packet_seq"] += 2
    envelope = check_operating_envelope(frame, profile)
    timing = diagnose_frame_timing(frame, profile)
    assert "speed_kph" in set(envelope["channel"])
    assert any(item["code"] == "TIMESTAMP_GAP" for item in timing["diagnostics"])
    assert timing["sequence_gaps"] >= 1


def test_invalid_baseline_timestamps_do_not_create_kinematic_relationship():
    profile = load_profile(ROOT / "config" / "channel_profiles.json")
    frame = load_clean().iloc[:640].copy()
    frame.loc[100, "time_s"] = frame.loc[99, "time_s"]
    baseline = fit_trusted_baseline(frame, profile)
    assert "speed_derivative_vs_accelerometer" not in baseline.relationships


def test_profile_rejects_non_object_json_root(tmp_path):
    path = tmp_path / "list-profile.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ProfileError, match="root must be a JSON object"):
        load_profile(path)
