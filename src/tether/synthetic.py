"""Deterministic, clearly-labelled synthetic motorsport telemetry examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_telemetry(rows: int = 1024, random_state: int = 7, inject_faults: bool = False) -> pd.DataFrame:
    """Create illustrative lap-like signals; these are not real team telemetry."""
    rng = np.random.default_rng(random_state)
    time_s = np.arange(rows, dtype=np.float64) / 20.0
    phase = 2.0 * np.pi * time_s / 11.5

    speed = 190.0 + 82.0 * np.sin(phase) + 7.0 * np.sin(phase * 2.2)
    speed = np.clip(speed + rng.normal(0.0, 0.18, rows), 55.0, 325.0)
    throttle = np.clip(55.0 + 43.0 * np.sin(phase + 0.7) + rng.normal(0.0, 1.3, rows), 0.0, 100.0)
    brake = np.clip(42.0 - 39.0 * np.sin(phase + 0.7) + rng.normal(0.0, 1.0, rows), 0.0, 100.0)
    gear = np.clip(np.rint(speed / 43.0), 1, 8).astype(int)
    rpm_slope = 52.0 - gear * 2.5
    rpm = np.clip(2500.0 + speed * rpm_slope + rng.normal(0.0, 24.0, rows), 2500.0, 19_000.0)
    longitudinal_accel = np.gradient(speed / 3.6, time_s) + rng.normal(0.0, 0.06, rows)
    coolant = 91.0 + 2.8 * np.sin(time_s / 20.0) + 0.035 * time_s + rng.normal(0.0, 0.08, rows)
    oil_pressure = 4.2 + 0.35 * np.sin(phase * 0.5) + rng.normal(0.0, 0.025, rows)

    frame = pd.DataFrame(
        {
            "packet_seq": np.arange(rows, dtype=np.int64),
            "time_s": time_s.round(4),
            "speed_kph": speed.round(3),
            "throttle_pct": throttle.round(3),
            "brake_pct": brake.round(3),
            "rpm": rpm.round(1),
            "gear": gear,
            "longitudinal_accel_mps2": longitudinal_accel.round(3),
            "coolant_temp_c": coolant.round(3),
            "oil_pressure_bar": oil_pressure.round(3),
        }
    )

    if inject_faults:
        # Faults are deliberately aligned to analysis windows for a clear demo.
        frame.loc[256:319, "coolant_temp_c"] += np.linspace(0.0, 8.5, 64)
        frame.loc[512:575, "oil_pressure_bar"] = 4.15
        frame.loc[[704, 719, 733], "throttle_pct"] = [0.0, 100.0, 0.0]
        frame.loc[448:511, "rpm"] += np.linspace(0.0, 1200.0, 64)
        noise_rng = np.random.default_rng(random_state + 984)
        frame.loc[832:895, "speed_kph"] += noise_rng.normal(0.0, 30.0, 64)
        frame.loc[930:944, "rpm"] = np.nan

    return frame
