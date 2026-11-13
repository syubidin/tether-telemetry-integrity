"""Operating-envelope and offline timing diagnostics for batch telemetry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tether.profiles import TelemetryProfile


def check_operating_envelope(frame: pd.DataFrame, profile: TelemetryProfile) -> pd.DataFrame:
    findings: list[dict[str, Any]] = []
    for name, channel in profile.channels.items():
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        outside = (finite < channel.minimum) | (finite > channel.maximum)
        count = int(outside.sum())
        if count:
            findings.append({
                "channel": name,
                "unit": channel.unit,
                "out_of_envelope": count,
                "total_finite": int(finite.size),
                "minimum_observed": round(float(finite.min()), 4),
                "maximum_observed": round(float(finite.max()), 4),
                "expected_minimum": channel.minimum,
                "expected_maximum": channel.maximum,
                "status": "REVIEW",
            })
    return pd.DataFrame(findings)


def diagnose_frame_timing(frame: pd.DataFrame, profile: TelemetryProfile) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    nominal = 1.0 / profile.sample_rate_hz
    if profile.timestamp_column not in frame:
        diagnostics.append({"code": "TIMESTAMP_MISSING", "message": "No timestamp channel; nominal sample rate is assumed."})
        return {"diagnostics": diagnostics, "sample_rate_hz": None, "sequence_gaps": 0}

    times = pd.to_numeric(frame[profile.timestamp_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(times)
    if not valid.all():
        diagnostics.append({"code": "TIMESTAMP_INVALID", "count": int((~valid).sum())})
    diffs = np.diff(times)
    regressions = int(np.count_nonzero(diffs <= 0))
    if regressions:
        diagnostics.append({"code": "TIMESTAMP_NOT_MONOTONIC", "count": regressions})
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    gap_count = int(np.count_nonzero(positive > nominal * 1.5))
    if gap_count:
        diagnostics.append({"code": "TIMESTAMP_GAP", "count": gap_count, "nominal_interval_s": round(nominal, 6)})
    observed_rate = 1.0 / float(np.median(positive)) if positive.size else None

    sequence_gaps = 0
    if profile.sequence_column in frame:
        seq = pd.to_numeric(frame[profile.sequence_column], errors="coerce").to_numpy(dtype=float)
        seq_diff = np.diff(seq)
        valid_seq = np.isfinite(seq_diff) & (seq_diff > 0)
        sequence_gaps = int(np.maximum(seq_diff[valid_seq] - 1, 0).sum())
        if sequence_gaps:
            diagnostics.append({"code": "SEQUENCE_GAP", "missing_packets": sequence_gaps})
        duplicate_or_reordered = int(np.count_nonzero(~valid_seq))
        if duplicate_or_reordered:
            diagnostics.append({"code": "SEQUENCE_NOT_MONOTONIC", "count": duplicate_or_reordered})

    return {
        "diagnostics": diagnostics,
        "sample_rate_hz": round(observed_rate, 3) if observed_rate else None,
        "nominal_sample_rate_hz": profile.sample_rate_hz,
        "sequence_gaps": sequence_gaps,
    }
