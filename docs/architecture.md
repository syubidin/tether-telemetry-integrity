# Architecture

```text
                 ┌──────────────────────────────────────────┐
                 │  CSV Upload / Synthetic Demo / API Packet │
                 └─────────────────────┬────────────────────┘
                                       │
                    bounded validation│contract checks
                                       ▼
                 ┌──────────────────────────────────────────┐
                 │     Channel Profile + Operating Envelope │
                 └─────────────────────┬────────────────────┘
                                       │
                      ┌────────────────┴───────────────┐
                      ▼                                ▼
          ┌────────────────────────┐       ┌─────────────────────────┐
          │ Signal-local ML + Rules│       │ Trusted Baseline Checks │
          │ RF / dropout / stuck   │       │ RPM↔speed, speed↔accel  │
          └────────────┬───────────┘       └─────────────┬───────────┘
                       └────────────────┬───────────────┘
                                        ▼
                         ┌────────────────────────────┐
                         │ HEALTHY / REVIEW / ALERT   │
                         │ human-in-the-loop findings │
                         └────────────────────────────┘
```

## Components

- `validation.py`: defensive CSV parsing and column selection limits.
- `features.py` / `model.py`: small, reproducible time-series feature extraction and Random Forest.
- `profiles.py`: bounded JSON channel contract with units, sample rates, envelopes and relationship definitions.
- `baseline.py`: robust baseline fitting; linear relationships can be conditioned on a discrete state such as gear. Kinematic fitting can select a small integer sample lag.
- `cross_sensor.py`: per-window residual screening. It returns `HEALTHY` or `REVIEW`, not a mechanical diagnosis.
- `analysis.py`: combines local ML findings and cross-sensor findings. A healthy configured cross-sensor relation may de-escalate a local ML `REVIEW` candidate to `HEALTHY`; deterministic dropout / stuck `ALERT` findings are not downgraded. Missing or mismatched evidence never suppresses a finding.
- `streaming.py`: bounded one-packet-at-a-time processor; sequence and timestamp checks happen before a sample enters the window.
- `api.py`: optional FastAPI adapter with bearer auth, a strict versioned JSON payload, channel allowlist, finite-value bounds, 8 KiB request limit and operational counters.
- `native.py` + `cpp/`: optional C++17 residual kernel, exposed by a small C ABI and loaded with `ctypes`; NumPy remains the fallback.
- `app.py`: Streamlit UI with profile selection, separate baseline upload, batch findings, cross-sensor diagnostics and packet replay.

## Data flow / contracts

`contracts/telemetry.schema.json` defines API packet `1.0.0`. `contracts/model-card.json` records model release `1.1.0`, the data and feature contract versions, training data provenance and prohibited use. Profile schema is currently `1.0.0`; profiles themselves carry an independent profile ID and version. A baseline is valid only for the exact same profile ID and version used for scoring.

## Bounded resource behaviour

- Dashboard upload: 5 MB, 20,000 rows, 40 columns.
- Profile JSON: 64 KiB.
- API packet: 8 KiB and at most 40 channel values.
- Stream processor: one fixed-size in-memory deque (default 64 packets) and bounded counters.
- Metrics latency sample: most recent 512 API packet-processing durations.
- C++ residual kernel: input length 2..1,000,000; Python adapter makes contiguous float64 arrays before crossing the C ABI.
- Public HDF conversion: first 20,000 rows and at most 40 numeric channel arrays.

## Cross-sensor interpretation

RPM is compared with road speed within a gear group, which avoids fitting one line across gear changes. The kinematic relationship computes a central-difference estimate of longitudinal acceleration from wheel / vehicle speed and compares it with the measured accelerometer after a small lag search. This is a diagnostic residual, not a complete vehicle-dynamics model: wheel slip, tire radius, grade, filtering, sign conventions and sensor mounting can all affect it.

Throttle and speed are intentionally not linearly constrained. Their mapping depends on driver intent, gear state, brake use, traction and other dynamics. Add further relationships only where there is a physical model, a clear operating context and validated reference data.

## Failure philosophy

- Invalid contract / malformed packet: reject.
- Duplicate, out-of-order sequence or non-increasing timestamp: reject and count.
- Missing sequence or time interval: accept the current packet but report the discontinuity.
- Insufficient profile samples or missing channel: skip that relationship.
- High residual: `REVIEW`; do not infer mechanical root cause.
- Missing baseline or version mismatch: disable cross-sensor de-escalation.
- Missing native library: transparently fall back to NumPy.
