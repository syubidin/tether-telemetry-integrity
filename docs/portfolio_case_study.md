# Portfolio case study — TETHER v1.1.0

## One-line summary

Built a lightweight, cross-sensor telemetry integrity system with Python ML, incremental packet processing and an optional C++17 numerical kernel.

## Problem

Telemetry consumers can make incorrect engineering decisions when a sensor has dropout, frozen values, drift, spikes, noise, packet loss or timestamp discontinuity. A naïve anomaly detector also produces false positives when normal vehicle behaviour changes multiple channels together.

## Solution

TETHER combines a compact Random Forest and deterministic signal rules with versioned channel profiles and trusted cross-sensor baselines. It checks RPM against speed within gear groups, compares speed-derived acceleration with measured longitudinal acceleration, and retains ambiguous residual shifts for human review. A bounded packet processor diagnoses sequence / timestamp gaps. A small C++17 kernel computes the speed-acceleration residual through `ctypes`, while Python / NumPy remains the portable fallback.

## Engineering implementation

- Python package separated into validation, profile, feature, model, baseline, analysis, streaming and reporting modules.
- Streamlit dashboard supports two explicit profiles and a separate trusted baseline upload.
- FastAPI ingestion is optional, bearer-token protected, versioned, channel-allowlisted and body-size bounded.
- Monitoring endpoint exposes bounded counters plus p50 / p95 packet processing latency.
- C++17 core is independently built and smoke-tested through CMake / CTest.
- CI includes Python tests, static lint, Bandit, pip-audit, Gitleaks, and native build checks.
- Third-party dataset is kept outside the repo; HDF export is capped at 20k rows / 40 channels.

## Reproducible synthetic experiment

A deterministic 30-minute synthetic stream is split into an initial 60% trusted baseline and a final 40% evaluation segment (12 minutes). Each scenario injects one labelled positive 64-sample window.

**Signal-local tests**

| Fault | Precision | Recall | F1 | FP windows | Detection delay |
|---|---:|---:|---:|---:|---:|
| DROPOUT | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |
| STUCK | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |
| DRIFT | 0.50 | 1.00 | 0.67 | 1 | 3.2 s |
| SPIKE | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |
| NOISE | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |

**Cross-sensor tests**

| Fault | Precision | Recall | F1 | Detection delay |
|---|---:|---:|---:|---:|
| RPM drift (+500 rpm) | 1.00 | 1.00 | 1.00 | 3.2 s |
| Accelerometer bias (+1.8 m/s²) | 1.00 | 1.00 | 1.00 | 3.2 s |

On the clean held-out synthetic segment, hard `ALERT` events were 0.0/hour and cross-sensor `REVIEW` findings were 0.0/hour. However, the signal-local model generated **3,585 review candidates/hour** across selected numeric channels. This is a significant operational limitation: the uncalibrated synthetic classifier still produces too many candidates for a useful engineer-facing queue. The DRIFT scenario also generated one false-positive window (precision 0.50). These values make the next optimization target explicit rather than hiding candidate load behind the hard-alert metric.

The detection delay is measured at completion of a non-overlapping 64-sample window (3.2 seconds at 20 Hz). There is one positive window per injection, so this is a regression experiment, not a statistically strong estimate. Exact metadata and limitations are in `reports/injected_fault_metrics.json`.

## Public-data validation status

The application includes a converter and channel profile for the CC BY 4.0 Vehicle CAN bus dataset on Zenodo. The raw HDF file could not be downloaded in the build environment; no real public-data results are claimed in this release. The dataset is passenger-car data with no ground-truth sensor-fault labels, so real fault metrics require separate fault injection or labelled cases.

## Next experiment plan

1. Run the HDF converter on one public trip after verifying signal units and channel paths.
2. Fit baseline on a separate clean trip; evaluate residual stability on unseen trips and manoeuvres.
3. Inject bias, drift, stuck-at, dropout, spike, noise and timing defects separately, at multiple magnitudes and locations.
4. Report per-fault window precision / recall / F1, false reviews per hour, detection delay distribution, and results by operating regime.
5. Profile CPU and memory on a low-end device; compare C++ and NumPy output parity and runtime.
6. Only then consider authenticated production ingestion, persistent monitoring and deployment hardening.

## Interview walkthrough

1. Explain why trustworthy input data is a software reliability problem, not just a model problem.
2. Show how gear-conditioned RPM/speed relationships reduce false positives from gear changes.
3. Explain why the acceleration residual includes lag search and why a large residual is review-only.
4. Demonstrate sequence gaps, duplicate packets and timestamp regressions in packet replay.
5. Compare C++ and NumPy paths and explain why the native library is optional.
6. Show how CI checks dependencies, secrets, static quality and C++ build health.
7. Be transparent that real public dataset validation remains a follow-up; show the exact reproducible workflow.

## Suggested CV bullet

> Developed TETHER, a CPU-first telemetry-integrity prototype combining Python time-series ML, gear-conditioned cross-sensor baselines, bounded real-time ingestion, sequence/timestamp diagnostics, an authenticated FastAPI endpoint and an optional C++17 signal-processing kernel; added reproducible fault-injection evaluation and CI security checks.
