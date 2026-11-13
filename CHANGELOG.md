# Changelog

## 1.1.0 — telemetry reliability expansion

- Added versioned generic track-car and public vehicle-CAN channel profiles with units, sampling rates, operating envelopes and expected signal relationships.
- Added robust trusted-baseline fitting and cross-sensor residual checks for gear-conditioned RPM/speed and speed-derived acceleration versus longitudinal acceleration.
- Added bounded packet-by-packet ingestion with duplicate, out-of-order, sequence-gap, timestamp-gap and timestamp-regression diagnostics.
- Added an optional FastAPI bearer-token endpoint with a versioned packet contract, channel allowlist, payload bounds and process-local latency/counter monitoring.
- Added an optional C++17 residual kernel with a portable NumPy fallback.
- Added a bounded HDF-to-CSV adapter and documented CC BY 4.0 public vehicle-CAN validation workflow; raw third-party data is not bundled.
- Added synthetic clean/fault injection reporting for per-fault precision, recall, F1, false positives per window, hard alerts/hour, review candidates/hour and detection delay.
- Added GitHub Actions for tests, lint, dependency audit, Bandit, Gitleaks and native C++ smoke testing.
- Tightened the decision policy: ML-only findings remain `REVIEW`; only deterministic dropout/stuck checks can create hard `ALERT` findings.

### Known limitations

- The bundled evaluation is synthetic-only. Real public HDF ingestion has not been executed in the build environment, and the selected public dataset contains no labelled sensor-fault events.
- The current synthetic model creates a high review-candidate volume on clean held-out data; see `reports/injected_fault_metrics.json` before using this as a deployment claim.
- The API is a prototype boundary, not a production security deployment. TLS, rate limiting, rotating identities, durable monitoring and persistence policies remain deployment responsibilities.
- Streamlit UI runtime was not smoke-tested in the build environment; install `requirements.txt` and run the local app to verify rendering.
