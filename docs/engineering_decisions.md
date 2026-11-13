# Engineering decisions and edge cases

| Concern | Decision |
|---|---|
| Low-end hardware | CPU-only Random Forest, 64 trees, max depth 10, bounded windows; no GPU or database |
| Optional native path | One C++17 numeric kernel through `ctypes`; deterministic NumPy fallback |
| Trusted baseline leakage | Fit on a separately declared known-good recording / segment, never the suspect data being scored |
| Vehicle state dependence | Fit RPM–speed separately by gear; avoid one global regression across gear changes |
| Sensor alignment | Small, explicitly bounded kinematic lag search (maximum ±8 samples in parser; profiles use ±2) |
| Cross-sensor ambiguity | Residual deviations are `REVIEW`, not a fault diagnosis |
| Throttle / speed | No fixed linear expectation; driver intent and vehicle state make it context-dependent |
| Missing channel | Skip only affected relationship; do not invent values |
| Insufficient group samples | Skip that gear / group until minimum observations are available |
| Timestamp invalidity | Batch timing reports non-finite / non-monotonic timestamps; streaming rejects regression |
| Packet loss | Missing sequence values increment a gap counter and emit a diagnostic |
| Duplicate / out-of-order packet | Reject; keep accepted buffer order unchanged |
| Oversized payload | Enforce body, row, column, channel and HDF sampling limits before analysis |
| Unknown API channel | Reject against a fixed allowlist |
| Non-finite / extreme API value | Reject at request validation and again in the stream processor |
| Profile mismatch | Baseline ID/version mismatch raises an error and cannot downgrade alerts |
| ML escalation | A non-normal ML result remains review-only; the uncalibrated synthetic model never creates a hard alert |
| Hard alert policy | Only high-signal deterministic dropout / stuck rules can currently escalate to `ALERT` |
| Cross-sensor de-escalation | A non-normal ML `REVIEW` candidate can return to `HEALTHY` only when its configured relationship is healthy in the same window |
| Low-confidence model output | Use review instead of forcing a hard classification |
| Model artifact safety | Do not deserialize uploaded `pickle` / `joblib` artifacts |
| Confidential telemetry | Do not use sensitive team data in the public demo; app code does not persist uploads |
| Dataset rights | Keep source data outside repo; retain the original dataset licence and attribution |

## Why cross-sensor correlation reduces false alarms

A signal can change sharply while the underlying system is healthy. A speed drop while RPM drops in the expected gear-conditioned relationship can be legitimate deceleration; a growing speed/acceleration residual, by contrast, may indicate misalignment, bias or a measurement issue. The second signal is contextual evidence, not proof. TETHER only de-escalates a local ML review candidate when the corresponding configured relationship is healthy for the same window; deterministic dropout / stuck alerts remain intact.

## Why the profile matters

A value without a unit, sample rate, coordinate convention or operating range is not enough for a reliable comparison. Profile definitions are versioned and bounded. A production vehicle would need measured calibration information and per-channel sampling / filtering details, not the illustrative generic ranges shipped here.

## Streaming semantics

A packet with a missing sequence number is accepted after emitting a gap diagnostic. This preserves observed data rather than silently fabricating missing samples. Duplicate or out-of-order packets are rejected. A non-increasing timestamp is also rejected. The fixed-size deque is cleared only when a complete analysis window is emitted; at most one incomplete window remains in memory.

## Evaluation limits

The checked-in report uses a 30-minute deterministic synthetic session with a separate first-60% baseline and held-out final-40% segment. One positive 64-sample window is injected per fault scenario. Perfect window precision / recall in this tiny controlled test is not evidence of generalized fault-detection performance. Public unlabelled vehicle data can validate normal-operation residual distributions and false-review rates, but fault precision / recall requires ground-truth labels or controlled, documented fault injection.
