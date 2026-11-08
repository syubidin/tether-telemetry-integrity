# Architecture

## Runtime flow

1. **Input boundary:** a built-in synthetic CSV or user-selected CSV enters through the Streamlit UI.
2. **Validation:** `validation.py` checks the byte limit, UTF-8 encoding, CSV shape, header uniqueness, row count and column count before constructing a dataframe.
3. **Channel selection:** mostly-numeric fields are detected. Known time/index and discrete state channels are excluded from automatic selection.
4. **Windowing:** each selected channel is split into 64-sample windows. A tail shorter than 32 samples is merged into its preceding window; a channel shorter than 32 samples is skipped with a warning.
5. **Feature extraction:** `features.py` computes scale-aware features that tolerate missing and non-finite values.
6. **Deterministic gate:** dropout and stuck-at signatures with strong evidence are handled by simple rules.
7. **ML inference:** remaining windows are passed to one batched Random Forest call.
8. **Triage:** definite rule or high-scoring ML findings become alerts; ambiguous predictions and all uncorroborated drift predictions remain review candidates.
9. **Presentation:** Streamlit shows findings, quality notes, metrics and a bounded signal plot.

## Why rules + ML?

A model is not the right place to make every decision. A window with 30% missing samples has an obvious integrity issue and is best handled by a transparent rule. More ambiguous derivative/noise patterns benefit from the ML classifier. This reduces complexity and makes the highest-signal checks easier to test.

## Why drift is review-only

A single-channel trend cannot reliably tell the difference between sensor drift and legitimate changes in speed, throttle or temperature. TETHER therefore does not automatically escalate `DRIFT`. A production version should compare against a trusted baseline and correlated channels before promoting that candidate to an alert.

## Scaling path

The current process is intentionally bounded and synchronous. For larger data volumes, keep the same pure analysis functions and move ingestion/inference behind a separately authenticated API, then add queueing, streaming windows, model versioning, persistence policy and observability. Do not add distributed infrastructure until measurement shows the single-process design is insufficient.
