# Engineering decisions and edge cases

| Concern | Decision |
|---|---|
| Low-end hardware | CPU-only Random Forest, 64 trees, depth 10, 64-sample windows |
| Startup cost | Train from compact, seeded synthetic examples once per Streamlit process and cache the model |
| Model artifact safety | Do not deserialize user-provided `pickle`/`joblib` files |
| Large uploads | 5 MB, 20,000-row and 40-column bounds; batched predictions |
| Empty input | Reject empty file, empty header, and header-only CSV |
| Malformed CSV | Reject duplicate/blank headers, ragged rows, invalid UTF-8 and binary/null bytes |
| Duplicate rows | Preserve them, but surface a data-quality note |
| Missing values | Treat NaN and infinities as missing; dropout is rule-based at 20% per window |
| All-missing window | Becomes dropout instead of raising a numeric exception |
| Constant channel | Flatline/uniqueness rule detects likely stuck-at behaviour |
| Short signal | Skip fewer than 32 samples; explain why |
| Short final window | Merge into previous window so samples are not silently dropped |
| Time order | Preserve upload row order; do not silently sort, interpolate or resample |
| Discrete state channels | Exclude gear/DRS/pit-limiter from automatic selection to avoid false stuck labels |
| Unusual trend | Keep drift as review-only until reference evidence exists |
| Low-confidence output | Use review rather than forcing a hard decision |
| Physical limits | Only show advisory checks for a few recognised channel names; do not pretend limits are universal |
| Confidential data | No persistence in app code; public demo still receives uploads on hosted server, so only synthetic/public data is recommended |
| CSV formula injection | No result CSV export is provided in this prototype |
| Cross-session privacy | No shared global dataframe or cached upload; only the model and fixed demo data are cached |

## Trade-offs

- A generic univariate model cannot understand every sensor's physical meaning. Real deployment requires per-channel units, sample rates, operating envelopes and trusted reference relationships.
- Synthetic evaluation is useful for repeatable tests, but it is not evidence of production accuracy.
- The 55% class-score threshold is a triage heuristic, not a statistically calibrated confidence threshold.
- Drift detection is intentionally conservative to avoid treating legitimate vehicle dynamics as a sensor failure.
