# Security Policy

## Supported use

TETHER is a local research / portfolio prototype for synthetic or appropriately licensed public telemetry. Do not upload confidential team telemetry to a public deployment. The project is not safety-certified and must not be connected to vehicle control or used to make a safety-critical decision.

## Implemented safeguards

- CSV: 5 MB, 20,000 rows and 40 columns maximum; malformed encodings, duplicate headers and ragged rows are rejected.
- Profile: UTF-8 JSON only, 64 KiB maximum, explicit schema version and validated channel relationships.
- Streaming API: bearer token must contain at least 32 characters; constant-time token comparison; strict contract version; fixed channel allowlist; finite / magnitude-bounded numeric values; at most 40 channels; 8 KiB request body limit.
- Stream processor: rejects invalid channel names, NaN / infinity, duplicate sequences, out-of-order sequences and timestamp regressions.
- No user-supplied model pickle / joblib loading.
- Uploaded CSVs and API packets are processed in memory; no application persistence is implemented.
- GitHub Actions checks dependencies, source patterns / secrets, Python static analysis and the optional C++ build.

## Known limitations

The API is a prototype: no TLS termination, rate limiting, external secret manager, per-client identity, token rotation / revocation, durable audit log or production observability. Streamlit hosting receives user-uploaded bytes even though application code does not persist them. Use only non-sensitive data in a public deployment.

The numeric channel allowlist and profile envelopes are illustrative. They are not a complete schema for every vehicle. Production use requires a vehicle-specific data contract and security review.

## Reporting a vulnerability

Please do not publish secrets, private telemetry or an exploit in a public issue. Contact the repository maintainer privately with a concise reproduction, affected version and impact. Rotate any credential that may have been exposed.
