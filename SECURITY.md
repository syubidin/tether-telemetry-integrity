# Security policy

## Scope

TETHER is an educational prototype. It is not safety-critical, is not designed for confidential team telemetry, and must not be connected to a vehicle or live engineering workflow without a formal review.

## Current protections

- Bounded uploads (5 MB), rows (20,000) and columns (40).
- UTF-8 and CSV shape validation; rejects malformed, ragged, duplicate-header and null-byte content.
- No user-controlled file paths or model artifact loading.
- No `pickle`/`joblib` deserialization of user-supplied files.
- No app-side persistence of uploaded telemetry.
- No credentials, API tokens or external secrets required.
- No CSV export of user-controlled cell values.

## Reporting a vulnerability

Please do not publish sensitive exploit details in a public issue. For a real deployment, provide a private security contact and a documented response process before accepting sensitive data. This repository currently has no dedicated security response SLA.

## Production prerequisites

Authentication and authorization, tenant isolation, encrypted data storage where required, audit logging, retention and deletion policies, dependency scanning, runtime monitoring, rate limiting, threat modelling, incident response and independent safety/security reviews.
