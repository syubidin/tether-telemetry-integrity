# Operations and monitoring

TETHER is currently a single-process prototype. It has no message broker, database, durable packet store or distributed worker pool.

## Metrics

The authenticated `GET /v1/telemetry/metrics` response contains:

- accepted and rejected packet counters;
- sequence-gap count, timestamp-gap count, duplicate packets, out-of-order packets and timestamp regressions;
- currently buffered samples;
- sample rate;
- rolling p50 / p95 processing duration over the most recent 512 packet ingests.

The latency sample is bounded and process-local. It is not a long-term time series, and metrics reset when the process restarts.

## Operational checks before production

- Emit structured logs with request ID, contract version, profile ID and outcome codes; never log the bearer token or raw telemetry payloads by default.
- Export metrics to an authenticated monitoring platform and alert on rising rejection / gap rates, latency, buffer starvation and unexpected alert volume.
- Track false-review / alert rates per operating hour and by vehicle / profile version.
- Establish an explicit retention policy and data classification before adding persistence.
- Put the API behind TLS, network restrictions, a managed secret store and a real rate limiter.
- Use multiple credentials with rotation / revocation rather than one static shared bearer token.
- Add durable queues only when measured throughput or availability requirements justify them.
- Validate p50 / p95 / p99 latency, memory ceiling and recovery behaviour under load and packet bursts.

The current API has a bearer-token gate, body size limit, fixed channel allowlist, strict payload contract and process-local counters. It is **not** a production security boundary by itself; it lacks TLS termination, rate limiting, credential identity / rotation, durable audit trails and external monitoring integration.
