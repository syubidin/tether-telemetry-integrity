from fastapi.testclient import TestClient

from tether.api import create_app

TOKEN = "test-token-that-is-at-least-thirty-two-characters-long"


def test_api_requires_bearer_token_and_accepts_valid_packet():
    client = TestClient(create_app(api_token=TOKEN, sample_rate_hz=20))
    payload = {
        "contract_version": "1.0.0",
        "packet_seq": 0,
        "timestamp_s": 0.0,
        "values": {"speed_kph": 140.0},
    }
    assert client.get("/health").status_code == 200
    assert client.post("/v1/telemetry/packet", json=payload).status_code == 401
    response = client.post(
        "/v1/telemetry/packet", json=payload,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_api_rejects_unsupported_contract_and_oversized_body():
    client = TestClient(create_app(api_token=TOKEN))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    unsupported = {
        "contract_version": "99.0.0", "packet_seq": 0,
        "timestamp_s": 0.0, "values": {"speed_kph": 100.0},
    }
    assert client.post("/v1/telemetry/packet", json=unsupported, headers=headers).status_code == 422
    oversized = b"{" + b" " * 9000 + b"}"
    assert client.post("/v1/telemetry/packet", content=oversized, headers=headers).status_code == 413


def test_api_disables_ingestion_when_token_is_not_configured():
    client = TestClient(create_app(api_token=""))
    payload = {
        "contract_version": "1.0.0", "packet_seq": 0,
        "timestamp_s": 0.0, "values": {"speed_kph": 140.0},
    }
    response = client.post(
        "/v1/telemetry/packet", json=payload,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 503


def test_api_rejects_unknown_channels_and_reports_ingest_latency():
    client = TestClient(create_app(api_token=TOKEN, sample_rate_hz=20))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    bad = {
        "contract_version": "1.0.0", "packet_seq": 0,
        "timestamp_s": 0.0, "values": {"shell_command": 1.0},
    }
    assert client.post("/v1/telemetry/packet", json=bad, headers=headers).status_code == 422
    good = {
        "contract_version": "1.0.0", "packet_seq": 0,
        "timestamp_s": 0.0, "values": {"speed_kph": 120.0},
    }
    assert client.post("/v1/telemetry/packet", json=good, headers=headers).status_code == 200
    metrics = client.get("/v1/telemetry/metrics", headers=headers).json()
    assert metrics["ingest_latency_ms"]["sample_count"] == 1
    assert metrics["ingest_latency_ms"]["p95"] is not None


def test_api_rejects_non_finite_or_extreme_channel_values():
    client = TestClient(create_app(api_token=TOKEN))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {
        "contract_version": "1.0.0", "packet_seq": 0,
        "timestamp_s": 0.0, "values": {"speed_kph": 1e12},
    }
    assert client.post("/v1/telemetry/packet", json=payload, headers=headers).status_code == 422
