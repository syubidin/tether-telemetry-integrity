"""Authenticated, bounded FastAPI ingestion endpoint for the prototype."""

from __future__ import annotations

import math
import os
import secrets
import time
from collections import deque
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator
from starlette.middleware import Middleware
from starlette.responses import JSONResponse

from tether.streaming import StreamProcessor, TelemetryPacket

MAX_BODY_BYTES = 8192
DEFAULT_ALLOWED_CHANNELS = {
    "speed_kph", "rpm", "gear", "longitudinal_accel_mps2", "throttle_pct",
    "brake_pct", "coolant_temp_c", "oil_pressure_bar", "battery_voltage_v",
    "wheel_speed_fl_kph", "steering_angle_deg", "yaw_rate_dps",
    "lateral_accel_mps2", "roll_deg", "pitch_deg",
}


class BodyLimitMiddleware:
    """Pure-ASGI request-size guard that bounds buffered JSON before parsing."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared:
            try:
                if int(declared) > self.max_bytes:
                    await JSONResponse({"detail": "Request body too large."}, status_code=413)(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await JSONResponse({"detail": "Request body too large."}, status_code=413)(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        sent_body = False

        async def bounded_receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


class PacketPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = Field(min_length=1, max_length=16)
    packet_seq: StrictInt = Field(ge=0, le=2**63 - 1)
    timestamp_s: StrictFloat | StrictInt = Field(ge=0, le=1e12, allow_inf_nan=False)
    values: dict[str, StrictFloat | StrictInt | None] = Field(min_length=1, max_length=40)

    @field_validator("contract_version")
    @classmethod
    def supported_contract(cls, value: str) -> str:
        if value != "1.0.0":
            raise ValueError("Unsupported telemetry contract version.")
        return value

    @field_validator("values")
    @classmethod
    def safe_values(cls, values: dict[str, StrictFloat | StrictInt | None]) -> dict[str, StrictFloat | StrictInt | None]:
        for name, value in values.items():
            if not isinstance(name, str) or not name.isidentifier() or len(name) > 64:
                raise ValueError("Channel names must be safe identifiers up to 64 characters.")
            if value is not None and (not math.isfinite(float(value)) or abs(float(value)) > 1e9):
                raise ValueError("Channel values must be finite and within ±1e9.")
        return values


def create_app(
    api_token: str | None = None,
    sample_rate_hz: float = 20.0,
    allowed_channels: set[str] | None = None,
) -> FastAPI:
    token = api_token if api_token is not None else os.getenv("TETHER_API_TOKEN", "")
    processor = StreamProcessor(sample_rate_hz=sample_rate_hz)
    app = FastAPI(
        title="TETHER Ingestion API",
        version="1.1.0",
        docs_url=None,
        redoc_url=None,
        middleware=[Middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)],
    )
    app.state.processor = processor
    app.state.api_token = token
    app.state.allowed_channels = set(allowed_channels or DEFAULT_ALLOWED_CHANNELS)
    app.state.ingest_latencies_ms = deque(maxlen=512)

    def authenticate(authorization: str | None) -> None:
        if len(token) < 32:
            raise HTTPException(status_code=503, detail="Ingestion is disabled: configure a 32-character TETHER_API_TOKEN.")
        scheme, _, provided = (authorization or "").partition(" ")
        if scheme.casefold() != "bearer" or not provided or not secrets.compare_digest(provided, token):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "tether-ingestion", "contract_version": "1.0.0"}

    @app.post("/v1/telemetry/packet")
    async def ingest_packet(payload: PacketPayload, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        unknown = set(payload.values) - app.state.allowed_channels
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unsupported channel(s): {', '.join(sorted(unknown))}")
        started = time.perf_counter()
        result = processor.ingest(TelemetryPacket(payload.packet_seq, float(payload.timestamp_s), payload.values))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        app.state.ingest_latencies_ms.append(elapsed_ms)
        return {
            "accepted": result.accepted,
            "diagnostics": result.diagnostics,
            "buffered_samples": result.buffered_samples,
            "window_completed": result.completed_window is not None,
            "processing_ms": round(elapsed_ms, 3),
            "counters": processor.snapshot(),
        }

    @app.get("/v1/telemetry/metrics")
    async def metrics(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        latencies = sorted(app.state.ingest_latencies_ms)
        def percentile(values: list[float], quantile: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, int((len(values) - 1) * quantile))
            return round(values[index], 3)

        snapshot = processor.snapshot()
        snapshot["ingest_latency_ms"] = {
            "sample_count": len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        }
        return snapshot

    return app


app = create_app()
