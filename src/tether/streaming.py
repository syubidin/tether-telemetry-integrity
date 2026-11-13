"""Bounded incremental packet ingestion with sequence and timestamp diagnostics."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TelemetryPacket:
    packet_seq: int
    timestamp_s: float
    values: dict[str, float | None]


@dataclass
class StreamResult:
    accepted: bool
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    completed_window: pd.DataFrame | None = None
    buffered_samples: int = 0


class StreamProcessor:
    """Process one packet at a time while keeping memory bounded."""

    def __init__(
        self,
        sample_rate_hz: float,
        window_size: int = 64,
        max_channels: int = 40,
        timestamp_gap_factor: float = 1.5,
    ) -> None:
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive.")
        if not 8 <= window_size <= 4096 or not 1 <= max_channels <= 40:
            raise ValueError("window_size or max_channels is outside supported bounds.")
        if not math.isfinite(timestamp_gap_factor) or timestamp_gap_factor < 1.0:
            raise ValueError("timestamp_gap_factor must be finite and at least 1.0.")
        self.sample_rate_hz = sample_rate_hz
        self.expected_period_s = 1.0 / sample_rate_hz
        self.window_size = window_size
        self.max_channels = max_channels
        self.timestamp_gap_factor = timestamp_gap_factor
        self._window: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._last_seq: int | None = None
        self._last_timestamp: float | None = None
        self.counters = {
            "accepted_packets": 0,
            "rejected_packets": 0,
            "sequence_gaps": 0,
            "timestamp_gaps": 0,
            "duplicate_packets": 0,
            "out_of_order_packets": 0,
            "timestamp_regressions": 0,
        }

    def ingest(self, packet: TelemetryPacket) -> StreamResult:
        diagnostics: list[dict[str, Any]] = []
        if isinstance(packet.packet_seq, bool) or not isinstance(packet.packet_seq, int) or packet.packet_seq < 0:
            return self._reject("INVALID_SEQUENCE", "packet_seq must be a non-negative integer.")
        if (
            isinstance(packet.timestamp_s, bool)
            or not isinstance(packet.timestamp_s, (int, float))
            or not math.isfinite(packet.timestamp_s)
            or packet.timestamp_s < 0
        ):
            return self._reject("INVALID_TIMESTAMP", "timestamp_s must be a finite non-negative number.")
        if not isinstance(packet.values, dict) or not 1 <= len(packet.values) <= self.max_channels:
            return self._reject("INVALID_CHANNEL_COUNT", f"values must contain 1 to {self.max_channels} channels.")
        if any(not isinstance(name, str) or not name.isidentifier() or len(name) > 64 for name in packet.values):
            return self._reject("INVALID_CHANNEL_NAME", "Channel names must be safe identifiers up to 64 characters.")
        for name, value in packet.values.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                return self._reject("INVALID_CHANNEL_VALUE", f"Channel {name} must be finite numeric data or null.")

        if self._last_seq is not None:
            if packet.packet_seq == self._last_seq:
                self.counters["duplicate_packets"] += 1
                return self._reject("DUPLICATE_SEQUENCE", "Packet sequence was already ingested.")
            if packet.packet_seq < self._last_seq:
                self.counters["out_of_order_packets"] += 1
                return self._reject("OUT_OF_ORDER_SEQUENCE", "Out-of-order packet rejected; buffer order is preserved.")
            sequence_delta = packet.packet_seq - self._last_seq
            if sequence_delta > 1:
                missing = sequence_delta - 1
                self.counters["sequence_gaps"] += missing
                diagnostics.append({"code": "SEQUENCE_GAP", "missing_packets": missing})
            if self._last_timestamp is not None:
                time_delta = packet.timestamp_s - self._last_timestamp
                if time_delta <= 0:
                    self.counters["timestamp_regressions"] += 1
                    return self._reject("TIMESTAMP_REGRESSION", "Timestamp must increase; packet was rejected.")
                expected_delta = sequence_delta * self.expected_period_s
                if time_delta > max(self.expected_period_s * self.timestamp_gap_factor, expected_delta * self.timestamp_gap_factor):
                    self.counters["timestamp_gaps"] += 1
                    diagnostics.append({
                        "code": "TIMESTAMP_GAP",
                        "observed_interval_s": round(time_delta, 6),
                        "expected_interval_s": round(expected_delta, 6),
                    })

        row = {"packet_seq": packet.packet_seq, "time_s": packet.timestamp_s, **packet.values}
        self._window.append(row)
        self._last_seq = packet.packet_seq
        self._last_timestamp = packet.timestamp_s
        self.counters["accepted_packets"] += 1

        completed: pd.DataFrame | None = None
        if len(self._window) == self.window_size:
            completed = pd.DataFrame(list(self._window))
            self._window.clear()
        return StreamResult(True, diagnostics, completed, len(self._window))

    def _reject(self, code: str, message: str) -> StreamResult:
        self.counters["rejected_packets"] += 1
        return StreamResult(False, [{"code": code, "message": message}], None, len(self._window))

    def snapshot(self) -> dict[str, Any]:
        return {**self.counters, "buffered_samples": len(self._window), "sample_rate_hz": self.sample_rate_hz}
