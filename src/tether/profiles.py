"""Versioned channel profiles and trusted-baseline contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = "1.0.0"


class ProfileError(ValueError):
    """Raised when a channel profile is invalid or unsafe to use."""


@dataclass(frozen=True)
class ChannelProfile:
    name: str
    unit: str
    minimum: float
    maximum: float
    sample_rate_hz: float


@dataclass(frozen=True)
class RelationshipProfile:
    name: str
    kind: str
    x: str
    y: str
    condition_by: str | None = None
    residual_z_threshold: float = 5.0
    min_samples: int = 20
    max_lag_samples: int = 0


@dataclass(frozen=True)
class TelemetryProfile:
    profile_id: str
    version: str
    sample_rate_hz: float
    timestamp_column: str
    sequence_column: str
    channels: dict[str, ChannelProfile]
    relationships: tuple[RelationshipProfile, ...]


def load_profile(path: str | Path) -> TelemetryProfile:
    """Load a bounded JSON profile; no YAML/object deserialization is used."""
    file_path = Path(path)
    raw = file_path.read_bytes()
    if len(raw) > 64 * 1024:
        raise ProfileError("Profile file exceeds the 64 KiB limit.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError("Profile must be valid UTF-8 JSON.") from exc

    if not isinstance(payload, dict):
        raise ProfileError("Profile root must be a JSON object.")
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ProfileError("Unsupported channel profile schema_version.")
    sample_rate = _positive_number(payload.get("sample_rate_hz"), "sample_rate_hz")
    channels_raw = payload.get("channels")
    relationships_raw = payload.get("relationships", [])
    if not isinstance(channels_raw, dict) or not channels_raw:
        raise ProfileError("Profile must define at least one channel.")
    if not isinstance(relationships_raw, list) or len(relationships_raw) > 40:
        raise ProfileError("relationships must be a list of at most 40 items.")

    channels: dict[str, ChannelProfile] = {}
    for name, item in channels_raw.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(item, dict):
            raise ProfileError("Channel names must be identifiers with object definitions.")
        minimum = _finite_number(item.get("minimum"), f"{name}.minimum")
        maximum = _finite_number(item.get("maximum"), f"{name}.maximum")
        if minimum >= maximum:
            raise ProfileError(f"{name}: minimum must be lower than maximum.")
        channels[name] = ChannelProfile(
            name=name,
            unit=str(item.get("unit", ""))[:32],
            minimum=minimum,
            maximum=maximum,
            sample_rate_hz=_positive_number(item.get("sample_rate_hz", sample_rate), f"{name}.sample_rate_hz"),
        )

    relationships: list[RelationshipProfile] = []
    seen: set[str] = set()
    for item in relationships_raw:
        if not isinstance(item, dict):
            raise ProfileError("Each relationship must be an object.")
        name = str(item.get("name", ""))
        kind = str(item.get("kind", ""))
        x, y = str(item.get("x", "")), str(item.get("y", ""))
        condition = item.get("condition_by")
        if not name or name in seen or kind not in {"linear", "kinematic"}:
            raise ProfileError("Relationship names must be unique and kind must be linear or kinematic.")
        if x not in channels or y not in channels:
            raise ProfileError(f"{name}: x and y must refer to configured channels.")
        if condition is not None and condition not in channels:
            raise ProfileError(f"{name}: condition_by references an unknown channel.")
        threshold = _positive_number(item.get("residual_z_threshold", 5.0), f"{name}.threshold")
        min_samples = int(item.get("min_samples", 20))
        max_lag = int(item.get("max_lag_samples", 0))
        if not 8 <= min_samples <= 100_000 or not 0 <= max_lag <= 8:
            raise ProfileError(f"{name}: min_samples or max_lag_samples is outside supported bounds.")
        relationships.append(RelationshipProfile(name, kind, x, y, condition, threshold, min_samples, max_lag))
        seen.add(name)

    return TelemetryProfile(
        profile_id=str(payload.get("profile_id", "unnamed"))[:80],
        version=str(payload.get("version", "1.0.0"))[:40],
        sample_rate_hz=sample_rate,
        timestamp_column=str(payload.get("timestamp_column", "time_s"))[:64],
        sequence_column=str(payload.get("sequence_column", "packet_seq"))[:64],
        channels=channels,
        relationships=tuple(relationships),
    )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProfileError(f"{field} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{field} must be a finite number.") from exc
    if not (float("-inf") < number < float("inf")):
        raise ProfileError(f"{field} must be a finite number.")
    return number


def _positive_number(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number <= 0:
        raise ProfileError(f"{field} must be greater than zero.")
    return number
