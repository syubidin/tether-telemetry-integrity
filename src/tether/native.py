"""Optional C++ acceleration with a safe and deterministic NumPy fallback."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class _NativeCore:
    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path))
        function = self.library.tether_speed_accel_residuals
        function.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        function.restype = ctypes.c_int
        self.function = function

    def residuals(self, speed: np.ndarray, timestamps: np.ndarray, accel: np.ndarray) -> np.ndarray:
        output = np.full(speed.size, np.nan, dtype=np.float64)
        result = self.function(
            speed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            accel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            speed.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if result < 0:
            raise RuntimeError("Native telemetry kernel rejected its input.")
        return output


def _load_native() -> _NativeCore | None:
    explicit = os.getenv("TETHER_NATIVE_LIB")
    candidates = [Path(explicit)] if explicit else []
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            root / "build" / "libtether_core.so",
            root / "build" / "tether_core.dll",
            root / "build" / "libtether_core.dylib",
            root / "cpp" / "build" / "libtether_core.so",
            root / "cpp" / "build" / "tether_core.dll",
            root / "cpp" / "build" / "libtether_core.dylib",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return _NativeCore(candidate)
            except (OSError, AttributeError):
                continue
    return None


_NATIVE = _load_native()


def native_available() -> bool:
    return _NATIVE is not None


def speed_accel_residuals(
    speed_kph: np.ndarray, timestamps_s: np.ndarray, measured_accel_mps2: np.ndarray
) -> np.ndarray:
    speed = np.ascontiguousarray(speed_kph, dtype=np.float64)
    timestamps = np.ascontiguousarray(timestamps_s, dtype=np.float64)
    accel = np.ascontiguousarray(measured_accel_mps2, dtype=np.float64)
    if speed.ndim != 1 or timestamps.ndim != 1 or accel.ndim != 1:
        raise ValueError("Native residual inputs must be one-dimensional.")
    if not (speed.size == timestamps.size == accel.size):
        raise ValueError("Native residual inputs must have identical lengths.")
    if _NATIVE is not None and 2 <= speed.size <= 1_000_000:
        return _NATIVE.residuals(speed, timestamps, accel)

    output = np.full(speed.size, np.nan, dtype=np.float64)
    if speed.size < 2:
        return output
    velocity = speed / 3.6
    indices = np.arange(speed.size)
    left = np.maximum(indices - 1, 0)
    right = np.minimum(indices + 1, speed.size - 1)
    dt = timestamps[right] - timestamps[left]
    derived = np.full(speed.size, np.nan, dtype=np.float64)
    valid_dt = np.isfinite(dt) & (dt > 0)
    derived[valid_dt] = (velocity[right[valid_dt]] - velocity[left[valid_dt]]) / dt[valid_dt]
    valid = np.isfinite(derived) & np.isfinite(accel)
    output[valid] = derived[valid] - accel[valid]
    return output
