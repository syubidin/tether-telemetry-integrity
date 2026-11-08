"""Deterministic synthetic training data and a compact CPU-friendly classifier."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from tether.features import FEATURE_NAMES, extract_features

FAULT_TYPES = ("NORMAL", "DROPOUT", "STUCK", "DRIFT", "SPIKE", "NOISE")


def _base_signal(rng: np.random.Generator, size: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, size)
    phase_a, phase_b = rng.uniform(0, 2 * np.pi, size=2)
    amp = rng.uniform(0.18, 1.5)
    # Include slow and fast operating patterns: real sensor windows are not
    # guaranteed to contain a complete oscillation or a stationary baseline.
    frequency_a = rng.uniform(0.12, 2.8)
    frequency_b = rng.uniform(2.5, 7.0)
    signal = (
        amp * np.sin(2 * np.pi * frequency_a * t + phase_a)
        + rng.uniform(0.04, 0.35) * amp * np.sin(2 * np.pi * frequency_b * t + phase_b)
    )
    signal += rng.uniform(-0.65, 0.65) * amp * (t - 0.5)
    # Sensor channels often have a positive or negative operating baseline;
    # include offsets so training is not restricted to zero-centred signals.
    signal += rng.uniform(-8.0, 8.0) * amp
    signal += rng.normal(0.0, rng.uniform(0.015, 0.14) * amp, size)

    # Include realistic plateaus and gentle transitions in healthy signals.
    if rng.random() < 0.22:
        start = int(rng.integers(size // 4, size // 2))
        stop = min(size, start + int(rng.integers(3, 8)))
        signal[start:stop] = signal[start]
    return signal.astype(np.float64)


def make_training_data(
    samples_per_class: int = 180,
    window_size: int = 64,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a small reproducible training set from simulated sensor signals."""
    if samples_per_class < 20:
        raise ValueError("samples_per_class must be at least 20")
    if window_size < 32:
        raise ValueError("window_size must be at least 32")

    rng = np.random.default_rng(random_state)
    rows: list[list[float]] = []
    labels: list[str] = []

    for fault_type in FAULT_TYPES:
        for _ in range(samples_per_class):
            size = window_size
            signal = _base_signal(rng, size)
            scale = max(float(np.std(signal)), 0.2)

            if fault_type == "DROPOUT":
                missing_count = int(rng.integers(max(4, size // 10), max(5, size // 3)))
                missing_indices = rng.choice(size, size=min(missing_count, size - 1), replace=False)
                signal[missing_indices] = np.nan
                if rng.random() < 0.5:
                    start = int(rng.integers(0, size // 2))
                    signal[start : start + size // 5] = np.nan
            elif fault_type == "STUCK":
                start = int(rng.integers(0, size // 3))
                signal[start:] = signal[start]
            elif fault_type == "DRIFT":
                drift_strength = rng.uniform(2.5, 5.0) * scale
                signal += np.linspace(-0.15 * drift_strength, drift_strength, size)
            elif fault_type == "SPIKE":
                count = int(rng.integers(1, 4))
                indices = rng.choice(size, size=count, replace=False)
                signal[indices] += rng.choice([-1.0, 1.0], size=count) * rng.uniform(7.0, 14.0) * scale
            elif fault_type == "NOISE":
                signal += rng.normal(0.0, rng.uniform(0.45, 1.0) * scale, size)

            features = extract_features(signal)
            rows.append([features[name] for name in FEATURE_NAMES])
            labels.append(fault_type)

    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=str)


def train_model(random_state: int = 42) -> RandomForestClassifier:
    """Train a compact classifier in memory; no model file is loaded or trusted."""
    x_train, y_train = make_training_data(random_state=random_state)
    model = RandomForestClassifier(
        n_estimators=64,
        max_depth=10,
        min_samples_leaf=2,
        max_features=0.9,
        class_weight="balanced",
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model
