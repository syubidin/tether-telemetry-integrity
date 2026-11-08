import numpy as np

from tether.features import FEATURE_NAMES, extract_features


def test_features_are_finite_for_empty_and_all_missing_windows():
    for values in ([], [np.nan, np.nan], [np.inf, -np.inf]):
        features = extract_features(values)
        assert set(features) == set(FEATURE_NAMES)
        assert all(np.isfinite(value) for value in features.values())
    assert extract_features([np.nan, np.nan])["missing_ratio"] == 1.0


def test_flatline_signature_is_visible():
    values = np.concatenate([np.sin(np.linspace(0, 2, 20)), np.repeat(0.5, 44)])
    features = extract_features(values)
    assert features["longest_flatline_ratio"] > 0.5
    assert features["unique_ratio"] < 0.6


def test_missing_values_are_counted_without_crashing():
    values = np.sin(np.linspace(0, 5, 64))
    values[::3] = np.nan
    features = extract_features(values)
    assert 0.3 <= features["missing_ratio"] <= 0.35
