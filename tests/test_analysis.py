import numpy as np
import pandas as pd

from tether.analysis import analyze_dataframe
from tether.model import train_model


def test_deterministic_dropout_rule_flags_missing_window():
    model = train_model()
    values = np.sin(np.linspace(0, 8, 64))
    values[:20] = np.nan
    results, _ = analyze_dataframe(pd.DataFrame({"sensor_a": values}), ["sensor_a"], model)
    assert len(results) == 1
    assert results.iloc[0]["fault_type"] == "DROPOUT"
    assert results.iloc[0]["method"] == "RULE"


def test_deterministic_stuck_rule_flags_frozen_sensor():
    model = train_model()
    values = np.concatenate([np.sin(np.linspace(0, 4, 12)), np.repeat(0.25, 52)])
    results, _ = analyze_dataframe(pd.DataFrame({"sensor_a": values}), ["sensor_a"], model)
    assert results.iloc[0]["fault_type"] == "STUCK"


def test_short_sensor_is_skipped_with_explanation():
    model = train_model()
    results, warnings = analyze_dataframe(pd.DataFrame({"sensor_a": [1.0] * 20}), ["sensor_a"], model)
    assert results.empty
    assert any("insufficient data" in note for note in warnings)
