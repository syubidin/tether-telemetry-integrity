"""Small result-summary helpers shared by the UI and smoke checks."""

from __future__ import annotations

import pandas as pd


def summarize_results(results: pd.DataFrame) -> dict[str, int]:
    if results.empty:
        return {"sensors": 0, "windows": 0, "alerts": 0, "healthy": 0, "review": 0}
    return {
        "sensors": int(results["sensor"].nunique()),
        "windows": int(len(results)),
        "alerts": int((results["status"] == "ALERT").sum()),
        "healthy": int((results["status"] == "HEALTHY").sum()),
        "review": int((results["status"] == "REVIEW").sum()),
    }
