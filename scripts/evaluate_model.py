"""Reproducible hold-out evaluation for the synthetic training distribution."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tether.model import make_training_data
from tether.features import FEATURE_NAMES


def main() -> None:
    x, y = make_training_data(samples_per_class=220, random_state=2026)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=2026, stratify=y
    )
    # Evaluate a held-out model on generated examples that were not seen in fit.
    from sklearn.ensemble import RandomForestClassifier

    evaluator = RandomForestClassifier(
        n_estimators=64, max_depth=10, min_samples_leaf=2, max_features=0.9,
        class_weight="balanced", n_jobs=1, random_state=42,
    )
    evaluator.fit(x_train, y_train)
    predictions = evaluator.predict(x_test)
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Hold-out accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, digits=3, zero_division=0))
    print("Reminder: these metrics measure only the synthetic data generator, not real vehicle telemetry.")


if __name__ == "__main__":
    main()
