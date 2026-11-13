"""Regenerate source-controlled synthetic examples and the trusted demo baseline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tether.synthetic import make_demo_telemetry


def main() -> None:
    clean = make_demo_telemetry(inject_faults=False)
    faulted = make_demo_telemetry(inject_faults=True)
    trusted = make_demo_telemetry(rows=2048, random_state=19, inject_faults=False)
    clean.to_csv(ROOT / "data" / "demo_clean.csv", index=False)
    faulted.to_csv(ROOT / "data" / "demo_faults.csv", index=False)
    trusted.to_csv(ROOT / "data" / "trusted_baseline.csv", index=False)
    print("Generated clean demo, fault-injected demo, and clearly-labelled synthetic baseline.")


if __name__ == "__main__":
    main()
