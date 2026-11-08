"""Regenerate the checked-in synthetic CSV examples."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tether.synthetic import make_demo_telemetry

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

make_demo_telemetry(inject_faults=False).to_csv(DATA_DIR / "demo_clean.csv", index=False)
make_demo_telemetry(inject_faults=True).to_csv(DATA_DIR / "demo_faults.csv", index=False)
print(f"Wrote synthetic examples to {DATA_DIR}")
