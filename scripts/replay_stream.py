"""Replay demo CSV rows as individual packets through the incremental processor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tether.streaming import StreamProcessor, TelemetryPacket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "demo_clean.csv"))
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.limit <= 20_000:
        parser.error("--limit must be between 1 and 20,000")

    frame = pd.read_csv(args.input, nrows=args.limit)
    processor = StreamProcessor(args.sample_rate)
    channels = [name for name in frame.columns if name not in {"packet_seq", "time_s"}]
    windows = 0
    for row in frame.to_dict(orient="records"):
        seq, timestamp = int(row["packet_seq"]), float(row["time_s"])
        values = {name: (None if pd.isna(row[name]) else float(row[name])) for name in channels}
        result = processor.ingest(TelemetryPacket(seq, timestamp, values))
        if not result.accepted:
            print("REJECT", result.diagnostics)
        if result.diagnostics:
            print("DIAGNOSTIC", result.diagnostics)
        windows += int(result.completed_window is not None)
    print({"windows_emitted": windows, **processor.snapshot()})


if __name__ == "__main__":
    main()
