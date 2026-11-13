"""Export a bounded sample from the CC BY 4.0 Zenodo vehicle HDF dataset to CSV.

The source files are not bundled. Use --help for the data citation and limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to one downloaded .hdf/.h5 trip")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--sample-rate", type=float, default=20.0)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--column-map", default="{}", help="JSON object mapping HDF dataset paths to canonical TETHER columns")
    args = parser.parse_args()
    if not 1 <= args.max_rows <= 20_000 or not 0 < args.sample_rate <= 1000:
        parser.error("--max-rows must be 1..20,000 and --sample-rate must be 0..1,000 Hz")
    try:
        mapping = json.loads(args.column_map)
    except json.JSONDecodeError as exc:
        parser.error(f"--column-map must be valid JSON: {exc}")
    if not isinstance(mapping, dict) or len(mapping) > 40:
        parser.error("--column-map must be a JSON object with at most 40 entries")

    try:
        import h5py
    except ImportError:
        parser.error("Install the optional reader with: pip install -r requirements-public-data.txt")

    columns: dict[str, np.ndarray] = {}
    with h5py.File(args.input, "r") as handle:
        def visit(path: str, item) -> None:
            if not isinstance(item, h5py.Dataset) or len(columns) >= 40:
                return
            if item.dtype.kind not in "iuf" or item.ndim not in (1, 2):
                return
            if item.ndim == 2 and item.shape[1] > 4:
                return
            raw = np.asarray(item[: args.max_rows])
            if raw.ndim == 2:
                for index in range(raw.shape[1]):
                    key = f"{path}_{index}".replace("/", "_")
                    columns[key] = raw[:, index].astype(np.float64, copy=False)
            else:
                columns[path.replace("/", "_")] = raw.astype(np.float64, copy=False)

        handle.visititems(visit)

    if not columns:
        parser.error("No numeric one-dimensional or small-vector datasets were found in the HDF file.")
    lengths: dict[int, int] = {}
    for values in columns.values():
        lengths[len(values)] = lengths.get(len(values), 0) + 1
    target_length = min(args.max_rows, max(lengths, key=lengths.get))
    frame = pd.DataFrame({name: values[:target_length] for name, values in columns.items() if len(values) >= target_length})
    renamed: dict[str, str] = {}
    for source, destination in mapping.items():
        source_key = str(source).replace("/", "_")
        if source_key not in frame.columns:
            parser.error(f"Mapped HDF path not found: {source}")
        if not isinstance(destination, str) or not destination.isidentifier():
            parser.error("Mapped canonical column names must be valid identifiers.")
        renamed[source_key] = destination
    frame = frame.rename(columns=renamed)
    if "time_s" not in frame.columns:
        frame.insert(0, "time_s", np.arange(len(frame), dtype=np.float64) / args.sample_rate)
        print("NOTE: time_s was generated from nominal 20 Hz sampling; it is not a captured source timestamp.")
    if "packet_seq" not in frame.columns:
        frame.insert(0, "packet_seq", np.arange(len(frame), dtype=np.int64))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Exported {len(frame):,} rows × {len(frame.columns)} columns to {output}")
    print("Columns:", ", ".join(frame.columns))
    print("Map the actual wheel-speed and longitudinal-acceleration signals only after inspecting their units and HDF paths.")


if __name__ == "__main__":
    main()
