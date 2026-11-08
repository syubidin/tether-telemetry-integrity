"""Defensive CSV validation for untrusted user-provided telemetry."""

from __future__ import annotations

import csv
import io
from typing import BinaryIO

import pandas as pd

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 20_000
MAX_COLUMNS = 40
TIME_OR_INDEX_COLUMNS = {
    "time", "time_s", "timestamp", "timestamp_s", "elapsed_s", "lap_time",
    "lap_distance_m", "distance_m", "sample_id", "frame", "index", "row_id",
    "gear", "drs", "pit_limiter", "flag", "state",
}


class TelemetryValidationError(ValueError):
    """Raised when uploaded telemetry does not meet the accepted CSV contract."""


def read_telemetry_csv(source: bytes | BinaryIO) -> pd.DataFrame:
    """Read a bounded UTF-8 CSV after validating its shape and header."""
    raw = source if isinstance(source, bytes) else source.read(MAX_UPLOAD_BYTES + 1)
    if not isinstance(raw, bytes):
        raise TelemetryValidationError("The uploaded file could not be read as bytes.")
    if not raw:
        raise TelemetryValidationError("The CSV file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise TelemetryValidationError("File too large. The maximum upload size is 5 MB.")
    if b"\x00" in raw:
        raise TelemetryValidationError("The file contains binary or null-byte content; upload a UTF-8 CSV.")

    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise TelemetryValidationError("Unsupported encoding. Save the file as UTF-8 CSV and try again.") from exc

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, None)
        if not header:
            raise TelemetryValidationError("The CSV needs a header row and at least one data row.")
        columns = [item.strip() for item in header]
        if len(columns) > MAX_COLUMNS:
            raise TelemetryValidationError(f"Too many columns. Maximum supported: {MAX_COLUMNS}.")
        if any(not item for item in columns):
            raise TelemetryValidationError("Every CSV column must have a non-empty name.")
        normalized = [item.casefold() for item in columns]
        if len(normalized) != len(set(normalized)):
            raise TelemetryValidationError("Duplicate column names are not allowed (case-insensitive).")

        row_count = 0
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(columns):
                raise TelemetryValidationError("The CSV has rows with a different number of fields than its header.")
            row_count += 1
            if row_count > MAX_ROWS:
                raise TelemetryValidationError(f"Too many rows. Maximum supported: {MAX_ROWS:,}.")
    except csv.Error as exc:
        raise TelemetryValidationError("The CSV structure is malformed. Check quoting and delimiters.") from exc

    if row_count == 0:
        raise TelemetryValidationError("The CSV has a header but no data rows.")

    try:
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1, on_bad_lines="error")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, ValueError) as exc:
        raise TelemetryValidationError("Could not parse the CSV. Check its delimiter and row structure.") from exc

    frame.columns = columns
    if frame.empty:
        raise TelemetryValidationError("No usable data rows were found.")
    if frame.shape[1] != len(columns) or len(frame) > MAX_ROWS:
        raise TelemetryValidationError("The parsed CSV exceeded supported dimensions.")
    return frame


def numeric_sensor_columns(frame: pd.DataFrame) -> list[str]:
    """Return mostly-numeric columns, excluding obvious time/index fields."""
    selected: list[str] = []
    for column in frame.columns:
        if str(column).strip().casefold() in TIME_OR_INDEX_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(frame[column].dtype):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        non_empty = frame[column].notna().sum()
        if non_empty == 0:
            continue
        numeric_count = int(values.notna().sum())
        if numeric_count >= max(1, int(0.8 * non_empty)):
            selected.append(str(column))
    return selected


def validate_sensor_selection(frame: pd.DataFrame, sensor_columns: list[str]) -> None:
    """Validate selected names against the already parsed dataframe."""
    if not sensor_columns:
        raise TelemetryValidationError("Select at least one numeric sensor column.")
    if len(sensor_columns) > MAX_COLUMNS:
        raise TelemetryValidationError(f"Select no more than {MAX_COLUMNS} sensor columns.")
    if len(set(sensor_columns)) != len(sensor_columns):
        raise TelemetryValidationError("A sensor column was selected more than once.")
    unknown = [name for name in sensor_columns if name not in frame.columns]
    if unknown:
        raise TelemetryValidationError("One or more selected sensors are not present in this dataset.")
