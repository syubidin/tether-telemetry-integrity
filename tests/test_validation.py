import pytest

from tether.validation import (
    MAX_UPLOAD_BYTES,
    TelemetryValidationError,
    numeric_sensor_columns,
    read_telemetry_csv,
)


def test_reads_utf8_csv_and_ignores_time_column():
    frame = read_telemetry_csv(b"time_s,speed_kph,mode\n0,100,green\n1,120,green\n")
    assert list(frame.columns) == ["time_s", "speed_kph", "mode"]
    assert numeric_sensor_columns(frame) == ["speed_kph"]


def test_rejects_empty_and_oversized_files():
    with pytest.raises(TelemetryValidationError, match="empty"):
        read_telemetry_csv(b"")
    with pytest.raises(TelemetryValidationError, match="5 MB"):
        read_telemetry_csv(b"a\n" + b"1" * MAX_UPLOAD_BYTES)


def test_rejects_duplicate_or_empty_headers():
    with pytest.raises(TelemetryValidationError, match="Duplicate"):
        read_telemetry_csv(b"Speed,speed\n1,2\n")
    with pytest.raises(TelemetryValidationError, match="non-empty"):
        read_telemetry_csv(b"speed,,rpm\n1,2,3\n")


def test_rejects_ragged_csv_and_binary_content():
    with pytest.raises(TelemetryValidationError, match="different number"):
        read_telemetry_csv(b"a,b\n1\n")
    with pytest.raises(TelemetryValidationError, match="null-byte"):
        read_telemetry_csv(b"a,b\n1,\x00\n")


def test_rejects_non_utf8_content():
    with pytest.raises(TelemetryValidationError, match="UTF-8"):
        read_telemetry_csv(b"speed\n\xff\n")
