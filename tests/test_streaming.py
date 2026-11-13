from tether.streaming import StreamProcessor, TelemetryPacket


def packet(seq, timestamp):
    return TelemetryPacket(seq, timestamp, {"speed_kph": 120.0, "rpm": 7000.0})


def test_stream_emits_fixed_size_windows_and_bounds_memory():
    processor = StreamProcessor(sample_rate_hz=20, window_size=8)
    results = [processor.ingest(packet(index, index * 0.05)) for index in range(17)]
    assert sum(result.completed_window is not None for result in results) == 2
    assert processor.snapshot()["buffered_samples"] == 1
    assert processor.snapshot()["accepted_packets"] == 17


def test_stream_detects_sequence_and_timestamp_gaps():
    processor = StreamProcessor(sample_rate_hz=20, window_size=8)
    assert processor.ingest(packet(0, 0.0)).accepted
    assert processor.ingest(packet(1, 0.05)).accepted
    sequence_gap = processor.ingest(packet(3, 0.15))
    assert sequence_gap.accepted
    assert sequence_gap.diagnostics[0]["code"] == "SEQUENCE_GAP"
    timestamp_gap = processor.ingest(packet(4, 0.40))
    assert timestamp_gap.accepted
    assert any(item["code"] == "TIMESTAMP_GAP" for item in timestamp_gap.diagnostics)
    assert processor.snapshot()["sequence_gaps"] == 1
    assert processor.snapshot()["timestamp_gaps"] == 1


def test_stream_rejects_duplicate_out_of_order_and_invalid_packets():
    processor = StreamProcessor(sample_rate_hz=20, window_size=8)
    assert processor.ingest(packet(10, 1.0)).accepted
    duplicate = processor.ingest(packet(10, 1.05))
    old = processor.ingest(packet(9, 1.02))
    bad = processor.ingest(TelemetryPacket(11, 1.1, {"rpm": float("nan")}))
    assert not duplicate.accepted and duplicate.diagnostics[0]["code"] == "DUPLICATE_SEQUENCE"
    assert not old.accepted and old.diagnostics[0]["code"] == "OUT_OF_ORDER_SEQUENCE"
    assert not bad.accepted and bad.diagnostics[0]["code"] == "INVALID_CHANNEL_VALUE"
    snapshot = processor.snapshot()
    assert snapshot["duplicate_packets"] == 1
    assert snapshot["out_of_order_packets"] == 1
    assert snapshot["rejected_packets"] == 3


def test_stream_rejects_boolean_timestamp_and_invalid_gap_factor():
    processor = StreamProcessor(sample_rate_hz=20, window_size=8)
    result = processor.ingest(TelemetryPacket(0, True, {"speed_kph": 100.0}))
    assert not result.accepted
    assert result.diagnostics[0]["code"] == "INVALID_TIMESTAMP"
    import pytest
    with pytest.raises(ValueError, match="timestamp_gap_factor"):
        StreamProcessor(sample_rate_hz=20, timestamp_gap_factor=0.5)
