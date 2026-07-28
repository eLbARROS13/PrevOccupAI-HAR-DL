"""Tests for complete-window construction and metadata alignment."""

import numpy as np

from prevoccupai_har.windowing import create_windows, iter_window_bounds


def test_exact_50_percent_overlap_bounds() -> None:
    bounds = tuple(
        iter_window_bounds(
            10_000,
            window_size_samples=5_000,
            step_size_samples=2_500,
        )
    )

    assert bounds == ((0, 5_000), (2_500, 7_500), (5_000, 10_000))


def test_partial_tail_is_not_emitted() -> None:
    bounds = tuple(
        iter_window_bounds(
            9_999,
            window_size_samples=5_000,
            step_size_samples=2_500,
        )
    )

    assert bounds == ((0, 5_000), (2_500, 7_500))


def test_windows_and_provenance_stay_aligned() -> None:
    signal = np.arange(30, dtype=np.float32).reshape(10, 3)

    windows, metadata = create_windows(
        signal,
        window_size_samples=4,
        step_size_samples=2,
        subject_id="P003",
        recording_id="recording-1",
        main_label="walking",
        sub_activity_label="walk_medium",
        sensor_stream_id="stream-1",
        sensor_side="unverified",
        preprocessing_status="raw",
        quality_status="accepted",
    )

    assert windows.shape == (4, 4, 3)
    assert len(metadata) == windows.shape[0]
    assert all(item.sensor_side == "unverified" for item in metadata)
    assert [(item.start_sample, item.end_sample_exclusive) for item in metadata] == [
        (0, 4),
        (2, 6),
        (4, 8),
        (6, 10),
    ]
    np.testing.assert_array_equal(windows[1], signal[2:6])
