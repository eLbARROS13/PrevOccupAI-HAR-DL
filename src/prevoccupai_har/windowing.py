"""Deterministic sliding windows with explicit provenance metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WindowMetadata:
    """Provenance required to trace a window to its source recording."""

    subject_id: str
    recording_id: str
    main_label: str
    sub_activity_label: str
    sensor_stream_id: str
    sensor_side: str
    start_sample: int
    end_sample_exclusive: int
    preprocessing_status: str
    quality_status: str

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable metadata record."""
        return asdict(self)


def iter_window_bounds(
    n_samples: int,
    *,
    window_size_samples: int,
    step_size_samples: int,
) -> Iterator[tuple[int, int]]:
    """Yield complete half-open window bounds; partial tails are never emitted."""
    if n_samples < 0:
        raise ValueError("Sample count cannot be negative")
    if window_size_samples <= 0 or step_size_samples <= 0:
        raise ValueError("Window and step sizes must be positive")
    if n_samples < window_size_samples:
        return
    for start in range(0, n_samples - window_size_samples + 1, step_size_samples):
        yield start, start + window_size_samples


def create_windows(
    signal: NDArray[np.floating],
    *,
    window_size_samples: int,
    step_size_samples: int,
    subject_id: str,
    recording_id: str,
    main_label: str,
    sub_activity_label: str,
    sensor_stream_id: str,
    sensor_side: str,
    preprocessing_status: str,
    quality_status: str,
) -> tuple[NDArray[np.floating], tuple[WindowMetadata, ...]]:
    """Create complete windows and aligned provenance records from one signal."""
    values = np.asarray(signal)
    if values.ndim != 2:
        raise ValueError("Signal must have shape (samples, channels)")
    bounds = tuple(
        iter_window_bounds(
            values.shape[0],
            window_size_samples=window_size_samples,
            step_size_samples=step_size_samples,
        )
    )
    if not bounds:
        empty = np.empty((0, window_size_samples, values.shape[1]), dtype=values.dtype)
        return empty, ()

    windows = np.stack([values[start:end] for start, end in bounds])
    metadata = tuple(
        WindowMetadata(
            subject_id=subject_id,
            recording_id=recording_id,
            main_label=main_label,
            sub_activity_label=sub_activity_label,
            sensor_stream_id=sensor_stream_id,
            sensor_side=sensor_side,
            start_sample=start,
            end_sample_exclusive=end,
            preprocessing_status=preprocessing_status,
            quality_status=quality_status,
        )
        for start, end in bounds
    )
    return windows, metadata
