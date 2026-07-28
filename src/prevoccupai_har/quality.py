"""Transparent signal-quality checks for reconstructed muscleBAN ACC data.

The recovered quality-assessment repository is treated as provenance evidence,
not executable authority.  This module expresses its main criteria as pure,
typed functions, corrects known implementation defects, and makes unresolved
calibration explicit through an ``UNDETERMINED`` status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatVector = NDArray[np.float64]
BooleanVector = NDArray[np.bool_]


class QualityStatus(StrEnum):
    """Possible outcomes for a criterion set or recording."""

    GOOD = "GOOD"
    BAD = "BAD"
    UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True)
class QualityThresholds:
    """Numerical thresholds used by the reconstructed quality checks."""

    sampling_rate_hz: int = 1000
    sequence_counter_modulus: int | None = 32768
    acceleration_range_g: float = 8.0
    gravity_m_s2: float = 9.80665
    missing_fraction_max: float = 0.01
    out_of_range_fraction_max: float = 0.0
    flatline_fraction_max: float = 0.01
    flatline_duration_seconds: float = 10.0
    flatline_absolute_tolerance_m_s2: float = 1e-6
    spike_event_fraction_max: float = 0.01
    spike_amplitude_threshold_m_s2: float = 4.0
    spike_max_width_samples: int = 3
    spike_local_std_max_m_s2: float = 2.0
    spike_local_window_milliseconds: float = 100.0
    startup_sample_count: int = 30
    startup_amplitude_threshold_m_s2: float = 10.0
    saturation_fraction_max: float = 0.01
    saturation_absolute_tolerance_m_s2: float = 1e-4
    minimum_variance_m_s2_squared: float | None = None

    def __post_init__(self) -> None:
        """Reject nonsensical thresholds before any signal is assessed."""
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.sequence_counter_modulus is not None and self.sequence_counter_modulus < 2:
            raise ValueError("sequence_counter_modulus must be at least two")
        if self.acceleration_range_g <= 0 or self.gravity_m_s2 <= 0:
            raise ValueError("physical range and gravity must be positive")
        for name in (
            "missing_fraction_max",
            "out_of_range_fraction_max",
            "flatline_fraction_max",
            "spike_event_fraction_max",
            "saturation_fraction_max",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "flatline_duration_seconds",
            "flatline_absolute_tolerance_m_s2",
            "spike_amplitude_threshold_m_s2",
            "spike_local_std_max_m_s2",
            "spike_local_window_milliseconds",
            "startup_amplitude_threshold_m_s2",
            "saturation_absolute_tolerance_m_s2",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.spike_max_width_samples < 1:
            raise ValueError("spike_max_width_samples must be at least one")
        if self.startup_sample_count < 0:
            raise ValueError("startup_sample_count must be non-negative")
        if (
            self.minimum_variance_m_s2_squared is not None
            and self.minimum_variance_m_s2_squared < 0
        ):
            raise ValueError("minimum_variance_m_s2_squared must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QualityThresholds":
        """Build thresholds from a JSON-compatible mapping."""
        return cls(**dict(values))

    @property
    def physical_limit_m_s2(self) -> float:
        """Return the positive accelerometer range endpoint in SI units."""
        return self.acceleration_range_g * self.gravity_m_s2


@dataclass(frozen=True)
class QualityAssessmentConfiguration:
    """Versioned metadata and thresholds for a quality reconstruction."""

    schema_version: int
    name: str
    authoritative: bool
    controls_inclusion: bool
    source: Mapping[str, Any]
    notes: tuple[str, ...]
    thresholds: QualityThresholds


@dataclass(frozen=True)
class SequenceGapReport:
    """Missing-sample evidence derived from a strictly increasing counter."""

    observed_samples: int
    expected_samples: int
    missing_samples: int
    missing_fraction: float
    gap_after_indices: tuple[int, ...]
    gap_sizes: tuple[int, ...]


@dataclass(frozen=True)
class SpikeReport:
    """Detected narrow impulse events and their marked samples."""

    event_count: int
    event_fraction: float
    sample_fraction: float
    event_start_indices: tuple[int, ...]
    sample_mask: BooleanVector


@dataclass(frozen=True)
class AxisQualityReport:
    """Quality metrics and decision for one accelerometer axis."""

    sample_count: int
    sequence_missing_samples: int
    missing_fraction: float
    nonfinite_fraction: float
    out_of_range_fraction: float
    flatline_sample_fraction: float
    spike_event_count: int
    spike_event_fraction: float
    spike_sample_fraction: float
    saturation_fraction: float
    variance_m_s2_squared: float
    variance_below_threshold: bool | None
    status: QualityStatus
    failed_criteria: tuple[str, ...]
    undetermined_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with stable field names."""
        return {
            "sample_count": self.sample_count,
            "sequence_missing_samples": self.sequence_missing_samples,
            "missing_fraction": self.missing_fraction,
            "nonfinite_fraction": self.nonfinite_fraction,
            "out_of_range_fraction": self.out_of_range_fraction,
            "flatline_sample_fraction": self.flatline_sample_fraction,
            "spike_event_count": self.spike_event_count,
            "spike_event_fraction": self.spike_event_fraction,
            "spike_sample_fraction": self.spike_sample_fraction,
            "saturation_fraction": self.saturation_fraction,
            "variance_m_s2_squared": self.variance_m_s2_squared,
            "variance_below_threshold": self.variance_below_threshold,
            "status": self.status.value,
            "failed_criteria": list(self.failed_criteria),
            "undetermined_criteria": list(self.undetermined_criteria),
        }


@dataclass(frozen=True)
class AccelerometerQualityReport:
    """Aggregate result for a three-axis accelerometer recording."""

    status: QualityStatus
    axes: Mapping[str, AxisQualityReport]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible aggregate report."""
        return {
            "status": self.status.value,
            "axes": {name: report.to_dict() for name, report in self.axes.items()},
        }


def load_quality_configuration(path: Path) -> QualityAssessmentConfiguration:
    """Load a versioned quality configuration from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return QualityAssessmentConfiguration(
        schema_version=int(payload["schema_version"]),
        name=str(payload["name"]),
        authoritative=bool(payload["authoritative"]),
        controls_inclusion=bool(payload["controls_inclusion"]),
        source=dict(payload["source"]),
        notes=tuple(str(note) for note in payload.get("notes", ())),
        thresholds=QualityThresholds.from_mapping(payload["thresholds"]),
    )


def _float_vector(values: ArrayLike, *, name: str) -> FloatVector:
    """Convert an array-like object to a non-empty one-dimensional vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    return vector


def detect_sequence_gaps(
    sequence_numbers: ArrayLike,
    *,
    modulus: int | None = None,
) -> SequenceGapReport:
    """Detect gaps and use the expected counter length as the denominator.

    The recovered implementation divides missing samples by the observed row
    count.  Here, a counter gap contributes to the expected count, so two
    missing samples among five expected samples are reported as 2/5.  When a
    modulus is supplied, the counter's maximum-to-zero transition is treated
    as one normal increment.
    """
    sequence = _float_vector(sequence_numbers, name="sequence_numbers")
    if not np.all(np.isfinite(sequence)):
        raise ValueError("sequence_numbers must be finite")
    rounded = np.rint(sequence)
    if not np.array_equal(sequence, rounded):
        raise ValueError("sequence_numbers must be integer-valued")
    integer_sequence = rounded.astype(np.int64)
    if modulus is None:
        differences = np.diff(integer_sequence)
        if np.any(differences <= 0):
            raise ValueError("sequence_numbers must be strictly increasing")
    else:
        if modulus < 2:
            raise ValueError("modulus must be at least two")
        if np.any(integer_sequence < 0) or np.any(integer_sequence >= modulus):
            raise ValueError("sequence_numbers must lie in [0, modulus)")
        differences = np.mod(np.diff(integer_sequence), modulus)
        if np.any(differences == 0):
            raise ValueError("sequence_numbers contain a repeated counter value")

    gap_indices = np.flatnonzero(differences > 1)
    gap_sizes_array = differences[gap_indices] - 1
    missing_samples = int(gap_sizes_array.sum())
    expected_samples = int(sequence.size + missing_samples)
    return SequenceGapReport(
        observed_samples=int(sequence.size),
        expected_samples=expected_samples,
        missing_samples=missing_samples,
        missing_fraction=missing_samples / expected_samples,
        gap_after_indices=tuple(int(index) for index in gap_indices),
        gap_sizes=tuple(int(size) for size in gap_sizes_array),
    )


def detect_out_of_range_mask(
    data: ArrayLike,
    *,
    physical_limit_m_s2: float,
) -> BooleanVector:
    """Mark finite values outside the symmetric physical sensor range."""
    vector = _float_vector(data, name="data")
    finite = np.isfinite(vector)
    return finite & (np.abs(vector) > physical_limit_m_s2)


def detect_saturation_mask(
    data: ArrayLike,
    *,
    physical_limit_m_s2: float,
    absolute_tolerance_m_s2: float,
) -> BooleanVector:
    """Mark finite values at either physical range endpoint."""
    vector = _float_vector(data, name="data")
    return np.isfinite(vector) & np.isclose(
        np.abs(vector),
        physical_limit_m_s2,
        rtol=0.0,
        atol=absolute_tolerance_m_s2,
    )


def _true_runs(mask: BooleanVector) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous true runs."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def detect_flatline_mask(
    data: ArrayLike,
    *,
    sampling_rate_hz: int,
    minimum_duration_seconds: float,
    absolute_tolerance_m_s2: float,
) -> BooleanVector:
    """Mark complete constant runs meeting a duration threshold."""
    vector = _float_vector(data, name="data")
    minimum_samples = int(np.ceil(minimum_duration_seconds * sampling_rate_hz))
    if minimum_samples < 2:
        raise ValueError("flatline duration must correspond to at least two samples")

    finite_pairs = np.isfinite(vector[:-1]) & np.isfinite(vector[1:])
    flat_edges = finite_pairs & (np.abs(np.diff(vector)) <= absolute_tolerance_m_s2)
    result = np.zeros(vector.size, dtype=bool)
    for edge_start, edge_stop in _true_runs(flat_edges):
        sample_count = edge_stop - edge_start + 1
        if sample_count >= minimum_samples:
            result[edge_start : edge_stop + 1] = True
    return result


def detect_impulsive_spikes(
    data: ArrayLike,
    *,
    sampling_rate_hz: int,
    amplitude_threshold_m_s2: float,
    maximum_width_samples: int,
    local_std_max_m_s2: float,
    local_window_milliseconds: float,
    startup_sample_count: int,
    startup_amplitude_threshold_m_s2: float,
) -> SpikeReport:
    """Detect narrow excursions that return to their local baseline.

    Candidate events are bounded by consecutive, opposite-direction jumps.
    They must return close to the preceding baseline within at most
    ``maximum_width_samples`` and occur in a locally low-variance region.
    This operational definition avoids the recovered implementation's
    contradictory minimum-width and maximum-width filters.
    """
    vector = _float_vector(data, name="data")
    finite_pairs = np.isfinite(vector[:-1]) & np.isfinite(vector[1:])
    differences = np.diff(vector)
    jump_indices = np.flatnonzero(
        finite_pairs & (np.abs(differences) >= amplitude_threshold_m_s2)
    )
    local_half_window = max(
        1,
        int(np.ceil(local_window_milliseconds * sampling_rate_hz / 1000.0)),
    )
    spike_mask = np.zeros(vector.size, dtype=bool)

    for first_jump, second_jump in zip(jump_indices[:-1], jump_indices[1:]):
        width = int(second_jump - first_jump)
        if width < 1 or width > maximum_width_samples:
            continue
        if differences[first_jump] * differences[second_jump] >= 0:
            continue

        event_start = int(first_jump + 1)
        event_stop = int(second_jump + 1)
        baseline_before = vector[first_jump]
        baseline_after = vector[second_jump + 1]
        if abs(baseline_after - baseline_before) > amplitude_threshold_m_s2:
            continue
        baseline = 0.5 * (baseline_before + baseline_after)
        if np.max(np.abs(vector[event_start:event_stop] - baseline)) < amplitude_threshold_m_s2:
            continue

        left = vector[max(0, event_start - local_half_window) : event_start]
        right = vector[event_stop : min(vector.size, event_stop + local_half_window)]
        local_context = np.concatenate((left, right))
        local_context = local_context[np.isfinite(local_context)]
        if local_context.size and np.std(local_context, ddof=0) > local_std_max_m_s2:
            continue
        spike_mask[event_start:event_stop] = True

    startup_stop = min(startup_sample_count, vector.size)
    startup_indices = np.flatnonzero(
        np.isfinite(vector[:startup_stop])
        & (np.abs(vector[:startup_stop]) > startup_amplitude_threshold_m_s2)
    )
    spike_mask[startup_indices] = True

    event_runs = _true_runs(spike_mask)
    event_starts = tuple(start for start, _ in event_runs)
    event_count = len(event_runs)
    return SpikeReport(
        event_count=event_count,
        event_fraction=event_count / vector.size,
        sample_fraction=float(np.mean(spike_mask)),
        event_start_indices=event_starts,
        sample_mask=spike_mask,
    )


def assess_axis_quality(
    data: ArrayLike,
    sequence_numbers: ArrayLike,
    thresholds: QualityThresholds,
) -> AxisQualityReport:
    """Assess one axis and report all failed or unresolved criteria."""
    vector = _float_vector(data, name="data")
    gaps = detect_sequence_gaps(
        sequence_numbers,
        modulus=thresholds.sequence_counter_modulus,
    )
    if vector.size != gaps.observed_samples:
        raise ValueError("data and sequence_numbers must have equal length")

    finite = np.isfinite(vector)
    nonfinite_samples = int((~finite).sum())
    missing_fraction = (
        gaps.missing_samples + nonfinite_samples
    ) / gaps.expected_samples
    nonfinite_fraction = nonfinite_samples / vector.size

    out_of_range_mask = detect_out_of_range_mask(
        vector,
        physical_limit_m_s2=thresholds.physical_limit_m_s2,
    )
    flatline_mask = detect_flatline_mask(
        vector,
        sampling_rate_hz=thresholds.sampling_rate_hz,
        minimum_duration_seconds=thresholds.flatline_duration_seconds,
        absolute_tolerance_m_s2=thresholds.flatline_absolute_tolerance_m_s2,
    )
    spikes = detect_impulsive_spikes(
        vector,
        sampling_rate_hz=thresholds.sampling_rate_hz,
        amplitude_threshold_m_s2=thresholds.spike_amplitude_threshold_m_s2,
        maximum_width_samples=thresholds.spike_max_width_samples,
        local_std_max_m_s2=thresholds.spike_local_std_max_m_s2,
        local_window_milliseconds=thresholds.spike_local_window_milliseconds,
        startup_sample_count=thresholds.startup_sample_count,
        startup_amplitude_threshold_m_s2=thresholds.startup_amplitude_threshold_m_s2,
    )
    saturation_mask = detect_saturation_mask(
        vector,
        physical_limit_m_s2=thresholds.physical_limit_m_s2,
        absolute_tolerance_m_s2=thresholds.saturation_absolute_tolerance_m_s2,
    )

    finite_values = vector[finite]
    variance = (
        float(np.var(finite_values, ddof=0)) if finite_values.size else float("nan")
    )
    if thresholds.minimum_variance_m_s2_squared is None:
        variance_below_threshold: bool | None = None
    else:
        variance_below_threshold = bool(
            not np.isfinite(variance)
            or variance < thresholds.minimum_variance_m_s2_squared
        )

    out_of_range_fraction = float(np.mean(out_of_range_mask))
    flatline_fraction = float(np.mean(flatline_mask))
    saturation_fraction = float(np.mean(saturation_mask))
    failed: list[str] = []
    if missing_fraction > thresholds.missing_fraction_max:
        failed.append("missing_fraction")
    if out_of_range_fraction > thresholds.out_of_range_fraction_max:
        failed.append("out_of_range_fraction")
    if flatline_fraction > thresholds.flatline_fraction_max:
        failed.append("flatline_sample_fraction")
    if spikes.event_fraction > thresholds.spike_event_fraction_max:
        failed.append("spike_event_fraction")
    if saturation_fraction > thresholds.saturation_fraction_max:
        failed.append("saturation_fraction")
    if variance_below_threshold is True:
        failed.append("variance_below_threshold")

    undetermined: list[str] = []
    if variance_below_threshold is None:
        undetermined.append("minimum_variance_not_calibrated")

    if failed:
        status = QualityStatus.BAD
    elif undetermined:
        status = QualityStatus.UNDETERMINED
    else:
        status = QualityStatus.GOOD

    return AxisQualityReport(
        sample_count=int(vector.size),
        sequence_missing_samples=gaps.missing_samples,
        missing_fraction=float(missing_fraction),
        nonfinite_fraction=float(nonfinite_fraction),
        out_of_range_fraction=out_of_range_fraction,
        flatline_sample_fraction=flatline_fraction,
        spike_event_count=spikes.event_count,
        spike_event_fraction=spikes.event_fraction,
        spike_sample_fraction=spikes.sample_fraction,
        saturation_fraction=saturation_fraction,
        variance_m_s2_squared=variance,
        variance_below_threshold=variance_below_threshold,
        status=status,
        failed_criteria=tuple(failed),
        undetermined_criteria=tuple(undetermined),
    )


def assess_accelerometer_quality(
    samples: ArrayLike,
    sequence_numbers: ArrayLike,
    thresholds: QualityThresholds,
    *,
    axis_names: Sequence[str] = ("x", "y", "z"),
) -> AccelerometerQualityReport:
    """Assess a sample-by-axis matrix and aggregate axis decisions."""
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("samples must be a two-dimensional sample-by-axis matrix")
    if matrix.shape[1] != len(axis_names):
        raise ValueError("axis_names must match the number of sample columns")
    if len(set(axis_names)) != len(axis_names):
        raise ValueError("axis_names must be unique")

    axes = {
        name: assess_axis_quality(matrix[:, index], sequence_numbers, thresholds)
        for index, name in enumerate(axis_names)
    }
    statuses = {report.status for report in axes.values()}
    if QualityStatus.BAD in statuses:
        status = QualityStatus.BAD
    elif QualityStatus.UNDETERMINED in statuses:
        status = QualityStatus.UNDETERMINED
    else:
        status = QualityStatus.GOOD
    return AccelerometerQualityReport(status=status, axes=axes)
