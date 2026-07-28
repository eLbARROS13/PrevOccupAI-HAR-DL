"""Temporal-consistency diagnostics for ordered HAR window predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Sequence

import numpy as np

from .evaluation import confusion_matrix_from_predictions
from .windowing import WindowMetadata


@dataclass(frozen=True)
class TemporalConsistencyMetrics:
    """Fragmentation and transition diagnostics over contiguous window sequences."""

    window_count: int
    contiguous_sequence_count: int
    adjacent_pair_count: int
    reference_transition_count: int
    predicted_transition_count: int
    transition_event_disagreement_count: int
    reference_transition_rate: float | None
    predicted_transition_rate: float | None
    transition_event_disagreement_rate: float | None
    reference_run_count: int
    predicted_run_count: int
    excess_predicted_run_count: int
    short_predicted_run_count: int
    short_predicted_run_fraction: float
    median_predicted_run_length_windows: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class TemporalPredictionEvaluation:
    """Overall and participant-level temporal diagnostics for one prediction vector."""

    class_labels: tuple[str, ...]
    window_size_samples: int
    expected_step_size_samples: int
    short_run_max_windows: int
    overall_metrics: TemporalConsistencyMetrics
    per_participant_metrics: dict[str, TemporalConsistencyMetrics]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "class_labels": list(self.class_labels),
            "window_size_samples": self.window_size_samples,
            "expected_step_size_samples": self.expected_step_size_samples,
            "short_run_max_windows": self.short_run_max_windows,
            "overall_metrics": self.overall_metrics.as_dict(),
            "per_participant_metrics": {
                participant: metrics.as_dict()
                for participant, metrics in self.per_participant_metrics.items()
            },
        }


def _validate_metadata(
    metadata: tuple[WindowMetadata, ...],
    *,
    expected_step_size_samples: int,
) -> int:
    if not metadata:
        raise ValueError("Temporal evaluation requires at least one window")
    if expected_step_size_samples <= 0:
        raise ValueError("Expected step size must be positive")
    window_sizes = {
        record.end_sample_exclusive - record.start_sample for record in metadata
    }
    if len(window_sizes) != 1 or next(iter(window_sizes)) <= 0:
        raise ValueError("All temporal-evaluation windows must have one positive length")
    for record in metadata:
        if not all(
            (
                record.subject_id,
                record.recording_id,
                record.main_label,
                record.sensor_stream_id,
            )
        ):
            raise ValueError("Temporal sequence provenance fields cannot be empty")
        if record.start_sample < 0:
            raise ValueError("Window starts cannot be negative")
    return next(iter(window_sizes))


def _build_contiguous_sequences(
    metadata: tuple[WindowMetadata, ...],
    *,
    expected_step_size_samples: int,
) -> tuple[tuple[int, ...], ...]:
    grouped_indices: dict[tuple[str, str, str], list[int]] = {}
    for index, record in enumerate(metadata):
        sequence_key = (
            record.subject_id,
            record.recording_id,
            record.sensor_stream_id,
        )
        grouped_indices.setdefault(sequence_key, []).append(index)

    sequences: list[tuple[int, ...]] = []
    for sequence_key in sorted(grouped_indices):
        ordered = sorted(
            grouped_indices[sequence_key],
            key=lambda index: metadata[index].start_sample,
        )
        current = [ordered[0]]
        for previous_index, current_index in pairwise(ordered):
            step = (
                metadata[current_index].start_sample
                - metadata[previous_index].start_sample
            )
            if step < expected_step_size_samples:
                raise ValueError(
                    "Windows within a recording stream overlap more than declared or duplicate"
                )
            if step == expected_step_size_samples:
                current.append(current_index)
            else:
                sequences.append(tuple(current))
                current = [current_index]
        sequences.append(tuple(current))
    return tuple(sequences)


def _run_lengths(labels: Sequence[str]) -> tuple[int, ...]:
    if not labels:
        return ()
    lengths: list[int] = []
    current_label = labels[0]
    current_length = 1
    for label in labels[1:]:
        if label == current_label:
            current_length += 1
        else:
            lengths.append(current_length)
            current_label = label
            current_length = 1
    lengths.append(current_length)
    return tuple(lengths)


def _metrics_for_sequences(
    reference_labels: tuple[str, ...],
    predicted_labels: tuple[str, ...],
    sequences: tuple[tuple[int, ...], ...],
    *,
    short_run_max_windows: int,
) -> TemporalConsistencyMetrics:
    reference_transition_count = 0
    predicted_transition_count = 0
    transition_event_disagreement_count = 0
    adjacent_pair_count = 0
    reference_run_lengths: list[int] = []
    predicted_run_lengths: list[int] = []

    for sequence in sequences:
        sequence_reference = tuple(reference_labels[index] for index in sequence)
        sequence_predictions = tuple(predicted_labels[index] for index in sequence)
        reference_run_lengths.extend(_run_lengths(sequence_reference))
        predicted_run_lengths.extend(_run_lengths(sequence_predictions))
        for left, right in pairwise(sequence):
            reference_changed = reference_labels[left] != reference_labels[right]
            prediction_changed = predicted_labels[left] != predicted_labels[right]
            reference_transition_count += int(reference_changed)
            predicted_transition_count += int(prediction_changed)
            transition_event_disagreement_count += int(
                reference_changed != prediction_changed
            )
            adjacent_pair_count += 1

    def _rate(count: int) -> float | None:
        if adjacent_pair_count == 0:
            return None
        return count / adjacent_pair_count

    short_run_count = sum(
        length <= short_run_max_windows for length in predicted_run_lengths
    )
    return TemporalConsistencyMetrics(
        window_count=sum(len(sequence) for sequence in sequences),
        contiguous_sequence_count=len(sequences),
        adjacent_pair_count=adjacent_pair_count,
        reference_transition_count=reference_transition_count,
        predicted_transition_count=predicted_transition_count,
        transition_event_disagreement_count=transition_event_disagreement_count,
        reference_transition_rate=_rate(reference_transition_count),
        predicted_transition_rate=_rate(predicted_transition_count),
        transition_event_disagreement_rate=_rate(
            transition_event_disagreement_count
        ),
        reference_run_count=len(reference_run_lengths),
        predicted_run_count=len(predicted_run_lengths),
        excess_predicted_run_count=(
            len(predicted_run_lengths) - len(reference_run_lengths)
        ),
        short_predicted_run_count=short_run_count,
        short_predicted_run_fraction=short_run_count / len(predicted_run_lengths),
        median_predicted_run_length_windows=float(np.median(predicted_run_lengths)),
    )


def evaluate_temporal_predictions(
    predicted_labels: Sequence[str],
    metadata: Sequence[WindowMetadata],
    class_labels: Sequence[str],
    *,
    expected_step_size_samples: int,
    short_run_max_windows: int = 1,
) -> TemporalPredictionEvaluation:
    """Evaluate ordered predictions without bridging provenance gaps or streams.

    Reference labels come from window metadata. Adjacency is recognised only when
    two windows share participant, recording, and sensor-stream provenance and their
    start samples differ by exactly ``expected_step_size_samples``.
    """
    records = tuple(metadata)
    predictions = tuple(map(str, predicted_labels))
    labels = tuple(map(str, class_labels))
    if len(predictions) != len(records):
        raise ValueError("One prediction is required per metadata record")
    if short_run_max_windows <= 0:
        raise ValueError("Short-run threshold must be positive")
    window_size_samples = _validate_metadata(
        records,
        expected_step_size_samples=expected_step_size_samples,
    )
    reference_labels = tuple(record.main_label for record in records)
    confusion_matrix_from_predictions(reference_labels, predictions, labels)
    sequences = _build_contiguous_sequences(
        records,
        expected_step_size_samples=expected_step_size_samples,
    )
    overall = _metrics_for_sequences(
        reference_labels,
        predictions,
        sequences,
        short_run_max_windows=short_run_max_windows,
    )

    per_participant: dict[str, TemporalConsistencyMetrics] = {}
    for participant in sorted({record.subject_id for record in records}):
        participant_indices = tuple(
            index
            for index, record in enumerate(records)
            if record.subject_id == participant
        )
        participant_metadata = tuple(records[index] for index in participant_indices)
        participant_reference = tuple(
            reference_labels[index] for index in participant_indices
        )
        participant_predictions = tuple(
            predictions[index] for index in participant_indices
        )
        participant_sequences = _build_contiguous_sequences(
            participant_metadata,
            expected_step_size_samples=expected_step_size_samples,
        )
        per_participant[participant] = _metrics_for_sequences(
            participant_reference,
            participant_predictions,
            participant_sequences,
            short_run_max_windows=short_run_max_windows,
        )

    return TemporalPredictionEvaluation(
        class_labels=labels,
        window_size_samples=window_size_samples,
        expected_step_size_samples=expected_step_size_samples,
        short_run_max_windows=short_run_max_windows,
        overall_metrics=overall,
        per_participant_metrics=per_participant,
    )
