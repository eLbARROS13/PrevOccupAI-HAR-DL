"""Tests for provenance-bounded temporal-consistency diagnostics."""

from __future__ import annotations

import pytest

from prevoccupai_har.temporal_evaluation import evaluate_temporal_predictions
from prevoccupai_har.windowing import WindowMetadata


def _metadata(
    *,
    participant: str,
    recording: str,
    label: str,
    starts: tuple[int, ...],
    stream: str = "SYNTHETIC_LEFT",
) -> tuple[WindowMetadata, ...]:
    return tuple(
        WindowMetadata(
            subject_id=participant,
            recording_id=recording,
            main_label=label,
            sub_activity_label=f"synthetic_{label}",
            sensor_stream_id=stream,
            sensor_side="synthetic",
            start_sample=start,
            end_sample_exclusive=start + 5_000,
            preprocessing_status="synthetic",
            quality_status="synthetic",
        )
        for start in starts
    )


def test_temporal_metrics_quantify_one_window_fragmentation() -> None:
    records = _metadata(
        participant="SYNTHETIC_P001",
        recording="R1",
        label="sitting",
        starts=(0, 2_500, 5_000, 7_500, 10_000),
    ) + _metadata(
        participant="SYNTHETIC_P001",
        recording="R2",
        label="walking",
        starts=(0, 2_500),
    )
    predictions = (
        "sitting",
        "sitting",
        "standing",
        "sitting",
        "sitting",
        "walking",
        "walking",
    )

    evaluation = evaluate_temporal_predictions(
        predictions,
        records,
        ("sitting", "standing", "walking"),
        expected_step_size_samples=2_500,
    )

    metrics = evaluation.overall_metrics
    assert metrics.window_count == 7
    assert metrics.contiguous_sequence_count == 2
    assert metrics.adjacent_pair_count == 5
    assert metrics.reference_transition_count == 0
    assert metrics.predicted_transition_count == 2
    assert metrics.transition_event_disagreement_count == 2
    assert metrics.predicted_transition_rate == pytest.approx(0.4)
    assert metrics.transition_event_disagreement_rate == pytest.approx(0.4)
    assert metrics.reference_run_count == 2
    assert metrics.predicted_run_count == 4
    assert metrics.excess_predicted_run_count == 2
    assert metrics.short_predicted_run_count == 1
    assert metrics.short_predicted_run_fraction == pytest.approx(0.25)
    assert metrics.median_predicted_run_length_windows == pytest.approx(2.0)


def test_gap_and_recording_boundaries_are_not_bridged() -> None:
    records = _metadata(
        participant="SYNTHETIC_P001",
        recording="R1",
        label="sitting",
        starts=(0, 2_500, 7_500),
    )

    evaluation = evaluate_temporal_predictions(
        ("sitting", "sitting", "standing"),
        records,
        ("sitting", "standing", "walking"),
        expected_step_size_samples=2_500,
    )

    metrics = evaluation.overall_metrics
    assert metrics.contiguous_sequence_count == 2
    assert metrics.adjacent_pair_count == 1
    assert metrics.predicted_transition_count == 0


def test_reference_and_prediction_transitions_are_distinguished() -> None:
    records = (
        _metadata(
            participant="SYNTHETIC_P001",
            recording="R1",
            label="sitting",
            starts=(0,),
        )
        + _metadata(
            participant="SYNTHETIC_P001",
            recording="R1",
            label="standing",
            starts=(2_500, 5_000),
        )
    )

    evaluation = evaluate_temporal_predictions(
        ("sitting", "sitting", "standing"),
        records,
        ("sitting", "standing", "walking"),
        expected_step_size_samples=2_500,
    )

    metrics = evaluation.overall_metrics
    assert metrics.reference_transition_count == 1
    assert metrics.predicted_transition_count == 1
    assert metrics.transition_event_disagreement_count == 2
    assert metrics.excess_predicted_run_count == 0


def test_participant_metrics_never_bridge_participants() -> None:
    records = _metadata(
        participant="SYNTHETIC_P001",
        recording="R1",
        label="sitting",
        starts=(0, 2_500),
    ) + _metadata(
        participant="SYNTHETIC_P002",
        recording="R1",
        label="walking",
        starts=(0, 2_500),
    )

    evaluation = evaluate_temporal_predictions(
        ("sitting", "standing", "walking", "walking"),
        records,
        ("sitting", "standing", "walking"),
        expected_step_size_samples=2_500,
    )

    assert set(evaluation.per_participant_metrics) == {
        "SYNTHETIC_P001",
        "SYNTHETIC_P002",
    }
    assert evaluation.overall_metrics.adjacent_pair_count == 2
    assert evaluation.per_participant_metrics[
        "SYNTHETIC_P001"
    ].predicted_transition_count == 1
    assert evaluation.per_participant_metrics[
        "SYNTHETIC_P002"
    ].predicted_transition_count == 0


def test_isolated_windows_report_undefined_transition_rates() -> None:
    records = _metadata(
        participant="SYNTHETIC_P001",
        recording="R1",
        label="sitting",
        starts=(0, 5_000),
    )

    evaluation = evaluate_temporal_predictions(
        ("sitting", "standing"),
        records,
        ("sitting", "standing", "walking"),
        expected_step_size_samples=2_500,
    )

    assert evaluation.overall_metrics.adjacent_pair_count == 0
    assert evaluation.overall_metrics.predicted_transition_rate is None
    assert evaluation.overall_metrics.transition_event_disagreement_rate is None


def test_invalid_prediction_or_window_geometry_is_rejected() -> None:
    records = _metadata(
        participant="SYNTHETIC_P001",
        recording="R1",
        label="sitting",
        starts=(0, 1_000),
    )
    with pytest.raises(ValueError, match="overlap more than declared"):
        evaluate_temporal_predictions(
            ("sitting", "sitting"),
            records,
            ("sitting", "standing", "walking"),
            expected_step_size_samples=2_500,
        )
    with pytest.raises(ValueError, match="outside the declared class set"):
        evaluate_temporal_predictions(
            ("unknown", "sitting"),
            records,
            ("sitting", "standing", "walking"),
            expected_step_size_samples=1_000,
        )
    with pytest.raises(ValueError, match="One prediction"):
        evaluate_temporal_predictions(
            ("sitting",),
            records,
            ("sitting", "standing", "walking"),
            expected_step_size_samples=1_000,
        )
