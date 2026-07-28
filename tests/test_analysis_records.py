"""Tests for deterministic analyses derived from prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prevoccupai_har.analysis_records import (
    build_prediction_analysis_record,
    load_prediction_analysis_record,
    write_prediction_analysis_record,
)


def _prediction_artifact(path: Path) -> None:
    rows = [
        {
            "window_index": 0,
            "participant_id": "SYNTHETIC_VALIDATION_A",
            "recording_key_sha256": "1" * 64,
            "sensor_stream_key_sha256": "2" * 64,
            "main_label": "sitting",
            "sub_activity_label": "synthetic_sitting",
            "sensor_side": "synthetic",
            "start_sample": 0,
            "end_sample_exclusive": 5000,
            "preprocessing_status": "synthetic",
            "quality_status": "synthetic",
            "predicted_label": "sitting",
            "logits": [3.0, 1.0, 0.0],
        },
        {
            "window_index": 1,
            "participant_id": "SYNTHETIC_VALIDATION_A",
            "recording_key_sha256": "1" * 64,
            "sensor_stream_key_sha256": "2" * 64,
            "main_label": "sitting",
            "sub_activity_label": "synthetic_sitting",
            "sensor_side": "synthetic",
            "start_sample": 2500,
            "end_sample_exclusive": 7500,
            "preprocessing_status": "synthetic",
            "quality_status": "synthetic",
            "predicted_label": "standing",
            "logits": [1.0, 3.0, 0.0],
        },
        {
            "window_index": 2,
            "participant_id": "SYNTHETIC_VALIDATION_A",
            "recording_key_sha256": "1" * 64,
            "sensor_stream_key_sha256": "2" * 64,
            "main_label": "walking",
            "sub_activity_label": "synthetic_walking",
            "sensor_side": "synthetic",
            "start_sample": 5000,
            "end_sample_exclusive": 10000,
            "preprocessing_status": "synthetic",
            "quality_status": "synthetic",
            "predicted_label": "walking",
            "logits": [0.0, 1.0, 4.0],
        },
    ]
    from prevoccupai_har.provenance import sha256_canonical_json

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "synthetic-predictions-1",
                "created_at_utc": "2026-07-15T19:00:00Z",
                "experiment_id": "synthetic-experiment",
                "purpose": "synthetic_validation",
                "scientific_result": False,
                "holdout_accessed": False,
                "source_revision": "unversioned_workspace_software_test",
                "training_run_id": "synthetic-training-1",
                "training_result_sha256": "a" * 64,
                "model_configuration_sha256": "b" * 64,
                "learned_preprocessing_sha256": "c" * 64,
                "model_state_sha256": "d" * 64,
                "class_labels": ["sitting", "standing", "walking"],
                "validation_subjects": ["SYNTHETIC_VALIDATION_A"],
                "logit_dtype": "float32",
                "window_count": 3,
                "prediction_payload_sha256": sha256_canonical_json(rows),
                "windows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_analysis_round_trip_and_exact_metrics(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "analysis.json"
    _prediction_artifact(predictions)

    record = build_prediction_analysis_record(
        analysis_id="synthetic-analysis-1",
        created_at_utc="2026-07-15T19:10:00Z",
        prediction_artifact_path=predictions,
        calibration_bin_count=5,
        expected_step_size_samples=2500,
    )
    write_prediction_analysis_record(output, record)
    loaded = load_prediction_analysis_record(output)

    assert loaded.window_count == 3
    assert loaded.participant_count == 1
    assert loaded.analysis_payload["classification"]["overall_metrics"][
        "accuracy"
    ] == pytest.approx(2 / 3)
    assert loaded.analysis_payload["calibration"]["requested_bin_count"] == 5
    assert loaded.analysis_payload["temporal"]["overall_metrics"][
        "predicted_transition_count"
    ] == 2
    assert loaded.analysis_payload["temporal"]["overall_metrics"][
        "reference_transition_count"
    ] == 1
    assert loaded.scientific_result is False
    assert loaded.holdout_accessed is False
    with pytest.raises(FileExistsError):
        write_prediction_analysis_record(output, record)


def test_analysis_payload_is_deterministic(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    _prediction_artifact(predictions)
    first = build_prediction_analysis_record(
        analysis_id="synthetic-analysis-1",
        created_at_utc="2026-07-15T19:10:00Z",
        prediction_artifact_path=predictions,
        calibration_bin_count=5,
        expected_step_size_samples=2500,
    )
    second = build_prediction_analysis_record(
        analysis_id="synthetic-analysis-2",
        created_at_utc="2026-07-15T19:11:00Z",
        prediction_artifact_path=predictions,
        calibration_bin_count=5,
        expected_step_size_samples=2500,
    )

    assert first.analysis_payload == second.analysis_payload
    assert first.analysis_payload_sha256 == second.analysis_payload_sha256


def test_tampered_analysis_payload_is_rejected(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "analysis.json"
    _prediction_artifact(predictions)
    record = build_prediction_analysis_record(
        analysis_id="synthetic-analysis-1",
        created_at_utc="2026-07-15T19:10:00Z",
        prediction_artifact_path=predictions,
        expected_step_size_samples=2500,
    )
    write_prediction_analysis_record(output, record)
    decoded = json.loads(output.read_text(encoding="utf-8"))
    decoded["analysis_payload"]["classification"]["overall_metrics"][
        "accuracy"
    ] = 1.0
    output.write_text(json.dumps(decoded), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_prediction_analysis_record(output)


def test_invalid_analysis_settings_are_rejected(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    _prediction_artifact(predictions)
    with pytest.raises(ValueError, match="At least two calibration bins"):
        build_prediction_analysis_record(
            analysis_id="synthetic-analysis-1",
            created_at_utc="2026-07-15T19:10:00Z",
            prediction_artifact_path=predictions,
            calibration_bin_count=1,
            expected_step_size_samples=2500,
        )
