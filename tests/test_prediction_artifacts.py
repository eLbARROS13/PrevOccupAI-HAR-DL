"""Tests for immutable, development-only prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from prevoccupai_har.prediction_artifacts import (
    build_prediction_artifact_record,
    load_prediction_artifact_record,
    predict_logits,
    sha256_model_state,
    write_prediction_artifact_record,
)
from prevoccupai_har.training import TrainingPurpose, TrainingRunScope
from prevoccupai_har.windowing import WindowMetadata


def _model() -> nn.Module:
    model = nn.Sequential(nn.Flatten(), nn.Linear(6, 3, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            )
        )
    return model


def _metadata() -> tuple[WindowMetadata, ...]:
    labels = ("sitting", "standing", "walking")
    return tuple(
        WindowMetadata(
            subject_id="SYNTHETIC_VALIDATION_A",
            recording_id="PRIVATE_RECORDING_NAME",
            main_label=label,
            sub_activity_label=f"synthetic_{label}",
            sensor_stream_id="PRIVATE_DEVICE_STREAM",
            sensor_side="synthetic",
            start_sample=index * 2,
            end_sample_exclusive=index * 2 + 2,
            preprocessing_status="synthetic",
            quality_status="synthetic",
        )
        for index, label in enumerate(labels)
    )


def _training_result(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": "synthetic-training-1",
                "experiment_id": "synthetic-experiment",
                "purpose": "synthetic_validation",
                "scientific_result": False,
                "holdout_accessed": False,
                "source_revision": "unversioned_workspace_software_test",
                "model_configuration_sha256": "a" * 64,
                "learned_preprocessing_sha256": "b" * 64,
                "validation_subjects": ["SYNTHETIC_VALIDATION_A"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_ordered_inference_is_immutable_and_restores_training_mode() -> None:
    model = _model()
    model.train()
    inputs = np.asarray(
        [
            [[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [4.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    logits = predict_logits(
        model,
        inputs,
        expected_output_classes=3,
        batch_size=2,
    )

    assert model.training is True
    assert logits.shape == (3, 3)
    assert logits.flags.writeable is False
    np.testing.assert_array_equal(np.argmax(logits, axis=1), [0, 1, 2])


def test_model_state_hash_changes_with_tensor_content() -> None:
    model = _model()
    first = sha256_model_state(model)
    with torch.no_grad():
        model[1].weight[0, 0] += 1.0
    second = sha256_model_state(model)
    assert first != second
    assert len(sha256_model_state(nn.BatchNorm1d(3))) == 64


def test_prediction_artifact_round_trip_hides_raw_sequence_names(tmp_path: Path) -> None:
    training_path = tmp_path / "training.json"
    output_path = tmp_path / "predictions.json"
    _training_result(training_path)
    model = _model()
    logits = np.asarray(
        [[3.0, 1.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 5.0]],
        dtype=np.float32,
    )
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=("SYNTHETIC_TRAIN_A",),
        validation_subjects=("SYNTHETIC_VALIDATION_A",),
    )

    record = build_prediction_artifact_record(
        run_id="synthetic-predictions-1",
        created_at_utc="2026-07-15T19:00:00Z",
        logits=logits,
        metadata=_metadata(),
        class_labels=("sitting", "standing", "walking"),
        model=model,
        training_result_path=training_path,
        scope=scope,
    )
    write_prediction_artifact_record(output_path, record)
    loaded = load_prediction_artifact_record(output_path)

    np.testing.assert_array_equal(loaded.logits_array(), logits)
    assert loaded.true_labels == ("sitting", "standing", "walking")
    assert loaded.predicted_labels == loaded.true_labels
    assert loaded.window_metadata()[0].recording_id != "PRIVATE_RECORDING_NAME"
    saved = output_path.read_text(encoding="utf-8")
    assert "PRIVATE_RECORDING_NAME" not in saved
    assert "PRIVATE_DEVICE_STREAM" not in saved
    assert '"holdout_accessed": false' in saved
    with pytest.raises(FileExistsError):
        write_prediction_artifact_record(output_path, record)


def test_prediction_artifact_rejects_training_or_holdout_like_scope(tmp_path: Path) -> None:
    training_path = tmp_path / "training.json"
    _training_result(training_path)
    model = _model()
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=("SYNTHETIC_TRAIN_A",),
        validation_subjects=("SYNTHETIC_VALIDATION_A",),
    )
    wrong_metadata = tuple(
        record.__class__(
            **{
                **record.as_dict(),
                "subject_id": "SYNTHETIC_TRAIN_A",
            }
        )
        for record in _metadata()
    )

    with pytest.raises(ValueError, match="exactly match validation"):
        build_prediction_artifact_record(
            run_id="synthetic-predictions-1",
            created_at_utc="2026-07-15T19:00:00Z",
            logits=np.zeros((3, 3), dtype=np.float32),
            metadata=wrong_metadata,
            class_labels=("sitting", "standing", "walking"),
            model=model,
            training_result_path=training_path,
            scope=scope,
        )


def test_tampered_prediction_payload_is_rejected(tmp_path: Path) -> None:
    training_path = tmp_path / "training.json"
    output_path = tmp_path / "predictions.json"
    _training_result(training_path)
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=("SYNTHETIC_TRAIN_A",),
        validation_subjects=("SYNTHETIC_VALIDATION_A",),
    )
    record = build_prediction_artifact_record(
        run_id="synthetic-predictions-1",
        created_at_utc="2026-07-15T19:00:00Z",
        logits=np.eye(3, dtype=np.float32),
        metadata=_metadata(),
        class_labels=("sitting", "standing", "walking"),
        model=_model(),
        training_result_path=training_path,
        scope=scope,
    )
    write_prediction_artifact_record(output_path, record)
    decoded = json.loads(output_path.read_text(encoding="utf-8"))
    decoded["windows"][0]["logits"][0] = -99.0
    output_path.write_text(json.dumps(decoded), encoding="utf-8")

    with pytest.raises(ValueError, match="argmax|digest"):
        load_prediction_artifact_record(output_path)


def test_inference_rejects_wrong_model_output_shape() -> None:
    with pytest.raises(ValueError, match="declared class count"):
        predict_logits(
            nn.Sequential(nn.Flatten(), nn.Linear(6, 2)),
            np.zeros((1, 3, 2), dtype=np.float32),
            expected_output_classes=3,
            batch_size=1,
        )
