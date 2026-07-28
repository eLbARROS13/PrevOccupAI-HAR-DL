"""Synthetic tests for governed window datasets and fold tensor assembly."""

from pathlib import Path

import numpy as np
import pytest

from prevoccupai_har.dataset import (
    DatasetAssemblyPurpose,
    assemble_development_fold_tensors,
    build_window_dataset,
)
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.splits import SubjectPartition
from prevoccupai_har.windowing import WindowMetadata


ROOT = Path(__file__).resolve().parents[1]
CLASS_LABELS = ("sitting", "standing", "walking")


def _synthetic_dataset(
    subjects: tuple[str, ...] = (
        "SYNTHETIC_TRAIN_A",
        "SYNTHETIC_TRAIN_B",
        "SYNTHETIC_VALIDATION_A",
    ),
):
    windows = []
    metadata = []
    for subject_index, subject in enumerate(subjects):
        for label_index, label in enumerate(CLASS_LABELS):
            window = np.full((8, 3), subject_index * 10 + label_index, dtype=np.float32)
            window[:, 1] += np.arange(8, dtype=np.float32)
            windows.append(window)
            metadata.append(
                WindowMetadata(
                    subject_id=subject,
                    recording_id=f"{subject}-{label}",
                    main_label=label,
                    sub_activity_label=f"synthetic-{label}",
                    sensor_stream_id=f"{subject}-stream",
                    sensor_side="synthetic_side",
                    start_sample=0,
                    end_sample_exclusive=8,
                    preprocessing_status="synthetic",
                    quality_status="synthetic",
                )
            )
    return build_window_dataset(np.stack(windows), metadata, CLASS_LABELS)


def _synthetic_partition() -> SubjectPartition:
    return SubjectPartition(
        training=("SYNTHETIC_TRAIN_A", "SYNTHETIC_TRAIN_B"),
        validation=("SYNTHETIC_VALIDATION_A",),
        holdout=("SYNTHETIC_HOLDOUT_A",),
        fold_index=0,
    )


def test_window_dataset_encodes_labels_and_freezes_arrays() -> None:
    dataset = _synthetic_dataset()

    assert dataset.windows.shape == (9, 8, 3)
    assert dataset.label_indices.tolist() == [0, 1, 2] * 3
    assert dataset.windows.flags.writeable is False
    assert dataset.label_indices.flags.writeable is False


def test_window_dataset_rejects_duplicate_provenance() -> None:
    dataset = _synthetic_dataset(("SYNTHETIC_TRAIN_A",))
    duplicate_metadata = list(dataset.metadata)
    duplicate_metadata[1] = duplicate_metadata[0]

    with pytest.raises(ValueError, match="Duplicate window provenance"):
        build_window_dataset(dataset.windows, duplicate_metadata, CLASS_LABELS)


def test_window_dataset_rejects_blank_class_labels() -> None:
    dataset = _synthetic_dataset(("SYNTHETIC_TRAIN_A",))

    with pytest.raises(ValueError, match="non-empty and unique"):
        build_window_dataset(dataset.windows, dataset.metadata, ("sitting", "", "walking"))


def test_fold_assembly_is_train_only_channels_first_and_deterministic() -> None:
    dataset = _synthetic_dataset()
    partition = _synthetic_partition()

    first = assemble_development_fold_tensors(
        dataset,
        partition,
        purpose=DatasetAssemblyPurpose.SYNTHETIC_VALIDATION,
    )
    second = assemble_development_fold_tensors(
        dataset,
        partition,
        purpose=DatasetAssemblyPurpose.SYNTHETIC_VALIDATION,
    )

    assert first.training_inputs.shape == (6, 3, 8)
    assert first.validation_inputs.shape == (3, 3, 8)
    np.testing.assert_allclose(first.training_inputs.mean(axis=(0, 2)), 0.0, atol=1e-6)
    assert np.all(first.validation_inputs[:, 0, :] > 1.0)
    np.testing.assert_array_equal(first.training_inputs, second.training_inputs)
    np.testing.assert_array_equal(first.validation_inputs, second.validation_inputs)
    assert first.standardizer_state == second.standardizer_state
    assert first.training_inputs.flags.writeable is False
    with pytest.raises(TypeError):
        first.standardizer_state["fit_subject_count"] = 99


def test_synthetic_assembly_rejects_real_or_holdout_subjects() -> None:
    real_dataset = _synthetic_dataset(("P003", "P004"))
    real_partition = SubjectPartition(
        training=("P003",),
        validation=("P004",),
        holdout=("P001",),
        fold_index=0,
    )
    with pytest.raises(ValueError, match="synthetic participant identifiers"):
        assemble_development_fold_tensors(
            real_dataset,
            real_partition,
            purpose=DatasetAssemblyPurpose.SYNTHETIC_VALIDATION,
        )

    holdout_dataset = _synthetic_dataset(
        ("SYNTHETIC_TRAIN_A", "SYNTHETIC_VALIDATION_A", "SYNTHETIC_HOLDOUT_A")
    )
    with pytest.raises(PermissionError, match="cannot receive hold-out"):
        assemble_development_fold_tensors(
            holdout_dataset,
            _synthetic_partition(),
            purpose=DatasetAssemblyPurpose.SYNTHETIC_VALIDATION,
        )


def test_development_assembly_is_authorized_for_the_exact_real_cohort() -> None:
    protocol = load_protocol(ROOT / "configs/mban_protocol.json")
    dataset = _synthetic_dataset(protocol.development_participants)
    partition = SubjectPartition(
        training=protocol.development_participants[:-1],
        validation=(protocol.development_participants[-1],),
        holdout=protocol.holdout_participants,
        fold_index=0,
    )

    tensors = assemble_development_fold_tensors(
        dataset,
        partition,
        purpose=DatasetAssemblyPurpose.DEVELOPMENT_SELECTION,
        protocol=protocol,
    )

    assert {item.subject_id for item in tensors.training_metadata} == set(
        partition.training
    )
    assert {item.subject_id for item in tensors.validation_metadata} == set(
        partition.validation
    )
    assert not (
        {item.subject_id for item in tensors.training_metadata}
        | {item.subject_id for item in tensors.validation_metadata}
    ) & set(protocol.holdout_participants)
