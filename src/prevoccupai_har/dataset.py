"""Governed window datasets and leakage-safe development-fold tensor assembly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .preprocessing import TrainOnlyChannelStandardizer
from .protocol import ProtocolConfiguration
from .splits import SubjectPartition
from .windowing import WindowMetadata


class DatasetAssemblyPurpose(str, Enum):
    """Permitted purposes for model-input tensor assembly."""

    SYNTHETIC_VALIDATION = "synthetic_validation"
    DEVELOPMENT_SELECTION = "development_selection"


@dataclass(frozen=True)
class WindowDataset:
    """Aligned sample-first windows, immutable metadata, and fixed class indices."""

    windows: NDArray[np.float32]
    metadata: tuple[WindowMetadata, ...]
    class_labels: tuple[str, ...]
    label_indices: NDArray[np.int64]

    @property
    def participant_ids(self) -> tuple[str, ...]:
        """Return the participant identifier aligned to each window."""
        return tuple(item.subject_id for item in self.metadata)


@dataclass(frozen=True)
class DevelopmentFoldTensors:
    """Channels-first tensors assembled without loading external hold-out windows."""

    fold_index: int
    class_labels: tuple[str, ...]
    training_inputs: NDArray[np.float32]
    training_targets: NDArray[np.int64]
    training_metadata: tuple[WindowMetadata, ...]
    validation_inputs: NDArray[np.float32]
    validation_targets: NDArray[np.int64]
    validation_metadata: tuple[WindowMetadata, ...]
    standardizer_state: Mapping[str, object]


def _readonly_array(values: NDArray, *, dtype: np.dtype) -> NDArray:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _freeze_standardizer_state(state: Mapping[str, object]) -> Mapping[str, object]:
    frozen = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in state.items()
    }
    return MappingProxyType(frozen)


def build_window_dataset(
    windows: NDArray[np.floating],
    metadata: Sequence[WindowMetadata],
    class_labels: Sequence[str],
) -> WindowDataset:
    """Validate and freeze aligned windows and their scientific provenance."""
    values = np.asarray(windows)
    records = tuple(metadata)
    labels = tuple(map(str, class_labels))
    if values.ndim != 3:
        raise ValueError("Windows must have shape (windows, samples, channels)")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("Window datasets cannot have empty dimensions")
    if len(records) != values.shape[0]:
        raise ValueError("Window and metadata counts do not match")
    if not np.isfinite(values).all():
        raise ValueError("Window values must be finite")
    if (
        not labels
        or len(labels) != len(set(labels))
        or any(not label.strip() for label in labels)
    ):
        raise ValueError("Class labels must be non-empty and unique")
    label_index = {label: index for index, label in enumerate(labels)}
    provenance_keys: set[tuple[object, ...]] = set()
    encoded_labels: list[int] = []
    for record in records:
        if not all(
            (
                record.subject_id,
                record.recording_id,
                record.main_label,
                record.sub_activity_label,
                record.sensor_stream_id,
                record.sensor_side,
                record.preprocessing_status,
                record.quality_status,
            )
        ):
            raise ValueError("Window provenance fields cannot be empty")
        if record.main_label not in label_index:
            raise ValueError(f"Undeclared main label: {record.main_label}")
        if record.start_sample < 0 or record.end_sample_exclusive <= record.start_sample:
            raise ValueError("Window sample bounds are invalid")
        if record.end_sample_exclusive - record.start_sample != values.shape[1]:
            raise ValueError("Window metadata bounds disagree with the tensor length")
        provenance_key = (
            record.subject_id,
            record.recording_id,
            record.sensor_stream_id,
            record.start_sample,
            record.end_sample_exclusive,
        )
        if provenance_key in provenance_keys:
            raise ValueError("Duplicate window provenance was detected")
        provenance_keys.add(provenance_key)
        encoded_labels.append(label_index[record.main_label])
    return WindowDataset(
        windows=_readonly_array(values, dtype=np.dtype(np.float32)),
        metadata=records,
        class_labels=labels,
        label_indices=_readonly_array(
            np.asarray(encoded_labels, dtype=np.int64),
            dtype=np.dtype(np.int64),
        ),
    )


def _validate_assembly_scope(
    dataset: WindowDataset,
    partition: SubjectPartition,
    purpose: DatasetAssemblyPurpose,
    protocol: ProtocolConfiguration | None,
) -> None:
    observed = set(dataset.participant_ids)
    training = set(partition.training)
    validation = set(partition.validation)
    holdout = set(partition.holdout)
    partition.validate()
    if observed & holdout:
        raise PermissionError("Development tensor assembly cannot receive hold-out windows")
    if observed != training | validation:
        raise ValueError("Dataset subjects must exactly match the development fold")

    if purpose is DatasetAssemblyPurpose.SYNTHETIC_VALIDATION:
        if protocol is not None:
            raise ValueError("Synthetic assembly must not receive a scientific protocol")
        if any(
            not subject.startswith("SYNTHETIC_")
            for subject in observed | training | validation | holdout
        ):
            raise ValueError("Synthetic assembly requires synthetic participant identifiers")
        return
    if purpose is not DatasetAssemblyPurpose.DEVELOPMENT_SELECTION:
        raise TypeError("Assembly purpose must be a DatasetAssemblyPurpose value")
    if protocol is None:
        raise ValueError("Development assembly requires a scientific protocol")
    if not protocol.training_authorized:
        raise PermissionError("The protocol does not authorize scientific dataset assembly")
    partition.validate(expected_development=protocol.development_participants)
    if holdout != set(protocol.holdout_participants):
        raise PermissionError("Fold and protocol hold-out cohorts differ")


def assemble_development_fold_tensors(
    dataset: WindowDataset,
    partition: SubjectPartition,
    *,
    purpose: DatasetAssemblyPurpose,
    protocol: ProtocolConfiguration | None = None,
) -> DevelopmentFoldTensors:
    """Fit standardization on training windows and assemble channels-first tensors."""
    _validate_assembly_scope(dataset, partition, purpose, protocol)
    participants = np.asarray(dataset.participant_ids, dtype=object)
    training_mask = np.isin(participants, np.asarray(partition.training, dtype=object))
    validation_mask = np.isin(participants, np.asarray(partition.validation, dtype=object))
    training_labels = set(dataset.label_indices[training_mask].tolist())
    validation_labels = set(dataset.label_indices[validation_mask].tolist())
    expected_labels = set(range(len(dataset.class_labels)))
    if training_labels != expected_labels or validation_labels != expected_labels:
        raise ValueError("Training and validation must each contain every declared class")

    standardizer = TrainOnlyChannelStandardizer.for_subjects(partition.training)
    standardized_training = standardizer.fit_transform(
        dataset.windows[training_mask],
        participants[training_mask].tolist(),
    )
    standardized_validation = standardizer.transform(dataset.windows[validation_mask])
    training_inputs = _readonly_array(
        standardized_training.transpose(0, 2, 1),
        dtype=np.dtype(np.float32),
    )
    validation_inputs = _readonly_array(
        standardized_validation.transpose(0, 2, 1),
        dtype=np.dtype(np.float32),
    )
    return DevelopmentFoldTensors(
        fold_index=partition.fold_index,
        class_labels=dataset.class_labels,
        training_inputs=training_inputs,
        training_targets=_readonly_array(
            dataset.label_indices[training_mask],
            dtype=np.dtype(np.int64),
        ),
        training_metadata=tuple(
            record for record, selected in zip(dataset.metadata, training_mask, strict=True) if selected
        ),
        validation_inputs=validation_inputs,
        validation_targets=_readonly_array(
            dataset.label_indices[validation_mask],
            dtype=np.dtype(np.int64),
        ),
        validation_metadata=tuple(
            record
            for record, selected in zip(dataset.metadata, validation_mask, strict=True)
            if selected
        ),
        standardizer_state=_freeze_standardizer_state(standardizer.state_dict()),
    )
