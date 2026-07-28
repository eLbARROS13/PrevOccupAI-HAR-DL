"""Tests for memory-bounded participant-disjoint window-store training."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prevoccupai_har.modeling import (  # noqa: E402
    CompactCnn1D,
    CompactCnnArchitecture,
    OptimizationConfiguration,
)
from prevoccupai_har.preprocessing import TrainOnlyChannelStandardizer  # noqa: E402
from prevoccupai_har.protocol import load_protocol  # noqa: E402
from prevoccupai_har.streaming_training import (  # noqa: E402
    fit_classifier_streaming,
    fit_streaming_channel_standardizer,
    indices_for_subjects,
    metadata_for_indices,
    predict_classifier_streaming,
)
from prevoccupai_har.training import (  # noqa: E402
    TrainingPurpose,
    TrainingRunScope,
)
from prevoccupai_har.window_store import (  # noqa: E402
    DevelopmentWindowStore,
    METADATA_DTYPE,
)


def _store() -> DevelopmentWindowStore:
    generator = np.random.default_rng(20260716)
    subjects = ("P003", "P004", "P005", "P006")
    labels = np.tile(np.arange(3, dtype=np.int64), 16)
    windows = generator.normal(size=(48, 64, 3)).astype(np.float32)
    metadata = np.empty(48, dtype=METADATA_DTYPE)
    for index in range(48):
        participant = subjects[index // 12]
        label = ("sitting", "standing", "walking")[int(labels[index])]
        metadata[index] = (
            participant,
            f"recording-{index // 3:02d}",
            label,
            f"{label}_task",
            "DEVICE000001",
            "left",
            f"segment-{index:03d}.npy",
            index * 32,
            index * 32 + 64,
            "approved",
            "GOOD",
        )
    return DevelopmentWindowStore(
        windows=windows,
        labels=labels,
        metadata=metadata,
        index={"development_participants": list(subjects)},
    )


def _model() -> CompactCnn1D:
    return CompactCnn1D(
        CompactCnnArchitecture(
            input_channels=3,
            expected_samples=64,
            convolution_channels=(4, 8),
            kernel_sizes=(7, 5),
            strides=(2, 2),
            max_pool_sizes=(2, 2),
            batch_normalization=True,
            dropout_probability=0.1,
            initialization="pytorch_module_defaults_seeded",
            output_classes=3,
        )
    )


def test_streaming_standardizer_matches_direct_statistics() -> None:
    store = _store()
    indices = indices_for_subjects(store, ("P003", "P004", "P005"))
    standardizer = fit_streaming_channel_standardizer(
        store,
        indices,
        allowed_training_subjects=("P003", "P004", "P005"),
        chunk_windows=5,
    )
    expected = np.asarray(store.windows[indices], dtype=np.float64)

    np.testing.assert_allclose(standardizer.mean_, expected.mean(axis=(0, 1)))
    np.testing.assert_allclose(standardizer.scale_, expected.std(axis=(0, 1)))


def test_streaming_training_and_prediction_preserve_partition_order() -> None:
    store = _store()
    protocol = replace(
        load_protocol("configs/mban_protocol.json"),
        development_participants=("P003", "P004", "P005", "P006"),
        holdout_participants=("P001",),
    )
    scope = TrainingRunScope(
        purpose=TrainingPurpose.DEVELOPMENT_SELECTION,
        training_subjects=("P003", "P004", "P005"),
        validation_subjects=("P006",),
    )
    training = indices_for_subjects(store, scope.training_subjects)
    validation = indices_for_subjects(store, scope.validation_subjects)
    standardizer = fit_streaming_channel_standardizer(
        store,
        training,
        allowed_training_subjects=scope.training_subjects,
        chunk_windows=4,
    )
    optimization = OptimizationConfiguration(
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=6,
        maximum_epochs=2,
        early_stopping_patience=2,
        early_stopping_minimum_delta=0.0,
    )
    model = _model()
    outcome = fit_classifier_streaming(
        model,
        store,
        training,
        validation,
        standardizer,
        output_classes=3,
        optimization=optimization,
        seed=1103,
        scope=scope,
        protocol=protocol,
    )
    prediction = predict_classifier_streaming(
        model,
        store,
        validation,
        standardizer,
        output_classes=3,
        batch_size=6,
        seed=1103,
    )
    metadata = metadata_for_indices(store, prediction.row_indices)

    assert len(outcome.history) == 2
    assert np.array_equal(prediction.row_indices, validation)
    assert prediction.logits.shape == (validation.size, 3)
    assert {row.subject_id for row in metadata} == {"P006"}


def test_streaming_training_rejects_validation_fitted_preprocessing() -> None:
    store = _store()
    validation = indices_for_subjects(store, ("P006",))
    standardizer = TrainOnlyChannelStandardizer.for_subjects(("P006",))
    standardizer.mean_ = np.zeros(3)
    standardizer.scale_ = np.ones(3)

    with pytest.raises(ValueError, match="Preprocessing authorization"):
        fit_classifier_streaming(
            _model(),
            store,
            indices_for_subjects(store, ("P003", "P004", "P005")),
            validation,
            standardizer,
            output_classes=3,
            optimization=OptimizationConfiguration(1e-3, 1e-4, 6, 1, 1, 0.0),
            seed=1103,
            scope=TrainingRunScope(
                purpose=TrainingPurpose.DEVELOPMENT_SELECTION,
                training_subjects=("P003", "P004", "P005"),
                validation_subjects=("P006",),
            ),
            protocol=replace(
                load_protocol("configs/mban_protocol.json"),
                development_participants=("P003", "P004", "P005", "P006"),
                holdout_participants=("P001",),
            ),
        )
