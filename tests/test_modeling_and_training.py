"""Synthetic-only tests for the compact PyTorch model and trainer."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prevoccupai_har.modeling import (  # noqa: E402
    CompactDilatedTcn,
    CompactCnn1D,
    CompactCnnArchitecture,
    CompactTcnArchitecture,
    OptimizationConfiguration,
    build_compact_cnn_1d,
    build_compact_tcn,
    build_time_series_classifier,
    load_compact_cnn_experiment_configuration,
    load_compact_tcn_experiment_configuration,
    load_time_series_experiment_configuration,
)
from prevoccupai_har.protocol import load_protocol  # noqa: E402
from prevoccupai_har.results import (  # noqa: E402
    build_training_result_record,
    load_training_result_record,
    write_training_result_record,
)
from prevoccupai_har.training import (  # noqa: E402
    TrainingHistoryEntry,
    TrainingOutcome,
    TrainingPurpose,
    TrainingRunScope,
    _as_validated_tensors,
    fit_classifier,
    set_reproducible_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_architecture() -> CompactCnnArchitecture:
    return CompactCnnArchitecture(
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


def _small_optimization() -> OptimizationConfiguration:
    return OptimizationConfiguration(
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=6,
        maximum_epochs=3,
        early_stopping_patience=2,
        early_stopping_minimum_delta=0.0,
    )


def _small_tcn_architecture() -> CompactTcnArchitecture:
    return CompactTcnArchitecture(
        input_channels=3,
        expected_samples=64,
        channels=4,
        stem_kernel_size=3,
        stem_stride=1,
        kernel_size=3,
        dilations=(1, 2),
        batch_normalization=True,
        dropout_probability=0.1,
        padding="symmetric_same",
        initialization="pytorch_module_defaults_seeded",
        output_classes=3,
    )


def test_checked_in_configuration_builds_compact_model() -> None:
    configuration = load_compact_cnn_experiment_configuration(ROOT / "configs/cnn_1d.json")

    model = build_compact_cnn_1d(configuration)
    output = model(torch.zeros(2, 3, 500))

    assert output.shape == (2, 3)
    assert 1_000 < model.trainable_parameter_count < 100_000
    assert configuration.status == "frozen_for_development"
    assert configuration.class_labels == ("sitting", "standing", "walking")
    assert configuration.scientific_training_authorized_by_this_config is False


def test_model_rejects_wrong_window_shape() -> None:
    model = CompactCnn1D(_small_architecture())

    with pytest.raises(ValueError, match="wrong window length"):
        model(torch.zeros(2, 3, 63))


def test_checked_in_tcn_is_compact_and_has_declared_receptive_field() -> None:
    configuration = load_compact_tcn_experiment_configuration(
        ROOT / "configs/tcn_1d.json"
    )
    generic_configuration = load_time_series_experiment_configuration(
        ROOT / "configs/tcn_1d.json"
    )

    model = build_compact_tcn(configuration)
    generic_model = build_time_series_classifier(generic_configuration)

    assert model(torch.zeros(2, 3, 500)).shape == (2, 3)
    assert isinstance(generic_model, CompactDilatedTcn)
    assert configuration.architecture.receptive_field_samples == 369
    assert model.trainable_parameter_count == 2_307
    assert configuration.status == "frozen_for_development"
    assert configuration.scientific_training_authorized_by_this_config is False


def test_tcn_rejects_non_doubling_dilation_schedule() -> None:
    invalid = CompactTcnArchitecture(
        input_channels=3,
        expected_samples=64,
        channels=4,
        stem_kernel_size=3,
        stem_stride=1,
        kernel_size=3,
        dilations=(1, 3),
        batch_normalization=True,
        dropout_probability=0.1,
        padding="symmetric_same",
        initialization="pytorch_module_defaults_seeded",
        output_classes=3,
    )

    with pytest.raises(ValueError, match="ordered sequence"):
        invalid.validate()


def test_tcn_uses_the_shared_validation_only_trainer() -> None:
    generator = np.random.default_rng(65537)
    model = CompactDilatedTcn(_small_tcn_architecture())
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=("SYNTHETIC_TRAIN_A",),
        validation_subjects=("SYNTHETIC_VALIDATION_A",),
    )
    optimization = OptimizationConfiguration(
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=6,
        maximum_epochs=1,
        early_stopping_patience=1,
        early_stopping_minimum_delta=0.0,
    )

    outcome = fit_classifier(
        model,
        generator.normal(size=(12, 3, 64)).astype(np.float32),
        np.tile(np.arange(3), 4).astype(np.int64),
        generator.normal(size=(6, 3, 64)).astype(np.float32),
        np.tile(np.arange(3), 2).astype(np.int64),
        output_classes=3,
        optimization=optimization,
        seed=1103,
        scope=scope,
    )

    assert outcome.best_epoch == 1
    assert len(outcome.history) == 1


def test_synthetic_training_is_reproducible() -> None:
    generator = np.random.default_rng(104729)
    training_inputs = generator.normal(size=(18, 3, 64)).astype(np.float32)
    training_targets = np.tile(np.arange(3), 6).astype(np.int64)
    validation_inputs = generator.normal(size=(9, 3, 64)).astype(np.float32)
    validation_targets = np.tile(np.arange(3), 3).astype(np.int64)
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=("SYNTHETIC_TRAIN_A", "SYNTHETIC_TRAIN_B"),
        validation_subjects=("SYNTHETIC_VALIDATION_A",),
    )

    outcomes = []
    states = []
    for _ in range(2):
        set_reproducible_seed(1103)
        model = CompactCnn1D(_small_architecture())
        outcome = fit_classifier(
            model,
            training_inputs,
            training_targets,
            validation_inputs,
            validation_targets,
            output_classes=3,
            optimization=_small_optimization(),
            seed=1103,
            scope=scope,
        )
        outcomes.append(outcome)
        states.append({name: value.detach().clone() for name, value in model.state_dict().items()})

    assert outcomes[0] == outcomes[1]
    for parameter_name in states[0]:
        torch.testing.assert_close(states[0][parameter_name], states[1][parameter_name])


def test_trainer_copies_read_only_model_inputs_without_warning() -> None:
    inputs = np.zeros((3, 3, 64), dtype=np.float32)
    targets = np.arange(3, dtype=np.int64)
    inputs.setflags(write=False)
    targets.setflags(write=False)

    input_tensor, target_tensor = _as_validated_tensors(
        inputs,
        targets,
        output_classes=3,
    )

    input_tensor[0, 0, 0] = 1.0
    target_tensor[0] = 2
    assert inputs[0, 0, 0] == 0.0
    assert targets[0] == 0


def test_synthetic_scope_rejects_real_participant_identifiers() -> None:
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=("P003",),
        validation_subjects=("SYNTHETIC_VALIDATION_A",),
    )

    with pytest.raises(ValueError, match="synthetic subject identifiers"):
        scope.validate()


def test_development_training_is_denied_while_protocol_gate_is_closed() -> None:
    protocol = replace(
        load_protocol(ROOT / "configs/mban_protocol.json"),
        training_authorized=False,
    )
    scope = TrainingRunScope(
        purpose=TrainingPurpose.DEVELOPMENT_SELECTION,
        training_subjects=protocol.development_participants[:-1],
        validation_subjects=(protocol.development_participants[-1],),
    )

    with pytest.raises(PermissionError, match="does not authorize scientific training"):
        scope.validate(protocol)


def test_synthetic_result_record_is_explicit_and_immutable(tmp_path: Path) -> None:
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=("SYNTHETIC_TRAIN_A",),
        validation_subjects=("SYNTHETIC_VALIDATION_A",),
    )
    outcome = TrainingOutcome(
        seed=1103,
        best_epoch=1,
        stopped_early=False,
        history=(
            TrainingHistoryEntry(
                epoch=1,
                training_loss=1.1,
                validation_loss=1.2,
                validation_macro_f1=0.2,
                validation_balanced_accuracy=1 / 3,
            ),
        ),
    )
    output_path = tmp_path / "synthetic_result.json"
    record = build_training_result_record(
        run_id="synthetic-smoke-1103",
        created_at_utc="2026-07-15T18:00:00Z",
        experiment_id="cnn_1d_single_stream_v1",
        source_revision="unversioned_workspace_software_test",
        model_configuration_path=ROOT / "configs/cnn_1d.json",
        model_trainable_parameter_count=1234,
        learned_preprocessing_state={
            "transform": "TrainOnlyChannelStandardizer",
            "fit_subject_count": 1,
            "mean": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        scope=scope,
        outcome=outcome,
    )

    write_training_result_record(output_path, record)

    saved = output_path.read_text(encoding="utf-8")
    assert '"scientific_result": false' in saved
    assert '"holdout_accessed": false' in saved
    assert '"data_provenance": null' in saved
    assert '"learned_preprocessing_sha256":' in saved
    assert load_training_result_record(output_path) == record
    with pytest.raises(FileExistsError):
        write_training_result_record(output_path, record)
