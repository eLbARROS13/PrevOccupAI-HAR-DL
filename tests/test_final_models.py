"""Tests for post-development model freezing without hold-out access."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from prevoccupai_har.classical_baseline import (  # noqa: E402
    RandomForestReconstructionConfiguration,
)
from prevoccupai_har.feature_store import DevelopmentFeatureMatrix  # noqa: E402
from prevoccupai_har.final_models import (  # noqa: E402
    FinalTrainingSettings,
    SeedEpochDecision,
    arithmetic_mean_probabilities,
    finite_history,
    fit_classifier_fixed_epochs_streaming,
    fit_final_random_forest,
    load_model_state_npz,
    load_random_forest_pipeline,
    write_model_state_npz,
    write_random_forest_pipeline,
)
from prevoccupai_har.modeling import (  # noqa: E402
    CompactCnn1D,
    CompactCnnArchitecture,
    OptimizationConfiguration,
)
from prevoccupai_har.preprocessing import TrainOnlyChannelStandardizer  # noqa: E402
from prevoccupai_har.protocol import load_protocol  # noqa: E402
from prevoccupai_har.provenance import sha256_canonical_json  # noqa: E402
from prevoccupai_har.streaming_training import (  # noqa: E402
    fit_streaming_channel_standardizer,
)
from prevoccupai_har.window_store import (  # noqa: E402
    DevelopmentWindowStore,
    METADATA_DTYPE,
)


SEEDS = (1103, 2207, 3301, 4409, 5519)


def _settings(
    *,
    development: tuple[str, ...] = ("P003", "P004"),
    selected_feature_count: int = 3,
) -> FinalTrainingSettings:
    del selected_feature_count
    decisions = tuple(
        SeedEpochDecision(
            random_seed=seed,
            fold_indices=tuple(range(5)),
            fold_best_epochs=(1, 2, 3, 4, 5),
            training_result_sha256=(str(index) * 64 for index in ("a", "b", "c", "d", "e")),
            fixed_epoch_count=3,
        )
        for seed in SEEDS
    )
    # Materialize generator-backed fields before constructing the immutable record.
    decisions = tuple(
        replace(value, training_result_sha256=tuple(value.training_result_sha256))
        for value in decisions
    )
    provisional = FinalTrainingSettings(
        schema_version=1,
        settings_id="synthetic-final-settings-v1",
        purpose="final_model_settings_freeze",
        scientific_result=False,
        holdout_accessed=False,
        selected_candidate_id="synthetic-cnn-v1",
        class_labels=("sitting", "standing", "walking"),
        development_participants=development,
        holdout_participants=("P001",),
        epoch_decisions=decisions,
        rf_hyperparameters={
            "criterion": "gini",
            "max_depth": 5,
            "n_estimators": 20,
        },
        rf_modal_fold_count=3,
        rf_tie_count=1,
        rf_selected_grid_index_zero_based=0,
        input_hashes={"synthetic_input_sha256": "f" * 64},
        development_source_revision=f"tree-sha256:{'1' * 64}",
        rf_development_source_revision=f"tree-sha256:{'2' * 64}",
        settings_payload_sha256="0" * 64,
    )
    settings = replace(
        provisional,
        settings_payload_sha256=sha256_canonical_json(provisional._payload()),
    )
    settings.validate()
    return settings


def _store() -> DevelopmentWindowStore:
    generator = np.random.default_rng(20260717)
    windows = generator.normal(size=(24, 64, 3)).astype(np.float32)
    labels = np.tile(np.arange(3, dtype=np.int64), 8)
    metadata = np.empty(24, dtype=METADATA_DTYPE)
    for index in range(24):
        participant = ("P003", "P004")[index // 12]
        label = ("sitting", "standing", "walking")[int(labels[index])]
        metadata[index] = (
            participant,
            f"recording-{index:02d}",
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
        index={"development_participants": ["P003", "P004"]},
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


def test_fixed_epoch_refit_uses_all_development_rows_without_validation() -> None:
    store = _store()
    protocol = replace(
        load_protocol("configs/mban_protocol.json"),
        development_participants=("P003", "P004"),
        holdout_participants=("P001",),
    )
    indices = np.arange(store.windows.shape[0], dtype=np.int64)
    standardizer = fit_streaming_channel_standardizer(
        store,
        indices,
        allowed_training_subjects=protocol.development_participants,
    )
    outcome = fit_classifier_fixed_epochs_streaming(
        _model(),
        store,
        indices,
        standardizer,
        output_classes=3,
        optimization=OptimizationConfiguration(1e-3, 1e-4, 6, 3, 2, 0.0),
        random_seed=1103,
        fixed_epoch_count=2,
        protocol=protocol,
    )

    assert outcome.fixed_epoch_count == 2
    assert len(outcome.history) == 2
    assert finite_history(outcome.history)
    assert all(not hasattr(entry, "validation_loss") for entry in outcome.history)

    with pytest.raises(ValueError, match="every development window"):
        fit_classifier_fixed_epochs_streaming(
            _model(),
            store,
            indices[:-1],
            standardizer,
            output_classes=3,
            optimization=OptimizationConfiguration(1e-3, 1e-4, 6, 3, 2, 0.0),
            random_seed=1103,
            fixed_epoch_count=2,
            protocol=protocol,
        )


def test_model_state_round_trip_preserves_exact_payload(tmp_path: Path) -> None:
    model = _model()
    from prevoccupai_har.prediction_artifacts import sha256_model_state

    expected = sha256_model_state(model)
    output = tmp_path / "model_state.npz"
    write_model_state_npz(output, model)
    reloaded = load_model_state_npz(output, _model(), expected_payload_sha256=expected)

    assert sha256_model_state(reloaded) == expected
    with pytest.raises(FileExistsError):
        write_model_state_npz(output, model)


def test_probability_ensemble_is_seed_order_invariant_and_strict() -> None:
    logits = {
        seed: np.asarray([[0.0, float(index), -1.0], [1.0, 0.0, float(index)]])
        for index, seed in enumerate(SEEDS)
    }
    forward = arithmetic_mean_probabilities(logits, expected_seeds=SEEDS)
    reverse = arithmetic_mean_probabilities(logits, expected_seeds=tuple(reversed(SEEDS)))

    np.testing.assert_array_equal(forward, reverse)
    np.testing.assert_allclose(forward.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    with pytest.raises(ValueError, match="exactly match"):
        arithmetic_mean_probabilities(
            {key: value for key, value in logits.items() if key != 5519},
            expected_seeds=SEEDS,
        )


def test_final_rf_refits_balancing_and_pipeline_on_development_only(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(19)
    features: list[np.ndarray] = []
    labels: list[str] = []
    subactivities: list[str] = []
    participants: list[str] = []
    for participant_index, participant in enumerate(("P003", "P004")):
        for class_index, label in enumerate(("sitting", "standing", "walking")):
            for repeat_index in range(10):
                row = generator.normal(size=6)
                row[class_index] += 4.0 + participant_index * 0.1
                features.append(row)
                labels.append(label)
                subactivities.append(f"{label}_task")
                participants.append(participant)
    dataset = DevelopmentFeatureMatrix(
        features=np.asarray(features, dtype=np.float64),
        labels=tuple(labels),
        subactivity_labels=tuple(subactivities),
        participant_ids=tuple(participants),
        feature_names=tuple(f"feature_{index}" for index in range(6)),
        manifest={},
    )
    configuration = RandomForestReconstructionConfiguration(
        schema_version=1,
        experiment_id="synthetic-rf-v1",
        status="synthetic_validation_only",
        expected_candidate_feature_count=6,
        variance_threshold=0.0,
        absolute_correlation_threshold=0.99,
        selected_feature_count=3,
        balancing_strategy="per_participant_per_subactivity_training_fold_only",
        criteria=("gini",),
        estimator_counts=(20,),
        maximum_depths=(5,),
        inner_group_folds=2,
        inner_selection_metric="accuracy",
        n_jobs=1,
        random_seed=42,
    )
    protocol = replace(
        load_protocol("configs/mban_protocol.json"),
        development_participants=("P003", "P004"),
        holdout_participants=("P001",),
    )
    fitted = fit_final_random_forest(
        dataset=dataset,
        configuration=configuration,
        settings=_settings(),
        protocol=protocol,
    )
    output = tmp_path / "final_rf.joblib"
    write_random_forest_pipeline(output, fitted.pipeline)
    reloaded = load_random_forest_pipeline(output)

    assert len(fitted.selected_feature_names) == 3
    np.testing.assert_array_equal(
        fitted.pipeline.predict(dataset.features),
        reloaded.predict(dataset.features),
    )
