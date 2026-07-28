"""Tests for the fold-local, participant-grouped Random Forest comparator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from prevoccupai_har.classical_baseline import (  # noqa: E402
    DEVELOPMENT_SELECTION,
    SYNTHETIC_VALIDATION,
    AbsoluteCorrelationFilter,
    RandomForestReconstructionConfiguration,
    ScientificFeatureProvenance,
    build_leakage_safe_random_forest_pipeline,
    evaluate_random_forest_development_folds,
    load_random_forest_development_record,
    load_random_forest_reconstruction_configuration,
    select_fold_local_balanced_training_rows,
    write_random_forest_development_record,
)
from prevoccupai_har.model_selection import SelectionFold  # noqa: E402
from prevoccupai_har.protocol import load_protocol  # noqa: E402
from prevoccupai_har.splits import build_validation_folds  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CREATED_AT = "2026-07-15T21:00:00Z"
CLASS_LABELS = ("sitting", "standing", "walking")
HOLDOUT = ("SYNTHETIC_HOLDOUT_NOT_LOADED",)


def _small_configuration(
    tmp_path: Path,
) -> tuple[Path, RandomForestReconstructionConfiguration]:
    decoded = json.loads(
        (ROOT / "configs/rf_baseline.json").read_text(encoding="utf-8")
    )
    decoded["input"]["expected_candidate_feature_count"] = 12
    decoded["fold_local_pipeline"]["anova_selected_feature_count"] = 5
    decoded["fold_local_pipeline"]["hyperparameter_grid"] = {
        "criterion": ["gini"],
        "n_estimators": [20],
        "max_depth": [5],
    }
    configuration_path = tmp_path / "rf_smoke_configuration.json"
    configuration_path.write_text(
        json.dumps(decoded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        configuration_path,
        load_random_forest_reconstruction_configuration(configuration_path),
    )


def _synthetic_feature_matrix() -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[SelectionFold, ...],
]:
    generator = np.random.default_rng(20260715)
    participants = tuple(f"SYNTHETIC_P{index:03d}" for index in range(1, 7))
    labels: list[str] = []
    subactivity_labels: list[str] = []
    participant_ids: list[str] = []
    rows: list[np.ndarray] = []
    for participant_index, participant in enumerate(participants):
        participant_offset = (participant_index - 2.5) * 0.03
        for class_index, label in enumerate(CLASS_LABELS):
            for repeat_index in range(8):
                row = generator.normal(0.0, 1.0, size=12)
                row[0] += class_index * 4.0 + participant_offset
                row[1] += (class_index == 1) * 3.0
                row[2] += (class_index == 2) * 3.0
                rows.append(row)
                labels.append(label)
                subactivity_labels.append(f"{label}_task_{repeat_index % 2}")
                participant_ids.append(participant)
    partitions = build_validation_folds(
        participants,
        HOLDOUT,
        n_splits=3,
        random_seed=42,
    )
    folds = tuple(
        SelectionFold(
            fold_index=partition.fold_index,
            training_subjects=partition.training,
            validation_subjects=partition.validation,
            holdout_subjects=partition.holdout,
        )
        for partition in partitions
    )
    return (
        np.asarray(rows, dtype=np.float64),
        tuple(labels),
        tuple(subactivity_labels),
        tuple(participant_ids),
        tuple(f"feature_{index:02d}" for index in range(12)),
        folds,
    )


def _evaluate(tmp_path: Path):
    configuration_path, configuration = _small_configuration(tmp_path)
    features, labels, subactivity_labels, participant_ids, feature_names, folds = (
        _synthetic_feature_matrix()
    )
    record = evaluate_random_forest_development_folds(
        run_id="synthetic-rf-contract-v1",
        created_at_utc=CREATED_AT,
        configuration_path=configuration_path,
        configuration=configuration,
        features=features,
        labels=labels,
        subactivity_labels=subactivity_labels,
        participant_ids=participant_ids,
        feature_names=feature_names,
        class_labels=CLASS_LABELS,
        folds=folds,
        purpose=SYNTHETIC_VALIDATION,
        source_revision="unversioned_workspace_software_test",
    )
    return record, configuration_path, configuration


def test_pipeline_keeps_every_learned_selector_inside_grouped_search(
    tmp_path: Path,
) -> None:
    _, configuration = _small_configuration(tmp_path)
    pipeline = build_leakage_safe_random_forest_pipeline(
        configuration,
        random_seed=configuration.random_seed,
    )

    assert tuple(pipeline.named_steps) == (
        "variance_filter",
        "correlation_filter",
        "anova_selector",
        "random_forest",
    )
    assert pipeline.named_steps["anova_selector"].k == 5

    training_features = np.asarray(
        [[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]]
    )
    validation_features = np.asarray([[1.0, 1.0], [2.0, 2.0]])
    correlation_filter = AbsoluteCorrelationFilter(threshold=0.9).fit(
        training_features
    )
    assert correlation_filter.transform(validation_features).shape[1] == 2


def test_balancing_uses_only_candidate_training_rows() -> None:
    labels = (
        "sitting",
        "sitting",
        "standing",
        "standing",
        "sitting",
        "sitting",
        "sitting",
        "standing",
        "standing",
        "standing",
        "walking",
    )
    subactivities = (
        "desk",
        "desk",
        "still",
        "still",
        "desk",
        "desk",
        "desk",
        "still",
        "still",
        "still",
        "gait",
    )
    participants = ("A",) * 4 + ("B",) * 6 + ("VALIDATION",)
    selected, class_counts, quotas = select_fold_local_balanced_training_rows(
        labels=labels,
        subactivity_labels=subactivities,
        participant_ids=participants,
        candidate_indices=tuple(range(10)),
        class_labels=("sitting", "standing"),
        random_seed=42,
    )

    assert set(selected) <= set(range(10))
    assert 10 not in selected
    assert class_counts == {"sitting": 4, "standing": 4}
    assert quotas == {"desk": 2, "still": 2}


def test_synthetic_outer_folds_are_participant_disjoint_and_round_trip(
    tmp_path: Path,
) -> None:
    record, _, _ = _evaluate(tmp_path)
    output_path = tmp_path / "rf_development_record.json"
    write_random_forest_development_record(output_path, record)
    loaded = load_random_forest_development_record(output_path)

    assert loaded == record
    assert loaded.scientific_result is False
    assert loaded.holdout_accessed is False
    assert len(loaded.folds) == 3
    assert {
        participant
        for fold in loaded.folds
        for participant in fold.validation_subjects
    } == {f"SYNTHETIC_P{index:03d}" for index in range(1, 7)}
    for fold in loaded.folds:
        assert not set(fold.training_subjects) & set(fold.validation_subjects)
        assert len(fold.selected_feature_names) == 5
        assert fold.balanced_training_row_count <= fold.training_row_count
        assert set(fold.balanced_training_class_counts) == set(CLASS_LABELS)
        assert fold.balancing_quota_by_subactivity
        assert fold.validation_metrics["overall_metrics"]["accuracy"] > 0.75
    with pytest.raises(FileExistsError):
        write_random_forest_development_record(output_path, record)


def test_artifact_tampering_and_configuration_substitution_are_rejected(
    tmp_path: Path,
) -> None:
    record, configuration_path, configuration = _evaluate(tmp_path)
    output_path = tmp_path / "rf_development_record.json"
    write_random_forest_development_record(output_path, record)
    decoded = json.loads(output_path.read_text(encoding="utf-8"))
    decoded["folds"][0]["predictions"][0]["predicted_label"] = "walking"
    output_path.write_text(json.dumps(decoded), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics disagree|digest does not match"):
        load_random_forest_development_record(output_path)

    features, labels, subactivity_labels, participant_ids, feature_names, folds = (
        _synthetic_feature_matrix()
    )
    substituted = replace(configuration, estimator_counts=(21,))
    with pytest.raises(ValueError, match="disagrees with its governed file"):
        evaluate_random_forest_development_folds(
            run_id="synthetic-rf-substitution-test",
            created_at_utc=CREATED_AT,
            configuration_path=configuration_path,
            configuration=substituted,
            features=features,
            labels=labels,
            subactivity_labels=subactivity_labels,
            participant_ids=participant_ids,
            feature_names=feature_names,
            class_labels=CLASS_LABELS,
            folds=folds,
            purpose=SYNTHETIC_VALIDATION,
            source_revision="unversioned_workspace_software_test",
        )


def test_closed_protocol_and_holdout_feature_rows_fail_closed(
    tmp_path: Path,
) -> None:
    configuration_path, configuration = _small_configuration(tmp_path)
    features, labels, subactivity_labels, participant_ids, feature_names, folds = (
        _synthetic_feature_matrix()
    )
    protocol_path = ROOT / "configs/mban_protocol.json"
    closed_protocol = replace(load_protocol(protocol_path), training_authorized=False)
    with pytest.raises(PermissionError, match="does not authorize"):
        evaluate_random_forest_development_folds(
            run_id="scientific-rf-not-authorized",
            created_at_utc=CREATED_AT,
            configuration_path=configuration_path,
            configuration=configuration,
            features=features,
            labels=labels,
            subactivity_labels=subactivity_labels,
            participant_ids=participant_ids,
            feature_names=feature_names,
            class_labels=CLASS_LABELS,
            folds=folds,
            purpose="development_selection",
            source_revision="0" * 40,
            protocol=closed_protocol,
            protocol_configuration_path=protocol_path,
        )

    holdout_features = np.vstack([features, features[0]])
    holdout_labels = labels + (labels[0],)
    holdout_subactivities = subactivity_labels + (subactivity_labels[0],)
    holdout_participants = participant_ids + (HOLDOUT[0],)
    with pytest.raises(PermissionError, match="hold-out"):
        evaluate_random_forest_development_folds(
            run_id="synthetic-rf-holdout-injection",
            created_at_utc=CREATED_AT,
            configuration_path=configuration_path,
            configuration=configuration,
            features=holdout_features,
            labels=holdout_labels,
            subactivity_labels=holdout_subactivities,
            participant_ids=holdout_participants,
            feature_names=feature_names,
            class_labels=CLASS_LABELS,
            folds=folds,
            purpose=SYNTHETIC_VALIDATION,
            source_revision="unversioned_workspace_software_test",
        )


def test_scientific_path_reloads_the_governed_protocol(tmp_path: Path) -> None:
    """Exercise the scientific protocol-file guard before a real RF run."""
    configuration_path, configuration = _small_configuration(tmp_path)
    protocol_path = ROOT / "configs/mban_protocol.json"
    protocol = load_protocol(protocol_path)
    generator = np.random.default_rng(20260717)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    subactivities: list[str] = []
    participants: list[str] = []
    for participant_index, participant in enumerate(protocol.development_participants):
        for class_index, class_label in enumerate(CLASS_LABELS):
            for repeat_index in range(8):
                row = generator.normal(0.0, 1.0, size=12)
                row[0] += class_index * 4.0 + participant_index * 0.001
                row[1] += (class_index == 1) * 3.0
                row[2] += (class_index == 2) * 3.0
                rows.append(row)
                labels.append(class_label)
                subactivities.append(
                    f"{class_label}_task_{repeat_index % 2}"
                )
                participants.append(participant)
    partitions = build_validation_folds(
        protocol.development_participants,
        protocol.holdout_participants,
        n_splits=5,
        random_seed=42,
    )
    folds = tuple(
        SelectionFold(
            fold_index=partition.fold_index,
            training_subjects=partition.training,
            validation_subjects=partition.validation,
            holdout_subjects=partition.holdout,
        )
        for partition in partitions
    )
    provenance = ScientificFeatureProvenance(
        raw_recording_manifest_sha256="0" * 64,
        segmentation_manifest_sha256="1" * 64,
        quality_manifest_sha256="2" * 64,
        split_manifest_sha256="3" * 64,
        signal_preprocessing_configuration_sha256="4" * 64,
        feature_extraction_configuration_sha256="5" * 64,
        feature_matrix_file_sha256="6" * 64,
    )

    record = evaluate_random_forest_development_folds(
        run_id="scientific-rf-protocol-reload-test",
        created_at_utc=CREATED_AT,
        configuration_path=configuration_path,
        configuration=configuration,
        features=np.asarray(rows, dtype=np.float64),
        labels=tuple(labels),
        subactivity_labels=tuple(subactivities),
        participant_ids=tuple(participants),
        feature_names=tuple(f"feature_{index:02d}" for index in range(12)),
        class_labels=CLASS_LABELS,
        folds=folds,
        purpose=DEVELOPMENT_SELECTION,
        source_revision="tree-sha256:" + "7" * 64,
        protocol=protocol,
        protocol_configuration_path=protocol_path,
        data_provenance=provenance,
    )

    assert record.scientific_result is True
    assert record.holdout_accessed is False
    assert len(record.folds) == 5
