#!/usr/bin/env python3
"""Exercise the leakage-safe Random Forest comparator on generated features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prevoccupai_har.classical_baseline import (
    SYNTHETIC_VALIDATION,
    evaluate_random_forest_development_folds,
    load_random_forest_reconstruction_configuration,
    write_random_forest_development_record,
)
from prevoccupai_har.model_selection import SelectionFold
from prevoccupai_har.provenance import sha256_file
from prevoccupai_har.splits import build_validation_folds


ROOT = Path(__file__).resolve().parents[1]
CLASS_LABELS = ("sitting", "standing", "walking")
DEVELOPMENT_PARTICIPANTS = tuple(
    f"SYNTHETIC_P{index:03d}" for index in range(1, 7)
)
HOLDOUT_PARTICIPANTS = ("SYNTHETIC_HOLDOUT_NOT_LOADED",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run grouped nested validation on generated features only. This command "
            "cannot produce a scientific result or evaluate the external hold-out."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rf_baseline_synthetic_smoke.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260715)
    return parser


def _synthetic_features(
    *,
    seed: int,
    feature_count: int,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Create a deterministic, class-informative feature matrix with no raw data."""
    if seed < 0:
        raise ValueError("Synthetic-data seed cannot be negative")
    if feature_count < 3:
        raise ValueError("Synthetic RF validation requires at least three features")
    generator = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    subactivity_labels: list[str] = []
    participant_ids: list[str] = []
    for participant_index, participant in enumerate(DEVELOPMENT_PARTICIPANTS):
        participant_offset = (participant_index - 2.5) * 0.03
        for class_index, label in enumerate(CLASS_LABELS):
            for subactivity_index in range(2):
                repeat_count = 4 + (
                    participant_index + class_index + subactivity_index
                ) % 3
                for _ in range(repeat_count):
                    row = generator.normal(0.0, 1.0, size=feature_count)
                    row[0] += class_index * 4.0 + participant_offset
                    row[1] += (class_index == 1) * 3.0
                    row[2] += (class_index == 2) * 3.0
                    rows.append(row)
                    labels.append(label)
                    subactivity_labels.append(
                        f"{label}_task_{subactivity_index}"
                    )
                    participant_ids.append(participant)
    return (
        np.asarray(rows, dtype=np.float64),
        tuple(labels),
        tuple(subactivity_labels),
        tuple(participant_ids),
        tuple(f"synthetic_feature_{index:02d}" for index in range(feature_count)),
    )


def _folds() -> tuple[SelectionFold, ...]:
    partitions = build_validation_folds(
        DEVELOPMENT_PARTICIPANTS,
        HOLDOUT_PARTICIPANTS,
        n_splits=3,
        random_seed=42,
    )
    return tuple(
        SelectionFold(
            fold_index=partition.fold_index,
            training_subjects=partition.training,
            validation_subjects=partition.validation,
            holdout_subjects=partition.holdout,
        )
        for partition in partitions
    )


def main() -> int:
    """Run the synthetic nested evaluation and exclusively write its record."""
    arguments = _parser().parse_args()
    configuration = load_random_forest_reconstruction_configuration(arguments.config)
    features, labels, subactivity_labels, participant_ids, feature_names = (
        _synthetic_features(
            seed=arguments.data_seed,
            feature_count=configuration.expected_candidate_feature_count,
        )
    )
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    record = evaluate_random_forest_development_folds(
        run_id=f"synthetic-rf-smoke-{arguments.data_seed}",
        created_at_utc=created_at.isoformat().replace("+00:00", "Z"),
        configuration_path=arguments.config,
        configuration=configuration,
        features=features,
        labels=labels,
        subactivity_labels=subactivity_labels,
        participant_ids=participant_ids,
        feature_names=feature_names,
        class_labels=CLASS_LABELS,
        folds=_folds(),
        purpose=SYNTHETIC_VALIDATION,
        source_revision="unversioned_workspace_software_test",
    )
    write_random_forest_development_record(arguments.output, record)
    accuracies = tuple(
        float(fold.validation_metrics["overall_metrics"]["accuracy"])
        for fold in record.folds
    )
    print(
        json.dumps(
            {
                "development_participant_count": len(DEVELOPMENT_PARTICIPANTS),
                "feature_matrix_sha256": record.feature_matrix_sha256,
                "fold_count": len(record.folds),
                "holdout_accessed": record.holdout_accessed,
                "mean_synthetic_outer_accuracy": float(np.mean(accuracies)),
                "model_configuration_sha256": record.model_configuration_sha256,
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256_file(arguments.output),
                "scientific_result": record.scientific_result,
                "synthetic_data_seed": arguments.data_seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
