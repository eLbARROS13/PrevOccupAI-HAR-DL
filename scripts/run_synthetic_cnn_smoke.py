#!/usr/bin/env python3
"""Run a deterministic synthetic-only smoke test of the compact CNN pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prevoccupai_har.dataset import (
    DatasetAssemblyPurpose,
    assemble_development_fold_tensors,
    build_window_dataset,
)
from prevoccupai_har.modeling import (
    build_compact_cnn_1d,
    load_compact_cnn_experiment_configuration,
)
from prevoccupai_har.prediction_artifacts import (
    build_prediction_artifact_record,
    predict_logits,
    write_prediction_artifact_record,
)
from prevoccupai_har.results import (
    build_training_result_record,
    write_training_result_record,
)
from prevoccupai_har.splits import SubjectPartition
from prevoccupai_har.training import (
    TrainingPurpose,
    TrainingRunScope,
    fit_classifier,
    set_reproducible_seed,
)
from prevoccupai_har.windowing import WindowMetadata


ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the compact CNN on generated arrays only. This command cannot "
            "produce a scientific result or access the external hold-out cohort."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/cnn_1d.json",
        help="Compact-CNN configuration file.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Optional immutable validation-logit artifact written after training.",
    )
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser


def _synthetic_windows(
    *,
    seed: int,
    sample_count: int,
    subject_ids: tuple[str, ...],
    class_labels: tuple[str, ...],
    windows_per_class: int,
) -> tuple[np.ndarray, tuple[WindowMetadata, ...]]:
    """Generate sample-first arrays with clearly synthetic window provenance."""
    generator = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, sample_count, endpoint=False, dtype=np.float32)
    windows: list[np.ndarray] = []
    metadata: list[WindowMetadata] = []
    for subject_id in subject_ids:
        for class_index, class_label in enumerate(class_labels):
            for repeat_index in range(windows_per_class):
                window = generator.normal(
                    loc=0.0,
                    scale=0.2,
                    size=(sample_count, 3),
                ).astype(np.float32)
                window[:, class_index] += np.sin(
                    2 * np.pi * (class_index + 1) * time
                )
                recording_id = (
                    f"{subject_id}-{class_label}-synthetic-{repeat_index:03d}"
                )
                windows.append(window)
                metadata.append(
                    WindowMetadata(
                        subject_id=subject_id,
                        recording_id=recording_id,
                        main_label=class_label,
                        sub_activity_label=f"synthetic-{class_label}",
                        sensor_stream_id=f"{recording_id}-stream",
                        sensor_side="synthetic_side",
                        start_sample=0,
                        end_sample_exclusive=sample_count,
                        preprocessing_status="synthetic_software_test",
                        quality_status="synthetic_software_test",
                    )
                )
    return np.stack(windows), tuple(metadata)


def main() -> int:
    """Execute the synthetic smoke test and write an immutable result record."""
    arguments = _build_parser().parse_args()
    if arguments.epochs <= 0:
        raise ValueError("Epoch count must be positive")
    configuration = load_compact_cnn_experiment_configuration(arguments.config)
    if configuration.status != "synthetic_validation_only":
        raise PermissionError("The smoke command requires a synthetic-only configuration")

    optimization = replace(
        configuration.optimization,
        maximum_epochs=min(arguments.epochs, configuration.optimization.maximum_epochs),
        early_stopping_patience=min(
            arguments.epochs,
            configuration.optimization.early_stopping_patience,
        ),
    )
    partition = SubjectPartition(
        training=("SYNTHETIC_TRAIN_A", "SYNTHETIC_TRAIN_B"),
        validation=("SYNTHETIC_VALIDATION_A",),
        holdout=("SYNTHETIC_HOLDOUT_NOT_LOADED",),
        fold_index=0,
    )
    training_windows, training_metadata = _synthetic_windows(
        seed=arguments.seed,
        sample_count=configuration.architecture.expected_samples,
        subject_ids=partition.training,
        class_labels=configuration.class_labels,
        windows_per_class=6,
    )
    validation_windows, validation_metadata = _synthetic_windows(
        seed=arguments.seed + 1,
        sample_count=configuration.architecture.expected_samples,
        subject_ids=partition.validation,
        class_labels=configuration.class_labels,
        windows_per_class=4,
    )
    dataset = build_window_dataset(
        np.concatenate((training_windows, validation_windows), axis=0),
        training_metadata + validation_metadata,
        configuration.class_labels,
    )
    fold = assemble_development_fold_tensors(
        dataset,
        partition,
        purpose=DatasetAssemblyPurpose.SYNTHETIC_VALIDATION,
    )
    scope = TrainingRunScope(
        purpose=TrainingPurpose.SYNTHETIC_VALIDATION,
        training_subjects=partition.training,
        validation_subjects=partition.validation,
    )
    set_reproducible_seed(arguments.seed)
    model = build_compact_cnn_1d(configuration)
    outcome = fit_classifier(
        model,
        fold.training_inputs,
        fold.training_targets,
        fold.validation_inputs,
        fold.validation_targets,
        output_classes=configuration.architecture.output_classes,
        optimization=optimization,
        seed=arguments.seed,
        scope=scope,
        device=arguments.device,
    )
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    record = build_training_result_record(
        run_id=f"synthetic-cnn-smoke-{arguments.seed}",
        created_at_utc=created_at.isoformat().replace("+00:00", "Z"),
        experiment_id=configuration.experiment_id,
        source_revision="unversioned_workspace_software_test",
        model_configuration_path=arguments.config,
        model_trainable_parameter_count=model.trainable_parameter_count,
        learned_preprocessing_state=fold.standardizer_state,
        scope=scope,
        outcome=outcome,
    )
    write_training_result_record(arguments.output, record)
    prediction_output: str | None = None
    if arguments.predictions_output is not None:
        logits = predict_logits(
            model,
            fold.validation_inputs,
            expected_output_classes=configuration.architecture.output_classes,
            batch_size=configuration.optimization.batch_size,
            device=arguments.device,
        )
        prediction_record = build_prediction_artifact_record(
            run_id=f"synthetic-cnn-predictions-{arguments.seed}",
            created_at_utc=created_at.isoformat().replace("+00:00", "Z"),
            logits=logits,
            metadata=fold.validation_metadata,
            class_labels=configuration.class_labels,
            model=model,
            training_result_path=arguments.output,
            scope=scope,
        )
        write_prediction_artifact_record(
            arguments.predictions_output,
            prediction_record,
        )
        prediction_output = str(arguments.predictions_output.resolve())
    print(
        json.dumps(
            {
                "best_epoch": outcome.best_epoch,
                "holdout_accessed": False,
                "output": str(arguments.output.resolve()),
                "predictions_output": prediction_output,
                "scientific_result": False,
                "standardization_fit_subject_count": fold.standardizer_state[
                    "fit_subject_count"
                ],
                "trainable_parameter_count": model.trainable_parameter_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
