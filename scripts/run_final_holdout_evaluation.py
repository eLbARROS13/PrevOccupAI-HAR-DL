#!/usr/bin/env python3
"""Execute the one authorized external evaluation of the frozen CNN and RF."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from prevoccupai_har.calibration import evaluate_calibration
from prevoccupai_har.evaluation import evaluate_predictions
from prevoccupai_har.final_evaluation import (
    build_holdout_window_store_after_claim,
    execute_claim_gated_holdout,
    load_final_standardizer,
    load_holdout_feature_matrix_after_claim,
    metadata_records,
    verify_modality_count_alignment,
)
from prevoccupai_har.final_manifest import (
    EXPECTED_SEEDS,
    FinalModelFreezeManifest,
    verify_final_model_freeze_files,
)
from prevoccupai_har.final_models import (
    arithmetic_mean_probabilities,
    load_model_state_npz,
    load_random_forest_pipeline,
)
from prevoccupai_har.holdout import load_holdout_evaluation_policy
from prevoccupai_har.modeling import (
    build_time_series_classifier,
    load_time_series_experiment_configuration,
)
from prevoccupai_har.protocol import ProtocolConfiguration, load_protocol
from prevoccupai_har.provenance import sha256_canonical_json, sha256_file
from prevoccupai_har.signal_preprocessing import (
    load_signal_preprocessing_configuration,
)
from prevoccupai_har.source_snapshot import load_source_tree_manifest
from prevoccupai_har.statistical_evaluation import (
    compare_paired_participant_metrics,
    holm_adjust,
    participant_bootstrap_mean_interval,
)
from prevoccupai_har.streaming_training import predict_classifier_streaming
from prevoccupai_har.temporal_evaluation import evaluate_temporal_predictions


ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-freeze",
        type=Path,
        default=ROOT / "artifacts/final_models/v1/final_model_freeze_manifest.json",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/mban_protocol.json"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs/holdout_evaluation_policy.json",
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=ROOT / "extended_mban_har_paper/notes/statistical_analysis_plan.md",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/candidate_snapshot_manifest.json",
    )
    parser.add_argument(
        "--signal-preprocessing",
        type=Path,
        default=ROOT / "configs/mban_signal_preprocessing.json",
    )
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs/cnn_1d.json"
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "artifacts/final_evaluation/v1/access_ledger.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/final_evaluation/v1/result",
    )
    parser.add_argument(
        "--failure-record",
        type=Path,
        default=ROOT / "artifacts/final_evaluation/v1/failure_record.json",
    )
    parser.add_argument("--accessed-at-utc")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser


def _require_frozen_input(
    manifest: FinalModelFreezeManifest,
    key: str,
    path: Path,
) -> None:
    expected = manifest.input_hashes.get(key)
    if expected is None:
        raise ValueError(f"Final model freeze lacks required input binding: {key}")
    if sha256_file(path) != expected:
        raise ValueError(f"Frozen prerequisite changed: {key}")


def _validate_prerequisites(
    arguments: argparse.Namespace,
) -> tuple[ProtocolConfiguration, FinalModelFreezeManifest]:
    if arguments.threads <= 0:
        raise ValueError("PyTorch thread count must be positive")
    protocol = load_protocol(arguments.protocol)
    freeze = verify_final_model_freeze_files(arguments.model_freeze)
    if protocol.development_participants != freeze.development_participants or (
        protocol.holdout_participants != freeze.holdout_participants
    ):
        raise ValueError("Protocol and final model freeze cohorts differ")
    if protocol.main_labels != freeze.class_labels:
        raise ValueError("Protocol and final model freeze class orders differ")
    _require_frozen_input(
        freeze, "protocol_configuration_sha256", arguments.protocol
    )
    _require_frozen_input(
        freeze, "candidate_manifest_sha256", arguments.candidate_manifest
    )
    _require_frozen_input(
        freeze, "signal_preprocessing_sha256", arguments.signal_preprocessing
    )
    _require_frozen_input(
        freeze, "selected_model_configuration_sha256", arguments.model_config
    )
    _require_frozen_input(
        freeze, "statistical_analysis_plan_sha256", arguments.analysis_plan
    )
    _require_frozen_input(
        freeze,
        "final_stage_source_manifest_sha256",
        arguments.source_manifest,
    )
    source = load_source_tree_manifest(
        arguments.source_manifest,
        root=ROOT,
        verify_current_tree=True,
    )
    if str(source["source_revision"]) != freeze.final_stage_source_revision:
        raise ValueError("Current final-stage source differs from the model freeze")
    configuration = load_time_series_experiment_configuration(arguments.model_config)
    if configuration.experiment_id != freeze.selected_candidate_id:
        raise ValueError("Model configuration differs from the selected candidate")
    if configuration.class_labels != freeze.class_labels:
        raise ValueError("Model configuration and frozen class orders differ")
    return protocol, freeze


def _write_npz_exclusive(path: Path, **arrays: Any) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _label_indices(labels: tuple[str, ...], class_labels: tuple[str, ...]) -> np.ndarray:
    index = {label: position for position, label in enumerate(class_labels)}
    unknown = set(labels) - set(class_labels)
    if unknown:
        raise ValueError(f"Labels outside frozen class order: {sorted(unknown)}")
    return np.asarray([index[label] for label in labels], dtype=np.int64)


def _participant_metric_values(
    evaluation: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    return {
        participant: float(values[metric])
        for participant, values in evaluation.items()
    }


def _artifact_manifest(directory: Path, names: tuple[str, ...]) -> dict[str, Any]:
    files = {
        name: {
            "sha256": sha256_file(directory / name),
            "size_bytes": (directory / name).stat().st_size,
        }
        for name in names
    }
    payload = {
        "status": "complete_final_external_evaluation_artifacts",
        "raw_signal_windows_retained": False,
        "files": files,
    }
    return {
        "schema_version": 1,
        **payload,
        "payload_sha256": sha256_canonical_json(payload),
    }


def _evaluate_after_claim(
    *,
    output_directory: Path,
    claim: Mapping[str, Any],
    arguments: argparse.Namespace,
    protocol: ProtocolConfiguration,
    freeze: FinalModelFreezeManifest,
) -> dict[str, Any]:
    """Read hold-out values, predict once, and emit only path-free results."""
    preprocessing = load_signal_preprocessing_configuration(
        arguments.signal_preprocessing
    )
    temporary_store_path = output_directory / "temporary_holdout_window_store"
    raw_store = build_holdout_window_store_after_claim(
        segment_root=arguments.segments_root,
        candidate_manifest_path=arguments.candidate_manifest,
        preprocessing=preprocessing,
        protocol=protocol,
        claim_record=claim,
        output_directory=temporary_store_path,
    )
    feature_manifest_path = output_directory / "holdout_feature_manifest.json"
    feature_dataset = load_holdout_feature_matrix_after_claim(
        feature_root=arguments.feature_root,
        candidate_manifest_path=arguments.candidate_manifest,
        protocol=protocol,
        claim_record=claim,
        manifest_output_path=feature_manifest_path,
    )
    alignment = verify_modality_count_alignment(
        raw_store,
        feature_dataset,
        class_labels=freeze.class_labels,
        holdout_participants=freeze.holdout_participants,
    )
    configuration = load_time_series_experiment_configuration(arguments.model_config)
    row_indices = np.arange(raw_store.windows.shape[0], dtype=np.int64)
    raw_truth = tuple(str(value) for value in raw_store.metadata["main_label"])
    raw_participants = tuple(
        str(value) for value in raw_store.metadata["participant_id"]
    )
    if not np.array_equal(
        raw_store.labels,
        _label_indices(raw_truth, freeze.class_labels),
    ):
        raise ValueError("Raw hold-out labels and metadata disagree")

    model_base = Path(arguments.model_freeze).resolve().parent
    logits_by_seed: dict[int, np.ndarray] = {}
    individual_seed_results: dict[str, Any] = {}
    for reference in freeze.dl_refits:
        standardizer, sample_stride = load_final_standardizer(
            model_base / reference.preprocessing_state,
            development_participants=freeze.development_participants,
            expected_payload_sha256=reference.preprocessing_state_payload_sha256,
        )
        model = build_time_series_classifier(configuration)
        load_model_state_npz(
            model_base / reference.model_state,
            model,
            expected_payload_sha256=reference.model_state_payload_sha256,
        )
        prediction = predict_classifier_streaming(
            model,
            raw_store,
            row_indices,
            standardizer,
            output_classes=len(freeze.class_labels),
            batch_size=configuration.optimization.batch_size,
            seed=reference.random_seed,
            device=arguments.device,
            sample_stride=sample_stride,
        )
        logits_by_seed[reference.random_seed] = prediction.logits
        seed_labels = tuple(
            freeze.class_labels[int(value)]
            for value in prediction.logits.argmax(axis=1)
        )
        individual_seed_results[str(reference.random_seed)] = evaluate_predictions(
            raw_truth,
            seed_labels,
            raw_participants,
            freeze.class_labels,
        ).as_dict()
        del prediction, model, standardizer
        gc.collect()

    probabilities = arithmetic_mean_probabilities(
        logits_by_seed,
        expected_seeds=EXPECTED_SEEDS,
    )
    dl_prediction_indices = probabilities.argmax(axis=1).astype(np.int64)
    dl_predictions = tuple(
        freeze.class_labels[int(value)] for value in dl_prediction_indices
    )
    dl_evaluation = evaluate_predictions(
        raw_truth,
        dl_predictions,
        raw_participants,
        freeze.class_labels,
    )
    dl_calibration = evaluate_calibration(
        probabilities,
        raw_truth,
        freeze.class_labels,
        bin_count=int(freeze.analysis_settings["calibration_bin_count"]),
    )
    temporal_metadata = metadata_records(raw_store)
    dl_temporal = evaluate_temporal_predictions(
        dl_predictions,
        temporal_metadata,
        freeze.class_labels,
        expected_step_size_samples=int(
            freeze.analysis_settings["expected_step_size_samples"]
        ),
        short_run_max_windows=int(freeze.analysis_settings["short_run_max_windows"]),
    )

    rf_pipeline = load_random_forest_pipeline(
        model_base / freeze.random_forest.fitted_pipeline
    )
    rf_classes = tuple(map(str, rf_pipeline.classes_))
    if rf_classes != freeze.class_labels:
        raise ValueError("Reloaded RF probability columns differ from frozen class order")
    rf_probabilities = np.asarray(
        rf_pipeline.predict_proba(feature_dataset.features), dtype=np.float64
    )
    if (
        rf_probabilities.shape
        != (feature_dataset.features.shape[0], len(freeze.class_labels))
        or not np.isfinite(rf_probabilities).all()
        or not np.allclose(
            rf_probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError("RF probabilities are invalid")
    rf_predictions = tuple(map(str, rf_pipeline.predict(feature_dataset.features)))
    rf_prediction_indices = rf_probabilities.argmax(axis=1).astype(np.int64)
    if rf_predictions != tuple(
        freeze.class_labels[int(value)] for value in rf_prediction_indices
    ):
        raise RuntimeError("RF hard predictions and probability argmax disagree")
    rf_evaluation = evaluate_predictions(
        feature_dataset.labels,
        rf_predictions,
        feature_dataset.participant_ids,
        freeze.class_labels,
    )
    rf_calibration = evaluate_calibration(
        rf_probabilities,
        feature_dataset.labels,
        freeze.class_labels,
        bin_count=int(freeze.analysis_settings["calibration_bin_count"]),
    )

    paired: dict[str, Any] = {}
    raw_p_values: list[float] = []
    for metric in ("macro_f1", "balanced_accuracy"):
        comparison = compare_paired_participant_metrics(
            _participant_metric_values(dl_evaluation.per_participant_metrics, metric),
            _participant_metric_values(rf_evaluation.per_participant_metrics, metric),
            confidence_level=0.95,
            resample_count=10_000,
            random_seed=1103,
        )
        paired[metric] = comparison.as_dict()
        raw_p_values.append(comparison.exact_sign_flip_p_value)
    adjusted = holm_adjust(raw_p_values)
    for metric, adjusted_value in zip(
        ("macro_f1", "balanced_accuracy"), adjusted, strict=True
    ):
        paired[metric]["holm_adjusted_p_value"] = adjusted_value
    paired["interpretation"] = (
        "descriptive_with_four_independent_participants; minimum attainable "
        "two-sided exact p-value is 0.125"
    )

    uncertainty: dict[str, Any] = {}
    for model_name, evaluation in (
        ("dl_ensemble", dl_evaluation),
        ("random_forest", rf_evaluation),
    ):
        uncertainty[model_name] = {
            metric: participant_bootstrap_mean_interval(
                _participant_metric_values(evaluation.per_participant_metrics, metric),
                confidence_level=0.95,
                resample_count=10_000,
                random_seed=1103,
            ).as_dict()
            for metric in ("macro_f1", "balanced_accuracy")
        }

    dl_predictions_path = output_directory / "dl_predictions.npz"
    ordered_seeds = np.asarray(EXPECTED_SEEDS, dtype=np.int64)
    stacked_logits = np.stack(
        [logits_by_seed[int(seed)] for seed in ordered_seeds], axis=0
    )
    _write_npz_exclusive(
        dl_predictions_path,
        class_labels=np.asarray(freeze.class_labels),
        participant_ids=np.asarray(raw_participants),
        true_label_indices=_label_indices(raw_truth, freeze.class_labels),
        predicted_label_indices=dl_prediction_indices,
        mean_probabilities=probabilities,
        random_seeds=ordered_seeds,
        seed_logits=stacked_logits,
        recording_ids=np.asarray([value.recording_id for value in temporal_metadata]),
        sensor_stream_ids=np.asarray(
            [value.sensor_stream_id for value in temporal_metadata]
        ),
        start_samples=np.asarray(
            [value.start_sample for value in temporal_metadata], dtype=np.int64
        ),
        end_samples_exclusive=np.asarray(
            [value.end_sample_exclusive for value in temporal_metadata],
            dtype=np.int64,
        ),
    )
    rf_predictions_path = output_directory / "rf_predictions.npz"
    _write_npz_exclusive(
        rf_predictions_path,
        class_labels=np.asarray(freeze.class_labels),
        participant_ids=np.asarray(feature_dataset.participant_ids),
        true_label_indices=_label_indices(
            feature_dataset.labels, freeze.class_labels
        ),
        predicted_label_indices=rf_prediction_indices,
        probabilities=rf_probabilities,
        subactivity_labels=np.asarray(feature_dataset.subactivity_labels),
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete_single_external_holdout_evaluation",
        "scientific_result": True,
        "holdout_accessed": True,
        "authorization_id": claim["authorization_id"],
        "accessed_at_utc": claim["accessed_at_utc"],
        "created_at_utc": _utc_now(),
        "holdout_participants": list(freeze.holdout_participants),
        "class_labels": list(freeze.class_labels),
        "selected_dl_candidate": freeze.selected_candidate_id,
        "primary_prediction": freeze.primary_prediction,
        "ensemble_method": freeze.ensemble_method,
        "analysis_settings": freeze.analysis_settings,
        "modality_alignment": alignment,
        "dl_ensemble": {
            "classification": dl_evaluation.as_dict(),
            "calibration": dl_calibration.as_dict(),
            "temporal": dl_temporal.as_dict(),
            "individual_seed_classification": individual_seed_results,
            "model_count": len(freeze.dl_refits),
            "serialized_model_size_bytes": sum(
                (model_base / value.model_state).stat().st_size
                for value in freeze.dl_refits
            ),
        },
        "random_forest": {
            "classification": rf_evaluation.as_dict(),
            "calibration": rf_calibration.as_dict(),
            "temporal": {
                "status": (
                    "not_computed_exact_feature_row_sequence_provenance_unavailable"
                ),
                "raw_feature_rowwise_alignment_claimed": False,
            },
            "serialized_model_size_bytes": (
                model_base / freeze.random_forest.fitted_pipeline
            ).stat().st_size,
        },
        "participant_grouped_uncertainty": uncertainty,
        "paired_dl_minus_rf": paired,
        "prediction_artifacts": {
            "dl": {
                "relative_name": dl_predictions_path.name,
                "sha256": sha256_file(dl_predictions_path),
            },
            "rf": {
                "relative_name": rf_predictions_path.name,
                "sha256": sha256_file(rf_predictions_path),
            },
        },
        "input_hashes": {
            "access_ledger_sha256": sha256_file(arguments.ledger),
            "policy_sha256": sha256_file(arguments.policy),
            "protocol_configuration_sha256": sha256_file(arguments.protocol),
            "model_freeze_manifest_sha256": sha256_file(arguments.model_freeze),
            "statistical_analysis_plan_sha256": sha256_file(arguments.analysis_plan),
            "final_stage_source_manifest_sha256": sha256_file(
                arguments.source_manifest
            ),
            "candidate_manifest_sha256": sha256_file(arguments.candidate_manifest),
            "signal_preprocessing_sha256": sha256_file(
                arguments.signal_preprocessing
            ),
        },
        "interpretation_constraints": [
            "The four hold-out participants are the independent evaluation units.",
            "Window-level confusion counts are descriptive, not independent samples.",
            "Bootstrap intervals are highly unstable with fewer than five participants.",
            "CNN and RF row orders are authoritative within modality but not paired rowwise.",
            "RF temporal diagnostics are unavailable without exact feature-row sequence provenance.",
        ],
    }
    result_path = output_directory / "final_result.json"
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")

    del raw_store, feature_dataset, rf_pipeline
    gc.collect()
    shutil.rmtree(temporary_store_path)
    artifact_names = (
        "dl_predictions.npz",
        "final_result.json",
        "holdout_feature_manifest.json",
        "rf_predictions.npz",
    )
    artifacts = _artifact_manifest(output_directory, artifact_names)
    artifact_manifest_path = output_directory / "artifact_manifest.json"
    with artifact_manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(artifacts, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    return {
        "artifact_manifest": artifact_manifest_path.name,
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "primary_result": result_path.name,
        "primary_result_sha256": sha256_file(result_path),
        "holdout_window_count": alignment["raw_window_count"],
        "feature_row_count": alignment["feature_row_count"],
    }


def main() -> int:
    """Validate frozen prerequisites, consume the claim, and evaluate once."""
    arguments = _parser().parse_args()
    protocol, freeze = _validate_prerequisites(arguments)
    policy = load_holdout_evaluation_policy(arguments.policy)
    torch.set_num_threads(arguments.threads)
    accessed_at = arguments.accessed_at_utc or _utc_now()

    completion = execute_claim_gated_holdout(
        protocol=protocol,
        policy=policy,
        protocol_configuration_path=arguments.protocol,
        model_freeze_manifest_path=arguments.model_freeze,
        statistical_analysis_plan_path=arguments.analysis_plan,
        ledger_path=arguments.ledger,
        accessed_at_utc=accessed_at,
        final_output_directory=arguments.output,
        failure_record_path=arguments.failure_record,
        evaluator=lambda directory, claim: _evaluate_after_claim(
            output_directory=directory,
            claim=claim,
            arguments=arguments,
            protocol=protocol,
            freeze=freeze,
        ),
    )
    print(json.dumps(completion, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
