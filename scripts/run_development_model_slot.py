#!/usr/bin/env python3
"""Run one immutable candidate/fold/seed development-selection slot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from prevoccupai_har.analysis_records import (
    build_prediction_analysis_record,
    write_prediction_analysis_record,
)
from prevoccupai_har.model_selection import load_development_selection_plan
from prevoccupai_har.modeling import (
    build_time_series_classifier,
    load_model_input_sample_stride,
    load_time_series_experiment_configuration,
)
from prevoccupai_har.prediction_artifacts import (
    build_prediction_artifact_record,
    sha256_model_state,
    write_prediction_artifact_record,
)
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_canonical_json, sha256_file
from prevoccupai_har.results import (
    ScientificDataProvenance,
    build_training_result_record,
    write_training_result_record,
)
from prevoccupai_har.source_snapshot import load_source_tree_manifest
from prevoccupai_har.streaming_training import (
    fit_classifier_streaming,
    fit_streaming_channel_standardizer,
    indices_for_subjects,
    metadata_for_indices,
    predict_classifier_streaming,
)
from prevoccupai_har.training import (
    TrainingPurpose,
    TrainingRunScope,
    set_reproducible_seed,
)
from prevoccupai_har.window_store import load_development_window_store


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--window-store",
        type=Path,
        default=ROOT / "artifacts/windows/approved_development_v1",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/mban_protocol.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/subject_split_manifest_authorized.json",
    )
    parser.add_argument(
        "--device-provenance",
        type=Path,
        default=ROOT / "artifacts/data_audit/development_device_provenance.json",
    )
    parser.add_argument(
        "--approved-segments",
        type=Path,
        default=ROOT / "artifacts/data_audit/approved_development_segments.json",
    )
    parser.add_argument(
        "--signal-preprocessing",
        type=Path,
        default=ROOT / "configs/mban_signal_preprocessing.json",
    )
    parser.add_argument(
        "--segmentation-contract",
        type=Path,
        default=ROOT / "configs/mban_segmentation_contract.json",
    )
    return parser


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_preprocessing(path: Path, state: dict[str, object]) -> str:
    payload_sha256 = sha256_canonical_json(state)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "fold_training_only_preprocessing",
                "state": state,
                "state_payload_sha256": payload_sha256,
            },
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    return payload_sha256


def _write_model_state(path: Path, model: torch.nn.Module) -> None:
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in sorted(model.state_dict().items())
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def main() -> int:
    """Execute one frozen slot and atomically publish its artifact chain."""
    arguments = _parser().parse_args()
    if arguments.threads <= 0:
        raise ValueError("PyTorch thread count must be positive")
    plan = load_development_selection_plan(arguments.plan)
    configuration = load_time_series_experiment_configuration(arguments.config)
    candidate = next(
        value
        for value in plan.candidates
        if value.experiment_id == configuration.experiment_id
    )
    if candidate.model_configuration_sha256 != sha256_file(arguments.config):
        raise ValueError("Model configuration differs from the frozen candidate")
    fold = next(value for value in plan.folds if value.fold_index == arguments.fold)
    if arguments.seed not in plan.random_seeds:
        raise ValueError("Random seed is absent from the frozen selection plan")
    source_manifest = load_source_tree_manifest(
        arguments.source_manifest,
        root=ROOT,
        verify_current_tree=True,
    )
    source_revision = str(source_manifest["source_revision"])
    protocol = load_protocol(arguments.protocol)
    scope = TrainingRunScope(
        purpose=TrainingPurpose.DEVELOPMENT_SELECTION,
        training_subjects=fold.training_subjects,
        validation_subjects=fold.validation_subjects,
    )
    scope.validate(protocol)
    store = load_development_window_store(arguments.window_store)
    training_indices = indices_for_subjects(store, fold.training_subjects)
    validation_indices = indices_for_subjects(store, fold.validation_subjects)
    sample_stride = load_model_input_sample_stride(arguments.config)
    standardizer = fit_streaming_channel_standardizer(
        store,
        training_indices,
        allowed_training_subjects=fold.training_subjects,
        sample_stride=sample_stride,
    )

    run_slug = f"{configuration.experiment_id}-f{fold.fold_index:02d}-s{arguments.seed}"
    final_directory = (
        arguments.output_root
        / configuration.experiment_id
        / f"fold_{fold.fold_index:02d}"
        / f"seed_{arguments.seed}"
    ).resolve()
    if final_directory.exists():
        raise FileExistsError(f"Immutable run directory already exists: {final_directory}")
    partial_directory = final_directory.with_name(
        f".{final_directory.name}.partial-{os.getpid()}"
    )
    partial_directory.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(arguments.threads)
        set_reproducible_seed(arguments.seed)
        model = build_time_series_classifier(configuration)
        created_at = _utc_now()
        outcome = fit_classifier_streaming(
            model,
            store,
            training_indices,
            validation_indices,
            standardizer,
            output_classes=len(plan.class_labels),
            optimization=configuration.optimization,
            seed=arguments.seed,
            scope=scope,
            protocol=protocol,
            device=arguments.device,
            sample_stride=sample_stride,
        )
        preprocessing_state = {
            **standardizer.state_dict(),
            "source_sampling_rate_hz": 1000,
            "model_sampling_rate_hz": 1000 // sample_stride,
            "sample_stride": sample_stride,
        }
        preprocessing_payload_sha256 = _write_preprocessing(
            partial_directory / "preprocessing_state.json",
            preprocessing_state,
        )
        data_provenance = ScientificDataProvenance(
            raw_recording_manifest_sha256=sha256_file(arguments.device_provenance),
            segmentation_manifest_sha256=sha256_file(arguments.approved_segments),
            quality_manifest_sha256=sha256_file(arguments.approved_segments),
            split_manifest_sha256=sha256_file(arguments.split_manifest),
            window_store_index_sha256=sha256_file(
                arguments.window_store / "index.json"
            ),
            signal_preprocessing_configuration_sha256=sha256_file(
                arguments.signal_preprocessing
            ),
            segmentation_contract_configuration_sha256=sha256_file(
                arguments.segmentation_contract
            ),
        )
        training_path = partial_directory / "training_result.json"
        training_record = build_training_result_record(
            run_id=f"train-{run_slug}",
            created_at_utc=created_at,
            experiment_id=configuration.experiment_id,
            source_revision=source_revision,
            model_configuration_path=arguments.config,
            model_trainable_parameter_count=sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            learned_preprocessing_state=preprocessing_state,
            scope=scope,
            outcome=outcome,
            protocol=protocol,
            protocol_configuration_path=arguments.protocol,
            data_provenance=data_provenance,
        )
        if training_record.learned_preprocessing_sha256 != preprocessing_payload_sha256:
            raise RuntimeError("Training record and preprocessing payload disagree")
        write_training_result_record(training_path, training_record)

        prediction = predict_classifier_streaming(
            model,
            store,
            validation_indices,
            standardizer,
            output_classes=len(plan.class_labels),
            batch_size=configuration.optimization.batch_size,
            seed=arguments.seed,
            device=arguments.device,
            sample_stride=sample_stride,
        )
        prediction_path = partial_directory / "prediction_artifact.json"
        prediction_record = build_prediction_artifact_record(
            run_id=f"prediction-{run_slug}",
            created_at_utc=created_at,
            logits=prediction.logits,
            metadata=metadata_for_indices(store, prediction.row_indices),
            class_labels=plan.class_labels,
            model=model,
            training_result_path=training_path,
            scope=scope,
            protocol=protocol,
        )
        write_prediction_artifact_record(prediction_path, prediction_record)

        analysis_path = partial_directory / "analysis_record.json"
        analysis_record = build_prediction_analysis_record(
            analysis_id=f"analysis-{run_slug}",
            created_at_utc=created_at,
            prediction_artifact_path=prediction_path,
            calibration_bin_count=int(plan.analysis_settings["calibration_bin_count"]),
            expected_step_size_samples=int(
                plan.analysis_settings["expected_step_size_samples"]
            ),
            short_run_max_windows=int(
                plan.analysis_settings["short_run_max_windows"]
            ),
        )
        write_prediction_analysis_record(analysis_path, analysis_record)
        model_state_path = partial_directory / "model_state.npz"
        _write_model_state(model_state_path, model)
        if sha256_model_state(model) != prediction_record.model_state_sha256:
            raise RuntimeError("Retained model state changed after prediction")
        run_entry = {
            "schema_version": 1,
            "candidate_id": configuration.experiment_id,
            "fold_index": fold.fold_index,
            "random_seed": arguments.seed,
            "holdout_accessed": False,
            "source_revision": source_revision,
            "selection_plan_sha256": sha256_file(arguments.plan),
            "training_result": "training_result.json",
            "training_result_sha256": sha256_file(training_path),
            "preprocessing_state": "preprocessing_state.json",
            "preprocessing_state_sha256": sha256_file(
                partial_directory / "preprocessing_state.json"
            ),
            "prediction_artifact": "prediction_artifact.json",
            "prediction_artifact_sha256": sha256_file(prediction_path),
            "analysis_record": "analysis_record.json",
            "analysis_record_sha256": sha256_file(analysis_path),
            "model_state": "model_state.npz",
            "model_state_file_sha256": sha256_file(model_state_path),
            "model_state_payload_sha256": prediction_record.model_state_sha256,
        }
        with (partial_directory / "run_entry.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(run_entry, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        partial_directory.rename(final_directory)
    except BaseException:
        shutil.rmtree(partial_directory, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "best_epoch": outcome.best_epoch,
                "candidate_id": configuration.experiment_id,
                "fold_index": fold.fold_index,
                "holdout_accessed": False,
                "output": str(final_directory),
                "random_seed": arguments.seed,
                "run_entry_sha256": sha256_file(final_directory / "run_entry.json"),
                "stopped_early": outcome.stopped_early,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
