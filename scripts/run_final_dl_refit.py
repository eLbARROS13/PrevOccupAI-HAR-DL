#!/usr/bin/env python3
"""Run one fixed-epoch DL refit on all development participants."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from prevoccupai_har.final_models import (
    build_final_dl_refit_record,
    fit_classifier_fixed_epochs_streaming,
    load_final_dl_refit_record,
    load_final_training_settings,
    load_model_state_npz,
    write_final_dl_refit_record,
    write_model_state_npz,
)
from prevoccupai_har.modeling import (
    build_time_series_classifier,
    load_model_input_sample_stride,
    load_time_series_experiment_configuration,
)
from prevoccupai_har.prediction_artifacts import sha256_model_state
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_canonical_json, sha256_file
from prevoccupai_har.source_snapshot import load_source_tree_manifest
from prevoccupai_har.streaming_training import (
    fit_streaming_channel_standardizer,
    indices_for_subjects,
)
from prevoccupai_har.training import set_reproducible_seed
from prevoccupai_har.window_store import load_development_window_store


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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "artifacts/final_models/v1/final_training_settings.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/cnn_1d.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/mban_protocol.json",
    )
    parser.add_argument(
        "--window-store",
        type=Path,
        default=ROOT / "artifacts/windows/approved_development_v1",
    )
    parser.add_argument(
        "--selection-bundle",
        type=Path,
        default=ROOT / "artifacts/development_selection/v1/model_selection_bundle.json",
    )
    parser.add_argument(
        "--approved-segments",
        type=Path,
        default=ROOT / "artifacts/data_audit/approved_development_segments.json",
    )
    parser.add_argument(
        "--device-provenance",
        type=Path,
        default=ROOT / "artifacts/data_audit/development_device_provenance.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/subject_split_manifest_authorized.json",
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
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser


def _write_preprocessing(path: Path, state: dict[str, object]) -> str:
    payload_sha256 = sha256_canonical_json(state)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "all_development_train_only_preprocessing",
                "holdout_accessed": False,
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


def main() -> int:
    """Fit one seed and atomically publish its immutable refit directory."""
    arguments = _parser().parse_args()
    if arguments.threads <= 0:
        raise ValueError("PyTorch thread count must be positive")
    settings = load_final_training_settings(arguments.settings)
    epoch_decision = next(
        (value for value in settings.epoch_decisions if value.random_seed == arguments.seed),
        None,
    )
    if epoch_decision is None:
        raise ValueError("Requested seed is absent from frozen final settings")
    source = load_source_tree_manifest(
        arguments.source_manifest,
        root=ROOT,
        verify_current_tree=True,
    )
    configuration = load_time_series_experiment_configuration(arguments.config)
    if configuration.experiment_id != settings.selected_candidate_id:
        raise ValueError("Model configuration is not the selected DL candidate")
    if sha256_file(arguments.config) != settings.input_hashes[
        "selected_model_configuration_sha256"
    ]:
        raise ValueError("Selected model configuration changed after settings freeze")
    protocol = load_protocol(arguments.protocol)
    if (
        protocol.development_participants != settings.development_participants
        or protocol.holdout_participants != settings.holdout_participants
    ):
        raise ValueError("Protocol cohorts differ from frozen final settings")
    store = load_development_window_store(
        arguments.window_store,
        verify_file_hashes=True,
    )
    training_indices = indices_for_subjects(store, protocol.development_participants)
    if training_indices.size != store.windows.shape[0]:
        raise ValueError("Final refit does not cover the complete development store")
    sample_stride = load_model_input_sample_stride(arguments.config)
    standardizer = fit_streaming_channel_standardizer(
        store,
        training_indices,
        allowed_training_subjects=protocol.development_participants,
        sample_stride=sample_stride,
    )

    final_directory = (arguments.output_root / f"seed_{arguments.seed}").resolve()
    if final_directory.exists():
        raise FileExistsError(f"Immutable final-refit directory exists: {final_directory}")
    partial_directory = final_directory.with_name(
        f".{final_directory.name}.partial-{os.getpid()}"
    )
    partial_directory.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(arguments.threads)
        set_reproducible_seed(arguments.seed)
        model = build_time_series_classifier(configuration)
        created_at = _utc_now()
        outcome = fit_classifier_fixed_epochs_streaming(
            model,
            store,
            training_indices,
            standardizer,
            output_classes=len(settings.class_labels),
            optimization=configuration.optimization,
            random_seed=arguments.seed,
            fixed_epoch_count=epoch_decision.fixed_epoch_count,
            protocol=protocol,
            device=arguments.device,
            sample_stride=sample_stride,
        )
        preprocessing_state = {
            **standardizer.state_dict(),
            "source_sampling_rate_hz": protocol.sampling_rate_hz,
            "model_sampling_rate_hz": protocol.sampling_rate_hz // sample_stride,
            "sample_stride": sample_stride,
        }
        preprocessing_path = partial_directory / "preprocessing_state.json"
        preprocessing_payload_sha256 = _write_preprocessing(
            preprocessing_path,
            preprocessing_state,
        )
        model_state_path = partial_directory / "model_state.npz"
        write_model_state_npz(model_state_path, model)
        model_state_payload_sha256 = sha256_model_state(model)

        reloaded_model = build_time_series_classifier(configuration)
        load_model_state_npz(
            model_state_path,
            reloaded_model,
            expected_payload_sha256=model_state_payload_sha256,
        )
        input_hashes = {
            "final_training_settings_sha256": sha256_file(arguments.settings),
            "model_configuration_sha256": sha256_file(arguments.config),
            "protocol_configuration_sha256": sha256_file(arguments.protocol),
            "window_store_index_sha256": sha256_file(
                arguments.window_store / "index.json"
            ),
            "selection_bundle_sha256": sha256_file(arguments.selection_bundle),
            "approved_segments_sha256": sha256_file(arguments.approved_segments),
            "device_provenance_sha256": sha256_file(arguments.device_provenance),
            "split_manifest_sha256": sha256_file(arguments.split_manifest),
            "signal_preprocessing_sha256": sha256_file(
                arguments.signal_preprocessing
            ),
            "segmentation_contract_sha256": sha256_file(
                arguments.segmentation_contract
            ),
            "final_stage_source_manifest_sha256": sha256_file(
                arguments.source_manifest
            ),
        }
        record = build_final_dl_refit_record(
            refit_id=f"final-{configuration.experiment_id}-seed-{arguments.seed}",
            created_at_utc=created_at,
            experiment_id=configuration.experiment_id,
            final_stage_source_revision=str(source["source_revision"]),
            settings=settings,
            epoch_decision=epoch_decision,
            protocol=protocol,
            development_window_count=int(store.windows.shape[0]),
            development_class_counts={
                str(key): int(value)
                for key, value in store.index["window_counts_per_class"].items()
            },
            sample_stride=sample_stride,
            trainable_parameter_count=sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            input_hashes=input_hashes,
            preprocessing_state_payload_sha256=preprocessing_payload_sha256,
            preprocessing_state_file_sha256=sha256_file(preprocessing_path),
            model_state_payload_sha256=model_state_payload_sha256,
            model_state_file_sha256=sha256_file(model_state_path),
            outcome=outcome,
        )
        record_path = partial_directory / "refit_record.json"
        write_final_dl_refit_record(record_path, record)
        if load_final_dl_refit_record(record_path) != record:
            raise RuntimeError("Final DL refit record changed on reload")
        entry = {
            "schema_version": 1,
            "experiment_id": configuration.experiment_id,
            "random_seed": arguments.seed,
            "fixed_epoch_count": epoch_decision.fixed_epoch_count,
            "holdout_accessed": False,
            "source_revision": source["source_revision"],
            "refit_record": "refit_record.json",
            "refit_record_sha256": sha256_file(record_path),
            "preprocessing_state": "preprocessing_state.json",
            "preprocessing_state_sha256": sha256_file(preprocessing_path),
            "model_state": "model_state.npz",
            "model_state_file_sha256": sha256_file(model_state_path),
            "model_state_payload_sha256": model_state_payload_sha256,
        }
        with (partial_directory / "refit_entry.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(entry, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        partial_directory.rename(final_directory)
    except BaseException:
        shutil.rmtree(partial_directory, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "elapsed_seconds": outcome.elapsed_seconds,
                "fixed_epoch_count": epoch_decision.fixed_epoch_count,
                "holdout_accessed": False,
                "output": str(final_directory),
                "random_seed": arguments.seed,
                "refit_entry_sha256": sha256_file(
                    final_directory / "refit_entry.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
