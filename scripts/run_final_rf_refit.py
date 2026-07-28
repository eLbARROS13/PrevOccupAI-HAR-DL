#!/usr/bin/env python3
"""Refit the frozen Random Forest pipeline on all development participants."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prevoccupai_har.classical_baseline import (
    load_random_forest_reconstruction_configuration,
)
from prevoccupai_har.feature_store import load_approved_development_feature_matrix
from prevoccupai_har.final_models import (
    build_final_random_forest_record,
    fit_final_random_forest,
    load_final_random_forest_record,
    load_final_training_settings,
    load_random_forest_pipeline,
    write_final_random_forest_record,
    write_random_forest_pipeline,
)
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_file
from prevoccupai_har.source_snapshot import load_source_tree_manifest


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
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT / "artifacts/final_models/v1/final_training_settings.json",
    )
    parser.add_argument(
        "--rf-development-record",
        type=Path,
        default=ROOT / "artifacts/development_selection/v1/rf_development_record.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/rf_baseline.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/mban_protocol.json"
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/candidate_snapshot_manifest.json",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/approved_development_feature_matrices.json",
    )
    parser.add_argument(
        "--window-store-index",
        type=Path,
        default=ROOT / "artifacts/windows/approved_development_v1/index.json",
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
        "--feature-extraction",
        type=Path,
        default=ROOT / "configs/mban_tsfel_feature_reconstruction.json",
    )
    return parser


def main() -> int:
    """Fit and atomically publish the all-development RF artifact."""
    arguments = _parser().parse_args()
    settings = load_final_training_settings(arguments.settings)
    source = load_source_tree_manifest(
        arguments.source_manifest,
        root=ROOT,
        verify_current_tree=True,
    )
    protocol = load_protocol(arguments.protocol)
    configuration = load_random_forest_reconstruction_configuration(arguments.config)
    if sha256_file(arguments.config) != settings.input_hashes["rf_configuration_sha256"]:
        raise ValueError("RF configuration changed after settings freeze")
    dataset = load_approved_development_feature_matrix(
        feature_root=arguments.feature_root,
        candidate_manifest_path=arguments.candidate_manifest,
        window_store_index_path=arguments.window_store_index,
        protocol=protocol,
    )
    recorded_manifest = json.loads(arguments.feature_manifest.read_text(encoding="utf-8"))
    if recorded_manifest != dataset.manifest:
        raise ValueError("Development feature manifest differs from loaded matrices")

    final_directory = arguments.output.resolve()
    if final_directory.exists():
        raise FileExistsError(f"Immutable final-RF directory exists: {final_directory}")
    partial_directory = final_directory.with_name(
        f".{final_directory.name}.partial-{os.getpid()}"
    )
    partial_directory.mkdir(parents=True, exist_ok=False)
    try:
        fitted = fit_final_random_forest(
            dataset=dataset,
            configuration=configuration,
            settings=settings,
            protocol=protocol,
        )
        pipeline_path = partial_directory / "fitted_pipeline.joblib"
        write_random_forest_pipeline(pipeline_path, fitted.pipeline)
        reloaded = load_random_forest_pipeline(pipeline_path)
        original_predictions = fitted.pipeline.predict(dataset.features)
        reloaded_predictions = reloaded.predict(dataset.features)
        if not np.array_equal(original_predictions, reloaded_predictions):
            raise RuntimeError("Final RF predictions changed after pipeline reload")

        input_hashes = {
            "final_training_settings_sha256": sha256_file(arguments.settings),
            "rf_development_record_sha256": sha256_file(
                arguments.rf_development_record
            ),
            "rf_configuration_sha256": sha256_file(arguments.config),
            "protocol_configuration_sha256": sha256_file(arguments.protocol),
            "candidate_manifest_sha256": sha256_file(arguments.candidate_manifest),
            "feature_manifest_sha256": sha256_file(arguments.feature_manifest),
            "window_store_index_sha256": sha256_file(arguments.window_store_index),
            "approved_segments_sha256": sha256_file(arguments.approved_segments),
            "device_provenance_sha256": sha256_file(arguments.device_provenance),
            "split_manifest_sha256": sha256_file(arguments.split_manifest),
            "signal_preprocessing_sha256": sha256_file(
                arguments.signal_preprocessing
            ),
            "feature_extraction_sha256": sha256_file(arguments.feature_extraction),
            "final_stage_source_manifest_sha256": sha256_file(
                arguments.source_manifest
            ),
        }
        record = build_final_random_forest_record(
            refit_id="final-rf-tsfel-development-v1",
            created_at_utc=_utc_now(),
            experiment_id=configuration.experiment_id,
            final_stage_source_revision=str(source["source_revision"]),
            settings=settings,
            protocol=protocol,
            dataset=dataset,
            fitted=fitted,
            input_hashes=input_hashes,
            fitted_pipeline_file_sha256=sha256_file(pipeline_path),
        )
        record_path = partial_directory / "refit_record.json"
        write_final_random_forest_record(record_path, record)
        if load_final_random_forest_record(record_path) != record:
            raise RuntimeError("Final RF record changed after reload")
        entry = {
            "schema_version": 1,
            "experiment_id": configuration.experiment_id,
            "holdout_accessed": False,
            "source_revision": source["source_revision"],
            "refit_record": "refit_record.json",
            "refit_record_sha256": sha256_file(record_path),
            "fitted_pipeline": "fitted_pipeline.joblib",
            "fitted_pipeline_sha256": sha256_file(pipeline_path),
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
                "balanced_training_row_count": int(
                    fitted.balanced_training_indices.size
                ),
                "elapsed_seconds": fitted.elapsed_seconds,
                "holdout_accessed": False,
                "output": str(final_directory),
                "refit_entry_sha256": sha256_file(
                    final_directory / "refit_entry.json"
                ),
                "selected_feature_count": len(fitted.selected_feature_names),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
