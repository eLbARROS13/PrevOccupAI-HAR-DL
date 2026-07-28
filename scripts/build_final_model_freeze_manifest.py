#!/usr/bin/env python3
"""Build and strictly reload the complete pre-hold-out model freeze."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from prevoccupai_har.final_manifest import (
    build_final_model_freeze_manifest,
    write_final_model_freeze_manifest,
)
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> int:
    """Bind every final model and upstream identity into one immutable manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-source-manifest", type=Path, required=True)
    parser.add_argument(
        "--dl-refit-root",
        type=Path,
        default=ROOT / "artifacts/final_models/v1/dl_refits",
    )
    parser.add_argument(
        "--rf-refit-directory",
        type=Path,
        default=ROOT / "artifacts/final_models/v1/rf_refit",
    )
    arguments = parser.parse_args()
    dl_directories = [
        arguments.dl_refit_root / f"seed_{seed}"
        for seed in (1103, 2207, 3301, 4409, 5519)
    ]
    upstream = {
        "approved_dataset_configuration_sha256": ROOT
        / "configs/mban_processed_dataset_approval.json",
        "approved_segments_sha256": ROOT
        / "artifacts/data_audit/approved_development_segments.json",
        "candidate_manifest_sha256": ROOT
        / "artifacts/data_audit/candidate_snapshot_manifest.json",
        "development_environment_lock_sha256": ROOT
        / "artifacts/provenance/development_environment_lock_v1.json",
        "development_run_manifest_sha256": ROOT
        / "artifacts/development_selection/v1/development_run_manifest.json",
        "development_source_manifest_sha256": ROOT
        / "artifacts/provenance/source_tree_manifest_development_v1.json",
        "device_provenance_sha256": ROOT
        / "artifacts/data_audit/development_device_provenance.json",
        "exact_environment_requirements_sha256": ROOT
        / "requirements/development_environment_exact.txt",
        "feature_extraction_sha256": ROOT
        / "configs/mban_tsfel_feature_reconstruction.json",
        "feature_manifest": ROOT
        / "artifacts/data_audit/approved_development_feature_matrices.json",
        "final_execution_plan_sha256": ROOT
        / "artifacts/protocol/final_model_holdout_plan_predeclared_v1.json",
        "model_selection_bundle_sha256": ROOT
        / "artifacts/development_selection/v1/model_selection_bundle.json",
        "protocol_configuration_sha256": ROOT / "configs/mban_protocol.json",
        "rf_development_record_sha256": ROOT
        / "artifacts/development_selection/v1/rf_development_record.json",
        "rf_development_source_manifest_sha256": ROOT
        / "artifacts/provenance/source_tree_manifest_rf_development_v2.json",
        "signal_preprocessing_sha256": ROOT / "configs/mban_signal_preprocessing.json",
        "split_manifest_sha256": ROOT
        / "artifacts/data_audit/subject_split_manifest_authorized.json",
        "statistical_analysis_plan_sha256": ROOT
        / "extended_mban_har_paper/notes/statistical_analysis_plan.md",
        "window_store_index_sha256": ROOT
        / "artifacts/windows/approved_development_v1/index.json",
    }
    manifest = build_final_model_freeze_manifest(
        manifest_id="prevoccupai-final-model-freeze-v1",
        created_at_utc=_utc_now(),
        manifest_output_path=arguments.output,
        final_training_settings_path=ROOT
        / "artifacts/final_models/v1/final_training_settings.json",
        dl_refit_directories=dl_directories,
        rf_refit_directory=arguments.rf_refit_directory,
        selected_model_configuration_path=ROOT / "configs/cnn_1d.json",
        rf_configuration_path=ROOT / "configs/rf_baseline.json",
        final_stage_source_manifest_path=arguments.final_source_manifest,
        final_stage_source_root=ROOT,
        protocol=load_protocol(ROOT / "configs/mban_protocol.json"),
        upstream_artifacts=upstream,
    )
    write_final_model_freeze_manifest(arguments.output, manifest)
    print(
        json.dumps(
            {
                "dl_refit_count": len(manifest.dl_refits),
                "final_stage_source_revision": manifest.final_stage_source_revision,
                "holdout_accessed": False,
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256_file(arguments.output),
                "rf_selected_feature_count": len(
                    manifest.random_forest.selected_feature_names
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
