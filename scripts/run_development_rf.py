#!/usr/bin/env python3
"""Run the leakage-safe RF comparator on approved development features only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from prevoccupai_har.classical_baseline import (
    DEVELOPMENT_SELECTION,
    ScientificFeatureProvenance,
    evaluate_random_forest_development_folds,
    load_random_forest_reconstruction_configuration,
    write_random_forest_development_record,
)
from prevoccupai_har.feature_store import load_approved_development_feature_matrix
from prevoccupai_har.model_selection import load_development_selection_plan
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_file
from prevoccupai_har.source_snapshot import load_source_tree_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        "--window-store-index",
        type=Path,
        default=ROOT / "artifacts/windows/approved_development_v1/index.json",
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
    arguments = parser.parse_args()
    source = load_source_tree_manifest(
        arguments.source_manifest,
        root=ROOT,
        verify_current_tree=True,
    )
    protocol = load_protocol(arguments.protocol)
    configuration = load_random_forest_reconstruction_configuration(arguments.config)
    plan = load_development_selection_plan(arguments.plan)
    dataset = load_approved_development_feature_matrix(
        feature_root=arguments.feature_root,
        candidate_manifest_path=arguments.candidate_manifest,
        window_store_index_path=arguments.window_store_index,
        protocol=protocol,
    )
    recorded_manifest = json.loads(arguments.feature_manifest.read_text(encoding="utf-8"))
    if recorded_manifest != dataset.manifest:
        raise ValueError("Development feature manifest differs from the loaded matrices")
    created_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    record = evaluate_random_forest_development_folds(
        run_id="rf-tsfel-development-selection-v1",
        created_at_utc=created_at,
        configuration_path=arguments.config,
        configuration=configuration,
        features=dataset.features,
        labels=dataset.labels,
        subactivity_labels=dataset.subactivity_labels,
        participant_ids=dataset.participant_ids,
        feature_names=dataset.feature_names,
        class_labels=plan.class_labels,
        folds=plan.folds,
        purpose=DEVELOPMENT_SELECTION,
        source_revision=str(source["source_revision"]),
        protocol=protocol,
        protocol_configuration_path=arguments.protocol,
        data_provenance=ScientificFeatureProvenance(
            raw_recording_manifest_sha256=sha256_file(arguments.device_provenance),
            segmentation_manifest_sha256=sha256_file(arguments.approved_segments),
            quality_manifest_sha256=sha256_file(arguments.approved_segments),
            split_manifest_sha256=sha256_file(arguments.split_manifest),
            signal_preprocessing_configuration_sha256=sha256_file(
                arguments.signal_preprocessing
            ),
            feature_extraction_configuration_sha256=sha256_file(
                arguments.feature_extraction
            ),
            feature_matrix_file_sha256=sha256_file(arguments.feature_manifest),
        ),
    )
    write_random_forest_development_record(arguments.output, record)
    print(
        json.dumps(
            {
                "fold_count": len(record.folds),
                "holdout_accessed": False,
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256_file(arguments.output),
                "row_count": dataset.features.shape[0],
                "scientific_result": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
