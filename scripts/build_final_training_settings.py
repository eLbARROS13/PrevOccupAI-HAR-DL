#!/usr/bin/env python3
"""Freeze mechanically derived DL epochs and RF hyperparameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.final_models import (
    build_final_training_settings,
    write_final_training_settings,
)
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Build and exclusively write the post-development settings artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--development-run-manifest",
        type=Path,
        default=ROOT / "artifacts/development_selection/v1/development_run_manifest.json",
    )
    parser.add_argument(
        "--selection-bundle",
        type=Path,
        default=ROOT / "artifacts/development_selection/v1/model_selection_bundle.json",
    )
    parser.add_argument(
        "--rf-development-record",
        type=Path,
        default=ROOT / "artifacts/development_selection/v1/rf_development_record.json",
    )
    parser.add_argument(
        "--rf-config",
        type=Path,
        default=ROOT / "configs/rf_baseline.json",
    )
    parser.add_argument(
        "--final-execution-plan",
        type=Path,
        default=ROOT / "artifacts/protocol/final_model_holdout_plan_predeclared_v1.json",
    )
    parser.add_argument(
        "--selected-model-config",
        type=Path,
        default=ROOT / "configs/cnn_1d.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/mban_protocol.json",
    )
    arguments = parser.parse_args()
    settings = build_final_training_settings(
        settings_id="prevoccupai-final-training-settings-v1",
        development_run_manifest_path=arguments.development_run_manifest,
        model_selection_bundle_path=arguments.selection_bundle,
        rf_development_record_path=arguments.rf_development_record,
        rf_configuration_path=arguments.rf_config,
        final_execution_plan_path=arguments.final_execution_plan,
        selected_model_configuration_path=arguments.selected_model_config,
        protocol=load_protocol(arguments.protocol),
    )
    write_final_training_settings(arguments.output, settings)
    print(
        json.dumps(
            {
                "dl_fixed_epochs": {
                    str(value.random_seed): value.fixed_epoch_count
                    for value in settings.epoch_decisions
                },
                "holdout_accessed": False,
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256_file(arguments.output),
                "rf_hyperparameters": settings.rf_hyperparameters,
                "selected_candidate_id": settings.selected_candidate_id,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
