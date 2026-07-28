#!/usr/bin/env python3
"""Profile one configured HAR model using generated zeros only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from prevoccupai_har.modeling import (
    build_time_series_classifier,
    load_time_series_experiment_configuration,
)
from prevoccupai_har.profiling import (
    build_synthetic_complexity_profile_record,
    profile_model_complexity,
    write_synthetic_complexity_profile_record,
)
from prevoccupai_har.training import set_reproducible_seed


ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a configured compact model with generated zeros only. The "
            "hardware-specific artefact is not a participant-performance result."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/cnn_1d.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--timed-iterations", type=int, default=100)
    return parser


def main() -> int:
    """Create one immutable synthetic complexity-profile record."""
    arguments = _build_parser().parse_args()
    if arguments.threads <= 0:
        raise ValueError("PyTorch thread count must be positive")
    configuration = load_time_series_experiment_configuration(arguments.config)
    if configuration.status not in {"synthetic_validation_only", "frozen_for_development"}:
        raise PermissionError("The profiling command received an unsupported configuration")
    torch.set_num_threads(arguments.threads)
    set_reproducible_seed(arguments.seed)
    model = build_time_series_classifier(configuration)
    measurement = profile_model_complexity(
        model,
        input_shape=(
            1,
            configuration.architecture.input_channels,
            configuration.architecture.expected_samples,
        ),
        device=arguments.device,
        warmup_iterations=arguments.warmup_iterations,
        timed_iterations=arguments.timed_iterations,
    )
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    record = build_synthetic_complexity_profile_record(
        created_at_utc=created_at.isoformat().replace("+00:00", "Z"),
        experiment_id=configuration.experiment_id,
        source_revision="unversioned_workspace_software_test",
        model_configuration_path=arguments.config,
        random_seed=arguments.seed,
        device=arguments.device,
        measurement=measurement,
    )
    write_synthetic_complexity_profile_record(arguments.output, record)
    print(
        json.dumps(
            {
                "experiment_id": configuration.experiment_id,
                "holdout_accessed": False,
                "median_latency_ms": measurement.latency.median_ms,
                "output": str(arguments.output.resolve()),
                "scientific_result": False,
                "serialized_state_dict_bytes": measurement.serialized_state_dict_bytes,
                "trainable_parameter_count": measurement.trainable_parameter_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
