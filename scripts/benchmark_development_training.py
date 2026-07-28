#!/usr/bin/env python3
"""Benchmark development training throughput without producing predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from prevoccupai_har.model_selection import load_development_selection_plan
from prevoccupai_har.modeling import (
    build_time_series_classifier,
    load_model_input_sample_stride,
    load_time_series_experiment_configuration,
)
from prevoccupai_har.provenance import sha256_file
from prevoccupai_har.streaming_training import (
    benchmark_training_throughput,
    fit_streaming_channel_standardizer,
    indices_for_subjects,
)
from prevoccupai_har.window_store import load_development_window_store


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Measure a bounded number of training batches on one development fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--window-store",
        type=Path,
        default=ROOT / "artifacts/windows/approved_development_v1",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.threads <= 0:
        raise ValueError("PyTorch thread count must be positive")
    plan = load_development_selection_plan(arguments.plan)
    fold = next(value for value in plan.folds if value.fold_index == arguments.fold)
    configuration = load_time_series_experiment_configuration(arguments.config)
    sample_stride = load_model_input_sample_stride(arguments.config)
    if configuration.experiment_id not in {
        candidate.experiment_id for candidate in plan.candidates
    }:
        raise ValueError("Benchmark model is absent from the frozen plan")
    store = load_development_window_store(arguments.window_store)
    training_indices = indices_for_subjects(store, fold.training_subjects)
    standardizer_start = time.perf_counter()
    standardizer = fit_streaming_channel_standardizer(
        store,
        training_indices,
        allowed_training_subjects=fold.training_subjects,
        sample_stride=sample_stride,
    )
    standardizer_seconds = time.perf_counter() - standardizer_start
    torch.set_num_threads(arguments.threads)
    measurement = benchmark_training_throughput(
        build_time_series_classifier(configuration),
        store,
        training_indices,
        standardizer,
        optimization=configuration.optimization,
        seed=arguments.seed,
        maximum_batches=arguments.batches,
        sample_stride=sample_stride,
    )
    result = {
        "schema_version": 1,
        "status": "development_compute_feasibility_only",
        "scientific_result": False,
        "predictions_created": False,
        "holdout_accessed": False,
        "experiment_id": configuration.experiment_id,
        "fold_index": fold.fold_index,
        "random_seed": arguments.seed,
        "torch_threads": arguments.threads,
        "plan_sha256": sha256_file(arguments.plan),
        "model_configuration_sha256": sha256_file(arguments.config),
        "window_store_index_sha256": sha256_file(
            arguments.window_store / "index.json"
        ),
        "training_window_count": int(training_indices.size),
        "model_input_sample_stride": sample_stride,
        "standardizer_fit_seconds": standardizer_seconds,
        "measured_batches": measurement.measured_batches,
        "measured_examples": measurement.measured_examples,
        "elapsed_seconds": measurement.elapsed_seconds,
        "estimated_training_epoch_seconds": measurement.estimated_epoch_seconds,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
