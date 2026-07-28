#!/usr/bin/env python3
"""Build one immutable derived-analysis record from development predictions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from prevoccupai_har.analysis_records import (
    build_prediction_analysis_record,
    write_prediction_analysis_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate classification, uncalibrated probability, and temporal "
            "diagnostics from a development-prediction artifact. External hold-out "
            "artifacts are not accepted."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--step-size-samples", type=int, required=True)
    parser.add_argument("--short-run-max-windows", type=int, default=1)
    return parser


def main() -> int:
    """Build and write the requested derived-analysis record."""
    arguments = _parser().parse_args()
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    record = build_prediction_analysis_record(
        analysis_id=arguments.analysis_id,
        created_at_utc=created_at.isoformat().replace("+00:00", "Z"),
        prediction_artifact_path=arguments.predictions,
        calibration_bin_count=arguments.calibration_bins,
        expected_step_size_samples=arguments.step_size_samples,
        short_run_max_windows=arguments.short_run_max_windows,
    )
    write_prediction_analysis_record(arguments.output, record)
    print(
        json.dumps(
            {
                "analysis_id": record.analysis_id,
                "holdout_accessed": record.holdout_accessed,
                "output": str(arguments.output.resolve()),
                "prediction_artifact_sha256": record.prediction_artifact_sha256,
                "scientific_result": record.scientific_result,
                "window_count": record.window_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
