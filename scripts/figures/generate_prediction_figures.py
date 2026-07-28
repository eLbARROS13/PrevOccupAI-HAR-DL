#!/usr/bin/env python3
"""Generate prediction-bound manuscript figures and a checksum manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.figure_generation import generate_prediction_figure_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate vector confusion-matrix, participant-macro-F1, and "
            "calibration-reliability figures from one immutable analysis record."
        )
    )
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Generate the requested figure package without overwriting an output."""
    arguments = _parser().parse_args()
    record = generate_prediction_figure_package(
        analysis_record_path=arguments.analysis,
        output_directory=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "analysis_id": record.analysis_id,
                "figure_count": len(record.figures),
                "holdout_accessed": record.holdout_accessed,
                "output_directory": str(arguments.output_dir.resolve()),
                "scientific_result": record.scientific_result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
