#!/usr/bin/env python3
"""Generate deterministic figures and tables from a selection bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.selection_reporting import generate_selection_report_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired-seed, learning-curve, and complexity-performance vector "
            "figures plus CSV summaries from one validated selection bundle."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Generate the output package without overwriting an existing directory."""
    arguments = _parser().parse_args()
    record = generate_selection_report_package(
        selection_bundle_path=arguments.bundle,
        output_directory=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "bundle_id": record.bundle_id,
                "holdout_accessed": record.holdout_accessed,
                "output_count": len(record.outputs),
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
