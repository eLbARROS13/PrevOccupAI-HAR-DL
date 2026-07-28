#!/usr/bin/env python3
"""Build the frozen non-predictive model-selection plan."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from prevoccupai_har.model_selection import (
    build_development_selection_plan,
    write_development_selection_plan,
)
from prevoccupai_har.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hash-bound candidate/fold/seed plan. This command performs no "
            "training, produces no scientific result, and cannot access hold-out data."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/development_model_selection.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Build, exclusively write, and summarize the non-predictive plan."""
    arguments = _parser().parse_args()
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    plan = build_development_selection_plan(
        configuration_path=arguments.config,
        created_at_utc=created_at.isoformat().replace("+00:00", "Z"),
    )
    write_development_selection_plan(arguments.output, plan)
    print(
        json.dumps(
            {
                "expected_run_count": plan.expected_run_count,
                "holdout_accessed": plan.holdout_accessed,
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256_file(arguments.output),
                "plan_id": plan.plan_id,
                "scientific_result": plan.scientific_result,
                "training_authorized": plan.training_authorized,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
