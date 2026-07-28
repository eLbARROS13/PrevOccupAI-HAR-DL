#!/usr/bin/env python3
"""Build a complete provenance-bound repeated-seed/fold selection bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from prevoccupai_har.model_selection import (
    build_model_selection_bundle,
    write_model_selection_bundle,
)
from prevoccupai_har.provenance import sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an exact repeated-seed/fold artifact grid and apply the frozen "
            "development-only selection rule. External hold-out evidence is rejected."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Build, exclusively write, and summarize a complete selection bundle."""
    arguments = _parser().parse_args()
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    bundle = build_model_selection_bundle(
        bundle_id=arguments.bundle_id,
        created_at_utc=created_at.isoformat().replace("+00:00", "Z"),
        selection_plan_path=arguments.plan,
        run_manifest_path=arguments.run_manifest,
    )
    write_model_selection_bundle(arguments.output, bundle)
    print(
        json.dumps(
            {
                "bundle_id": bundle.bundle_id,
                "holdout_accessed": bundle.holdout_accessed,
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256_file(arguments.output),
                "run_count": bundle.run_count,
                "scientific_result": bundle.scientific_result,
                "selected_candidate_id": bundle.decision["selected_candidate_id"],
                "status": bundle.decision["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
