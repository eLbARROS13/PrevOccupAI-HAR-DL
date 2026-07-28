#!/usr/bin/env python3
"""Audit a candidate processed dataset snapshot without authorizing training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prevoccupai_har.candidate_snapshot import build_candidate_snapshot_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-root", type=Path, required=True)
    parser.add_argument("--qa-summary", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit" / "candidate_snapshot_manifest.json",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Create a structural comparison only; do not treat it as a frozen snapshot identity.",
    )
    return parser.parse_args()


def main() -> None:
    """Audit and save the candidate snapshot manifest."""
    args = parse_args()
    manifest = build_candidate_snapshot_manifest(
        args.segments_root,
        args.qa_summary,
        args.features_root,
        calculate_checksums=not args.skip_checksums,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": manifest["status"],
        "scientific_training_authorized": manifest["scientific_training_authorized"],
        "participants": manifest["participants"],
        "evaluated_segment_count": manifest["quality_summary"]["evaluated_segment_count"],
        "retained_segment_count": manifest["quality_summary"]["retained_segment_count"],
        "rejected_segment_count": manifest["quality_summary"]["rejected_segment_count"],
        "feature_matrix_window_count": manifest["feature_matrices"]["total_window_count"],
        "all_published_alignment_checks_pass": manifest[
            "all_published_alignment_checks_pass"
        ],
        "snapshot_content_sha256": manifest["snapshot_content_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

