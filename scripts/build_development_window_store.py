#!/usr/bin/env python3
"""Materialize the approved development-only ACC window store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.window_store import build_development_window_store


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-root", type=Path, required=True)
    parser.add_argument(
        "--approved-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/approved_development_segments.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/candidate_snapshot_manifest.json",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/mban_protocol.json"
    )
    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=ROOT / "configs/mban_signal_preprocessing.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = build_development_window_store(
        approved_manifest_path=args.approved_manifest,
        candidate_manifest_path=args.candidate_manifest,
        segment_root=args.segments_root,
        protocol_path=args.protocol,
        preprocessing_configuration_path=args.preprocessing,
        output_directory=args.output,
    )
    print(
        json.dumps(
            {
                "status": index["status"],
                "window_count": index["window_count"],
                "window_counts_per_class": index["window_counts_per_class"],
                "holdout_accessed": index["holdout_accessed"],
                "payload_sha256": index["payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
