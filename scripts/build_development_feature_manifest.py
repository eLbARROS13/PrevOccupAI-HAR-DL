#!/usr/bin/env python3
"""Verify development-only TSFEL inputs and write their immutable manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.feature_store import (
    load_approved_development_feature_matrix,
    write_development_feature_manifest,
)
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=ROOT / "artifacts/data_audit/candidate_snapshot_manifest.json",
    )
    parser.add_argument(
        "--window-store-index",
        type=Path,
        default=ROOT / "artifacts/windows/approved_development_v1/index.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/mban_protocol.json",
    )
    arguments = parser.parse_args()
    dataset = load_approved_development_feature_matrix(
        feature_root=arguments.feature_root,
        candidate_manifest_path=arguments.candidate_manifest,
        window_store_index_path=arguments.window_store_index,
        protocol=load_protocol(arguments.protocol),
    )
    write_development_feature_manifest(arguments.output, dataset)
    print(
        json.dumps(
            {
                "class_counts": dataset.manifest["class_counts"],
                "holdout_values_loaded": False,
                "manifest_sha256": sha256_file(arguments.output),
                "output": str(arguments.output.resolve()),
                "row_count": dataset.features.shape[0],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
