#!/usr/bin/env python3
"""Build approved development-only segment and side-provenance manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.approved_dataset import (
    build_approved_development_manifest,
    recover_development_device_provenance,
    write_json_exclusive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--device-mapping", type=Path, required=True)
    parser.add_argument("--device-provenance-output", type=Path, required=True)
    parser.add_argument("--development-manifest-output", type=Path, required=True)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def main() -> None:
    args = parse_args()
    approval = load_object(args.approval)
    device_mapping = load_object(args.device_mapping)
    provenance = recover_development_device_provenance(
        args.segments_root,
        args.raw_root,
        tuple(map(str, approval["development_participants"])),
        {
            str(device): str(side)
            for device, side in dict(device_mapping["device_to_side"]).items()
        },
    )
    manifest = build_approved_development_manifest(
        args.candidate_manifest,
        args.approval,
        provenance,
    )
    write_json_exclusive(args.device_provenance_output, provenance)
    write_json_exclusive(args.development_manifest_output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "development_segment_count": manifest["development_segment_count"],
                "retained_development_segment_count": manifest[
                    "retained_development_segment_count"
                ],
                "rejected_development_segment_count": manifest[
                    "rejected_development_segment_count"
                ],
                "device_provenance_group_count": manifest[
                    "device_provenance_group_count"
                ],
                "holdout_accessed": manifest["holdout_accessed"],
                "payload_sha256": manifest["payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
