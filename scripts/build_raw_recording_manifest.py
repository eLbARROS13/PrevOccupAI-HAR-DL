#!/usr/bin/env python3
"""Build a privacy-preserving, checksummed raw-recording manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prevoccupai_har.inventory import build_recording_manifest  # noqa: E402
from prevoccupai_har.protocol import load_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mban_protocol.json",
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit" / "raw_recording_manifest.json",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Create a structural inventory only; not suitable as a frozen snapshot identity.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save the manifest."""
    args = parse_args()
    protocol = load_protocol(args.config)
    manifest = build_recording_manifest(
        protocol,
        raw_root=args.raw_root,
        calculate_checksums=not args.skip_checksums,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        key: manifest[key]
        for key in (
            "recording_count",
            "total_file_size_bytes",
            "checksums_calculated",
            "manifest_content_sha256",
            "observed_participants",
            "missing_expected_participants",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

