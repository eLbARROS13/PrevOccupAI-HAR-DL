#!/usr/bin/env python3
"""Create an immutable source-tree identity for scientific runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.provenance import sha256_file
from prevoccupai_har.source_snapshot import (
    DEFAULT_SOURCE_PATTERNS,
    build_source_tree_manifest,
    write_source_tree_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Build, exclusively write, and summarize the source manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help=(
            "Root-relative glob to bind; repeat for a custom source scope. "
            "Defaults to the complete executable-source pattern set."
        ),
    )
    arguments = parser.parse_args()
    patterns = arguments.patterns or list(DEFAULT_SOURCE_PATTERNS)
    manifest = build_source_tree_manifest(ROOT, patterns=patterns)
    write_source_tree_manifest(arguments.output, manifest)
    print(
        json.dumps(
            {
                "file_count": manifest["file_count"],
                "manifest_sha256": sha256_file(arguments.output),
                "output": str(arguments.output.resolve()),
                "source_revision": manifest["source_revision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
