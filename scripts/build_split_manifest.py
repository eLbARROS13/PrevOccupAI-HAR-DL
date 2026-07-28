#!/usr/bin/env python3
"""Create deterministic subject-disjoint validation folds from the protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prevoccupai_har.protocol import load_protocol  # noqa: E402
from prevoccupai_har.provenance import sha256_canonical_json, sha256_file  # noqa: E402
from prevoccupai_har.splits import build_validation_folds  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mban_protocol.json",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit" / "subject_split_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    """Build, validate, and save split metadata without reading signal data."""
    args = parse_args()
    protocol = load_protocol(args.config)
    folds = build_validation_folds(
        protocol.development_participants,
        protocol.holdout_participants,
        n_splits=args.n_splits,
        random_seed=args.seed,
    )
    manifest = {
        "schema_version": 1,
        "status": "authorized_development_subject_split",
        "strategy": "fixed_external_holdout_with_subject_disjoint_development_folds",
        "random_seed": args.seed,
        "n_splits": args.n_splits,
        "training_authorized": protocol.training_authorized,
        "training_authorization_scope": protocol.training_authorization_scope,
        "holdout_access_authorized": protocol.holdout_access_authorized,
        "holdout_accessed": False,
        "training_blockers": list(protocol.training_blockers),
        "protocol_configuration_sha256": sha256_file(args.config),
        "folds": [fold.as_dict() for fold in folds],
    }
    manifest["payload_sha256"] = sha256_canonical_json(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
