#!/usr/bin/env python3
"""Assemble a complete immutable DL run grid for bundle construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prevoccupai_har.model_selection import load_development_selection_plan
from prevoccupai_har.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    plan = load_development_selection_plan(arguments.plan)
    expected = {
        (candidate.experiment_id, fold.fold_index, seed)
        for candidate in plan.candidates
        for fold in plan.folds
        for seed in plan.random_seeds
    }
    entries: dict[tuple[str, int, int], dict[str, object]] = {}
    source_revisions: set[str] = set()
    for entry_path in sorted(arguments.runs_root.rglob("run_entry.json")):
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        if not isinstance(entry, dict):
            raise TypeError(f"Run entry must be an object: {entry_path}")
        slot = (
            str(entry["candidate_id"]),
            int(entry["fold_index"]),
            int(entry["random_seed"]),
        )
        if slot not in expected:
            raise ValueError(f"Run entry occupies an undeclared slot: {slot}")
        if slot in entries:
            raise ValueError(f"Run slot is duplicated: {slot}")
        if entry.get("holdout_accessed") is not False:
            raise PermissionError("Development run unexpectedly accessed hold-out data")
        if entry.get("selection_plan_sha256") != sha256_file(arguments.plan):
            raise ValueError("Run entry does not bind the selected plan")
        source_revisions.add(str(entry["source_revision"]))
        run_directory = entry_path.parent
        paths = {}
        for field, digest_field in (
            ("training_result", "training_result_sha256"),
            ("prediction_artifact", "prediction_artifact_sha256"),
            ("analysis_record", "analysis_record_sha256"),
        ):
            path = run_directory / str(entry[field])
            if sha256_file(path) != str(entry[digest_field]):
                raise ValueError(f"Run artifact hash changed: {path}")
            paths[field] = str(path.resolve().relative_to(arguments.output.parent.resolve()))
        entries[slot] = {
            "candidate_id": slot[0],
            "fold_index": slot[1],
            "random_seed": slot[2],
            **paths,
        }
    missing = sorted(expected - set(entries))
    if missing:
        raise ValueError(f"Development run grid is incomplete; missing {len(missing)} slots")
    if len(source_revisions) != 1:
        raise ValueError("Complete run grid must share one immutable source revision")
    manifest = {
        "schema_version": 1,
        "purpose": plan.purpose,
        "holdout_accessed": False,
        "selection_plan_sha256": sha256_file(arguments.plan),
        "source_revision": next(iter(source_revisions)),
        "runs": [entries[slot] for slot in sorted(entries)],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "holdout_accessed": False,
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256_file(arguments.output),
                "run_count": len(entries),
                "source_revision": next(iter(source_revisions)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
