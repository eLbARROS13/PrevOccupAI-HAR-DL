"""Deterministic source-tree manifests for a non-Git project root."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .provenance import sha256_canonical_json, sha256_file


DEFAULT_SOURCE_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "configs/*.json",
    "pyproject.toml",
)


@dataclass(frozen=True)
class SourceFileRecord:
    """One path-relative source-file identity."""

    relative_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _selected_paths(root: Path, patterns: Iterable[str]) -> tuple[Path, ...]:
    paths = {
        path.resolve()
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def build_source_tree_manifest(
    root: Path | str,
    *,
    patterns: Iterable[str] = DEFAULT_SOURCE_PATTERNS,
) -> dict[str, object]:
    """Hash the governed executable source without traversing data or artifacts."""
    root_path = Path(root).resolve()
    selected_patterns = tuple(map(str, patterns))
    paths = _selected_paths(root_path, selected_patterns)
    if not paths:
        raise ValueError("Source-tree manifest selected no files")
    records = tuple(
        SourceFileRecord(
            relative_path=path.relative_to(root_path).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in paths
    )
    payload = {
        "patterns": list(selected_patterns),
        "files": [record.as_dict() for record in records],
    }
    payload_sha256 = sha256_canonical_json(payload)
    return {
        "schema_version": 1,
        "status": "immutable_source_tree_snapshot",
        "source_revision": f"tree-sha256:{payload_sha256}",
        "file_count": len(records),
        **payload,
        "payload_sha256": payload_sha256,
    }


def write_source_tree_manifest(path: Path | str, manifest: dict[str, object]) -> None:
    """Write a source manifest exclusively."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_source_tree_manifest(
    path: Path | str,
    *,
    root: Path | str | None = None,
    verify_current_tree: bool = False,
) -> dict[str, object]:
    """Load, validate, and optionally compare an immutable source manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Source-tree manifest must be a JSON object")
    if manifest.get("schema_version") != 1 or manifest.get("status") != (
        "immutable_source_tree_snapshot"
    ):
        raise ValueError("Source-tree manifest status or schema is unsupported")
    payload = {
        "patterns": manifest.get("patterns"),
        "files": manifest.get("files"),
    }
    observed = sha256_canonical_json(payload)
    if manifest.get("payload_sha256") != observed or manifest.get(
        "source_revision"
    ) != f"tree-sha256:{observed}":
        raise ValueError("Source-tree manifest payload identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or int(manifest.get("file_count", -1)) != len(files):
        raise ValueError("Source-tree manifest file count is invalid")
    if verify_current_tree:
        if root is None:
            raise ValueError("Current-tree verification requires a project root")
        patterns = manifest.get("patterns")
        if not isinstance(patterns, list):
            raise TypeError("Source-tree manifest patterns must be an array")
        current = build_source_tree_manifest(root, patterns=map(str, patterns))
        if current["source_revision"] != manifest["source_revision"]:
            raise ValueError("Current executable source differs from the frozen manifest")
    return manifest
