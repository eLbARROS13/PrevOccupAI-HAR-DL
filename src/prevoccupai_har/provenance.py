"""Small provenance primitives shared by scientific artifact contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
TREE_REVISION_PATTERN = re.compile(r"^tree-sha256:[0-9a-f]{64}$")


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of a file without retaining its path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_value(value: object) -> object:
    """Convert mappings and non-string sequences into canonical JSON containers."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_json_value(item) for item in value]
    return value


def sha256_canonical_json(value: object) -> str:
    """Hash a JSON-compatible value using deterministic keys and separators."""
    encoded = json.dumps(
        _canonical_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_reproducible_source_revision(value: str) -> bool:
    """Return whether a source identifier is an immutable Git or tree digest.

    The project root predates Git and contains several independent nested
    repositories.  Scientific runs therefore accept either a full Git commit or
    a deterministic manifest digest written as ``tree-sha256:<digest>``.
    """
    return bool(
        GIT_REVISION_PATTERN.fullmatch(value)
        or TREE_REVISION_PATTERN.fullmatch(value)
    )
