"""Tests for deterministic scientific-artifact provenance primitives."""

from types import MappingProxyType

import pytest

from prevoccupai_har.provenance import sha256_canonical_json


def test_canonical_json_digest_ignores_mapping_order_and_sequence_type() -> None:
    first = MappingProxyType({"scale": (1.0, 2.0), "mean": [0.0, 0.5]})
    second = {"mean": (0.0, 0.5), "scale": [1.0, 2.0]}

    assert sha256_canonical_json(first) == sha256_canonical_json(second)


def test_canonical_json_digest_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        sha256_canonical_json({"mean": [float("nan")]})
