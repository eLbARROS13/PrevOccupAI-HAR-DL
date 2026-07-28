"""Tests for the hold-out-sealed approved feature-matrix loader."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from prevoccupai_har.feature_store import load_approved_development_feature_matrix
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_file


def test_loader_reads_only_declared_development_matrices(tmp_path: Path) -> None:
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    participants = ("P003", "P004")
    entries: list[dict[str, object]] = []
    for participant_index, participant in enumerate(participants):
        values = np.zeros((3, 47), dtype=np.float64)
        values[:, :45] = participant_index + np.arange(45)
        values[:, 45] = (0, 1, 2)
        values[:, 46] = (0, 3, 7)
        path = feature_root / f"{participant}.npy"
        np.save(path, values, allow_pickle=False)
        entries.append(
            {
                "participant_id": participant,
                "relative_name": path.name,
                "sha256": sha256_file(path),
                "shape": [3, 47],
                "size_bytes": path.stat().st_size,
            }
        )
    # A malformed hold-out file proves that its values are never opened.
    (feature_root / "P001.npy").write_bytes(b"not-a-numpy-file")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "feature_matrices": {
                    "checksums_calculated": True,
                    "candidate_feature_count": 45,
                    "feature_columns": [f"feature_{index:02d}" for index in range(45)],
                    "matrices": entries,
                }
            }
        ),
        encoding="utf-8",
    )
    window_index_path = tmp_path / "window-index.json"
    window_index_path.write_text(
        json.dumps(
            {
                "window_counts_per_participant": {participant: 3 for participant in participants}
            }
        ),
        encoding="utf-8",
    )
    protocol = replace(
        load_protocol("configs/mban_protocol.json"),
        development_participants=participants,
        holdout_participants=("P001",),
    )

    dataset = load_approved_development_feature_matrix(
        feature_root=feature_root,
        candidate_manifest_path=candidate_path,
        window_store_index_path=window_index_path,
        protocol=protocol,
    )

    assert dataset.features.shape == (6, 45)
    assert dataset.manifest["holdout_values_loaded"] is False
    assert dataset.manifest["class_counts"] == {
        "sitting": 2,
        "standing": 2,
        "walking": 2,
    }
    assert set(dataset.participant_ids) == set(participants)
