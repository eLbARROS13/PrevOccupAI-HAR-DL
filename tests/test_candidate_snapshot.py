"""Tests for non-authoritative candidate processed-dataset audits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from prevoccupai_har.candidate_snapshot import build_candidate_snapshot_manifest


def _write_qa_summary(path: Path, *, status: str = "GOOD") -> None:
    fieldnames = [
        "original_file",
        "segment_name",
        "segment_duration_sec",
        "axis",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for axis in ("xAcc", "yAcc", "zAcc"):
            writer.writerow(
                {
                    "original_file": "P019_walking_file01_walk_slow.csv",
                    "segment_name": "complete_file",
                    "segment_duration_sec": "25.0",
                    "axis": axis,
                    "status": status if axis == "zAcc" else "GOOD",
                }
            )


def _write_feature_snapshot(root: Path) -> None:
    root.mkdir()
    columns = ["x_ACC_Mean", "y_ACC_Mean", "z_ACC_Mean", "main_label", "sub_label"]
    (root / "class_instances.json").write_text(
        json.dumps({"feature_cols": columns, "P019": {"7": 2}}),
        encoding="utf-8",
    )
    np.save(root / "P019.npy", np.array([[1.0, 2.0, 3.0, 2.0, 7.0]]))


def test_candidate_manifest_is_explicitly_non_authoritative(tmp_path: Path) -> None:
    segments = tmp_path / "segments"
    features = tmp_path / "features"
    segments.mkdir()
    qa_path = tmp_path / "qa.csv"
    _write_qa_summary(qa_path)
    _write_feature_snapshot(features)
    np.save(
        segments / "P019_walking_file01_walk_slow_GlobalSegment1.npy",
        np.zeros((25_000, 4)),
    )

    manifest = build_candidate_snapshot_manifest(
        segments,
        qa_path,
        features,
        calculate_checksums=True,
    )

    assert manifest["status"] == "candidate_snapshot_not_authoritative"
    assert manifest["scientific_training_authorized"] is False
    assert manifest["holdout_accessed"] is False
    assert manifest["quality_summary"]["evaluated_segment_count"] == 1
    assert manifest["segments"]["retained_array_count"] == 1
    assert manifest["feature_matrices"]["candidate_feature_count"] == 3
    assert manifest["snapshot_content_sha256"] is not None
    assert manifest["all_published_alignment_checks_pass"] is False


def test_candidate_manifest_rejects_filename_qa_disagreement(tmp_path: Path) -> None:
    segments = tmp_path / "segments"
    features = tmp_path / "features"
    segments.mkdir()
    qa_path = tmp_path / "qa.csv"
    _write_qa_summary(qa_path, status="BAD")
    _write_feature_snapshot(features)
    np.save(
        segments / "P019_walking_file01_walk_slow_GlobalSegment1.npy",
        np.zeros((25_000, 4)),
    )

    with pytest.raises(ValueError, match="Filename and QA status disagree"):
        build_candidate_snapshot_manifest(
            segments,
            qa_path,
            features,
            calculate_checksums=False,
        )

