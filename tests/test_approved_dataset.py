import json
from pathlib import Path

import numpy as np
import pytest

from prevoccupai_har.approved_dataset import (
    build_approved_development_manifest,
    normalise_raw_activity_directory,
    parse_segment_filename,
    recover_development_device_provenance,
)
from prevoccupai_har.provenance import sha256_canonical_json, sha256_file


def _acceleration(adc: int) -> float:
    return (adc - 32768) * (16 / 65536) * 9.80665


def _write_raw(path: Path, rows: np.ndarray) -> None:
    lines = [
        "# OpenSignals Text File Format\n",
        "# {}\n",
        "# EndOfHeader\n",
    ]
    for nseq, x, y, z in rows:
        lines.append(f"{nseq}\t0\t32000\t{x}\t{y}\t{z}\t32768\t32768\t32768\n")
    path.write_text("".join(lines), encoding="utf-8")


def _segment_rows(rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            rows[:, 0],
            [_acceleration(int(value)) for value in rows[:, 1]],
            [_acceleration(int(value)) for value in rows[:, 2]],
            [_acceleration(int(value)) for value in rows[:, 3]],
        )
    )


def test_segment_parser_freezes_labels_and_activity_tokens() -> None:
    parsed = parse_segment_filename(
        "P003_cabinets2_file02_drink_coffee_GlobalSegment1.npy"
    )
    assert parsed.participant_id == "P003"
    assert parsed.activity_token == "cabinets2"
    assert parsed.file_token == "file02"
    assert parsed.main_label == "standing"
    assert parsed.sub_activity_label == "cabinets_coffee_tea"
    assert parsed.quality_status == "GOOD"
    assert normalise_raw_activity_directory("cabinets_2") == "cabinets2"
    assert normalise_raw_activity_directory("walking_1") == "walking"


def test_exact_signatures_recover_device_side_without_loading_holdout(tmp_path: Path) -> None:
    segment_root = tmp_path / "segments"
    raw_root = tmp_path / "raw"
    activity_root = raw_root / "P003" / "walking_1"
    segment_root.mkdir()
    activity_root.mkdir(parents=True)

    left_rows = np.column_stack(
        (np.arange(20), 34000 + np.arange(20), 35000 + np.arange(20), 36000 + np.arange(20))
    )
    right_rows = np.column_stack(
        (np.arange(20), 37000 + np.arange(20), 38000 + np.arange(20), 39000 + np.arange(20))
    )
    _write_raw(activity_root / "opensignals_AAAAAAAAAAAA_2024-01-01_00-00-00.txt", left_rows)
    _write_raw(activity_root / "opensignals_BBBBBBBBBBBB_2024-01-01_00-00-01.txt", right_rows)
    np.save(
        segment_root / "P003_walking_file01_walk_slow_GlobalSegment1.npy",
        _segment_rows(right_rows[2:17]),
    )
    np.save(
        segment_root / "P003_walking_file02_walk_slow_GlobalSegment1.npy",
        _segment_rows(left_rows[1:16]),
    )

    result = recover_development_device_provenance(
        segment_root,
        raw_root,
        ("P003",),
        {"AAAAAAAAAAAA": "left", "BBBBBBBBBBBB": "right"},
    )

    assert result["holdout_values_loaded"] is False
    assert result["all_groups_resolved"] is True
    mapping = {entry["file_token"]: entry for entry in result["mappings"]}
    assert mapping["file01"]["sensor_side"] == "right"
    assert mapping["file02"]["sensor_side"] == "left"
    assert all(entry["matched_signature_count"] >= 3 for entry in mapping.values())


def test_approved_manifest_binds_snapshot_approval_and_development_only(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    approval_path = tmp_path / "approval.json"
    segment_entry = {
        "relative_name": "P003_walking_file01_walk_slow_GlobalSegment1.npy",
        "participant_id": "P003",
        "quality_status": "GOOD",
        "shape": [5000, 4],
        "dtype": "float64",
        "size_bytes": 160128,
        "sha256": "a" * 64,
    }
    candidate = {
        "snapshot_content_sha256": "b" * 64,
        "all_published_alignment_checks_pass": True,
        "segments": {"content_sha256": "c" * 64, "arrays": [segment_entry]},
        "quality_summary": {"sha256": "d" * 64},
    }
    approval = {
        "approved_snapshot_content_sha256": "b" * 64,
        "scientific_training_authorized_development": True,
        "holdout_access_authorized": False,
        "development_participants": ["P003"],
        "holdout_participants": ["P001"],
        "approved_on": "2026-07-16",
        "approval_basis": "test approval",
        "notes": [],
    }
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    provenance = {
        "authoritative": True,
        "holdout_values_loaded": False,
        "group_count": 1,
        "payload_sha256": "e" * 64,
        "mappings": [
            {
                "participant_id": "P003",
                "activity_token": "walking",
                "file_token": "file01",
                "device_stream_id": "BBBBBBBBBBBB",
                "sensor_side": "right",
                "evidence": "exact_nseq_and_inverse_adc_signature_match",
            }
        ],
    }

    manifest = build_approved_development_manifest(
        candidate_path,
        approval_path,
        provenance,
    )

    assert manifest["status"] == "author_approved_development_dataset"
    assert manifest["scientific_training_authorized"] is True
    assert manifest["authorization_scope"] == "development_selection_only"
    assert manifest["holdout_accessed"] is False
    assert manifest["segments"][0]["sensor_side"] == "right"
    assert manifest["source_identity"]["candidate_manifest_sha256"] == sha256_file(
        candidate_path
    )
    payload_without_hash = dict(manifest)
    assert payload_without_hash.pop("payload_sha256") == sha256_canonical_json(
        payload_without_hash
    )


def test_approval_digest_substitution_is_rejected(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    approval_path = tmp_path / "approval.json"
    candidate_path.write_text(
        json.dumps(
            {
                "snapshot_content_sha256": "a" * 64,
                "all_published_alignment_checks_pass": True,
            }
        ),
        encoding="utf-8",
    )
    approval_path.write_text(
        json.dumps(
            {
                "approved_snapshot_content_sha256": "b" * 64,
                "scientific_training_authorized_development": True,
                "holdout_access_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digests differ"):
        build_approved_development_manifest(
            candidate_path,
            approval_path,
            {},
        )
