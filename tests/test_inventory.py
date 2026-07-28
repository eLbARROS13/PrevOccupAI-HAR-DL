"""Tests for privacy-preserving recording manifests."""

import json
from pathlib import Path

from prevoccupai_har.inventory import build_recording_manifest
from prevoccupai_har.protocol import ProtocolConfiguration


def _write_opensignals_file(path: Path) -> None:
    metadata = {
        "ABCDEF012345": {
            "device": "muscleBAN",
            "sampling rate": 1000,
            "sensor": ["RAW", "gACC", "gACC", "gACC"],
        }
    }
    path.write_text(
        "# OpenSignals Text File Format\n# "
        + json.dumps(metadata)
        + "\n# EndOfHeader\n1 2 3 4\n",
        encoding="utf-8",
    )


def _protocol(raw_root: Path) -> ProtocolConfiguration:
    return ProtocolConfiguration.from_mapping(
        {
            "schema_version": 1,
            "dataset_name": "synthetic",
            "source_status": "test",
            "raw_data_root": str(raw_root),
            "participant_id_pattern": r"^P[0-9]{3}$",
            "development_participants": ["P003"],
            "holdout_participants": ["P001"],
            "required_activity_directory_bases": ["walking"],
            "main_labels": ["walking"],
            "muscleban_filename_pattern": r"^opensignals_[0-9A-F]{12}_.+[.]txt$",
            "muscleban_sampling_rate_hz": 1000,
            "accelerometer_channels": 3,
            "window": {
                "duration_seconds": 5,
                "overlap_fraction": 0.5,
                "expected_samples": 5000,
            },
            "quality_assessment_manifest": None,
            "device_to_side_mapping": None,
            "training_authorized": False,
            "training_blockers": ["test"],
        }
    )


def test_manifest_omits_private_filename_device_id_and_timestamp(tmp_path: Path) -> None:
    activity = tmp_path / "P003" / "walking"
    activity.mkdir(parents=True)
    private_filename = "opensignals_ABCDEF012345_2026-01-02_03-04-05.txt"
    _write_opensignals_file(activity / private_filename)

    manifest = build_recording_manifest(_protocol(tmp_path), calculate_checksums=True)
    rendered = json.dumps(manifest)

    assert manifest["recording_count"] == 1
    assert manifest["recordings"][0]["sha256"] is not None
    assert private_filename not in rendered
    assert "ABCDEF012345" not in rendered
    assert "2026-01-02" not in rendered
    assert manifest["recordings"][0]["recording_index"] == 1

