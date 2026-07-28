"""Tests for protocol validation."""

from pathlib import Path

import pytest

from prevoccupai_har.protocol import ProtocolConfiguration, load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_protocol_is_internally_consistent() -> None:
    protocol = load_protocol(PROJECT_ROOT / "configs" / "mban_protocol.json")

    assert protocol.sampling_rate_hz == 1000
    assert protocol.window.expected_samples == 5000
    assert protocol.window.step_samples == 2500
    assert len(protocol.development_participants) == 16
    assert protocol.holdout_participants == ("P001", "P002", "P016", "P018")
    assert protocol.segmentation_manifest is not None
    assert protocol.quality_assessment_manifest is not None
    assert protocol.device_to_side_mapping is not None
    assert protocol.signal_preprocessing_configuration is not None
    assert protocol.segmentation_contract_configuration is not None
    assert protocol.training_authorized is True
    assert protocol.training_authorization_scope == "development_selection_only"
    assert protocol.holdout_access_authorized is False
    assert protocol.source_status == "public_release_method_configuration"
    assert protocol.device_to_side_mapping.name == "mban_device_to_side.example.json"


def test_protocol_rejects_subject_overlap() -> None:
    value = {
        "schema_version": 1,
        "dataset_name": "synthetic",
        "source_status": "test",
        "raw_data_root": "raw",
        "participant_id_pattern": r"^P[0-9]{3}$",
        "development_participants": ["P001", "P002"],
        "holdout_participants": ["P002"],
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

    with pytest.raises(ValueError, match="overlap"):
        ProtocolConfiguration.from_mapping(value)


def test_protocol_refuses_training_without_governance_paths() -> None:
    value = {
        "schema_version": 1,
        "dataset_name": "synthetic",
        "source_status": "test",
        "raw_data_root": "raw",
        "participant_id_pattern": r"^P[0-9]{3}$",
        "development_participants": ["P001"],
        "holdout_participants": ["P002"],
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
        "segmentation_manifest": None,
        "device_to_side_mapping": None,
        "signal_preprocessing_configuration": None,
        "segmentation_contract_configuration": None,
        "training_authorized": True,
        "training_authorization_scope": "development_selection_only",
        "holdout_access_authorized": False,
        "training_blockers": [],
    }

    with pytest.raises(ValueError, match="governance paths"):
        ProtocolConfiguration.from_mapping(value)
