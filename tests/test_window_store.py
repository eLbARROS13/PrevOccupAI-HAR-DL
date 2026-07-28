import json
from pathlib import Path

import numpy as np
import pytest

from prevoccupai_har.provenance import sha256_canonical_json, sha256_file
from prevoccupai_har.window_store import (
    FILTER_TRANSIENT_SAMPLES,
    build_development_window_store,
    load_development_window_store,
)


def _write_protocol(path: Path, manifest_path: Path, preprocessing_path: Path) -> None:
    value = {
        "schema_version": 1,
        "dataset_name": "synthetic approved",
        "source_status": "approved",
        "raw_data_root": "raw",
        "participant_id_pattern": "^P[0-9]{3}$",
        "development_participants": ["P003"],
        "holdout_participants": ["P001"],
        "required_activity_directory_bases": ["walking"],
        "main_labels": ["sitting", "standing", "walking"],
        "muscleban_filename_pattern": "^opensignals_[0-9A-F]{12}_.+[.]txt$",
        "muscleban_sampling_rate_hz": 1000,
        "accelerometer_channels": 3,
        "window": {"duration_seconds": 5, "overlap_fraction": 0.5, "expected_samples": 5000},
        "quality_assessment_manifest": str(manifest_path),
        "segmentation_manifest": str(manifest_path),
        "device_to_side_mapping": "mapping.json",
        "signal_preprocessing_configuration": str(preprocessing_path),
        "segmentation_contract_configuration": "segmentation.json",
        "training_authorized": True,
        "training_authorization_scope": "development_selection_only",
        "holdout_access_authorized": False,
        "training_blockers": [],
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_and_reload_approved_window_store(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    artifacts = tmp_path / "artifacts"
    segments = tmp_path / "segments"
    configs.mkdir()
    artifacts.mkdir()
    segments.mkdir()
    manifest_path = artifacts / "approved.json"
    candidate_path = artifacts / "candidate.json"
    preprocessing_path = configs / "preprocessing.json"
    protocol_path = configs / "protocol.json"
    output = artifacts / "windows"

    sample_count = FILTER_TRANSIENT_SAMPLES + 5000
    time = np.arange(sample_count, dtype=np.float64) / 1000
    values = np.column_stack(
        (
            np.arange(sample_count) % 65536,
            9.80665 + np.sin(2 * np.pi * 2 * time),
            np.cos(2 * np.pi * 3 * time),
            np.sin(2 * np.pi * 5 * time),
        )
    )
    segment_name = "P003_walking_file01_walk_slow_GlobalSegment1.npy"
    segment_path = segments / segment_name
    np.save(segment_path, values)
    manifest = {
        "schema_version": 1,
        "status": "author_approved_development_dataset",
        "scientific_training_authorized": True,
        "holdout_accessed": False,
        "development_participants": ["P003"],
        "segments": [
            {
                "relative_name": segment_name,
                "participant_id": "P003",
                "main_label": "walking",
                "sub_activity_label": "walking_slow",
                "quality_status": "GOOD",
                "shape": list(values.shape),
                "dtype": "float64",
                "size_bytes": segment_path.stat().st_size,
                "sha256": sha256_file(segment_path),
                "recording_id": "a" * 64,
                "device_stream_id": "BBBBBBBBBBBB",
                "sensor_side": "right",
            }
        ],
    }
    manifest["payload_sha256"] = sha256_canonical_json(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = {
        "feature_matrices": {
            "matrices": [{"participant_id": "P003", "window_count": 1}]
        }
    }
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    preprocessing_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "synthetic_approved_preprocessing",
                "authoritative": True,
                "controls_dataset_generation": True,
                "source": {},
                "notes": [],
                "parameters": {
                    "sampling_rate_hz": 1000,
                    "median_kernel_samples": 11,
                    "butterworth_order": 3,
                    "motion_lowpass_cutoff_hz": 20,
                    "gravity_lowpass_cutoff_hz": 0.3,
                    "filter_application": "causal_sos",
                    "median_boundary": "zero_padding",
                    "sos_initial_state": "zeros",
                    "normalization": "none"
                }
            }
        ),
        encoding="utf-8",
    )
    _write_protocol(protocol_path, manifest_path, preprocessing_path)

    index = build_development_window_store(
        approved_manifest_path=manifest_path,
        candidate_manifest_path=candidate_path,
        segment_root=segments,
        protocol_path=protocol_path,
        preprocessing_configuration_path=preprocessing_path,
        output_directory=output,
    )
    store = load_development_window_store(output, verify_file_hashes=True)

    assert index["window_count"] == 1
    assert index["holdout_accessed"] is False
    assert store.windows.shape == (1, 5000, 3)
    assert store.labels.tolist() == [2]
    assert store.metadata[0]["participant_id"] == "P003"
    assert store.metadata[0]["start_sample"] == FILTER_TRANSIENT_SAMPLES
    assert np.isfinite(store.windows).all()


def test_window_store_refuses_changed_approved_segment(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_development_window_store(
            approved_manifest_path=tmp_path / "missing.json",
            candidate_manifest_path=tmp_path / "candidate.json",
            segment_root=tmp_path,
            protocol_path=tmp_path / "protocol.json",
            preprocessing_configuration_path=tmp_path / "preprocessing.json",
            output_directory=tmp_path / "windows",
        )
