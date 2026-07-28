"""Tests for claim ordering and modality-safe final hold-out evaluation."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prevoccupai_har.final_evaluation import (
    HoldoutFeatureMatrix,
    build_holdout_window_store_after_claim,
    execute_claim_gated_holdout,
    load_holdout_feature_matrix_after_claim,
    verify_modality_count_alignment,
)
from prevoccupai_har.holdout import HoldoutEvaluationPolicy
from prevoccupai_har.protocol import ProtocolConfiguration, load_protocol
from prevoccupai_har.provenance import sha256_file
from prevoccupai_har.signal_preprocessing import (
    load_signal_preprocessing_configuration,
)
from prevoccupai_har.window_store import METADATA_DTYPE, DevelopmentWindowStore
from prevoccupai_har.protocol import WindowSpecification


ROOT = Path(__file__).resolve().parents[1]


def _authorized_case(
    tmp_path: Path,
) -> tuple[ProtocolConfiguration, HoldoutEvaluationPolicy, dict[str, Path]]:
    paths = {
        "protocol": tmp_path / "protocol.json",
        "freeze": tmp_path / "freeze.json",
        "plan": tmp_path / "plan.md",
        "ledger": tmp_path / "ledger.json",
        "output": tmp_path / "final_result",
        "failure": tmp_path / "failure.json",
    }
    paths["protocol"].write_text('{"synthetic": true}\n', encoding="utf-8")
    paths["freeze"].write_text('{"frozen": true}\n', encoding="utf-8")
    paths["plan"].write_text("# Synthetic plan\n", encoding="utf-8")
    base = load_protocol(ROOT / "configs/mban_protocol.json")
    protocol = replace(base, training_authorized=True, training_blockers=())
    policy = HoldoutEvaluationPolicy(
        schema_version=1,
        status="authorized_once",
        evaluation_enabled=True,
        maximum_access_count=1,
        required_purpose="final_external_evaluation",
        holdout_participants=protocol.holdout_participants,
        authorization_id="synthetic-final-execution",
        protocol_configuration_sha256=sha256_file(paths["protocol"]),
        model_freeze_manifest_sha256=sha256_file(paths["freeze"]),
        statistical_analysis_plan_sha256=sha256_file(paths["plan"]),
    )
    return protocol, policy, paths


def _execute(
    protocol: ProtocolConfiguration,
    policy: HoldoutEvaluationPolicy,
    paths: dict[str, Path],
    evaluator: Any,
) -> dict[str, Any]:
    return execute_claim_gated_holdout(
        protocol=protocol,
        policy=policy,
        protocol_configuration_path=paths["protocol"],
        model_freeze_manifest_path=paths["freeze"],
        statistical_analysis_plan_path=paths["plan"],
        ledger_path=paths["ledger"],
        accessed_at_utc="2026-07-17T12:00:00Z",
        final_output_directory=paths["output"],
        failure_record_path=paths["failure"],
        evaluator=evaluator,
    )


def test_existing_ledger_blocks_before_evaluator(tmp_path: Path) -> None:
    protocol, policy, paths = _authorized_case(tmp_path)
    paths["ledger"].write_text('{"existing": true}\n', encoding="utf-8")
    calls = 0

    def evaluator(_directory: Path, _claim: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(PermissionError, match="already been claimed"):
        _execute(protocol, policy, paths, evaluator)
    assert calls == 0
    assert not paths["output"].exists()
    assert not paths["failure"].exists()


def test_prerequisite_hash_mismatch_blocks_before_claim_or_evaluator(
    tmp_path: Path,
) -> None:
    protocol, policy, paths = _authorized_case(tmp_path)
    policy = replace(policy, model_freeze_manifest_sha256="0" * 64)
    calls = 0

    def evaluator(_directory: Path, _claim: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(PermissionError, match="prerequisites changed"):
        _execute(protocol, policy, paths, evaluator)
    assert calls == 0
    assert not paths["ledger"].exists()
    assert not paths["output"].exists()


def test_successful_execution_invokes_evaluator_only_after_ledger(
    tmp_path: Path,
) -> None:
    protocol, policy, paths = _authorized_case(tmp_path)
    calls = 0

    def evaluator(directory: Path, claim: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert paths["ledger"].is_file()
        assert claim["state"] == "access_claimed_before_data_read"
        (directory / "synthetic_result.txt").write_text("complete\n", encoding="utf-8")
        return {"primary_result": "synthetic_result.txt"}

    completion = _execute(protocol, policy, paths, evaluator)

    assert calls == 1
    assert completion["status"] == "final_external_evaluation_completed"
    assert paths["ledger"].is_file()
    assert (paths["output"] / "synthetic_result.txt").is_file()
    assert (paths["output"] / "execution_record.json").is_file()
    assert not paths["failure"].exists()


def test_post_claim_failure_consumes_access_and_removes_partial_output(
    tmp_path: Path,
) -> None:
    protocol, policy, paths = _authorized_case(tmp_path)
    calls = 0

    def evaluator(directory: Path, _claim: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert paths["ledger"].is_file()
        (directory / "incomplete.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("synthetic loader failure")

    with pytest.raises(RuntimeError, match="synthetic loader failure"):
        _execute(protocol, policy, paths, evaluator)

    assert calls == 1
    assert paths["ledger"].is_file()
    assert not paths["output"].exists()
    assert paths["failure"].is_file()
    failure = paths["failure"].read_text(encoding="utf-8")
    assert "final_external_evaluation_failed_after_access_claim" in failure
    assert '"retry_permitted": false' in failure
    assert not list(tmp_path.glob(".*.partial-*"))


def test_modality_alignment_uses_counts_not_unproven_row_order() -> None:
    metadata = np.empty(4, dtype=METADATA_DTYPE)
    raw_rows = (
        ("P001", "r1", "sitting"),
        ("P001", "r2", "walking"),
        ("P002", "r3", "standing"),
        ("P002", "r4", "sitting"),
    )
    for index, (participant, recording, label) in enumerate(raw_rows):
        metadata[index] = (
            participant,
            recording,
            label,
            f"{label}_subactivity",
            "file01",
            "unk",
            f"segment-{index}.npy",
            0,
            5000,
            "synthetic",
            "GOOD",
        )
    store = DevelopmentWindowStore(
        windows=np.zeros((4, 2, 3), dtype=np.float32),
        labels=np.asarray([0, 2, 1, 0], dtype=np.int64),
        metadata=metadata,
        index={},
    )
    features = HoldoutFeatureMatrix(
        features=np.zeros((4, 45), dtype=np.float64),
        labels=("walking", "sitting", "sitting", "standing"),
        subactivity_labels=("w", "s", "s", "st"),
        participant_ids=("P001", "P001", "P002", "P002"),
        feature_names=tuple(f"f{index}" for index in range(45)),
        manifest={},
    )

    result = verify_modality_count_alignment(
        store,
        features,
        class_labels=("sitting", "standing", "walking"),
        holdout_participants=("P001", "P002"),
    )

    assert result["status"] == "participant_class_counts_match"
    assert result["raw_feature_rowwise_alignment_claimed"] is False
    assert result["paired_comparison_unit"] == "participant"


def test_synthetic_holdout_loaders_require_claim_and_match_counts(
    tmp_path: Path,
) -> None:
    segment_root = tmp_path / "segments"
    feature_root = tmp_path / "features"
    segment_root.mkdir()
    feature_root.mkdir()
    segment_path = (
        segment_root / "P001_sitting_file01_sitting_GlobalSegment1.npy"
    )
    segment = np.zeros((290, 4), dtype=np.float64)
    segment[:, 0] = np.arange(segment.shape[0])
    segment[:, 1:] = np.linspace(-1.0, 1.0, segment.shape[0])[:, None]
    np.save(segment_path, segment, allow_pickle=False)
    feature_path = feature_root / "P001.npy"
    feature_matrix = np.zeros((3, 47), dtype=np.float64)
    feature_matrix[:, 45] = 0
    feature_matrix[:, 46] = 0
    np.save(feature_path, feature_matrix, allow_pickle=False)
    candidate = {
        "segments": {
            "checksums_calculated": True,
            "arrays": [
                {
                    "participant_id": "P001",
                    "quality_status": "GOOD",
                    "relative_name": segment_path.name,
                    "sha256": sha256_file(segment_path),
                    "shape": list(segment.shape),
                    "dtype": str(segment.dtype),
                    "size_bytes": segment_path.stat().st_size,
                }
            ],
        },
        "feature_matrices": {
            "checksums_calculated": True,
            "feature_columns": [f"feature_{index}" for index in range(45)],
            "matrices": [
                {
                    "participant_id": "P001",
                    "relative_name": feature_path.name,
                    "sha256": sha256_file(feature_path),
                    "shape": list(feature_matrix.shape),
                    "dtype": str(feature_matrix.dtype),
                    "size_bytes": feature_path.stat().st_size,
                    "window_count": 3,
                }
            ],
        },
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8"
    )
    base = load_protocol(ROOT / "configs/mban_protocol.json")
    protocol = replace(
        base,
        holdout_participants=("P001",),
        window=WindowSpecification(
            duration_seconds=0.02,
            overlap_fraction=0.5,
            expected_samples=20,
        ),
    )
    preprocessing = load_signal_preprocessing_configuration(
        ROOT / "configs/mban_signal_preprocessing.json"
    )

    with pytest.raises(PermissionError, match="completed access claim"):
        build_holdout_window_store_after_claim(
            segment_root=segment_root,
            candidate_manifest_path=candidate_path,
            preprocessing=preprocessing,
            protocol=protocol,
            claim_record={},
            output_directory=tmp_path / "blocked_store",
        )

    claim = {
        "state": "access_claimed_before_data_read",
        "access_count": 1,
        "authorization_id": "synthetic-loader-test",
        "holdout_participants": ["P001"],
    }
    store = build_holdout_window_store_after_claim(
        segment_root=segment_root,
        candidate_manifest_path=candidate_path,
        preprocessing=preprocessing,
        protocol=protocol,
        claim_record=claim,
        output_directory=tmp_path / "holdout_store",
    )
    features = load_holdout_feature_matrix_after_claim(
        feature_root=feature_root,
        candidate_manifest_path=candidate_path,
        protocol=protocol,
        claim_record=claim,
        manifest_output_path=tmp_path / "feature_manifest.json",
    )
    alignment = verify_modality_count_alignment(
        store,
        features,
        class_labels=protocol.main_labels,
        holdout_participants=protocol.holdout_participants,
    )

    assert store.windows.shape == (3, 20, 3)
    assert set(map(str, store.metadata["sensor_side"])) == {"unk"}
    assert alignment["participant_class_counts"]["P001"]["sitting"] == 3
    assert alignment["raw_feature_rowwise_alignment_claimed"] is False
