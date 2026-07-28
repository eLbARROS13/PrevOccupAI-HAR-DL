"""Tests for the fail-closed, single-use hold-out access contract."""

from dataclasses import replace
from pathlib import Path

import pytest

from prevoccupai_har.holdout import (
    HoldoutEvaluationPolicy,
    claim_holdout_access,
    load_holdout_evaluation_policy,
)
from prevoccupai_har.protocol import load_protocol
from prevoccupai_har.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_public_holdout_policy_is_disabled_and_contains_no_authorization() -> None:
    protocol_path = ROOT / "configs/mban_protocol.json"
    policy_path = ROOT / "configs/holdout_evaluation_policy.json"
    protocol = load_protocol(protocol_path)
    policy = load_holdout_evaluation_policy(policy_path)
    assert policy.status == "disabled_pending_data_readiness"
    assert policy.evaluation_enabled is False
    assert policy.authorization_id is None
    assert policy.protocol_configuration_sha256 is None
    assert policy.model_freeze_manifest_sha256 is None
    assert policy.statistical_analysis_plan_sha256 is None

    with pytest.raises(PermissionError, match="policy is disabled"):
        claim_holdout_access(
            protocol=protocol,
            policy=policy,
            protocol_configuration_path=protocol_path,
            model_freeze_manifest_path=ROOT / "not-distributed.json",
            statistical_analysis_plan_path=ROOT / "docs/statistical_analysis_plan.md",
            ledger_path=ROOT / "must-not-be-created.json",
            accessed_at_utc="2026-07-21T12:00:00Z",
        )


def test_enabled_policy_requires_complete_hashes() -> None:
    policy = HoldoutEvaluationPolicy(
        schema_version=1,
        status="authorized_once",
        evaluation_enabled=True,
        maximum_access_count=1,
        required_purpose="final_external_evaluation",
        holdout_participants=("P001",),
        authorization_id="final-evaluation-1",
        protocol_configuration_sha256=None,
        model_freeze_manifest_sha256=None,
        statistical_analysis_plan_sha256=None,
    )

    with pytest.raises(ValueError, match="requires a valid"):
        policy.validate()


def test_holdout_claim_is_single_use_and_hash_bound(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    freeze_path = tmp_path / "model_freeze.json"
    analysis_path = tmp_path / "analysis_plan.md"
    protocol_path.write_text('{"synthetic": true}\n', encoding="utf-8")
    freeze_path.write_text('{"frozen": true}\n', encoding="utf-8")
    analysis_path.write_text("# Synthetic plan\n", encoding="utf-8")
    base_protocol = load_protocol(ROOT / "configs/mban_protocol.json")
    authorized_protocol = replace(
        base_protocol,
        quality_assessment_manifest=tmp_path / "qa.json",
        segmentation_manifest=tmp_path / "segments.json",
        device_to_side_mapping=tmp_path / "sides.json",
        training_authorized=True,
        training_blockers=(),
    )
    policy = HoldoutEvaluationPolicy(
        schema_version=1,
        status="authorized_once",
        evaluation_enabled=True,
        maximum_access_count=1,
        required_purpose="final_external_evaluation",
        holdout_participants=authorized_protocol.holdout_participants,
        authorization_id="synthetic-final-evaluation-1",
        protocol_configuration_sha256=sha256_file(protocol_path),
        model_freeze_manifest_sha256=sha256_file(freeze_path),
        statistical_analysis_plan_sha256=sha256_file(analysis_path),
    )
    ledger_path = tmp_path / "holdout_access_ledger.json"

    record = claim_holdout_access(
        protocol=authorized_protocol,
        policy=policy,
        protocol_configuration_path=protocol_path,
        model_freeze_manifest_path=freeze_path,
        statistical_analysis_plan_path=analysis_path,
        ledger_path=ledger_path,
        accessed_at_utc="2026-07-15T18:00:00Z",
    )

    assert record["access_count"] == 1
    assert record["state"] == "access_claimed_before_data_read"
    with pytest.raises(PermissionError, match="already been claimed"):
        claim_holdout_access(
            protocol=authorized_protocol,
            policy=policy,
            protocol_configuration_path=protocol_path,
            model_freeze_manifest_path=freeze_path,
            statistical_analysis_plan_path=analysis_path,
            ledger_path=ledger_path,
            accessed_at_utc="2026-07-15T18:00:01Z",
        )
