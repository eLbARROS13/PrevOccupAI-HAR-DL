"""Fail-closed, single-use authorization contract for final hold-out access."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .protocol import ProtocolConfiguration
from .provenance import sha256_file


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class HoldoutEvaluationPolicy:
    """Configuration that is disabled until every final-evaluation prerequisite exists."""

    schema_version: int
    status: str
    evaluation_enabled: bool
    maximum_access_count: int
    required_purpose: str
    holdout_participants: tuple[str, ...]
    authorization_id: str | None
    protocol_configuration_sha256: str | None
    model_freeze_manifest_sha256: str | None
    statistical_analysis_plan_sha256: str | None

    def validate(self) -> None:
        """Enforce either a fully disabled or fully specified single-use state."""
        if self.schema_version != 1:
            raise ValueError("Unsupported hold-out policy schema version")
        if self.maximum_access_count != 1:
            raise ValueError("The external hold-out policy must be single-use")
        if self.required_purpose != "final_external_evaluation":
            raise ValueError("Unsupported hold-out access purpose")
        if not self.holdout_participants:
            raise ValueError("The hold-out cohort cannot be empty")
        if len(set(self.holdout_participants)) != len(self.holdout_participants):
            raise ValueError("Hold-out participants contain duplicates")

        governed_values = (
            self.authorization_id,
            self.protocol_configuration_sha256,
            self.model_freeze_manifest_sha256,
            self.statistical_analysis_plan_sha256,
        )
        if self.status == "disabled_pending_data_readiness":
            if self.evaluation_enabled or any(value is not None for value in governed_values):
                raise ValueError("A disabled hold-out policy cannot contain authorization data")
            return
        if self.status != "authorized_once" or not self.evaluation_enabled:
            raise ValueError("Unsupported or internally inconsistent hold-out policy status")
        if self.authorization_id is None or AUTHORIZATION_ID_PATTERN.fullmatch(
            self.authorization_id
        ) is None:
            raise ValueError("An authorized policy requires a valid authorization identifier")
        for field_name, value in (
            ("protocol_configuration_sha256", self.protocol_configuration_sha256),
            ("model_freeze_manifest_sha256", self.model_freeze_manifest_sha256),
            ("statistical_analysis_plan_sha256", self.statistical_analysis_plan_sha256),
        ):
            if value is None or SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"Authorized policy requires a valid {field_name}")


def load_holdout_evaluation_policy(path: Path | str) -> HoldoutEvaluationPolicy:
    """Load and validate a hold-out evaluation policy JSON file."""
    policy_path = Path(path).resolve()
    decoded = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError(f"Expected a JSON object in {policy_path}")
    policy = HoldoutEvaluationPolicy(
        schema_version=int(decoded["schema_version"]),
        status=str(decoded["status"]),
        evaluation_enabled=decoded.get("evaluation_enabled") is True,
        maximum_access_count=int(decoded["maximum_access_count"]),
        required_purpose=str(decoded["required_purpose"]),
        holdout_participants=tuple(map(str, decoded["holdout_participants"])),
        authorization_id=(
            str(decoded["authorization_id"])
            if decoded.get("authorization_id") is not None
            else None
        ),
        protocol_configuration_sha256=(
            str(decoded["protocol_configuration_sha256"])
            if decoded.get("protocol_configuration_sha256") is not None
            else None
        ),
        model_freeze_manifest_sha256=(
            str(decoded["model_freeze_manifest_sha256"])
            if decoded.get("model_freeze_manifest_sha256") is not None
            else None
        ),
        statistical_analysis_plan_sha256=(
            str(decoded["statistical_analysis_plan_sha256"])
            if decoded.get("statistical_analysis_plan_sha256") is not None
            else None
        ),
    )
    policy.validate()
    return policy


def claim_holdout_access(
    *,
    protocol: ProtocolConfiguration,
    policy: HoldoutEvaluationPolicy,
    protocol_configuration_path: Path | str,
    model_freeze_manifest_path: Path | str,
    statistical_analysis_plan_path: Path | str,
    ledger_path: Path | str,
    accessed_at_utc: str,
) -> dict[str, Any]:
    """Atomically consume the single hold-out access before any data are read.

    Existing ledgers are never overwritten. A failed run still consumes its claim,
    which forces explicit human adjudication instead of silent test-set reuse.
    """
    protocol.validate()
    policy.validate()
    if not protocol.training_authorized:
        raise PermissionError("The scientific protocol does not authorize training or evaluation")
    if policy.status != "authorized_once" or not policy.evaluation_enabled:
        raise PermissionError("The hold-out evaluation policy is disabled")
    if tuple(sorted(policy.holdout_participants)) != tuple(
        sorted(protocol.holdout_participants)
    ):
        raise PermissionError("The policy hold-out cohort differs from the protocol")
    if UTC_PATTERN.fullmatch(accessed_at_utc) is None:
        raise ValueError("Access time must use second-resolution UTC with a Z suffix")

    observed_hashes = {
        "protocol_configuration_sha256": sha256_file(protocol_configuration_path),
        "model_freeze_manifest_sha256": sha256_file(model_freeze_manifest_path),
        "statistical_analysis_plan_sha256": sha256_file(statistical_analysis_plan_path),
    }
    expected_hashes = {
        "protocol_configuration_sha256": policy.protocol_configuration_sha256,
        "model_freeze_manifest_sha256": policy.model_freeze_manifest_sha256,
        "statistical_analysis_plan_sha256": policy.statistical_analysis_plan_sha256,
    }
    if observed_hashes != expected_hashes:
        raise PermissionError("One or more frozen hold-out prerequisites changed")

    record: dict[str, Any] = {
        "schema_version": 1,
        "authorization_id": policy.authorization_id,
        "accessed_at_utc": accessed_at_utc,
        "access_count": 1,
        "purpose": policy.required_purpose,
        "holdout_participants": list(policy.holdout_participants),
        "frozen_artifact_hashes": expected_hashes,
        "state": "access_claimed_before_data_read",
    }
    output_path = Path(ledger_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise PermissionError("Hold-out access has already been claimed") from error
    return record
