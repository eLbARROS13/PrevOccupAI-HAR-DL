"""Frozen repeated-seed/fold planning and provenance-bound model selection."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .protocol import load_protocol
from .provenance import (
    is_reproducible_source_revision,
    sha256_canonical_json,
    sha256_file,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SYNTHETIC_VALIDATION = "synthetic_validation"
DEVELOPMENT_SELECTION = "development_selection"


def _require_sha256(value: str, field_name: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA-256 value for {field_name}")


def _finite_unit_interval(value: object, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric vector")
    return float(statistics.fmean(values))


def _population_sd(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric vector")
    return float(statistics.pstdev(values))


@dataclass(frozen=True)
class SelectionCandidate:
    """One model candidate bound to configuration and complexity evidence."""

    experiment_id: str
    role: str
    model_configuration_sha256: str
    complexity_profile_sha256: str
    trainable_parameter_count: int

    def validate(self) -> None:
        """Validate candidate identity, role, hashes, and exact size."""
        if IDENTIFIER_PATTERN.fullmatch(self.experiment_id) is None:
            raise ValueError("Candidate experiment identifier is invalid")
        if self.role not in {"reference", "challenger"}:
            raise ValueError("Candidate role must be reference or challenger")
        _require_sha256(
            self.model_configuration_sha256,
            "candidate model_configuration_sha256",
        )
        _require_sha256(
            self.complexity_profile_sha256,
            "candidate complexity_profile_sha256",
        )
        if self.trainable_parameter_count <= 0:
            raise ValueError("Candidate parameter count must be positive")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SelectionFold:
    """One shared participant-disjoint development fold."""

    fold_index: int
    training_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    holdout_subjects: tuple[str, ...]

    def validate(self) -> None:
        """Validate non-empty, unique, pairwise-disjoint participant sets."""
        if self.fold_index < 0:
            raise ValueError("Fold index cannot be negative")
        sets = tuple(
            set(values)
            for values in (
                self.training_subjects,
                self.validation_subjects,
                self.holdout_subjects,
            )
        )
        if any(not values for values in sets):
            raise ValueError("Training, validation, and hold-out sets must be non-empty")
        if any(
            len(values) != len(set(values))
            for values in (
                self.training_subjects,
                self.validation_subjects,
                self.holdout_subjects,
            )
        ):
            raise ValueError("Fold participant identifiers contain duplicates")
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Fold participant sets must be pairwise disjoint")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "fold_index": self.fold_index,
            "training_subjects": list(self.training_subjects),
            "validation_subjects": list(self.validation_subjects),
            "holdout_subjects": list(self.holdout_subjects),
        }


@dataclass(frozen=True)
class ConservativeSelectionRule:
    """Predeclared rule controlling whether the larger challenger is selected."""

    primary_metric: str
    supporting_metric: str
    challenger_minimum_primary_gain: float
    challenger_maximum_supporting_loss: float
    minimum_nonnegative_seed_fraction: float

    def validate(self) -> None:
        """Require the frozen participant-level endpoints and valid thresholds."""
        if self.primary_metric != "participant_macro_f1":
            raise ValueError("Primary model-selection metric must be participant_macro_f1")
        if self.supporting_metric != "participant_balanced_accuracy":
            raise ValueError(
                "Supporting model-selection metric must be participant_balanced_accuracy"
            )
        if self.challenger_minimum_primary_gain < 0.0:
            raise ValueError("Minimum primary gain cannot be negative")
        if self.challenger_maximum_supporting_loss < 0.0:
            raise ValueError("Maximum supporting loss cannot be negative")
        if not 0.0 <= self.minimum_nonnegative_seed_fraction <= 1.0:
            raise ValueError("Seed-consistency fraction must be in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class DevelopmentSelectionPlan:
    """Non-predictive frozen plan for a complete candidate/fold/seed grid."""

    schema_version: int
    plan_id: str
    created_at_utc: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    training_authorized: bool
    selection_configuration_sha256: str
    protocol_configuration_sha256: str
    split_manifest_sha256: str
    window_store_index_sha256: str
    statistical_analysis_plan_sha256: str
    class_labels: tuple[str, ...]
    random_seeds: tuple[int, ...]
    candidates: tuple[SelectionCandidate, ...]
    folds: tuple[SelectionFold, ...]
    analysis_settings: dict[str, object]
    selection_rule: ConservativeSelectionRule
    expected_run_count: int
    plan_payload_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "purpose": self.purpose,
            "training_authorized": self.training_authorized,
            "selection_configuration_sha256": self.selection_configuration_sha256,
            "protocol_configuration_sha256": self.protocol_configuration_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "window_store_index_sha256": self.window_store_index_sha256,
            "statistical_analysis_plan_sha256": self.statistical_analysis_plan_sha256,
            "class_labels": list(self.class_labels),
            "random_seeds": list(self.random_seeds),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "folds": [fold.as_dict() for fold in self.folds],
            "analysis_settings": self.analysis_settings,
            "selection_rule": self.selection_rule.as_dict(),
            "expected_run_count": self.expected_run_count,
        }

    def validate(self) -> None:
        """Reject mutable, incomplete, hold-out-like, or unpaired selection plans."""
        if self.schema_version != 1:
            raise ValueError("Unsupported development-selection plan schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.plan_id) is None:
            raise ValueError("Selection plan identifier is invalid")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Creation time must use second-resolution UTC with a Z suffix")
        if self.purpose not in {
            SYNTHETIC_VALIDATION,
            DEVELOPMENT_SELECTION,
        }:
            raise ValueError("Selection-plan purpose is unsupported")
        if self.scientific_result or self.holdout_accessed:
            raise ValueError("A selection plan is non-predictive and cannot access hold-out data")
        for field_name in (
            "selection_configuration_sha256",
            "protocol_configuration_sha256",
            "split_manifest_sha256",
            "window_store_index_sha256",
            "statistical_analysis_plan_sha256",
            "plan_payload_sha256",
        ):
            _require_sha256(str(getattr(self, field_name)), field_name)
        if not self.class_labels or len(self.class_labels) != len(set(self.class_labels)):
            raise ValueError("Class labels must be non-empty and unique")
        if not self.random_seeds or len(self.random_seeds) != len(set(self.random_seeds)):
            raise ValueError("Random seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.random_seeds):
            raise ValueError("Random seeds cannot be negative")
        if len(self.candidates) != 2:
            raise ValueError("The frozen comparison requires exactly two candidates")
        for candidate in self.candidates:
            candidate.validate()
        if len({candidate.experiment_id for candidate in self.candidates}) != 2:
            raise ValueError("Candidate experiment identifiers must be unique")
        if {candidate.role for candidate in self.candidates} != {
            "reference",
            "challenger",
        }:
            raise ValueError("Exactly one reference and one challenger are required")
        if not self.folds:
            raise ValueError("At least one selection fold is required")
        for fold in self.folds:
            fold.validate()
        if tuple(fold.fold_index for fold in self.folds) != tuple(range(len(self.folds))):
            raise ValueError("Fold indices must be unique, ordered, and contiguous from zero")
        development_sets = {
            frozenset(fold.training_subjects) | frozenset(fold.validation_subjects)
            for fold in self.folds
        }
        holdout_sets = {frozenset(fold.holdout_subjects) for fold in self.folds}
        if len(development_sets) != 1 or len(holdout_sets) != 1:
            raise ValueError("All folds must share one development and hold-out cohort")
        validation_occurrences = [
            subject for fold in self.folds for subject in fold.validation_subjects
        ]
        development = next(iter(development_sets))
        if sorted(validation_occurrences) != sorted(development):
            raise ValueError("Every development participant must validate exactly once")
        if self.purpose == SYNTHETIC_VALIDATION and any(
            not subject.startswith("SYNTHETIC_")
            for subject in development | next(iter(holdout_sets))
        ):
            raise ValueError("Synthetic selection plans require synthetic participants")
        expected_settings = {
            "probability_transform": {
                "method": "softmax",
                "temperature": 1.0,
                "fitted": False,
            },
            "calibration_bin_count": self.analysis_settings.get(
                "calibration_bin_count"
            ),
            "expected_step_size_samples": self.analysis_settings.get(
                "expected_step_size_samples"
            ),
            "short_run_max_windows": self.analysis_settings.get(
                "short_run_max_windows"
            ),
        }
        if self.analysis_settings != expected_settings:
            raise ValueError("Selection-plan analysis settings are incomplete or unexpected")
        if int(self.analysis_settings["calibration_bin_count"]) < 2:
            raise ValueError("At least two calibration bins are required")
        if int(self.analysis_settings["expected_step_size_samples"]) <= 0 or int(
            self.analysis_settings["short_run_max_windows"]
        ) <= 0:
            raise ValueError("Temporal analysis settings must be positive")
        self.selection_rule.validate()
        expected_count = len(self.candidates) * len(self.folds) * len(self.random_seeds)
        if self.expected_run_count != expected_count:
            raise ValueError("Expected run count does not match the candidate/fold/seed grid")
        if sha256_canonical_json(self._payload()) != self.plan_payload_sha256:
            raise ValueError("Selection-plan payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "created_at_utc": self.created_at_utc,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            **self._payload(),
            "plan_payload_sha256": self.plan_payload_sha256,
        }


@dataclass(frozen=True)
class SelectionRunEvidence:
    """One fold/seed run whose complete artifact chain has been validated."""

    candidate_id: str
    fold_index: int
    random_seed: int
    training_run_id: str
    training_result_sha256: str
    prediction_run_id: str
    prediction_artifact_sha256: str
    analysis_id: str
    analysis_record_sha256: str
    model_state_sha256: str
    validation_subjects: tuple[str, ...]
    participant_metrics: dict[str, dict[str, float]]
    history: tuple[dict[str, float | int], ...]

    def validate(self) -> None:
        """Validate identifiers, hashes, participants, metrics, and learning history."""
        for value in (
            self.candidate_id,
            self.training_run_id,
            self.prediction_run_id,
            self.analysis_id,
        ):
            if IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise ValueError("Selection evidence contains an invalid identifier")
        if self.fold_index < 0 or self.random_seed < 0:
            raise ValueError("Selection evidence fold and seed must be non-negative")
        for field_name in (
            "training_result_sha256",
            "prediction_artifact_sha256",
            "analysis_record_sha256",
            "model_state_sha256",
        ):
            _require_sha256(str(getattr(self, field_name)), field_name)
        if not self.validation_subjects or len(self.validation_subjects) != len(
            set(self.validation_subjects)
        ):
            raise ValueError("Evidence validation subjects must be non-empty and unique")
        if set(self.participant_metrics) != set(self.validation_subjects):
            raise ValueError("Participant metrics must exactly match validation subjects")
        for metrics in self.participant_metrics.values():
            if set(metrics) != {"macro_f1", "balanced_accuracy"}:
                raise ValueError("Participant selection metrics are incomplete")
            _finite_unit_interval(metrics["macro_f1"], "participant macro F1")
            _finite_unit_interval(
                metrics["balanced_accuracy"], "participant balanced accuracy"
            )
        if not self.history:
            raise ValueError("Selection evidence requires a training history")
        epochs = tuple(int(entry["epoch"]) for entry in self.history)
        if epochs != tuple(range(1, len(self.history) + 1)):
            raise ValueError("Training-history epochs must be contiguous from one")
        for entry in self.history:
            if set(entry) != {
                "epoch",
                "training_loss",
                "validation_loss",
                "validation_macro_f1",
                "validation_balanced_accuracy",
            }:
                raise ValueError("Training-history entry is incomplete or unexpected")
            for loss_name in ("training_loss", "validation_loss"):
                loss = float(entry[loss_name])
                if not math.isfinite(loss) or loss < 0.0:
                    raise ValueError("Training losses must be finite and non-negative")
            _finite_unit_interval(
                entry["validation_macro_f1"], "history validation macro F1"
            )
            _finite_unit_interval(
                entry["validation_balanced_accuracy"],
                "history validation balanced accuracy",
            )

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "fold_index": self.fold_index,
            "random_seed": self.random_seed,
            "training_run_id": self.training_run_id,
            "training_result_sha256": self.training_result_sha256,
            "prediction_run_id": self.prediction_run_id,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "analysis_id": self.analysis_id,
            "analysis_record_sha256": self.analysis_record_sha256,
            "model_state_sha256": self.model_state_sha256,
            "validation_subjects": list(self.validation_subjects),
            "participant_metrics": self.participant_metrics,
            "history": list(self.history),
        }


@dataclass(frozen=True)
class ModelSelectionBundle:
    """Complete repeated-seed/fold evidence and deterministic selection decision."""

    schema_version: int
    bundle_id: str
    created_at_utc: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    selection_plan_sha256: str
    source_revision: str
    run_count: int
    run_evidence: tuple[SelectionRunEvidence, ...]
    candidate_summaries: dict[str, object]
    decision: dict[str, object]
    bundle_payload_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "run_evidence": [evidence.as_dict() for evidence in self.run_evidence],
            "candidate_summaries": self.candidate_summaries,
            "decision": self.decision,
        }

    def validate(self) -> None:
        """Validate result scope, artifact count, decision fields, and payload digest."""
        if self.schema_version != 1:
            raise ValueError("Unsupported model-selection bundle schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.bundle_id) is None:
            raise ValueError("Model-selection bundle identifier is invalid")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Creation time must use second-resolution UTC with a Z suffix")
        if self.purpose not in {
            SYNTHETIC_VALIDATION,
            DEVELOPMENT_SELECTION,
        }:
            raise ValueError("Model-selection bundle purpose is unsupported")
        expected_scientific = self.purpose == DEVELOPMENT_SELECTION
        if self.scientific_result is not expected_scientific:
            raise ValueError("Bundle scientific status disagrees with its purpose")
        if self.holdout_accessed:
            raise ValueError("A development-selection bundle cannot access hold-out data")
        _require_sha256(self.selection_plan_sha256, "selection_plan_sha256")
        _require_sha256(self.bundle_payload_sha256, "bundle_payload_sha256")
        if not self.source_revision:
            raise ValueError("Bundle source revision is required")
        if expected_scientific and not is_reproducible_source_revision(
            self.source_revision
        ):
            raise ValueError(
                "Scientific selection bundles require an immutable source revision"
            )
        if self.run_count != len(self.run_evidence) or self.run_count <= 0:
            raise ValueError("Bundle run count must match non-empty evidence")
        for evidence in self.run_evidence:
            evidence.validate()
        if set(self.decision) != {
            "status",
            "selected_candidate_id",
            "reference_candidate_id",
            "challenger_candidate_id",
            "primary_gain",
            "supporting_gain",
            "nonnegative_seed_fraction",
            "criteria",
        }:
            raise ValueError("Selection decision is incomplete or unexpected")
        if self.decision["selected_candidate_id"] not in self.candidate_summaries:
            raise ValueError("Selected candidate lacks a candidate summary")
        if sha256_canonical_json(self._payload()) != self.bundle_payload_sha256:
            raise ValueError("Model-selection bundle payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "created_at_utc": self.created_at_utc,
            "purpose": self.purpose,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            "selection_plan_sha256": self.selection_plan_sha256,
            "source_revision": self.source_revision,
            "run_count": self.run_count,
            **self._payload(),
            "bundle_payload_sha256": self.bundle_payload_sha256,
        }


def _load_json_object(path: Path) -> Mapping[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError(f"Expected a JSON object in {path}")
    return decoded


def _analysis_settings(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "probability_transform": {
            "method": "softmax",
            "temperature": 1.0,
            "fitted": False,
        },
        "calibration_bin_count": int(value["calibration_bin_count"]),
        "expected_step_size_samples": int(value["expected_step_size_samples"]),
        "short_run_max_windows": int(value["short_run_max_windows"]),
    }


def build_development_selection_plan(
    *,
    configuration_path: Path | str,
    created_at_utc: str,
) -> DevelopmentSelectionPlan:
    """Build the frozen 2-candidate, 5-fold, 5-seed plan without training."""
    config_path = Path(configuration_path).resolve()
    root = config_path.parent.parent
    decoded = _load_json_object(config_path)
    if int(decoded["schema_version"]) != 1:
        raise ValueError("Unsupported model-selection configuration schema version")
    purpose = str(decoded["purpose"])
    if purpose != DEVELOPMENT_SELECTION:
        raise ValueError("Checked-in selection configuration must target development selection")
    protocol_path = root / str(decoded["protocol_configuration"])
    split_path = root / str(decoded["split_manifest"])
    window_store_index_path = root / str(decoded["window_store_index"])
    statistical_plan_path = root / str(decoded["statistical_analysis_plan"])
    protocol = load_protocol(protocol_path)
    split = _load_json_object(split_path)
    window_store_index = _load_json_object(window_store_index_path)
    if split.get("training_authorized") is not protocol.training_authorized:
        raise ValueError("Split-manifest and protocol training-authorization states differ")
    if split.get("holdout_accessed") is not False or split.get(
        "holdout_access_authorized"
    ) is not False:
        raise ValueError("Development split manifest must keep hold-out access disabled")
    if split.get("protocol_configuration_sha256") != sha256_file(protocol_path):
        raise ValueError("Split manifest does not bind the current protocol")
    if window_store_index.get("status") != "approved_development_window_store":
        raise ValueError("Selection plan requires the approved development window store")
    if window_store_index.get("holdout_accessed") is not False:
        raise ValueError("Selection window store cannot access hold-out data")
    if set(map(str, window_store_index.get("development_participants", []))) != set(
        protocol.development_participants
    ):
        raise ValueError("Window store and protocol development cohorts differ")
    folds_value = split.get("folds")
    if not isinstance(folds_value, list):
        raise TypeError("Split manifest folds must be an array")
    folds = tuple(
        SelectionFold(
            fold_index=int(value["fold_index"]),
            training_subjects=tuple(map(str, value["training_participants"])),
            validation_subjects=tuple(map(str, value["validation_participants"])),
            holdout_subjects=tuple(map(str, value["holdout_participants"])),
        )
        for value in folds_value
    )
    if int(split["n_splits"]) != int(decoded["fold_count"]) or len(folds) != int(
        decoded["fold_count"]
    ):
        raise ValueError("Split manifest does not contain the declared fold count")
    if int(split["random_seed"]) != int(decoded["split_random_seed"]):
        raise ValueError("Split-manifest random seed disagrees with the selection config")
    if any(
        set(fold.training_subjects) | set(fold.validation_subjects)
        != set(protocol.development_participants)
        for fold in folds
    ):
        raise ValueError("Selection folds do not exactly partition the protocol development cohort")
    if any(
        set(fold.holdout_subjects) != set(protocol.holdout_participants)
        for fold in folds
    ):
        raise ValueError("Selection folds do not preserve the protocol hold-out cohort")
    candidate_values = decoded.get("candidate_models")
    if not isinstance(candidate_values, list):
        raise TypeError("Candidate model definitions must be an array")
    configured_seeds = tuple(int(seed) for seed in decoded["random_seeds"])
    candidates: list[SelectionCandidate] = []
    observed_class_labels: set[tuple[str, ...]] = set()
    for value in candidate_values:
        if not isinstance(value, Mapping):
            raise TypeError("Candidate model definition must be an object")
        model_path = root / str(value["model_configuration"])
        complexity_path = root / str(value["complexity_profile"])
        model = _load_json_object(model_path)
        complexity = _load_json_object(complexity_path)
        experiment_id = str(value["experiment_id"])
        if str(model["experiment_id"]) != experiment_id or str(
            complexity["experiment_id"]
        ) != experiment_id:
            raise ValueError("Candidate experiment identifiers disagree across artifacts")
        model_sha256 = sha256_file(model_path)
        if str(complexity["model_configuration_sha256"]) != model_sha256:
            raise ValueError("Complexity profile does not bind the candidate configuration")
        if complexity.get("holdout_accessed") is not False or complexity.get(
            "scientific_result"
        ) is not False:
            raise ValueError("Planning complexity profiles must be non-scientific")
        if tuple(int(seed) for seed in model["random_seeds"]) != configured_seeds:
            raise ValueError("Candidates must share the frozen random-seed sequence")
        input_value = model.get("input")
        if not isinstance(input_value, Mapping):
            raise TypeError("Candidate input definition must be an object")
        observed_class_labels.add(tuple(map(str, input_value["class_labels"])))
        measurement = complexity.get("measurement")
        if not isinstance(measurement, Mapping):
            raise TypeError("Complexity profile measurement must be an object")
        candidates.append(
            SelectionCandidate(
                experiment_id=experiment_id,
                role=str(value["role"]),
                model_configuration_sha256=model_sha256,
                complexity_profile_sha256=sha256_file(complexity_path),
                trainable_parameter_count=int(measurement["trainable_parameter_count"]),
            )
        )
    if observed_class_labels != {protocol.main_labels}:
        raise ValueError("Candidate class labels must exactly match the protocol")
    rule_value = decoded.get("selection_rule")
    if not isinstance(rule_value, Mapping):
        raise TypeError("Selection rule must be an object")
    rule = ConservativeSelectionRule(
        primary_metric=str(rule_value["primary_metric"]),
        supporting_metric=str(rule_value["supporting_metric"]),
        challenger_minimum_primary_gain=float(
            rule_value["challenger_minimum_primary_gain"]
        ),
        challenger_maximum_supporting_loss=float(
            rule_value["challenger_maximum_supporting_loss"]
        ),
        minimum_nonnegative_seed_fraction=float(
            rule_value["minimum_nonnegative_seed_fraction"]
        ),
    )
    settings_value = decoded.get("analysis_settings")
    if not isinstance(settings_value, Mapping):
        raise TypeError("Analysis settings must be an object")
    payload_fields = {
        "purpose": purpose,
        "training_authorized": protocol.training_authorized,
        "selection_configuration_sha256": sha256_file(config_path),
        "protocol_configuration_sha256": sha256_file(protocol_path),
        "split_manifest_sha256": sha256_file(split_path),
        "window_store_index_sha256": sha256_file(window_store_index_path),
        "statistical_analysis_plan_sha256": sha256_file(statistical_plan_path),
        "class_labels": list(protocol.main_labels),
        "random_seeds": list(configured_seeds),
        "candidates": [candidate.as_dict() for candidate in candidates],
        "folds": [fold.as_dict() for fold in folds],
        "analysis_settings": _analysis_settings(settings_value),
        "selection_rule": rule.as_dict(),
        "expected_run_count": len(candidates) * len(folds) * len(configured_seeds),
    }
    plan = DevelopmentSelectionPlan(
        schema_version=1,
        plan_id=str(decoded["plan_id"]),
        created_at_utc=created_at_utc,
        purpose=purpose,
        scientific_result=False,
        holdout_accessed=False,
        training_authorized=protocol.training_authorized,
        selection_configuration_sha256=str(
            payload_fields["selection_configuration_sha256"]
        ),
        protocol_configuration_sha256=str(
            payload_fields["protocol_configuration_sha256"]
        ),
        split_manifest_sha256=str(payload_fields["split_manifest_sha256"]),
        window_store_index_sha256=str(payload_fields["window_store_index_sha256"]),
        statistical_analysis_plan_sha256=str(
            payload_fields["statistical_analysis_plan_sha256"]
        ),
        class_labels=protocol.main_labels,
        random_seeds=configured_seeds,
        candidates=tuple(candidates),
        folds=folds,
        analysis_settings=dict(payload_fields["analysis_settings"]),
        selection_rule=rule,
        expected_run_count=int(payload_fields["expected_run_count"]),
        plan_payload_sha256=sha256_canonical_json(payload_fields),
    )
    plan.validate()
    return plan


def write_development_selection_plan(
    path: Path | str,
    plan: DevelopmentSelectionPlan,
) -> None:
    """Write a frozen selection plan exclusively."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(plan.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_development_selection_plan(path: Path | str) -> DevelopmentSelectionPlan:
    """Load and validate a frozen development-selection plan."""
    decoded = _load_json_object(Path(path))
    candidates = tuple(
        SelectionCandidate(
            experiment_id=str(value["experiment_id"]),
            role=str(value["role"]),
            model_configuration_sha256=str(value["model_configuration_sha256"]),
            complexity_profile_sha256=str(value["complexity_profile_sha256"]),
            trainable_parameter_count=int(value["trainable_parameter_count"]),
        )
        for value in decoded["candidates"]
    )
    folds = tuple(
        SelectionFold(
            fold_index=int(value["fold_index"]),
            training_subjects=tuple(map(str, value["training_subjects"])),
            validation_subjects=tuple(map(str, value["validation_subjects"])),
            holdout_subjects=tuple(map(str, value["holdout_subjects"])),
        )
        for value in decoded["folds"]
    )
    rule_value = decoded["selection_rule"]
    rule = ConservativeSelectionRule(
        primary_metric=str(rule_value["primary_metric"]),
        supporting_metric=str(rule_value["supporting_metric"]),
        challenger_minimum_primary_gain=float(
            rule_value["challenger_minimum_primary_gain"]
        ),
        challenger_maximum_supporting_loss=float(
            rule_value["challenger_maximum_supporting_loss"]
        ),
        minimum_nonnegative_seed_fraction=float(
            rule_value["minimum_nonnegative_seed_fraction"]
        ),
    )
    plan = DevelopmentSelectionPlan(
        schema_version=int(decoded["schema_version"]),
        plan_id=str(decoded["plan_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        training_authorized=decoded.get("training_authorized") is True,
        selection_configuration_sha256=str(
            decoded["selection_configuration_sha256"]
        ),
        protocol_configuration_sha256=str(
            decoded["protocol_configuration_sha256"]
        ),
        split_manifest_sha256=str(decoded["split_manifest_sha256"]),
        window_store_index_sha256=str(decoded["window_store_index_sha256"]),
        statistical_analysis_plan_sha256=str(
            decoded["statistical_analysis_plan_sha256"]
        ),
        class_labels=tuple(map(str, decoded["class_labels"])),
        random_seeds=tuple(int(seed) for seed in decoded["random_seeds"]),
        candidates=candidates,
        folds=folds,
        analysis_settings=dict(decoded["analysis_settings"]),
        selection_rule=rule,
        expected_run_count=int(decoded["expected_run_count"]),
        plan_payload_sha256=str(decoded["plan_payload_sha256"]),
    )
    plan.validate()
    return plan


def _resolve_artifact_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _evidence_from_artifacts(
    *,
    manifest_path: Path,
    entry: Mapping[str, object],
    plan: DevelopmentSelectionPlan,
) -> SelectionRunEvidence:
    from .analysis_records import load_prediction_analysis_record
    from .prediction_artifacts import load_prediction_artifact_record
    from .results import load_training_result_record

    candidate_id = str(entry["candidate_id"])
    fold_index = int(entry["fold_index"])
    random_seed = int(entry["random_seed"])
    candidates = {candidate.experiment_id: candidate for candidate in plan.candidates}
    folds = {fold.fold_index: fold for fold in plan.folds}
    if candidate_id not in candidates or fold_index not in folds:
        raise ValueError("Run manifest references an undeclared candidate or fold")
    candidate = candidates[candidate_id]
    fold = folds[fold_index]
    training_path = _resolve_artifact_path(manifest_path, entry["training_result"])
    prediction_path = _resolve_artifact_path(manifest_path, entry["prediction_artifact"])
    analysis_path = _resolve_artifact_path(manifest_path, entry["analysis_record"])
    training = load_training_result_record(training_path)
    prediction = load_prediction_artifact_record(prediction_path)
    analysis = load_prediction_analysis_record(analysis_path)
    expected_scientific = plan.purpose == DEVELOPMENT_SELECTION
    for record_name, purpose, scientific_result, holdout_accessed in (
        ("training", training.purpose, training.scientific_result, training.holdout_accessed),
        (
            "prediction",
            prediction.purpose,
            prediction.scientific_result,
            prediction.holdout_accessed,
        ),
        ("analysis", analysis.purpose, analysis.scientific_result, analysis.holdout_accessed),
    ):
        if purpose != plan.purpose or scientific_result is not expected_scientific:
            raise ValueError(f"{record_name} scope disagrees with the selection plan")
        if holdout_accessed:
            raise PermissionError(f"{record_name} unexpectedly claims hold-out access")
    if training.experiment_id != candidate_id or prediction.experiment_id != candidate_id:
        raise ValueError("Run experiment identifier disagrees with the candidate slot")
    if training.model_configuration_sha256 != candidate.model_configuration_sha256 or (
        prediction.model_configuration_sha256 != candidate.model_configuration_sha256
    ):
        raise ValueError("Run artifacts disagree with the candidate configuration")
    if training.trainable_parameter_count != candidate.trainable_parameter_count:
        raise ValueError("Training parameter count disagrees with the frozen candidate")
    if expected_scientific and (
        training.protocol_configuration_sha256
        != plan.protocol_configuration_sha256
        or training.data_provenance is None
        or training.data_provenance.split_manifest_sha256
        != plan.split_manifest_sha256
        or training.data_provenance.window_store_index_sha256
        != plan.window_store_index_sha256
    ):
        raise ValueError(
            "Scientific training provenance does not bind the plan protocol, split, and window store"
        )
    if training.random_seed != random_seed:
        raise ValueError("Training seed disagrees with the run slot")
    if training.training_subjects != fold.training_subjects or (
        training.validation_subjects != fold.validation_subjects
    ):
        raise ValueError("Training scope disagrees with the shared fold")
    if prediction.validation_subjects != fold.validation_subjects:
        raise ValueError("Prediction scope disagrees with the shared fold")
    if prediction.class_labels != plan.class_labels or analysis.class_labels != plan.class_labels:
        raise ValueError("Run class vocabulary disagrees with the selection plan")
    if prediction.training_run_id != training.run_id or (
        prediction.training_result_sha256 != sha256_file(training_path)
    ):
        raise ValueError("Prediction artifact does not bind the supplied training result")
    if analysis.prediction_run_id != prediction.run_id or (
        analysis.prediction_artifact_sha256 != sha256_file(prediction_path)
    ):
        raise ValueError("Analysis record does not bind the supplied prediction artifact")
    if analysis.model_state_sha256 != prediction.model_state_sha256:
        raise ValueError("Analysis and prediction model-state digests differ")
    if analysis.analysis_settings != plan.analysis_settings:
        raise ValueError("Run analysis settings disagree with the frozen plan")
    if len({training.source_revision, prediction.source_revision, analysis.source_revision}) != 1:
        raise ValueError("Run source revisions disagree across the artifact chain")
    classification = analysis.analysis_payload["classification"]
    if not isinstance(classification, Mapping):
        raise TypeError("Analysis classification payload must be an object")
    per_participant = classification["per_participant_metrics"]
    if not isinstance(per_participant, Mapping):
        raise TypeError("Per-participant metrics must be an object")
    participant_metrics = {
        str(participant): {
            "macro_f1": _finite_unit_interval(metrics["macro_f1"], "macro F1"),
            "balanced_accuracy": _finite_unit_interval(
                metrics["balanced_accuracy"], "balanced accuracy"
            ),
        }
        for participant, metrics in per_participant.items()
    }
    history = tuple(
        {
            "epoch": int(value["epoch"]),
            "training_loss": float(value["training_loss"]),
            "validation_loss": float(value["validation_loss"]),
            "validation_macro_f1": float(value["validation_macro_f1"]),
            "validation_balanced_accuracy": float(
                value["validation_balanced_accuracy"]
            ),
        }
        for value in training.history
    )
    evidence = SelectionRunEvidence(
        candidate_id=candidate_id,
        fold_index=fold_index,
        random_seed=random_seed,
        training_run_id=training.run_id,
        training_result_sha256=sha256_file(training_path),
        prediction_run_id=prediction.run_id,
        prediction_artifact_sha256=sha256_file(prediction_path),
        analysis_id=analysis.analysis_id,
        analysis_record_sha256=sha256_file(analysis_path),
        model_state_sha256=prediction.model_state_sha256,
        validation_subjects=fold.validation_subjects,
        participant_metrics=participant_metrics,
        history=history,
    )
    evidence.validate()
    return evidence


def _candidate_summaries(
    plan: DevelopmentSelectionPlan,
    evidence: Sequence[SelectionRunEvidence],
) -> dict[str, object]:
    development_subjects = {
        subject for fold in plan.folds for subject in fold.validation_subjects
    }
    summaries: dict[str, object] = {}
    for candidate in plan.candidates:
        candidate_evidence = [
            item for item in evidence if item.candidate_id == candidate.experiment_id
        ]
        per_seed: list[dict[str, object]] = []
        for seed in plan.random_seeds:
            seed_evidence = [
                item for item in candidate_evidence if item.random_seed == seed
            ]
            metrics_by_participant: dict[str, dict[str, float]] = {}
            for item in seed_evidence:
                overlap = set(metrics_by_participant) & set(item.participant_metrics)
                if overlap:
                    raise ValueError("A participant appears in multiple folds for one seed")
                metrics_by_participant.update(item.participant_metrics)
            if set(metrics_by_participant) != development_subjects:
                raise ValueError("One candidate/seed does not cover every development participant")
            per_seed.append(
                {
                    "random_seed": seed,
                    "participant_count": len(metrics_by_participant),
                    "participant_macro_f1_mean": _mean(
                        [value["macro_f1"] for value in metrics_by_participant.values()]
                    ),
                    "participant_balanced_accuracy_mean": _mean(
                        [
                            value["balanced_accuracy"]
                            for value in metrics_by_participant.values()
                        ]
                    ),
                }
            )
        macro_values = [float(value["participant_macro_f1_mean"]) for value in per_seed]
        balanced_values = [
            float(value["participant_balanced_accuracy_mean"]) for value in per_seed
        ]
        maximum_epoch = max(len(item.history) for item in candidate_evidence)
        learning_curve: list[dict[str, object]] = []
        for epoch in range(1, maximum_epoch + 1):
            entries = [
                item.history[epoch - 1]
                for item in candidate_evidence
                if len(item.history) >= epoch
            ]
            learning_curve.append(
                {
                    "epoch": epoch,
                    "contributing_run_count": len(entries),
                    "contributing_run_fraction": len(entries) / len(candidate_evidence),
                    "training_loss_mean": _mean(
                        [float(value["training_loss"]) for value in entries]
                    ),
                    "validation_loss_mean": _mean(
                        [float(value["validation_loss"]) for value in entries]
                    ),
                    "validation_macro_f1_mean": _mean(
                        [float(value["validation_macro_f1"]) for value in entries]
                    ),
                    "validation_balanced_accuracy_mean": _mean(
                        [
                            float(value["validation_balanced_accuracy"])
                            for value in entries
                        ]
                    ),
                }
            )
        summaries[candidate.experiment_id] = {
            "role": candidate.role,
            "trainable_parameter_count": candidate.trainable_parameter_count,
            "run_count": len(candidate_evidence),
            "fold_count": len(plan.folds),
            "seed_count": len(plan.random_seeds),
            "development_participant_count": len(development_subjects),
            "per_seed": per_seed,
            "participant_macro_f1_seed_mean": _mean(macro_values),
            "participant_macro_f1_seed_population_sd": _population_sd(macro_values),
            "participant_balanced_accuracy_seed_mean": _mean(balanced_values),
            "participant_balanced_accuracy_seed_population_sd": _population_sd(
                balanced_values
            ),
            "learning_curve": learning_curve,
            "learning_curve_warning": (
                "Later epochs average only runs that had not already stopped; "
                "contributing_run_count must accompany every point."
            ),
        }
    return summaries


def _selection_decision(
    plan: DevelopmentSelectionPlan,
    summaries: Mapping[str, object],
) -> dict[str, object]:
    reference = next(candidate for candidate in plan.candidates if candidate.role == "reference")
    challenger = next(
        candidate for candidate in plan.candidates if candidate.role == "challenger"
    )
    reference_summary = summaries[reference.experiment_id]
    challenger_summary = summaries[challenger.experiment_id]
    if not isinstance(reference_summary, Mapping) or not isinstance(
        challenger_summary, Mapping
    ):
        raise TypeError("Candidate summaries must be objects")
    primary_gain = float(challenger_summary["participant_macro_f1_seed_mean"]) - float(
        reference_summary["participant_macro_f1_seed_mean"]
    )
    supporting_gain = float(
        challenger_summary["participant_balanced_accuracy_seed_mean"]
    ) - float(reference_summary["participant_balanced_accuracy_seed_mean"])
    reference_per_seed = {
        int(value["random_seed"]): float(value["participant_macro_f1_mean"])
        for value in reference_summary["per_seed"]
    }
    challenger_per_seed = {
        int(value["random_seed"]): float(value["participant_macro_f1_mean"])
        for value in challenger_summary["per_seed"]
    }
    if set(reference_per_seed) != set(challenger_per_seed):
        raise ValueError("Candidate seed summaries are not paired")
    nonnegative_seed_fraction = sum(
        challenger_per_seed[seed] - reference_per_seed[seed] >= 0.0
        for seed in plan.random_seeds
    ) / len(plan.random_seeds)
    rule = plan.selection_rule
    criteria = {
        "minimum_primary_gain": {
            "threshold": rule.challenger_minimum_primary_gain,
            "observed": primary_gain,
            "passed": primary_gain >= rule.challenger_minimum_primary_gain,
        },
        "maximum_supporting_loss": {
            "threshold": -rule.challenger_maximum_supporting_loss,
            "observed": supporting_gain,
            "passed": supporting_gain >= -rule.challenger_maximum_supporting_loss,
        },
        "minimum_nonnegative_seed_fraction": {
            "threshold": rule.minimum_nonnegative_seed_fraction,
            "observed": nonnegative_seed_fraction,
            "passed": nonnegative_seed_fraction
            >= rule.minimum_nonnegative_seed_fraction,
        },
    }
    challenger_selected = all(bool(value["passed"]) for value in criteria.values())
    return {
        "status": (
            "scientific_development_selection"
            if plan.purpose == DEVELOPMENT_SELECTION
            else "synthetic_contract_validation_only"
        ),
        "selected_candidate_id": (
            challenger.experiment_id if challenger_selected else reference.experiment_id
        ),
        "reference_candidate_id": reference.experiment_id,
        "challenger_candidate_id": challenger.experiment_id,
        "primary_gain": primary_gain,
        "supporting_gain": supporting_gain,
        "nonnegative_seed_fraction": nonnegative_seed_fraction,
        "criteria": criteria,
    }


def build_model_selection_bundle(
    *,
    bundle_id: str,
    created_at_utc: str,
    selection_plan_path: Path | str,
    run_manifest_path: Path | str,
) -> ModelSelectionBundle:
    """Validate a complete artifact grid and apply the frozen conservative rule."""
    from .results import load_training_result_record

    plan_path = Path(selection_plan_path).resolve()
    manifest_path = Path(run_manifest_path).resolve()
    plan = load_development_selection_plan(plan_path)
    if plan.purpose == DEVELOPMENT_SELECTION and not plan.training_authorized:
        raise PermissionError(
            "The frozen development-selection plan records a closed training gate"
        )
    manifest = _load_json_object(manifest_path)
    if int(manifest["schema_version"]) != 1:
        raise ValueError("Unsupported selection-run manifest schema version")
    if str(manifest["purpose"]) != plan.purpose:
        raise ValueError("Run-manifest purpose disagrees with the selection plan")
    if str(manifest["selection_plan_sha256"]) != sha256_file(plan_path):
        raise ValueError("Run manifest does not bind the supplied selection plan")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or any(
        not isinstance(entry, Mapping) for entry in entries
    ):
        raise TypeError("Selection-run manifest runs must be an array of objects")
    evidence = tuple(
        _evidence_from_artifacts(
            manifest_path=manifest_path,
            entry=entry,
            plan=plan,
        )
        for entry in entries
    )
    observed_slots = {
        (item.candidate_id, item.fold_index, item.random_seed) for item in evidence
    }
    if len(observed_slots) != len(evidence):
        raise ValueError("Selection-run manifest contains duplicate grid slots")
    expected_slots = {
        (candidate.experiment_id, fold.fold_index, seed)
        for candidate in plan.candidates
        for fold in plan.folds
        for seed in plan.random_seeds
    }
    if observed_slots != expected_slots or len(evidence) != plan.expected_run_count:
        missing = sorted(expected_slots - observed_slots)
        extra = sorted(observed_slots - expected_slots)
        raise ValueError(
            "Selection evidence does not exactly fill the frozen grid: "
            f"missing={missing}, extra={extra}"
        )
    evidence = tuple(
        sorted(
            evidence,
            key=lambda item: (item.candidate_id, item.fold_index, item.random_seed),
        )
    )
    source_revisions = {
        load_training_result_record(
            _resolve_artifact_path(manifest_path, entry["training_result"])
        ).source_revision
        for entry in entries
    }
    if len(source_revisions) != 1:
        raise ValueError("All selection runs must use one source revision")
    summaries = _candidate_summaries(plan, evidence)
    decision = _selection_decision(plan, summaries)
    payload = {
        "run_evidence": [item.as_dict() for item in evidence],
        "candidate_summaries": summaries,
        "decision": decision,
    }
    bundle = ModelSelectionBundle(
        schema_version=1,
        bundle_id=bundle_id,
        created_at_utc=created_at_utc,
        purpose=plan.purpose,
        scientific_result=(
            plan.purpose == DEVELOPMENT_SELECTION
        ),
        holdout_accessed=False,
        selection_plan_sha256=sha256_file(plan_path),
        source_revision=next(iter(source_revisions)),
        run_count=len(evidence),
        run_evidence=evidence,
        candidate_summaries=summaries,
        decision=decision,
        bundle_payload_sha256=sha256_canonical_json(payload),
    )
    bundle.validate()
    return bundle


def write_model_selection_bundle(path: Path | str, bundle: ModelSelectionBundle) -> None:
    """Write a model-selection bundle exclusively."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(bundle.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_model_selection_bundle(path: Path | str) -> ModelSelectionBundle:
    """Load and validate a model-selection bundle."""
    decoded = _load_json_object(Path(path))
    evidence = tuple(
        SelectionRunEvidence(
            candidate_id=str(value["candidate_id"]),
            fold_index=int(value["fold_index"]),
            random_seed=int(value["random_seed"]),
            training_run_id=str(value["training_run_id"]),
            training_result_sha256=str(value["training_result_sha256"]),
            prediction_run_id=str(value["prediction_run_id"]),
            prediction_artifact_sha256=str(value["prediction_artifact_sha256"]),
            analysis_id=str(value["analysis_id"]),
            analysis_record_sha256=str(value["analysis_record_sha256"]),
            model_state_sha256=str(value["model_state_sha256"]),
            validation_subjects=tuple(map(str, value["validation_subjects"])),
            participant_metrics={
                str(participant): {
                    "macro_f1": float(metrics["macro_f1"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                }
                for participant, metrics in value["participant_metrics"].items()
            },
            history=tuple(dict(entry) for entry in value["history"]),
        )
        for value in decoded["run_evidence"]
    )
    bundle = ModelSelectionBundle(
        schema_version=int(decoded["schema_version"]),
        bundle_id=str(decoded["bundle_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        selection_plan_sha256=str(decoded["selection_plan_sha256"]),
        source_revision=str(decoded["source_revision"]),
        run_count=int(decoded["run_count"]),
        run_evidence=evidence,
        candidate_summaries=dict(decoded["candidate_summaries"]),
        decision=dict(decoded["decision"]),
        bundle_payload_sha256=str(decoded["bundle_payload_sha256"]),
    )
    bundle.validate()
    return bundle
