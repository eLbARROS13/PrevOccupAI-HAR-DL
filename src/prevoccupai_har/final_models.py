"""Frozen final-model settings and refits that cannot access hold-out data."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import joblib
    import sklearn
    import torch
    from sklearn.model_selection import ParameterGrid
    from sklearn.pipeline import Pipeline
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised without model extras
    raise ImportError(
        "PyTorch and scikit-learn are required for prevoccupai_har.final_models"
    ) from error

from .classical_baseline import (
    RandomForestDevelopmentRecord,
    RandomForestReconstructionConfiguration,
    _selected_feature_names,
    build_leakage_safe_random_forest_pipeline,
    load_random_forest_development_record,
    load_random_forest_reconstruction_configuration,
    select_fold_local_balanced_training_rows,
    sha256_feature_matrix,
)
from .feature_store import DevelopmentFeatureMatrix
from .model_selection import load_model_selection_bundle
from .modeling import OptimizationConfiguration
from .prediction_artifacts import sha256_model_state
from .preprocessing import TrainOnlyChannelStandardizer
from .protocol import ProtocolConfiguration
from .provenance import (
    is_reproducible_source_revision,
    sha256_canonical_json,
    sha256_file,
)
from .results import load_training_result_record
from .streaming_training import _loader
from .training import set_reproducible_seed
from .window_store import DevelopmentWindowStore


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
FINAL_REFIT_PURPOSE = "final_development_refit"


def _require_sha256(value: str, field_name: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA-256 value for {field_name}")


@dataclass(frozen=True)
class SeedEpochDecision:
    """One seed's exact five-fold evidence and mechanically derived epoch count."""

    random_seed: int
    fold_indices: tuple[int, ...]
    fold_best_epochs: tuple[int, ...]
    training_result_sha256: tuple[str, ...]
    fixed_epoch_count: int

    def validate(self) -> None:
        """Require a complete five-fold median decision."""
        if self.random_seed < 0:
            raise ValueError("Final-refit seed cannot be negative")
        if self.fold_indices != tuple(range(5)):
            raise ValueError("Epoch decisions require exactly folds 0 through 4")
        if len(self.fold_best_epochs) != 5 or any(
            value <= 0 for value in self.fold_best_epochs
        ):
            raise ValueError("Epoch decisions require five positive best epochs")
        if len(self.training_result_sha256) != 5:
            raise ValueError("Epoch decisions require five training-result hashes")
        for index, value in enumerate(self.training_result_sha256):
            _require_sha256(value, f"training_result_sha256[{index}]")
        expected = int(statistics.median(self.fold_best_epochs))
        if self.fixed_epoch_count != expected:
            raise ValueError("Fixed epoch count is not the exact five-fold median")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "random_seed": self.random_seed,
            "fold_indices": list(self.fold_indices),
            "fold_best_epochs": list(self.fold_best_epochs),
            "training_result_sha256": list(self.training_result_sha256),
            "fixed_epoch_count": self.fixed_epoch_count,
        }


@dataclass(frozen=True)
class FinalTrainingSettings:
    """Post-development settings frozen before any final model is fitted."""

    schema_version: int
    settings_id: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    selected_candidate_id: str
    class_labels: tuple[str, ...]
    development_participants: tuple[str, ...]
    holdout_participants: tuple[str, ...]
    epoch_decisions: tuple[SeedEpochDecision, ...]
    rf_hyperparameters: dict[str, object]
    rf_modal_fold_count: int
    rf_tie_count: int
    rf_selected_grid_index_zero_based: int
    input_hashes: dict[str, str]
    development_source_revision: str
    rf_development_source_revision: str
    settings_payload_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "class_labels": list(self.class_labels),
            "development_participants": list(self.development_participants),
            "holdout_participants": list(self.holdout_participants),
            "epoch_decisions": [value.as_dict() for value in self.epoch_decisions],
            "rf_hyperparameters": self.rf_hyperparameters,
            "rf_modal_fold_count": self.rf_modal_fold_count,
            "rf_tie_count": self.rf_tie_count,
            "rf_selected_grid_index_zero_based": (
                self.rf_selected_grid_index_zero_based
            ),
            "input_hashes": self.input_hashes,
            "development_source_revision": self.development_source_revision,
            "rf_development_source_revision": self.rf_development_source_revision,
        }

    def validate(self) -> None:
        """Reject non-mechanical settings or any implication of hold-out access."""
        if self.schema_version != 1:
            raise ValueError("Unsupported final-training-settings schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.settings_id) is None:
            raise ValueError("Final-training-settings identifier is invalid")
        if self.purpose != "final_model_settings_freeze":
            raise ValueError("Unsupported final-training-settings purpose")
        if self.scientific_result or self.holdout_accessed:
            raise ValueError("Settings freeze cannot be a result or access hold-out data")
        if not self.selected_candidate_id:
            raise ValueError("A selected DL candidate is required")
        if not self.class_labels or len(self.class_labels) != len(set(self.class_labels)):
            raise ValueError("Final class labels must be non-empty and unique")
        development = set(self.development_participants)
        holdout = set(self.holdout_participants)
        if (
            not development
            or not holdout
            or development & holdout
            or len(development) != len(self.development_participants)
            or len(holdout) != len(self.holdout_participants)
        ):
            raise ValueError("Final cohorts must be non-empty, unique, and disjoint")
        if tuple(value.random_seed for value in self.epoch_decisions) != (
            1103,
            2207,
            3301,
            4409,
            5519,
        ):
            raise ValueError("Final-refit decisions require the five predeclared seeds")
        for value in self.epoch_decisions:
            value.validate()
        if set(self.rf_hyperparameters) != {
            "criterion",
            "max_depth",
            "n_estimators",
        }:
            raise ValueError("Frozen RF hyperparameters are incomplete")
        if self.rf_modal_fold_count <= 0 or self.rf_tie_count <= 0:
            raise ValueError("RF modal-decision counts must be positive")
        if self.rf_selected_grid_index_zero_based < 0:
            raise ValueError("RF selected grid index cannot be negative")
        if not self.input_hashes:
            raise ValueError("Final settings require bound input hashes")
        for field_name, value in self.input_hashes.items():
            _require_sha256(value, field_name)
        if not is_reproducible_source_revision(self.development_source_revision) or not (
            is_reproducible_source_revision(self.rf_development_source_revision)
        ):
            raise ValueError("Final settings require immutable development revisions")
        _require_sha256(self.settings_payload_sha256, "settings_payload_sha256")
        if sha256_canonical_json(self._payload()) != self.settings_payload_sha256:
            raise ValueError("Final-training-settings payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "settings_id": self.settings_id,
            "purpose": self.purpose,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            **self._payload(),
            "settings_payload_sha256": self.settings_payload_sha256,
        }


def _load_json_object(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _modal_rf_hyperparameters(
    record: RandomForestDevelopmentRecord,
    configuration: RandomForestReconstructionConfiguration,
) -> tuple[dict[str, object], int, int, int]:
    counts = Counter(
        (
            str(fold.best_hyperparameters["criterion"]),
            fold.best_hyperparameters["max_depth"],
            int(fold.best_hyperparameters["n_estimators"]),
        )
        for fold in record.folds
    )
    modal_count = max(counts.values())
    tied = {key for key, value in counts.items() if value == modal_count}
    grid = tuple(
        ParameterGrid(
            {
                "criterion": list(configuration.criteria),
                "n_estimators": list(configuration.estimator_counts),
                "max_depth": list(configuration.maximum_depths),
            }
        )
    )
    for index, parameters in enumerate(grid):
        key = (
            str(parameters["criterion"]),
            parameters["max_depth"],
            int(parameters["n_estimators"]),
        )
        if key in tied:
            return (
                {
                    "criterion": key[0],
                    "max_depth": key[1],
                    "n_estimators": key[2],
                },
                modal_count,
                len(tied),
                index,
            )
    raise RuntimeError("RF modal tuple is absent from the exact configuration grid")


def build_final_training_settings(
    *,
    settings_id: str,
    development_run_manifest_path: Path | str,
    model_selection_bundle_path: Path | str,
    rf_development_record_path: Path | str,
    rf_configuration_path: Path | str,
    final_execution_plan_path: Path | str,
    selected_model_configuration_path: Path | str,
    protocol: ProtocolConfiguration,
) -> FinalTrainingSettings:
    """Derive every model-specific final setting from sealed development evidence."""
    run_manifest_path = Path(development_run_manifest_path).resolve()
    selection_path = Path(model_selection_bundle_path).resolve()
    rf_record_path = Path(rf_development_record_path).resolve()
    rf_config_path = Path(rf_configuration_path).resolve()
    final_plan_path = Path(final_execution_plan_path).resolve()
    model_config_path = Path(selected_model_configuration_path).resolve()
    run_manifest = _load_json_object(run_manifest_path)
    bundle = load_model_selection_bundle(selection_path)
    rf_record = load_random_forest_development_record(rf_record_path)
    rf_configuration = load_random_forest_reconstruction_configuration(rf_config_path)
    final_plan = _load_json_object(final_plan_path)
    protocol.validate()

    if run_manifest.get("holdout_accessed") is not False or bundle.holdout_accessed:
        raise PermissionError("Final settings cannot follow hold-out access")
    if rf_record.holdout_accessed or final_plan.get("holdout_accessed") is not False:
        raise PermissionError("Final settings require no-hold-out development evidence")
    if bundle.run_count != 50 or len(run_manifest.get("runs", ())) != 50:
        raise ValueError("Final settings require the complete 50-run DL grid")
    if bundle.selection_plan_sha256 != str(run_manifest["selection_plan_sha256"]):
        raise ValueError("Run manifest and selection bundle bind different plans")
    if bundle.source_revision != str(run_manifest["source_revision"]):
        raise ValueError("Run manifest and selection bundle bind different source revisions")
    selected_candidate = str(bundle.decision["selected_candidate_id"])
    model_configuration = _load_json_object(model_config_path)
    if str(model_configuration.get("experiment_id")) != selected_candidate:
        raise ValueError("Selected model configuration and bundle decision disagree")
    candidate_summary = bundle.candidate_summaries.get(selected_candidate)
    if not isinstance(candidate_summary, Mapping):
        raise ValueError("Selected candidate lacks a complete summary")

    evidence_by_slot = {
        (item.candidate_id, item.fold_index, item.random_seed): item
        for item in bundle.run_evidence
    }
    run_by_slot: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for value in run_manifest["runs"]:
        if not isinstance(value, Mapping):
            raise TypeError("Development run entries must be objects")
        slot = (
            str(value["candidate_id"]),
            int(value["fold_index"]),
            int(value["random_seed"]),
        )
        if slot in run_by_slot:
            raise ValueError("Development run manifest contains a duplicate slot")
        run_by_slot[slot] = value

    epoch_decisions: list[SeedEpochDecision] = []
    declared_seeds = tuple(int(value) for value in final_plan["dl_final_refit"]["random_seeds"])
    if declared_seeds != (1103, 2207, 3301, 4409, 5519):
        raise ValueError("Final execution plan changed the frozen seed sequence")
    for seed in declared_seeds:
        epochs: list[int] = []
        hashes: list[str] = []
        for fold in range(5):
            slot = (selected_candidate, fold, seed)
            run = run_by_slot.get(slot)
            evidence = evidence_by_slot.get(slot)
            if run is None or evidence is None:
                raise ValueError(f"Selected-candidate evidence is incomplete for {slot}")
            training_path = run_manifest_path.parent / str(run["training_result"])
            observed_hash = sha256_file(training_path)
            if observed_hash != evidence.training_result_sha256:
                raise ValueError("Selected training-result hash changed")
            training = load_training_result_record(training_path)
            if (
                training.experiment_id != selected_candidate
                or training.random_seed != seed
                or training.holdout_accessed
            ):
                raise ValueError("Selected training result has inconsistent scope")
            epochs.append(training.best_epoch)
            hashes.append(observed_hash)
        epoch_decisions.append(
            SeedEpochDecision(
                random_seed=seed,
                fold_indices=tuple(range(5)),
                fold_best_epochs=tuple(epochs),
                training_result_sha256=tuple(hashes),
                fixed_epoch_count=int(statistics.median(epochs)),
            )
        )

    rf_hyperparameters, modal_count, tie_count, grid_index = _modal_rf_hyperparameters(
        rf_record,
        rf_configuration,
    )
    payload = {
        "selected_candidate_id": selected_candidate,
        "class_labels": list(protocol.main_labels),
        "development_participants": list(protocol.development_participants),
        "holdout_participants": list(protocol.holdout_participants),
        "epoch_decisions": [value.as_dict() for value in epoch_decisions],
        "rf_hyperparameters": rf_hyperparameters,
        "rf_modal_fold_count": modal_count,
        "rf_tie_count": tie_count,
        "rf_selected_grid_index_zero_based": grid_index,
        "input_hashes": {
            "development_run_manifest_sha256": sha256_file(run_manifest_path),
            "model_selection_bundle_sha256": sha256_file(selection_path),
            "rf_development_record_sha256": sha256_file(rf_record_path),
            "rf_configuration_sha256": sha256_file(rf_config_path),
            "final_execution_plan_sha256": sha256_file(final_plan_path),
            "selected_model_configuration_sha256": sha256_file(model_config_path),
        },
        "development_source_revision": bundle.source_revision,
        "rf_development_source_revision": rf_record.source_revision,
    }
    settings = FinalTrainingSettings(
        schema_version=1,
        settings_id=settings_id,
        purpose="final_model_settings_freeze",
        scientific_result=False,
        holdout_accessed=False,
        selected_candidate_id=selected_candidate,
        class_labels=protocol.main_labels,
        development_participants=protocol.development_participants,
        holdout_participants=protocol.holdout_participants,
        epoch_decisions=tuple(epoch_decisions),
        rf_hyperparameters=rf_hyperparameters,
        rf_modal_fold_count=modal_count,
        rf_tie_count=tie_count,
        rf_selected_grid_index_zero_based=grid_index,
        input_hashes=dict(payload["input_hashes"]),
        development_source_revision=bundle.source_revision,
        rf_development_source_revision=rf_record.source_revision,
        settings_payload_sha256=sha256_canonical_json(payload),
    )
    settings.validate()
    return settings


def write_final_training_settings(
    path: Path | str,
    settings: FinalTrainingSettings,
) -> None:
    """Write final settings exclusively before model refitting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(settings.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_final_training_settings(path: Path | str) -> FinalTrainingSettings:
    """Load and validate frozen final-training settings."""
    decoded = _load_json_object(path)
    settings = FinalTrainingSettings(
        schema_version=int(decoded["schema_version"]),
        settings_id=str(decoded["settings_id"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        selected_candidate_id=str(decoded["selected_candidate_id"]),
        class_labels=tuple(map(str, decoded["class_labels"])),
        development_participants=tuple(map(str, decoded["development_participants"])),
        holdout_participants=tuple(map(str, decoded["holdout_participants"])),
        epoch_decisions=tuple(
            SeedEpochDecision(
                random_seed=int(value["random_seed"]),
                fold_indices=tuple(int(item) for item in value["fold_indices"]),
                fold_best_epochs=tuple(int(item) for item in value["fold_best_epochs"]),
                training_result_sha256=tuple(
                    map(str, value["training_result_sha256"])
                ),
                fixed_epoch_count=int(value["fixed_epoch_count"]),
            )
            for value in decoded["epoch_decisions"]
        ),
        rf_hyperparameters=dict(decoded["rf_hyperparameters"]),
        rf_modal_fold_count=int(decoded["rf_modal_fold_count"]),
        rf_tie_count=int(decoded["rf_tie_count"]),
        rf_selected_grid_index_zero_based=int(
            decoded["rf_selected_grid_index_zero_based"]
        ),
        input_hashes={str(key): str(value) for key, value in decoded["input_hashes"].items()},
        development_source_revision=str(decoded["development_source_revision"]),
        rf_development_source_revision=str(decoded["rf_development_source_revision"]),
        settings_payload_sha256=str(decoded["settings_payload_sha256"]),
    )
    settings.validate()
    return settings


@dataclass(frozen=True)
class FixedEpochHistoryEntry:
    """One all-development training epoch with no validation-derived quantity."""

    epoch: int
    training_loss: float


@dataclass(frozen=True)
class FixedEpochTrainingOutcome:
    """Final-refit training outcome with no checkpoint selection."""

    random_seed: int
    fixed_epoch_count: int
    history: tuple[FixedEpochHistoryEntry, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class FinalDlRefitRecord:
    """Immutable all-development DL refit with no validation or hold-out input."""

    schema_version: int
    refit_id: str
    created_at_utc: str
    purpose: str
    scientific_model_artifact: bool
    holdout_accessed: bool
    experiment_id: str
    final_stage_source_revision: str
    development_source_revision: str
    random_seed: int
    fixed_epoch_count: int
    fold_best_epochs: tuple[int, ...]
    development_participants: tuple[str, ...]
    sealed_holdout_participants: tuple[str, ...]
    development_window_count: int
    development_class_counts: dict[str, int]
    sample_stride: int
    trainable_parameter_count: int
    input_hashes: dict[str, str]
    preprocessing_state_payload_sha256: str
    preprocessing_state_file_sha256: str
    model_state_payload_sha256: str
    model_state_file_sha256: str
    history: tuple[FixedEpochHistoryEntry, ...]
    elapsed_seconds: float
    software_versions: dict[str, str]
    refit_payload_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "final_stage_source_revision": self.final_stage_source_revision,
            "development_source_revision": self.development_source_revision,
            "random_seed": self.random_seed,
            "fixed_epoch_count": self.fixed_epoch_count,
            "fold_best_epochs": list(self.fold_best_epochs),
            "development_participants": list(self.development_participants),
            "sealed_holdout_participants": list(self.sealed_holdout_participants),
            "development_window_count": self.development_window_count,
            "development_class_counts": self.development_class_counts,
            "sample_stride": self.sample_stride,
            "trainable_parameter_count": self.trainable_parameter_count,
            "input_hashes": self.input_hashes,
            "preprocessing_state_payload_sha256": (
                self.preprocessing_state_payload_sha256
            ),
            "preprocessing_state_file_sha256": self.preprocessing_state_file_sha256,
            "model_state_payload_sha256": self.model_state_payload_sha256,
            "model_state_file_sha256": self.model_state_file_sha256,
            "history": [asdict(value) for value in self.history],
            "elapsed_seconds": self.elapsed_seconds,
            "software_versions": self.software_versions,
        }

    def validate(self) -> None:
        """Reject incomplete refits or any validation/hold-out-like state."""
        if self.schema_version != 1:
            raise ValueError("Unsupported final-DL-refit schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.refit_id) is None:
            raise ValueError("Final-DL-refit identifier is invalid")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Final-DL-refit time must use second-resolution UTC")
        if self.purpose != FINAL_REFIT_PURPOSE or not self.scientific_model_artifact:
            raise ValueError("Final-DL-refit purpose or scientific status is invalid")
        if self.holdout_accessed:
            raise PermissionError("Final DL refit cannot access hold-out data")
        if not self.experiment_id:
            raise ValueError("Final DL refit requires an experiment identifier")
        if not is_reproducible_source_revision(self.final_stage_source_revision) or not (
            is_reproducible_source_revision(self.development_source_revision)
        ):
            raise ValueError("Final DL refit requires immutable source revisions")
        if self.random_seed not in {1103, 2207, 3301, 4409, 5519}:
            raise ValueError("Final DL refit seed is not predeclared")
        if (
            len(self.fold_best_epochs) != 5
            or int(statistics.median(self.fold_best_epochs)) != self.fixed_epoch_count
        ):
            raise ValueError("Final DL epoch count is not the five-fold median")
        development = set(self.development_participants)
        holdout = set(self.sealed_holdout_participants)
        if not development or not holdout or development & holdout:
            raise ValueError("Final DL development and hold-out cohorts are invalid")
        if (
            len(development) != len(self.development_participants)
            or len(holdout) != len(self.sealed_holdout_participants)
        ):
            raise ValueError("Final DL participant identifiers contain duplicates")
        if self.development_window_count <= 0:
            raise ValueError("Final DL refit requires development windows")
        if (
            not self.development_class_counts
            or sum(self.development_class_counts.values()) != self.development_window_count
            or any(value <= 0 for value in self.development_class_counts.values())
        ):
            raise ValueError("Final DL class counts are incomplete")
        if self.sample_stride <= 0 or self.trainable_parameter_count <= 0:
            raise ValueError("Final DL sample stride or parameter count is invalid")
        if not self.input_hashes:
            raise ValueError("Final DL refit requires bound input hashes")
        for field_name, value in {
            **self.input_hashes,
            "preprocessing_state_payload_sha256": self.preprocessing_state_payload_sha256,
            "preprocessing_state_file_sha256": self.preprocessing_state_file_sha256,
            "model_state_payload_sha256": self.model_state_payload_sha256,
            "model_state_file_sha256": self.model_state_file_sha256,
            "refit_payload_sha256": self.refit_payload_sha256,
        }.items():
            _require_sha256(value, field_name)
        if (
            len(self.history) != self.fixed_epoch_count
            or not finite_history(self.history)
            or self.elapsed_seconds <= 0.0
            or not math.isfinite(self.elapsed_seconds)
        ):
            raise ValueError("Final DL training history or elapsed time is invalid")
        if not self.software_versions:
            raise ValueError("Final DL refit requires software versions")
        if sha256_canonical_json(self._payload()) != self.refit_payload_sha256:
            raise ValueError("Final-DL-refit payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "refit_id": self.refit_id,
            "created_at_utc": self.created_at_utc,
            "purpose": self.purpose,
            "scientific_model_artifact": self.scientific_model_artifact,
            "holdout_accessed": self.holdout_accessed,
            **self._payload(),
            "refit_payload_sha256": self.refit_payload_sha256,
        }


def build_final_dl_refit_record(
    *,
    refit_id: str,
    created_at_utc: str,
    experiment_id: str,
    final_stage_source_revision: str,
    settings: FinalTrainingSettings,
    epoch_decision: SeedEpochDecision,
    protocol: ProtocolConfiguration,
    development_window_count: int,
    development_class_counts: Mapping[str, int],
    sample_stride: int,
    trainable_parameter_count: int,
    input_hashes: Mapping[str, str],
    preprocessing_state_payload_sha256: str,
    preprocessing_state_file_sha256: str,
    model_state_payload_sha256: str,
    model_state_file_sha256: str,
    outcome: FixedEpochTrainingOutcome,
) -> FinalDlRefitRecord:
    """Construct a final-DL record from an exact frozen epoch decision."""
    settings.validate()
    epoch_decision.validate()
    protocol.validate()
    if experiment_id != settings.selected_candidate_id:
        raise ValueError("Final DL experiment differs from the selected candidate")
    if epoch_decision not in settings.epoch_decisions:
        raise ValueError("Final DL epoch decision is absent from frozen settings")
    if (
        outcome.random_seed != epoch_decision.random_seed
        or outcome.fixed_epoch_count != epoch_decision.fixed_epoch_count
    ):
        raise ValueError("Final DL outcome differs from its frozen epoch decision")
    if tuple(protocol.development_participants) != settings.development_participants or tuple(
        protocol.holdout_participants
    ) != settings.holdout_participants:
        raise ValueError("Final DL protocol and frozen settings cohorts differ")
    payload = {
        "experiment_id": experiment_id,
        "final_stage_source_revision": final_stage_source_revision,
        "development_source_revision": settings.development_source_revision,
        "random_seed": epoch_decision.random_seed,
        "fixed_epoch_count": epoch_decision.fixed_epoch_count,
        "fold_best_epochs": list(epoch_decision.fold_best_epochs),
        "development_participants": list(settings.development_participants),
        "sealed_holdout_participants": list(settings.holdout_participants),
        "development_window_count": int(development_window_count),
        "development_class_counts": {
            str(key): int(value) for key, value in development_class_counts.items()
        },
        "sample_stride": int(sample_stride),
        "trainable_parameter_count": int(trainable_parameter_count),
        "input_hashes": {str(key): str(value) for key, value in input_hashes.items()},
        "preprocessing_state_payload_sha256": preprocessing_state_payload_sha256,
        "preprocessing_state_file_sha256": preprocessing_state_file_sha256,
        "model_state_payload_sha256": model_state_payload_sha256,
        "model_state_file_sha256": model_state_file_sha256,
        "history": [asdict(value) for value in outcome.history],
        "elapsed_seconds": outcome.elapsed_seconds,
        "software_versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    record = FinalDlRefitRecord(
        schema_version=1,
        refit_id=refit_id,
        created_at_utc=created_at_utc,
        purpose=FINAL_REFIT_PURPOSE,
        scientific_model_artifact=True,
        holdout_accessed=False,
        experiment_id=experiment_id,
        final_stage_source_revision=final_stage_source_revision,
        development_source_revision=settings.development_source_revision,
        random_seed=epoch_decision.random_seed,
        fixed_epoch_count=epoch_decision.fixed_epoch_count,
        fold_best_epochs=epoch_decision.fold_best_epochs,
        development_participants=settings.development_participants,
        sealed_holdout_participants=settings.holdout_participants,
        development_window_count=int(development_window_count),
        development_class_counts=dict(payload["development_class_counts"]),
        sample_stride=int(sample_stride),
        trainable_parameter_count=int(trainable_parameter_count),
        input_hashes=dict(payload["input_hashes"]),
        preprocessing_state_payload_sha256=preprocessing_state_payload_sha256,
        preprocessing_state_file_sha256=preprocessing_state_file_sha256,
        model_state_payload_sha256=model_state_payload_sha256,
        model_state_file_sha256=model_state_file_sha256,
        history=outcome.history,
        elapsed_seconds=outcome.elapsed_seconds,
        software_versions=dict(payload["software_versions"]),
        refit_payload_sha256=sha256_canonical_json(payload),
    )
    record.validate()
    return record


def write_final_dl_refit_record(path: Path | str, record: FinalDlRefitRecord) -> None:
    """Write an immutable final-DL-refit record exclusively."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_final_dl_refit_record(path: Path | str) -> FinalDlRefitRecord:
    """Load and validate an immutable final-DL-refit record."""
    decoded = _load_json_object(path)
    record = FinalDlRefitRecord(
        schema_version=int(decoded["schema_version"]),
        refit_id=str(decoded["refit_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        purpose=str(decoded["purpose"]),
        scientific_model_artifact=decoded.get("scientific_model_artifact") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        experiment_id=str(decoded["experiment_id"]),
        final_stage_source_revision=str(decoded["final_stage_source_revision"]),
        development_source_revision=str(decoded["development_source_revision"]),
        random_seed=int(decoded["random_seed"]),
        fixed_epoch_count=int(decoded["fixed_epoch_count"]),
        fold_best_epochs=tuple(int(value) for value in decoded["fold_best_epochs"]),
        development_participants=tuple(map(str, decoded["development_participants"])),
        sealed_holdout_participants=tuple(
            map(str, decoded["sealed_holdout_participants"])
        ),
        development_window_count=int(decoded["development_window_count"]),
        development_class_counts={
            str(key): int(value)
            for key, value in decoded["development_class_counts"].items()
        },
        sample_stride=int(decoded["sample_stride"]),
        trainable_parameter_count=int(decoded["trainable_parameter_count"]),
        input_hashes={str(key): str(value) for key, value in decoded["input_hashes"].items()},
        preprocessing_state_payload_sha256=str(
            decoded["preprocessing_state_payload_sha256"]
        ),
        preprocessing_state_file_sha256=str(decoded["preprocessing_state_file_sha256"]),
        model_state_payload_sha256=str(decoded["model_state_payload_sha256"]),
        model_state_file_sha256=str(decoded["model_state_file_sha256"]),
        history=tuple(
            FixedEpochHistoryEntry(
                epoch=int(value["epoch"]),
                training_loss=float(value["training_loss"]),
            )
            for value in decoded["history"]
        ),
        elapsed_seconds=float(decoded["elapsed_seconds"]),
        software_versions={
            str(key): str(value) for key, value in decoded["software_versions"].items()
        },
        refit_payload_sha256=str(decoded["refit_payload_sha256"]),
    )
    record.validate()
    return record


def fit_classifier_fixed_epochs_streaming(
    model: nn.Module,
    store: DevelopmentWindowStore,
    training_indices: NDArray[np.integer],
    standardizer: TrainOnlyChannelStandardizer,
    *,
    output_classes: int,
    optimization: OptimizationConfiguration,
    random_seed: int,
    fixed_epoch_count: int,
    protocol: ProtocolConfiguration,
    device: str = "cpu",
    sample_stride: int = 1,
) -> FixedEpochTrainingOutcome:
    """Fit exactly the frozen epoch count on all development participants only."""
    protocol.validate()
    optimization.validate()
    if not protocol.training_authorized or protocol.holdout_access_authorized:
        raise PermissionError("Protocol does not authorize a sealed final refit")
    if fixed_epoch_count <= 0 or fixed_epoch_count > optimization.maximum_epochs:
        raise ValueError("Fixed epoch count is outside the frozen optimization range")
    if output_classes != len(protocol.main_labels):
        raise ValueError("Final-refit output classes disagree with the protocol")
    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or len(np.unique(indices)) != indices.size:
        raise ValueError("Final-refit indices must be a non-empty unique vector")
    if not np.array_equal(np.sort(indices), np.arange(store.windows.shape[0])):
        raise ValueError("Final DL refit must use every development window exactly once")
    observed = frozenset(map(str, store.metadata["participant_id"][indices]))
    expected = frozenset(protocol.development_participants)
    if observed != expected or standardizer.allowed_training_subjects != expected:
        raise ValueError("Final DL refit must use exactly the development cohort")
    if observed & frozenset(protocol.holdout_participants):
        raise PermissionError("Final DL refit cannot receive hold-out participants")

    set_reproducible_seed(random_seed)
    loader = _loader(
        store,
        indices,
        standardizer,
        batch_size=optimization.batch_size,
        shuffle=True,
        seed=random_seed,
        sample_stride=sample_stride,
    )
    selected_device = torch.device(device)
    model.to(selected_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    history: list[FixedEpochHistoryEntry] = []
    started = time.perf_counter()
    for epoch in range(1, fixed_epoch_count + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for inputs, targets, _ in loader:
            inputs = inputs.to(selected_device)
            targets = targets.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if logits.shape != (targets.shape[0], output_classes):
                raise ValueError("Final-refit model output has an unexpected shape")
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Final-refit training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(targets.shape[0])
            total_examples += int(targets.shape[0])
        if total_examples != indices.size:
            raise RuntimeError("Final-refit epoch did not consume every development row")
        history.append(
            FixedEpochHistoryEntry(
                epoch=epoch,
                training_loss=total_loss / total_examples,
            )
        )
    elapsed = time.perf_counter() - started
    return FixedEpochTrainingOutcome(
        random_seed=random_seed,
        fixed_epoch_count=fixed_epoch_count,
        history=tuple(history),
        elapsed_seconds=elapsed,
    )


def write_model_state_npz(path: Path | str, model: nn.Module) -> None:
    """Write a deterministic-name NPZ representation exclusively."""
    output = Path(path)
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in sorted(model.state_dict().items())
    }
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_model_state_npz(
    path: Path | str,
    model: nn.Module,
    *,
    expected_payload_sha256: str | None = None,
) -> nn.Module:
    """Strictly reload an NPZ state and optionally verify its tensor payload hash."""
    with np.load(Path(path), allow_pickle=False) as archive:
        expected_names = tuple(sorted(model.state_dict()))
        if tuple(sorted(archive.files)) != expected_names:
            raise ValueError("Model-state tensor names changed")
        state = {
            name: torch.as_tensor(np.array(archive[name], copy=True))
            for name in expected_names
        }
    model.load_state_dict(state, strict=True)
    if expected_payload_sha256 is not None:
        _require_sha256(expected_payload_sha256, "expected_payload_sha256")
        if sha256_model_state(model) != expected_payload_sha256:
            raise ValueError("Reloaded model-state payload hash changed")
    return model


def arithmetic_mean_probabilities(
    logits_by_seed: Mapping[int, NDArray[np.floating]],
    *,
    expected_seeds: Sequence[int],
) -> NDArray[np.float64]:
    """Return an order-invariant float64 mean of softmax probabilities."""
    seeds = tuple(int(seed) for seed in expected_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Ensemble seeds must be non-empty and unique")
    if set(logits_by_seed) != set(seeds):
        raise ValueError("Ensemble logits do not exactly match the frozen seeds")
    arrays = [np.asarray(logits_by_seed[seed], dtype=np.float64) for seed in sorted(seeds)]
    if any(array.ndim != 2 or array.shape[0] == 0 for array in arrays):
        raise ValueError("Each ensemble logit matrix must be non-empty and two-dimensional")
    if len({array.shape for array in arrays}) != 1 or any(
        not np.isfinite(array).all() for array in arrays
    ):
        raise ValueError("Ensemble logits must be aligned and finite")
    probabilities = []
    for logits in arrays:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities.append(exponentials / exponentials.sum(axis=1, keepdims=True))
    mean = np.mean(np.stack(probabilities, axis=0), axis=0, dtype=np.float64)
    if not np.allclose(mean.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("Mean ensemble probabilities do not sum to one")
    mean.setflags(write=False)
    return mean


@dataclass(frozen=True)
class FinalRandomForestFit:
    """In-memory all-development RF refit and its training-only evidence."""

    pipeline: Pipeline
    selected_feature_names: tuple[str, ...]
    balanced_training_indices: NDArray[np.int64]
    balanced_training_class_counts: dict[str, int]
    balancing_quota_by_subactivity: dict[str, int]
    hyperparameters: dict[str, object]
    elapsed_seconds: float


@dataclass(frozen=True)
class FinalRandomForestRecord:
    """Immutable record for the all-development fitted RF pipeline."""

    schema_version: int
    refit_id: str
    created_at_utc: str
    purpose: str
    scientific_model_artifact: bool
    holdout_accessed: bool
    experiment_id: str
    final_stage_source_revision: str
    rf_development_source_revision: str
    development_participants: tuple[str, ...]
    sealed_holdout_participants: tuple[str, ...]
    class_labels: tuple[str, ...]
    random_seed: int
    hyperparameters: dict[str, object]
    development_row_count: int
    balanced_training_row_count: int
    balanced_training_indices_sha256: str
    balanced_training_class_counts: dict[str, int]
    balancing_quota_by_subactivity: dict[str, int]
    selected_feature_names: tuple[str, ...]
    feature_matrix_sha256: str
    input_hashes: dict[str, str]
    fitted_pipeline_file_sha256: str
    elapsed_seconds: float
    software_versions: dict[str, str]
    refit_payload_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "final_stage_source_revision": self.final_stage_source_revision,
            "rf_development_source_revision": self.rf_development_source_revision,
            "development_participants": list(self.development_participants),
            "sealed_holdout_participants": list(self.sealed_holdout_participants),
            "class_labels": list(self.class_labels),
            "random_seed": self.random_seed,
            "hyperparameters": self.hyperparameters,
            "development_row_count": self.development_row_count,
            "balanced_training_row_count": self.balanced_training_row_count,
            "balanced_training_indices_sha256": self.balanced_training_indices_sha256,
            "balanced_training_class_counts": self.balanced_training_class_counts,
            "balancing_quota_by_subactivity": self.balancing_quota_by_subactivity,
            "selected_feature_names": list(self.selected_feature_names),
            "feature_matrix_sha256": self.feature_matrix_sha256,
            "input_hashes": self.input_hashes,
            "fitted_pipeline_file_sha256": self.fitted_pipeline_file_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "software_versions": self.software_versions,
        }

    def validate(self) -> None:
        """Reject incomplete final RF artifacts or hold-out contamination."""
        if self.schema_version != 1:
            raise ValueError("Unsupported final-RF schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.refit_id) is None:
            raise ValueError("Final-RF identifier is invalid")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Final-RF time must use second-resolution UTC")
        if self.purpose != FINAL_REFIT_PURPOSE or not self.scientific_model_artifact:
            raise ValueError("Final-RF purpose or scientific status is invalid")
        if self.holdout_accessed:
            raise PermissionError("Final RF refit cannot access hold-out data")
        if not self.experiment_id:
            raise ValueError("Final RF requires an experiment identifier")
        if not is_reproducible_source_revision(self.final_stage_source_revision) or not (
            is_reproducible_source_revision(self.rf_development_source_revision)
        ):
            raise ValueError("Final RF requires immutable source revisions")
        development = set(self.development_participants)
        holdout = set(self.sealed_holdout_participants)
        if not development or not holdout or development & holdout:
            raise ValueError("Final RF cohorts are invalid")
        if not self.class_labels or len(self.class_labels) != len(set(self.class_labels)):
            raise ValueError("Final RF class labels are invalid")
        if self.random_seed != 42:
            raise ValueError("Final RF requires the predeclared seed 42")
        if set(self.hyperparameters) != {"criterion", "max_depth", "n_estimators"}:
            raise ValueError("Final RF hyperparameters are incomplete")
        if (
            self.development_row_count <= 0
            or not 0 < self.balanced_training_row_count <= self.development_row_count
        ):
            raise ValueError("Final RF row counts are invalid")
        if (
            not self.balanced_training_class_counts
            or sum(self.balanced_training_class_counts.values())
            != self.balanced_training_row_count
            or set(self.balanced_training_class_counts) != set(self.class_labels)
        ):
            raise ValueError("Final RF balanced class counts are invalid")
        if not self.balancing_quota_by_subactivity or any(
            value <= 0 for value in self.balancing_quota_by_subactivity.values()
        ):
            raise ValueError("Final RF balancing quotas are invalid")
        if not self.selected_feature_names or len(self.selected_feature_names) != len(
            set(self.selected_feature_names)
        ):
            raise ValueError("Final RF selected feature names are invalid")
        if not self.input_hashes:
            raise ValueError("Final RF requires bound input hashes")
        for field_name, value in {
            **self.input_hashes,
            "balanced_training_indices_sha256": self.balanced_training_indices_sha256,
            "feature_matrix_sha256": self.feature_matrix_sha256,
            "fitted_pipeline_file_sha256": self.fitted_pipeline_file_sha256,
            "refit_payload_sha256": self.refit_payload_sha256,
        }.items():
            _require_sha256(value, field_name)
        if self.elapsed_seconds <= 0.0 or not math.isfinite(self.elapsed_seconds):
            raise ValueError("Final RF elapsed time is invalid")
        if not self.software_versions:
            raise ValueError("Final RF requires software versions")
        if sha256_canonical_json(self._payload()) != self.refit_payload_sha256:
            raise ValueError("Final-RF payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "refit_id": self.refit_id,
            "created_at_utc": self.created_at_utc,
            "purpose": self.purpose,
            "scientific_model_artifact": self.scientific_model_artifact,
            "holdout_accessed": self.holdout_accessed,
            **self._payload(),
            "refit_payload_sha256": self.refit_payload_sha256,
        }


def _sha256_int64_indices(indices: NDArray[np.integer]) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    payload = {
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "bytes_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }
    return sha256_canonical_json(payload)


def build_final_random_forest_record(
    *,
    refit_id: str,
    created_at_utc: str,
    experiment_id: str,
    final_stage_source_revision: str,
    settings: FinalTrainingSettings,
    protocol: ProtocolConfiguration,
    dataset: DevelopmentFeatureMatrix,
    fitted: FinalRandomForestFit,
    input_hashes: Mapping[str, str],
    fitted_pipeline_file_sha256: str,
) -> FinalRandomForestRecord:
    """Construct the final RF record from a frozen all-development fit."""
    settings.validate()
    protocol.validate()
    if fitted.hyperparameters != settings.rf_hyperparameters:
        raise ValueError("Fitted RF hyperparameters differ from frozen settings")
    payload = {
        "experiment_id": experiment_id,
        "final_stage_source_revision": final_stage_source_revision,
        "rf_development_source_revision": settings.rf_development_source_revision,
        "development_participants": list(settings.development_participants),
        "sealed_holdout_participants": list(settings.holdout_participants),
        "class_labels": list(settings.class_labels),
        "random_seed": 42,
        "hyperparameters": fitted.hyperparameters,
        "development_row_count": int(dataset.features.shape[0]),
        "balanced_training_row_count": int(fitted.balanced_training_indices.size),
        "balanced_training_indices_sha256": _sha256_int64_indices(
            fitted.balanced_training_indices
        ),
        "balanced_training_class_counts": fitted.balanced_training_class_counts,
        "balancing_quota_by_subactivity": fitted.balancing_quota_by_subactivity,
        "selected_feature_names": list(fitted.selected_feature_names),
        "feature_matrix_sha256": final_rf_feature_matrix_sha256(
            dataset,
            settings.class_labels,
        ),
        "input_hashes": {str(key): str(value) for key, value in input_hashes.items()},
        "fitted_pipeline_file_sha256": fitted_pipeline_file_sha256,
        "elapsed_seconds": fitted.elapsed_seconds,
        "software_versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    record = FinalRandomForestRecord(
        schema_version=1,
        refit_id=refit_id,
        created_at_utc=created_at_utc,
        purpose=FINAL_REFIT_PURPOSE,
        scientific_model_artifact=True,
        holdout_accessed=False,
        experiment_id=experiment_id,
        final_stage_source_revision=final_stage_source_revision,
        rf_development_source_revision=settings.rf_development_source_revision,
        development_participants=settings.development_participants,
        sealed_holdout_participants=settings.holdout_participants,
        class_labels=settings.class_labels,
        random_seed=42,
        hyperparameters=dict(fitted.hyperparameters),
        development_row_count=int(dataset.features.shape[0]),
        balanced_training_row_count=int(fitted.balanced_training_indices.size),
        balanced_training_indices_sha256=str(
            payload["balanced_training_indices_sha256"]
        ),
        balanced_training_class_counts=dict(fitted.balanced_training_class_counts),
        balancing_quota_by_subactivity=dict(fitted.balancing_quota_by_subactivity),
        selected_feature_names=fitted.selected_feature_names,
        feature_matrix_sha256=str(payload["feature_matrix_sha256"]),
        input_hashes=dict(payload["input_hashes"]),
        fitted_pipeline_file_sha256=fitted_pipeline_file_sha256,
        elapsed_seconds=fitted.elapsed_seconds,
        software_versions=dict(payload["software_versions"]),
        refit_payload_sha256=sha256_canonical_json(payload),
    )
    record.validate()
    return record


def write_final_random_forest_record(
    path: Path | str,
    record: FinalRandomForestRecord,
) -> None:
    """Write an immutable final-RF record exclusively."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_final_random_forest_record(path: Path | str) -> FinalRandomForestRecord:
    """Load and validate an immutable final-RF record."""
    decoded = _load_json_object(path)
    record = FinalRandomForestRecord(
        schema_version=int(decoded["schema_version"]),
        refit_id=str(decoded["refit_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        purpose=str(decoded["purpose"]),
        scientific_model_artifact=decoded.get("scientific_model_artifact") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        experiment_id=str(decoded["experiment_id"]),
        final_stage_source_revision=str(decoded["final_stage_source_revision"]),
        rf_development_source_revision=str(decoded["rf_development_source_revision"]),
        development_participants=tuple(map(str, decoded["development_participants"])),
        sealed_holdout_participants=tuple(
            map(str, decoded["sealed_holdout_participants"])
        ),
        class_labels=tuple(map(str, decoded["class_labels"])),
        random_seed=int(decoded["random_seed"]),
        hyperparameters=dict(decoded["hyperparameters"]),
        development_row_count=int(decoded["development_row_count"]),
        balanced_training_row_count=int(decoded["balanced_training_row_count"]),
        balanced_training_indices_sha256=str(
            decoded["balanced_training_indices_sha256"]
        ),
        balanced_training_class_counts={
            str(key): int(value)
            for key, value in decoded["balanced_training_class_counts"].items()
        },
        balancing_quota_by_subactivity={
            str(key): int(value)
            for key, value in decoded["balancing_quota_by_subactivity"].items()
        },
        selected_feature_names=tuple(map(str, decoded["selected_feature_names"])),
        feature_matrix_sha256=str(decoded["feature_matrix_sha256"]),
        input_hashes={str(key): str(value) for key, value in decoded["input_hashes"].items()},
        fitted_pipeline_file_sha256=str(decoded["fitted_pipeline_file_sha256"]),
        elapsed_seconds=float(decoded["elapsed_seconds"]),
        software_versions={
            str(key): str(value) for key, value in decoded["software_versions"].items()
        },
        refit_payload_sha256=str(decoded["refit_payload_sha256"]),
    )
    record.validate()
    return record


def fit_final_random_forest(
    *,
    dataset: DevelopmentFeatureMatrix,
    configuration: RandomForestReconstructionConfiguration,
    settings: FinalTrainingSettings,
    protocol: ProtocolConfiguration,
) -> FinalRandomForestFit:
    """Refit balancing, selectors, and the frozen RF on all development rows."""
    configuration.validate()
    settings.validate()
    protocol.validate()
    if set(dataset.participant_ids) != set(protocol.development_participants):
        raise ValueError("Final RF dataset must exactly cover the development cohort")
    if set(dataset.participant_ids) & set(protocol.holdout_participants):
        raise PermissionError("Final RF dataset cannot contain hold-out participants")
    all_indices = np.arange(dataset.features.shape[0], dtype=np.int64)
    balanced, class_counts, quotas = select_fold_local_balanced_training_rows(
        labels=dataset.labels,
        subactivity_labels=dataset.subactivity_labels,
        participant_ids=dataset.participant_ids,
        candidate_indices=all_indices,
        class_labels=settings.class_labels,
        random_seed=configuration.random_seed,
    )
    pipeline = build_leakage_safe_random_forest_pipeline(
        configuration,
        random_seed=configuration.random_seed,
    )
    pipeline.set_params(
        random_forest__criterion=settings.rf_hyperparameters["criterion"],
        random_forest__n_estimators=settings.rf_hyperparameters["n_estimators"],
        random_forest__max_depth=settings.rf_hyperparameters["max_depth"],
    )
    started = time.perf_counter()
    pipeline.fit(
        dataset.features[balanced],
        np.asarray(dataset.labels, dtype=object)[balanced],
    )
    elapsed = time.perf_counter() - started
    selected_names = _selected_feature_names(pipeline, dataset.feature_names)
    if len(selected_names) != configuration.selected_feature_count:
        raise ValueError("Final RF did not retain the frozen selected-feature count")
    return FinalRandomForestFit(
        pipeline=pipeline,
        selected_feature_names=selected_names,
        balanced_training_indices=balanced,
        balanced_training_class_counts=class_counts,
        balancing_quota_by_subactivity=quotas,
        hyperparameters=dict(settings.rf_hyperparameters),
        elapsed_seconds=elapsed,
    )


def write_random_forest_pipeline(path: Path | str, pipeline: Pipeline) -> None:
    """Persist a fitted RF pipeline exclusively."""
    output = Path(path)
    with output.open("xb") as stream:
        joblib.dump(pipeline, stream, compress=3)


def load_random_forest_pipeline(path: Path | str) -> Pipeline:
    """Load a locally generated fitted RF pipeline."""
    with Path(path).open("rb") as stream:
        value = joblib.load(stream)
    if not isinstance(value, Pipeline):
        raise TypeError("Final RF artifact does not contain a scikit-learn Pipeline")
    return value


def final_rf_feature_matrix_sha256(
    dataset: DevelopmentFeatureMatrix,
    class_labels: Sequence[str],
) -> str:
    """Hash the exact all-development RF input for record construction."""
    return sha256_feature_matrix(
        dataset.features,
        labels=dataset.labels,
        subactivity_labels=dataset.subactivity_labels,
        participant_ids=dataset.participant_ids,
        feature_names=dataset.feature_names,
        class_labels=class_labels,
    )


def finite_history(history: Sequence[FixedEpochHistoryEntry]) -> bool:
    """Return whether fixed-epoch history is contiguous and finite."""
    return tuple(value.epoch for value in history) == tuple(range(1, len(history) + 1)) and all(
        math.isfinite(value.training_loss) and value.training_loss >= 0.0
        for value in history
    )


__all__ = [
    "FINAL_REFIT_PURPOSE",
    "FinalDlRefitRecord",
    "FinalRandomForestRecord",
    "FinalRandomForestFit",
    "FinalTrainingSettings",
    "FixedEpochHistoryEntry",
    "FixedEpochTrainingOutcome",
    "SeedEpochDecision",
    "arithmetic_mean_probabilities",
    "build_final_dl_refit_record",
    "build_final_random_forest_record",
    "build_final_training_settings",
    "final_rf_feature_matrix_sha256",
    "finite_history",
    "fit_classifier_fixed_epochs_streaming",
    "fit_final_random_forest",
    "load_final_training_settings",
    "load_final_dl_refit_record",
    "load_final_random_forest_record",
    "load_model_state_npz",
    "load_random_forest_pipeline",
    "write_final_training_settings",
    "write_final_dl_refit_record",
    "write_final_random_forest_record",
    "write_model_state_npz",
    "write_random_forest_pipeline",
]
