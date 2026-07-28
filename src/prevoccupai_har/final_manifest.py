"""Strict final-model freeze manifest built before hold-out authorization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .classical_baseline import (
    _selected_feature_names,
    load_random_forest_reconstruction_configuration,
)
from .final_models import (
    FinalDlRefitRecord,
    FinalRandomForestRecord,
    load_final_dl_refit_record,
    load_final_random_forest_record,
    load_final_training_settings,
    load_model_state_npz,
    load_random_forest_pipeline,
)
from .modeling import (
    build_time_series_classifier,
    load_time_series_experiment_configuration,
)
from .protocol import ProtocolConfiguration
from .provenance import sha256_canonical_json, sha256_file
from .source_snapshot import load_source_tree_manifest


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
EXPECTED_SEEDS = (1103, 2207, 3301, 4409, 5519)


def _require_sha256(value: str, field_name: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA-256 value for {field_name}")


@dataclass(frozen=True)
class FrozenDlRefitReference:
    """Path-free identities for one strictly reloaded DL refit."""

    random_seed: int
    fixed_epoch_count: int
    refit_record: str
    refit_record_sha256: str
    preprocessing_state: str
    preprocessing_state_sha256: str
    preprocessing_state_payload_sha256: str
    model_state: str
    model_state_file_sha256: str
    model_state_payload_sha256: str

    def validate(self) -> None:
        if self.random_seed not in EXPECTED_SEEDS or self.fixed_epoch_count <= 0:
            raise ValueError("Frozen DL reference seed or epoch count is invalid")
        if any(
            not value or Path(value).is_absolute() or ".." in Path(value).parts
            for value in (
                self.refit_record,
                self.preprocessing_state,
                self.model_state,
            )
        ):
            raise ValueError("Frozen DL references must be safe relative paths")
        for field_name in (
            "refit_record_sha256",
            "preprocessing_state_sha256",
            "preprocessing_state_payload_sha256",
            "model_state_file_sha256",
            "model_state_payload_sha256",
        ):
            _require_sha256(str(getattr(self, field_name)), field_name)

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "random_seed": self.random_seed,
            "fixed_epoch_count": self.fixed_epoch_count,
            "refit_record": self.refit_record,
            "refit_record_sha256": self.refit_record_sha256,
            "preprocessing_state": self.preprocessing_state,
            "preprocessing_state_sha256": self.preprocessing_state_sha256,
            "preprocessing_state_payload_sha256": (
                self.preprocessing_state_payload_sha256
            ),
            "model_state": self.model_state,
            "model_state_file_sha256": self.model_state_file_sha256,
            "model_state_payload_sha256": self.model_state_payload_sha256,
        }


@dataclass(frozen=True)
class FrozenRfReference:
    """Path-free identities for the strictly reloaded final RF pipeline."""

    refit_record: str
    refit_record_sha256: str
    fitted_pipeline: str
    fitted_pipeline_sha256: str
    selected_feature_names: tuple[str, ...]
    hyperparameters: dict[str, object]

    def validate(self) -> None:
        if any(
            not value or Path(value).is_absolute() or ".." in Path(value).parts
            for value in (self.refit_record, self.fitted_pipeline)
        ):
            raise ValueError("Frozen RF references must be safe relative paths")
        _require_sha256(self.refit_record_sha256, "rf_refit_record_sha256")
        _require_sha256(self.fitted_pipeline_sha256, "rf_fitted_pipeline_sha256")
        if not self.selected_feature_names or len(self.selected_feature_names) != len(
            set(self.selected_feature_names)
        ):
            raise ValueError("Frozen RF selected features are invalid")
        if set(self.hyperparameters) != {"criterion", "max_depth", "n_estimators"}:
            raise ValueError("Frozen RF hyperparameters are incomplete")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "refit_record": self.refit_record,
            "refit_record_sha256": self.refit_record_sha256,
            "fitted_pipeline": self.fitted_pipeline,
            "fitted_pipeline_sha256": self.fitted_pipeline_sha256,
            "selected_feature_names": list(self.selected_feature_names),
            "hyperparameters": self.hyperparameters,
        }


@dataclass(frozen=True)
class FinalModelFreezeManifest:
    """All final model identities frozen before the single external execution."""

    schema_version: int
    manifest_id: str
    created_at_utc: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    selected_candidate_id: str
    class_labels: tuple[str, ...]
    development_participants: tuple[str, ...]
    holdout_participants: tuple[str, ...]
    final_stage_source_revision: str
    historical_source_revisions: tuple[str, ...]
    ensemble_method: str
    primary_prediction: str
    dl_refits: tuple[FrozenDlRefitReference, ...]
    random_forest: FrozenRfReference
    analysis_settings: dict[str, object]
    input_hashes: dict[str, str]
    manifest_payload_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "class_labels": list(self.class_labels),
            "development_participants": list(self.development_participants),
            "holdout_participants": list(self.holdout_participants),
            "final_stage_source_revision": self.final_stage_source_revision,
            "historical_source_revisions": list(self.historical_source_revisions),
            "ensemble_method": self.ensemble_method,
            "primary_prediction": self.primary_prediction,
            "dl_refits": [value.as_dict() for value in self.dl_refits],
            "random_forest": self.random_forest.as_dict(),
            "analysis_settings": self.analysis_settings,
            "input_hashes": self.input_hashes,
        }

    def validate(self) -> None:
        """Reject incomplete, mutable, or post-hold-out model freezes."""
        if self.schema_version != 1:
            raise ValueError("Unsupported final-model-freeze schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.manifest_id) is None:
            raise ValueError("Final-model-freeze identifier is invalid")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Final-model-freeze time must use second-resolution UTC")
        if self.purpose != "final_model_freeze":
            raise ValueError("Unsupported final-model-freeze purpose")
        if self.scientific_result or self.holdout_accessed:
            raise ValueError("Model freeze cannot be a result or access hold-out data")
        if not self.selected_candidate_id:
            raise ValueError("Final model freeze requires a selected candidate")
        if not self.class_labels or len(self.class_labels) != len(set(self.class_labels)):
            raise ValueError("Final model class labels are invalid")
        development = set(self.development_participants)
        holdout = set(self.holdout_participants)
        if not development or not holdout or development & holdout:
            raise ValueError("Final model cohorts are invalid")
        if not self.final_stage_source_revision.startswith("tree-sha256:"):
            raise ValueError("Final model freeze requires an immutable final source")
        if len(self.historical_source_revisions) != 2 or any(
            not value.startswith("tree-sha256:") for value in self.historical_source_revisions
        ):
            raise ValueError("Final model freeze requires both historical revisions")
        if self.ensemble_method != "arithmetic_mean_of_five_softmax_probability_vectors":
            raise ValueError("Unsupported final DL ensemble method")
        if self.primary_prediction != "argmax_of_mean_probability":
            raise ValueError("Unsupported final DL primary prediction")
        if tuple(value.random_seed for value in self.dl_refits) != EXPECTED_SEEDS:
            raise ValueError("Final model freeze requires all five ordered DL refits")
        for value in self.dl_refits:
            value.validate()
        self.random_forest.validate()
        if self.analysis_settings != {
            "probability_calibration": "none",
            "temporal_smoothing": "none",
            "calibration_bin_count": 15,
            "expected_step_size_samples": 2500,
            "short_run_max_windows": 2,
        }:
            raise ValueError("Final analysis settings differ from the predeclared primary analysis")
        if not self.input_hashes:
            raise ValueError("Final model freeze requires bound inputs")
        for field_name, value in self.input_hashes.items():
            _require_sha256(value, field_name)
        _require_sha256(self.manifest_payload_sha256, "manifest_payload_sha256")
        if sha256_canonical_json(self._payload()) != self.manifest_payload_sha256:
            raise ValueError("Final-model-freeze payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "created_at_utc": self.created_at_utc,
            "purpose": self.purpose,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            **self._payload(),
            "manifest_payload_sha256": self.manifest_payload_sha256,
        }


def _safe_relative(path: Path, base: Path) -> str:
    relative = path.resolve().relative_to(base.resolve())
    if ".." in relative.parts:
        raise ValueError("Final-model artifact lies outside the manifest tree")
    return relative.as_posix()


def _load_preprocessing_state(
    path: Path,
    record: FinalDlRefitRecord,
) -> None:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Final preprocessing state must be a JSON object")
    state = decoded.get("state")
    if (
        decoded.get("status") != "all_development_train_only_preprocessing"
        or decoded.get("holdout_accessed") is not False
        or not isinstance(state, Mapping)
    ):
        raise ValueError("Final preprocessing state has an invalid scope")
    if sha256_canonical_json(state) != str(decoded.get("state_payload_sha256")):
        raise ValueError("Final preprocessing state payload changed")
    if str(decoded["state_payload_sha256"]) != record.preprocessing_state_payload_sha256:
        raise ValueError("Final preprocessing file and refit record disagree")
    mean = np.asarray(state.get("mean"), dtype=np.float64)
    scale = np.asarray(state.get("scale"), dtype=np.float64)
    if (
        mean.shape != (3,)
        or scale.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
        or int(state.get("fit_subject_count", -1)) != len(record.development_participants)
    ):
        raise ValueError("Final preprocessing statistics are invalid")


def build_final_model_freeze_manifest(
    *,
    manifest_id: str,
    created_at_utc: str,
    manifest_output_path: Path | str,
    final_training_settings_path: Path | str,
    dl_refit_directories: Sequence[Path | str],
    rf_refit_directory: Path | str,
    selected_model_configuration_path: Path | str,
    rf_configuration_path: Path | str,
    final_stage_source_manifest_path: Path | str,
    final_stage_source_root: Path | str,
    protocol: ProtocolConfiguration,
    upstream_artifacts: Mapping[str, Path | str],
) -> FinalModelFreezeManifest:
    """Strictly reload every fitted model and bind all pre-hold-out evidence."""
    output_path = Path(manifest_output_path).resolve()
    base = output_path.parent
    settings_path = Path(final_training_settings_path).resolve()
    model_config_path = Path(selected_model_configuration_path).resolve()
    rf_config_path = Path(rf_configuration_path).resolve()
    final_source_path = Path(final_stage_source_manifest_path).resolve()
    settings = load_final_training_settings(settings_path)
    protocol.validate()
    source = load_source_tree_manifest(
        final_source_path,
        root=final_stage_source_root,
        verify_current_tree=True,
    )
    configuration = load_time_series_experiment_configuration(model_config_path)
    if configuration.experiment_id != settings.selected_candidate_id:
        raise ValueError("Final DL configuration differs from frozen settings")
    if settings.development_participants != protocol.development_participants or (
        settings.holdout_participants != protocol.holdout_participants
    ):
        raise ValueError("Final settings and protocol cohorts differ")

    refit_references: list[FrozenDlRefitReference] = []
    observed_records: list[FinalDlRefitRecord] = []
    for directory_value in dl_refit_directories:
        directory = Path(directory_value).resolve()
        entry_path = directory / "refit_entry.json"
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
        if not isinstance(entry, Mapping) or entry.get("holdout_accessed") is not False:
            raise ValueError("Final DL refit entry has an invalid scope")
        record_path = directory / str(entry["refit_record"])
        preprocessing_path = directory / str(entry["preprocessing_state"])
        state_path = directory / str(entry["model_state"])
        if sha256_file(record_path) != str(entry["refit_record_sha256"]):
            raise ValueError("Final DL refit-record hash changed")
        if sha256_file(preprocessing_path) != str(entry["preprocessing_state_sha256"]):
            raise ValueError("Final DL preprocessing-file hash changed")
        if sha256_file(state_path) != str(entry["model_state_file_sha256"]):
            raise ValueError("Final DL model-state-file hash changed")
        record = load_final_dl_refit_record(record_path)
        if record.final_stage_source_revision != source["source_revision"]:
            raise ValueError("Final DL refit uses a different final-stage source revision")
        if record.input_hashes["final_training_settings_sha256"] != sha256_file(
            settings_path
        ):
            raise ValueError("Final DL refit does not bind the frozen settings")
        if record.input_hashes["final_stage_source_manifest_sha256"] != sha256_file(
            final_source_path
        ):
            raise ValueError("Final DL refit does not bind the final source manifest")
        _load_preprocessing_state(preprocessing_path, record)
        model = build_time_series_classifier(configuration)
        load_model_state_npz(
            state_path,
            model,
            expected_payload_sha256=record.model_state_payload_sha256,
        )
        refit_references.append(
            FrozenDlRefitReference(
                random_seed=record.random_seed,
                fixed_epoch_count=record.fixed_epoch_count,
                refit_record=_safe_relative(record_path, base),
                refit_record_sha256=sha256_file(record_path),
                preprocessing_state=_safe_relative(preprocessing_path, base),
                preprocessing_state_sha256=sha256_file(preprocessing_path),
                preprocessing_state_payload_sha256=(
                    record.preprocessing_state_payload_sha256
                ),
                model_state=_safe_relative(state_path, base),
                model_state_file_sha256=sha256_file(state_path),
                model_state_payload_sha256=record.model_state_payload_sha256,
            )
        )
        observed_records.append(record)
    refit_references = sorted(refit_references, key=lambda value: value.random_seed)
    observed_records = sorted(observed_records, key=lambda value: value.random_seed)
    if tuple(value.random_seed for value in refit_references) != EXPECTED_SEEDS:
        raise ValueError("Final DL refit set is incomplete or duplicated")
    if any(
        record.development_participants != settings.development_participants
        or record.sealed_holdout_participants != settings.holdout_participants
        for record in observed_records
    ):
        raise ValueError("Final DL refit cohorts differ from frozen settings")

    rf_directory = Path(rf_refit_directory).resolve()
    rf_entry = json.loads((rf_directory / "refit_entry.json").read_text(encoding="utf-8"))
    if not isinstance(rf_entry, Mapping) or rf_entry.get("holdout_accessed") is not False:
        raise ValueError("Final RF entry has an invalid scope")
    rf_record_path = rf_directory / str(rf_entry["refit_record"])
    pipeline_path = rf_directory / str(rf_entry["fitted_pipeline"])
    if sha256_file(rf_record_path) != str(rf_entry["refit_record_sha256"]):
        raise ValueError("Final RF record hash changed")
    if sha256_file(pipeline_path) != str(rf_entry["fitted_pipeline_sha256"]):
        raise ValueError("Final RF pipeline hash changed")
    rf_record = load_final_random_forest_record(rf_record_path)
    if rf_record.final_stage_source_revision != source["source_revision"]:
        raise ValueError("Final RF uses a different final-stage source revision")
    pipeline = load_random_forest_pipeline(pipeline_path)
    rf_configuration = load_random_forest_reconstruction_configuration(rf_config_path)
    selected_names = _selected_feature_names(
        pipeline,
        tuple(
            json.loads(
                Path(upstream_artifacts["feature_manifest"]).read_text(encoding="utf-8")
            )["feature_names"]
        ),
    )
    if selected_names != rf_record.selected_feature_names:
        raise ValueError("Reloaded final RF selected features changed")
    rf_reference = FrozenRfReference(
        refit_record=_safe_relative(rf_record_path, base),
        refit_record_sha256=sha256_file(rf_record_path),
        fitted_pipeline=_safe_relative(pipeline_path, base),
        fitted_pipeline_sha256=sha256_file(pipeline_path),
        selected_feature_names=selected_names,
        hyperparameters=dict(rf_record.hyperparameters),
    )
    if len(selected_names) != rf_configuration.selected_feature_count:
        raise ValueError("Final RF selected-feature count changed")

    input_hashes = {
        "final_training_settings_sha256": sha256_file(settings_path),
        "selected_model_configuration_sha256": sha256_file(model_config_path),
        "rf_configuration_sha256": sha256_file(rf_config_path),
        "final_stage_source_manifest_sha256": sha256_file(final_source_path),
        **{
            str(name): sha256_file(path)
            for name, path in sorted(upstream_artifacts.items())
        },
    }
    payload = {
        "selected_candidate_id": settings.selected_candidate_id,
        "class_labels": list(settings.class_labels),
        "development_participants": list(settings.development_participants),
        "holdout_participants": list(settings.holdout_participants),
        "final_stage_source_revision": str(source["source_revision"]),
        "historical_source_revisions": [
            settings.development_source_revision,
            settings.rf_development_source_revision,
        ],
        "ensemble_method": "arithmetic_mean_of_five_softmax_probability_vectors",
        "primary_prediction": "argmax_of_mean_probability",
        "dl_refits": [value.as_dict() for value in refit_references],
        "random_forest": rf_reference.as_dict(),
        "analysis_settings": {
            "probability_calibration": "none",
            "temporal_smoothing": "none",
            "calibration_bin_count": 15,
            "expected_step_size_samples": 2500,
            "short_run_max_windows": 2,
        },
        "input_hashes": input_hashes,
    }
    manifest = FinalModelFreezeManifest(
        schema_version=1,
        manifest_id=manifest_id,
        created_at_utc=created_at_utc,
        purpose="final_model_freeze",
        scientific_result=False,
        holdout_accessed=False,
        selected_candidate_id=settings.selected_candidate_id,
        class_labels=settings.class_labels,
        development_participants=settings.development_participants,
        holdout_participants=settings.holdout_participants,
        final_stage_source_revision=str(source["source_revision"]),
        historical_source_revisions=(
            settings.development_source_revision,
            settings.rf_development_source_revision,
        ),
        ensemble_method=str(payload["ensemble_method"]),
        primary_prediction=str(payload["primary_prediction"]),
        dl_refits=tuple(refit_references),
        random_forest=rf_reference,
        analysis_settings=dict(payload["analysis_settings"]),
        input_hashes=input_hashes,
        manifest_payload_sha256=sha256_canonical_json(payload),
    )
    manifest.validate()
    return manifest


def write_final_model_freeze_manifest(
    path: Path | str,
    manifest: FinalModelFreezeManifest,
) -> None:
    """Write the final model freeze exclusively."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_final_model_freeze_manifest(path: Path | str) -> FinalModelFreezeManifest:
    """Load and validate a final-model-freeze manifest."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Final-model-freeze manifest must be a JSON object")
    dl_refits = tuple(
        FrozenDlRefitReference(
            random_seed=int(value["random_seed"]),
            fixed_epoch_count=int(value["fixed_epoch_count"]),
            refit_record=str(value["refit_record"]),
            refit_record_sha256=str(value["refit_record_sha256"]),
            preprocessing_state=str(value["preprocessing_state"]),
            preprocessing_state_sha256=str(value["preprocessing_state_sha256"]),
            preprocessing_state_payload_sha256=str(
                value["preprocessing_state_payload_sha256"]
            ),
            model_state=str(value["model_state"]),
            model_state_file_sha256=str(value["model_state_file_sha256"]),
            model_state_payload_sha256=str(value["model_state_payload_sha256"]),
        )
        for value in decoded["dl_refits"]
    )
    rf_value = decoded["random_forest"]
    manifest = FinalModelFreezeManifest(
        schema_version=int(decoded["schema_version"]),
        manifest_id=str(decoded["manifest_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        selected_candidate_id=str(decoded["selected_candidate_id"]),
        class_labels=tuple(map(str, decoded["class_labels"])),
        development_participants=tuple(map(str, decoded["development_participants"])),
        holdout_participants=tuple(map(str, decoded["holdout_participants"])),
        final_stage_source_revision=str(decoded["final_stage_source_revision"]),
        historical_source_revisions=tuple(
            map(str, decoded["historical_source_revisions"])
        ),
        ensemble_method=str(decoded["ensemble_method"]),
        primary_prediction=str(decoded["primary_prediction"]),
        dl_refits=dl_refits,
        random_forest=FrozenRfReference(
            refit_record=str(rf_value["refit_record"]),
            refit_record_sha256=str(rf_value["refit_record_sha256"]),
            fitted_pipeline=str(rf_value["fitted_pipeline"]),
            fitted_pipeline_sha256=str(rf_value["fitted_pipeline_sha256"]),
            selected_feature_names=tuple(map(str, rf_value["selected_feature_names"])),
            hyperparameters=dict(rf_value["hyperparameters"]),
        ),
        analysis_settings=dict(decoded["analysis_settings"]),
        input_hashes={str(key): str(value) for key, value in decoded["input_hashes"].items()},
        manifest_payload_sha256=str(decoded["manifest_payload_sha256"]),
    )
    manifest.validate()
    return manifest


def verify_final_model_freeze_files(path: Path | str) -> FinalModelFreezeManifest:
    """Verify every relative model artifact hash without loading participant values."""
    manifest_path = Path(path).resolve()
    manifest = load_final_model_freeze_manifest(manifest_path)
    base = manifest_path.parent
    for refit in manifest.dl_refits:
        for relative, expected in (
            (refit.refit_record, refit.refit_record_sha256),
            (refit.preprocessing_state, refit.preprocessing_state_sha256),
            (refit.model_state, refit.model_state_file_sha256),
        ):
            if sha256_file(base / relative) != expected:
                raise ValueError("Frozen DL artifact hash changed")
    if sha256_file(base / manifest.random_forest.refit_record) != (
        manifest.random_forest.refit_record_sha256
    ):
        raise ValueError("Frozen RF record hash changed")
    if sha256_file(base / manifest.random_forest.fitted_pipeline) != (
        manifest.random_forest.fitted_pipeline_sha256
    ):
        raise ValueError("Frozen RF pipeline hash changed")
    return manifest


__all__ = [
    "FinalModelFreezeManifest",
    "FrozenDlRefitReference",
    "FrozenRfReference",
    "build_final_model_freeze_manifest",
    "load_final_model_freeze_manifest",
    "verify_final_model_freeze_files",
    "write_final_model_freeze_manifest",
]
