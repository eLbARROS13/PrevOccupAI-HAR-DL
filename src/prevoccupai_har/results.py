"""Immutable, provenance-rich records for development-only training outcomes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as error:  # pragma: no cover - exercised only without the DL extra
    raise ImportError(
        "PyTorch is required for prevoccupai_har.results; install the 'dl' extra"
    ) from error

from .protocol import ProtocolConfiguration
from .provenance import (
    is_reproducible_source_revision,
    sha256_canonical_json,
    sha256_file,
)
from .training import TrainingOutcome, TrainingPurpose, TrainingRunScope


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class ScientificDataProvenance:
    """Required governed-artifact hashes for a future scientific training run."""

    raw_recording_manifest_sha256: str
    segmentation_manifest_sha256: str
    quality_manifest_sha256: str
    split_manifest_sha256: str
    window_store_index_sha256: str
    signal_preprocessing_configuration_sha256: str
    segmentation_contract_configuration_sha256: str

    def validate(self) -> None:
        """Require lowercase SHA-256 values for every scientific input contract."""
        for field_name, value in asdict(self).items():
            if SHA256_PATTERN.fullmatch(str(value)) is None:
                raise ValueError(f"Invalid SHA-256 value for {field_name}")


@dataclass(frozen=True)
class TrainingResultRecord:
    """JSON-serialisable training record that never implies hold-out evaluation."""

    schema_version: int
    run_id: str
    created_at_utc: str
    experiment_id: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    source_revision: str
    model_configuration_sha256: str
    learned_preprocessing_sha256: str
    protocol_configuration_sha256: str | None
    data_provenance: ScientificDataProvenance | None
    random_seed: int
    training_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    trainable_parameter_count: int
    best_epoch: int
    stopped_early: bool
    history: tuple[dict[str, object], ...]
    software_versions: dict[str, str]

    def validate(self) -> None:
        """Reject ambiguous or scientifically under-provenanced records."""
        if self.schema_version != 1:
            raise ValueError("Unsupported training-result schema version")
        if RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("Run identifier contains unsupported characters")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Creation time must use second-resolution UTC with a Z suffix")
        if not self.experiment_id or not self.source_revision:
            raise ValueError("Experiment identifier and source revision are required")
        if SHA256_PATTERN.fullmatch(self.model_configuration_sha256) is None:
            raise ValueError("Model configuration digest is not a SHA-256 value")
        if SHA256_PATTERN.fullmatch(self.learned_preprocessing_sha256) is None:
            raise ValueError("Learned-preprocessing digest is not a SHA-256 value")
        if self.holdout_accessed:
            raise ValueError("Development training records cannot claim hold-out access")
        if self.random_seed < 0 or self.trainable_parameter_count <= 0:
            raise ValueError("Seed and trainable parameter count are invalid")
        if self.best_epoch < 1 or not self.history:
            raise ValueError("A validation-selected epoch history is required")
        if self.best_epoch not in {int(entry["epoch"]) for entry in self.history}:
            raise ValueError("Best epoch is absent from the training history")

        if self.purpose == TrainingPurpose.SYNTHETIC_VALIDATION.value:
            if self.scientific_result:
                raise ValueError("Synthetic validation cannot be a scientific result")
            if self.protocol_configuration_sha256 is not None or self.data_provenance is not None:
                raise ValueError("Synthetic records must not imply scientific data provenance")
        elif self.purpose == TrainingPurpose.DEVELOPMENT_SELECTION.value:
            if not self.scientific_result:
                raise ValueError("Authorized development selection is a scientific result")
            if self.protocol_configuration_sha256 is None or self.data_provenance is None:
                raise ValueError("Scientific development records require complete provenance")
            if SHA256_PATTERN.fullmatch(self.protocol_configuration_sha256) is None:
                raise ValueError("Protocol configuration digest is not a SHA-256 value")
            if not is_reproducible_source_revision(self.source_revision):
                raise ValueError(
                    "Scientific development records require an immutable source revision"
                )
            self.data_provenance.validate()
        else:
            raise ValueError(f"Unsupported training-result purpose: {self.purpose}")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        value = asdict(self)
        value["training_subjects"] = list(self.training_subjects)
        value["validation_subjects"] = list(self.validation_subjects)
        value["history"] = list(self.history)
        return value


def build_training_result_record(
    *,
    run_id: str,
    created_at_utc: str,
    experiment_id: str,
    source_revision: str,
    model_configuration_path: Path | str,
    model_trainable_parameter_count: int,
    learned_preprocessing_state: Mapping[str, object],
    scope: TrainingRunScope,
    outcome: TrainingOutcome,
    protocol: ProtocolConfiguration | None = None,
    protocol_configuration_path: Path | str | None = None,
    data_provenance: ScientificDataProvenance | None = None,
) -> TrainingResultRecord:
    """Build a record whose governance status follows the validated run scope."""
    scope.validate(protocol)
    scientific_result = scope.purpose is TrainingPurpose.DEVELOPMENT_SELECTION
    if scientific_result and protocol_configuration_path is None:
        raise ValueError("Scientific development records require the protocol file")
    if not scientific_result and protocol_configuration_path is not None:
        raise ValueError("Synthetic records must not receive a protocol file")
    history = tuple(asdict(entry) for entry in outcome.history)
    record = TrainingResultRecord(
        schema_version=1,
        run_id=run_id,
        created_at_utc=created_at_utc,
        experiment_id=experiment_id,
        purpose=scope.purpose.value,
        scientific_result=scientific_result,
        holdout_accessed=False,
        source_revision=source_revision,
        model_configuration_sha256=sha256_file(model_configuration_path),
        learned_preprocessing_sha256=sha256_canonical_json(
            learned_preprocessing_state
        ),
        protocol_configuration_sha256=(
            sha256_file(protocol_configuration_path)
            if protocol_configuration_path is not None
            else None
        ),
        data_provenance=data_provenance,
        random_seed=outcome.seed,
        training_subjects=scope.training_subjects,
        validation_subjects=scope.validation_subjects,
        trainable_parameter_count=model_trainable_parameter_count,
        best_epoch=outcome.best_epoch,
        stopped_early=outcome.stopped_early,
        history=history,
        software_versions={
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    )
    record.validate()
    return record


def write_training_result_record(
    path: Path | str,
    record: TrainingResultRecord,
) -> None:
    """Write a new result record exclusively; existing records are never overwritten."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(record.as_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_training_result_record(path: Path | str) -> TrainingResultRecord:
    """Load and validate an immutable development-only training record."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Training result must be a JSON object")
    provenance_value = decoded.get("data_provenance")
    if provenance_value is None:
        provenance = None
    elif isinstance(provenance_value, Mapping):
        provenance = ScientificDataProvenance(
            raw_recording_manifest_sha256=str(
                provenance_value["raw_recording_manifest_sha256"]
            ),
            segmentation_manifest_sha256=str(
                provenance_value["segmentation_manifest_sha256"]
            ),
            quality_manifest_sha256=str(
                provenance_value["quality_manifest_sha256"]
            ),
            split_manifest_sha256=str(
                provenance_value["split_manifest_sha256"]
            ),
            window_store_index_sha256=str(
                provenance_value["window_store_index_sha256"]
            ),
            signal_preprocessing_configuration_sha256=str(
                provenance_value["signal_preprocessing_configuration_sha256"]
            ),
            segmentation_contract_configuration_sha256=str(
                provenance_value["segmentation_contract_configuration_sha256"]
            ),
        )
    else:
        raise TypeError("Training-result data provenance must be an object or null")
    history_value = decoded.get("history")
    if not isinstance(history_value, list) or any(
        not isinstance(entry, Mapping) for entry in history_value
    ):
        raise TypeError("Training-result history must be an array of objects")
    software_versions = decoded.get("software_versions")
    if not isinstance(software_versions, Mapping):
        raise TypeError("Training-result software versions must be an object")
    record = TrainingResultRecord(
        schema_version=int(decoded["schema_version"]),
        run_id=str(decoded["run_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        experiment_id=str(decoded["experiment_id"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        source_revision=str(decoded["source_revision"]),
        model_configuration_sha256=str(decoded["model_configuration_sha256"]),
        learned_preprocessing_sha256=str(
            decoded["learned_preprocessing_sha256"]
        ),
        protocol_configuration_sha256=(
            None
            if decoded.get("protocol_configuration_sha256") is None
            else str(decoded["protocol_configuration_sha256"])
        ),
        data_provenance=provenance,
        random_seed=int(decoded["random_seed"]),
        training_subjects=tuple(map(str, decoded["training_subjects"])),
        validation_subjects=tuple(map(str, decoded["validation_subjects"])),
        trainable_parameter_count=int(decoded["trainable_parameter_count"]),
        best_epoch=int(decoded["best_epoch"]),
        stopped_early=decoded.get("stopped_early") is True,
        history=tuple(dict(entry) for entry in history_value),
        software_versions={
            str(key): str(value) for key, value in software_versions.items()
        },
    )
    record.validate()
    return record
