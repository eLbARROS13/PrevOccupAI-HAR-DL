"""Deterministic derived analyses from immutable development predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .calibration import evaluate_calibration, probabilities_from_logits
from .evaluation import evaluate_predictions
from .prediction_artifacts import load_prediction_artifact_record
from .provenance import sha256_canonical_json, sha256_file
from .temporal_evaluation import evaluate_temporal_predictions
from .training import TrainingPurpose


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ANALYSIS_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class PredictionAnalysisRecord:
    """Aggregate and participant analyses bound to one prediction artifact."""

    schema_version: int
    analysis_id: str
    created_at_utc: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    source_revision: str
    prediction_run_id: str
    prediction_artifact_sha256: str
    model_state_sha256: str
    class_labels: tuple[str, ...]
    participant_count: int
    window_count: int
    analysis_settings: dict[str, object]
    analysis_payload: dict[str, object]
    analysis_payload_sha256: str

    def validate(self) -> None:
        """Validate provenance, settings, internal counts, and payload integrity."""
        if self.schema_version != 1:
            raise ValueError("Unsupported prediction-analysis schema version")
        if ANALYSIS_ID_PATTERN.fullmatch(self.analysis_id) is None:
            raise ValueError("Analysis identifier contains unsupported characters")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Creation time must use second-resolution UTC with a Z suffix")
        if self.holdout_accessed:
            raise ValueError("Development analysis records cannot claim hold-out access")
        if not self.source_revision or not self.prediction_run_id:
            raise ValueError("Source revision and prediction run are required")
        for field_name in (
            "prediction_artifact_sha256",
            "model_state_sha256",
            "analysis_payload_sha256",
        ):
            if SHA256_PATTERN.fullmatch(str(getattr(self, field_name))) is None:
                raise ValueError(f"Invalid SHA-256 value for {field_name}")
        if (
            not self.class_labels
            or len(self.class_labels) != len(set(self.class_labels))
        ):
            raise ValueError("Class labels must be non-empty and unique")
        if self.participant_count <= 0 or self.window_count <= 0:
            raise ValueError("Participant and window counts must be positive")
        if self.purpose == TrainingPurpose.SYNTHETIC_VALIDATION.value:
            if self.scientific_result:
                raise ValueError("Synthetic analysis cannot be a scientific result")
        elif self.purpose == TrainingPurpose.DEVELOPMENT_SELECTION.value:
            if not self.scientific_result:
                raise ValueError("Development analysis is a scientific development result")
        else:
            raise ValueError(f"Unsupported analysis purpose: {self.purpose}")

        required_settings = {
            "probability_transform",
            "calibration_bin_count",
            "expected_step_size_samples",
            "short_run_max_windows",
        }
        if set(self.analysis_settings) != required_settings:
            raise ValueError("Prediction-analysis settings are incomplete or unexpected")
        probability_transform = self.analysis_settings["probability_transform"]
        if probability_transform != {
            "method": "softmax",
            "temperature": 1.0,
            "fitted": False,
        }:
            raise ValueError("Derived analysis must use uncalibrated softmax probabilities")
        if int(self.analysis_settings["calibration_bin_count"]) < 2:
            raise ValueError("At least two calibration bins are required")
        if int(self.analysis_settings["expected_step_size_samples"]) <= 0 or int(
            self.analysis_settings["short_run_max_windows"]
        ) <= 0:
            raise ValueError("Temporal-analysis settings must be positive")

        required_payloads = {"classification", "calibration", "temporal"}
        if set(self.analysis_payload) != required_payloads:
            raise ValueError("Prediction-analysis payload is incomplete or unexpected")
        classification = self.analysis_payload["classification"]
        calibration = self.analysis_payload["calibration"]
        temporal = self.analysis_payload["temporal"]
        if not all(isinstance(value, Mapping) for value in (classification, calibration, temporal)):
            raise TypeError("Derived analysis payloads must be mappings")
        if int(classification["overall_metrics"]["support"]) != self.window_count:
            raise ValueError("Classification support disagrees with the window count")
        if int(calibration["sample_count"]) != self.window_count:
            raise ValueError("Calibration sample count disagrees with the window count")
        if int(temporal["overall_metrics"]["window_count"]) != self.window_count:
            raise ValueError("Temporal window count disagrees with the source artifact")
        if len(classification["per_participant_metrics"]) != self.participant_count:
            raise ValueError("Participant metric count disagrees with the source artifact")
        observed_payload_hash = sha256_canonical_json(self.analysis_payload)
        if observed_payload_hash != self.analysis_payload_sha256:
            raise ValueError("Prediction-analysis payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "created_at_utc": self.created_at_utc,
            "purpose": self.purpose,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            "source_revision": self.source_revision,
            "prediction_run_id": self.prediction_run_id,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "model_state_sha256": self.model_state_sha256,
            "class_labels": list(self.class_labels),
            "participant_count": self.participant_count,
            "window_count": self.window_count,
            "analysis_settings": self.analysis_settings,
            "analysis_payload": self.analysis_payload,
            "analysis_payload_sha256": self.analysis_payload_sha256,
        }


def build_prediction_analysis_record(
    *,
    analysis_id: str,
    created_at_utc: str,
    prediction_artifact_path: Path | str,
    calibration_bin_count: int = 15,
    expected_step_size_samples: int,
    short_run_max_windows: int = 1,
) -> PredictionAnalysisRecord:
    """Regenerate deterministic development analyses from retained logits."""
    artifact = load_prediction_artifact_record(prediction_artifact_path)
    metadata = artifact.window_metadata()
    participant_ids = tuple(record.subject_id for record in metadata)
    classification = evaluate_predictions(
        artifact.true_labels,
        artifact.predicted_labels,
        participant_ids,
        artifact.class_labels,
    )
    probabilities = probabilities_from_logits(
        artifact.logits_array(),
        expected_class_count=len(artifact.class_labels),
        temperature=1.0,
    )
    calibration = evaluate_calibration(
        probabilities,
        artifact.true_labels,
        artifact.class_labels,
        bin_count=calibration_bin_count,
    )
    temporal = evaluate_temporal_predictions(
        artifact.predicted_labels,
        metadata,
        artifact.class_labels,
        expected_step_size_samples=expected_step_size_samples,
        short_run_max_windows=short_run_max_windows,
    )
    analysis_settings: dict[str, object] = {
        "probability_transform": {
            "method": "softmax",
            "temperature": 1.0,
            "fitted": False,
        },
        "calibration_bin_count": calibration_bin_count,
        "expected_step_size_samples": expected_step_size_samples,
        "short_run_max_windows": short_run_max_windows,
    }
    analysis_payload: dict[str, object] = {
        "classification": classification.as_dict(),
        "calibration": calibration.as_dict(),
        "temporal": temporal.as_dict(),
    }
    record = PredictionAnalysisRecord(
        schema_version=1,
        analysis_id=analysis_id,
        created_at_utc=created_at_utc,
        purpose=artifact.purpose,
        scientific_result=artifact.scientific_result,
        holdout_accessed=False,
        source_revision=artifact.source_revision,
        prediction_run_id=artifact.run_id,
        prediction_artifact_sha256=sha256_file(prediction_artifact_path),
        model_state_sha256=artifact.model_state_sha256,
        class_labels=artifact.class_labels,
        participant_count=len(set(participant_ids)),
        window_count=artifact.window_count,
        analysis_settings=analysis_settings,
        analysis_payload=analysis_payload,
        analysis_payload_sha256=sha256_canonical_json(analysis_payload),
    )
    record.validate()
    return record


def write_prediction_analysis_record(
    path: Path | str,
    record: PredictionAnalysisRecord,
) -> None:
    """Write a new derived-analysis record exclusively."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(
            record.as_dict(),
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


def load_prediction_analysis_record(path: Path | str) -> PredictionAnalysisRecord:
    """Load and validate a derived prediction-analysis record."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Prediction analysis must be a JSON object")
    record = PredictionAnalysisRecord(
        schema_version=int(decoded["schema_version"]),
        analysis_id=str(decoded["analysis_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        source_revision=str(decoded["source_revision"]),
        prediction_run_id=str(decoded["prediction_run_id"]),
        prediction_artifact_sha256=str(decoded["prediction_artifact_sha256"]),
        model_state_sha256=str(decoded["model_state_sha256"]),
        class_labels=tuple(map(str, decoded["class_labels"])),
        participant_count=int(decoded["participant_count"]),
        window_count=int(decoded["window_count"]),
        analysis_settings=dict(decoded["analysis_settings"]),
        analysis_payload=dict(decoded["analysis_payload"]),
        analysis_payload_sha256=str(decoded["analysis_payload_sha256"]),
    )
    record.validate()
    return record
