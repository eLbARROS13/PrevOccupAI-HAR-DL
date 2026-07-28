"""Immutable development-prediction artifacts bound to training and model state."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised only without the DL extra
    raise ImportError(
        "PyTorch is required for prevoccupai_har.prediction_artifacts; "
        "install the 'dl' extra"
    ) from error

from .evaluation import confusion_matrix_from_predictions
from .protocol import ProtocolConfiguration
from .provenance import (
    is_reproducible_source_revision,
    sha256_canonical_json,
    sha256_file,
)
from .training import TrainingPurpose, TrainingRunScope
from .windowing import WindowMetadata


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class PredictionWindowRecord:
    """One ordered logit row with pseudonymous, path-free window provenance."""

    window_index: int
    participant_id: str
    recording_key_sha256: str
    sensor_stream_key_sha256: str
    main_label: str
    sub_activity_label: str
    sensor_side: str
    start_sample: int
    end_sample_exclusive: int
    preprocessing_status: str
    quality_status: str
    predicted_label: str
    logits: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["logits"] = list(self.logits)
        return value


@dataclass(frozen=True)
class PredictionArtifactRecord:
    """Validation predictions that cannot imply external hold-out access."""

    schema_version: int
    run_id: str
    created_at_utc: str
    experiment_id: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    source_revision: str
    training_run_id: str
    training_result_sha256: str
    model_configuration_sha256: str
    learned_preprocessing_sha256: str
    model_state_sha256: str
    class_labels: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    logit_dtype: str
    window_count: int
    prediction_payload_sha256: str
    windows: tuple[PredictionWindowRecord, ...]

    def validate(self) -> None:
        """Reject ambiguous, mutable, or hold-out-like prediction records."""
        if self.schema_version != 1:
            raise ValueError("Unsupported prediction-artifact schema version")
        if RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("Prediction run identifier contains unsupported characters")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Creation time must use second-resolution UTC with a Z suffix")
        if not self.experiment_id or not self.training_run_id or not self.source_revision:
            raise ValueError("Experiment, training-run, and source identifiers are required")
        if self.holdout_accessed:
            raise ValueError("Development prediction artifacts cannot claim hold-out access")
        if self.logit_dtype != "float32":
            raise ValueError("Prediction artifacts must store float32 logits")
        for field_name in (
            "training_result_sha256",
            "model_configuration_sha256",
            "learned_preprocessing_sha256",
            "model_state_sha256",
            "prediction_payload_sha256",
        ):
            if SHA256_PATTERN.fullmatch(str(getattr(self, field_name))) is None:
                raise ValueError(f"Invalid SHA-256 value for {field_name}")
        if (
            not self.class_labels
            or len(self.class_labels) != len(set(self.class_labels))
            or any(not label for label in self.class_labels)
        ):
            raise ValueError("Class labels must be non-empty and unique")
        if (
            not self.validation_subjects
            or len(self.validation_subjects) != len(set(self.validation_subjects))
        ):
            raise ValueError("Validation subjects must be non-empty and unique")
        if self.purpose == TrainingPurpose.SYNTHETIC_VALIDATION.value:
            if self.scientific_result:
                raise ValueError("Synthetic predictions cannot be a scientific result")
            if any(
                not participant.startswith("SYNTHETIC_")
                for participant in self.validation_subjects
            ):
                raise ValueError("Synthetic prediction records require synthetic subjects")
        elif self.purpose == TrainingPurpose.DEVELOPMENT_SELECTION.value:
            if not self.scientific_result:
                raise ValueError("Development predictions are scientific development results")
            if not is_reproducible_source_revision(self.source_revision):
                raise ValueError(
                    "Scientific prediction records require an immutable source revision"
                )
        else:
            raise ValueError(f"Unsupported prediction-artifact purpose: {self.purpose}")
        if self.window_count != len(self.windows) or self.window_count <= 0:
            raise ValueError("Window count must match a non-empty prediction payload")
        if tuple(row.window_index for row in self.windows) != tuple(
            range(self.window_count)
        ):
            raise ValueError("Prediction window indices must be contiguous and ordered")
        observed_subjects = {row.participant_id for row in self.windows}
        if observed_subjects != set(self.validation_subjects):
            raise ValueError("Prediction participants must exactly match validation scope")
        window_sizes: set[int] = set()
        for row in self.windows:
            if not all(
                (
                    row.participant_id,
                    row.main_label,
                    row.sub_activity_label,
                    row.sensor_side,
                    row.preprocessing_status,
                    row.quality_status,
                )
            ):
                raise ValueError("Prediction window fields cannot be empty")
            if SHA256_PATTERN.fullmatch(row.recording_key_sha256) is None or (
                SHA256_PATTERN.fullmatch(row.sensor_stream_key_sha256) is None
            ):
                raise ValueError("Prediction sequence keys must be SHA-256 values")
            if row.start_sample < 0 or row.end_sample_exclusive <= row.start_sample:
                raise ValueError("Prediction window bounds are invalid")
            window_sizes.add(row.end_sample_exclusive - row.start_sample)
            if row.main_label not in self.class_labels:
                raise ValueError("Prediction reference label is undeclared")
            if len(row.logits) != len(self.class_labels) or not np.isfinite(
                np.asarray(row.logits, dtype=np.float64)
            ).all():
                raise ValueError("Every prediction row requires one finite logit per class")
            predicted_index = int(np.argmax(np.asarray(row.logits, dtype=np.float32)))
            if row.predicted_label != self.class_labels[predicted_index]:
                raise ValueError("Stored predicted label disagrees with the logit argmax")
        if len(window_sizes) != 1:
            raise ValueError("All prediction windows must have the same sample length")
        observed_payload_hash = sha256_canonical_json(
            [row.as_dict() for row in self.windows]
        )
        if observed_payload_hash != self.prediction_payload_sha256:
            raise ValueError("Prediction payload digest does not match its rows")

    @property
    def true_labels(self) -> tuple[str, ...]:
        """Return the ordered reference-label vector."""
        return tuple(row.main_label for row in self.windows)

    @property
    def predicted_labels(self) -> tuple[str, ...]:
        """Return the ordered predicted-label vector."""
        return tuple(row.predicted_label for row in self.windows)

    def logits_array(self) -> NDArray[np.float32]:
        """Return an immutable ordered float32 logit matrix."""
        values = np.asarray([row.logits for row in self.windows], dtype=np.float32)
        values.setflags(write=False)
        return values

    def window_metadata(self) -> tuple[WindowMetadata, ...]:
        """Reconstruct path-free metadata suitable for governed downstream metrics."""
        return tuple(
            WindowMetadata(
                subject_id=row.participant_id,
                recording_id=row.recording_key_sha256,
                main_label=row.main_label,
                sub_activity_label=row.sub_activity_label,
                sensor_stream_id=row.sensor_stream_key_sha256,
                sensor_side=row.sensor_side,
                start_sample=row.start_sample,
                end_sample_exclusive=row.end_sample_exclusive,
                preprocessing_status=row.preprocessing_status,
                quality_status=row.quality_status,
            )
            for row in self.windows
        )

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "experiment_id": self.experiment_id,
            "purpose": self.purpose,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            "source_revision": self.source_revision,
            "training_run_id": self.training_run_id,
            "training_result_sha256": self.training_result_sha256,
            "model_configuration_sha256": self.model_configuration_sha256,
            "learned_preprocessing_sha256": self.learned_preprocessing_sha256,
            "model_state_sha256": self.model_state_sha256,
            "class_labels": list(self.class_labels),
            "validation_subjects": list(self.validation_subjects),
            "logit_dtype": self.logit_dtype,
            "window_count": self.window_count,
            "prediction_payload_sha256": self.prediction_payload_sha256,
            "windows": [row.as_dict() for row in self.windows],
        }


def sha256_model_state(model: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a PyTorch state dictionary."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError("Model state entries must be tensors")
        contiguous = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        payload = (
            contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
        digest.update(len(metadata).to_bytes(8, byteorder="big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def predict_logits(
    model: nn.Module,
    inputs: Tensor | NDArray[np.floating],
    *,
    expected_output_classes: int,
    batch_size: int,
    device: str = "cpu",
) -> NDArray[np.float32]:
    """Run ordered, no-gradient inference and return immutable float32 logits."""
    if expected_output_classes < 2 or batch_size <= 0:
        raise ValueError("Output-class count and batch size must be valid")
    if isinstance(inputs, Tensor):
        input_tensor = inputs.detach().to(dtype=torch.float32).clone()
    else:
        input_tensor = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    if input_tensor.ndim != 3 or input_tensor.shape[0] == 0:
        raise ValueError("Inputs must have non-empty (windows, channels, samples) shape")
    if not torch.isfinite(input_tensor).all():
        raise ValueError("Prediction inputs must be finite")

    torch_device = torch.device(device)
    model.to(torch_device)
    was_training = model.training
    model.eval()
    outputs: list[Tensor] = []
    try:
        with torch.inference_mode():
            for start in range(0, input_tensor.shape[0], batch_size):
                batch = input_tensor[start : start + batch_size].to(torch_device)
                logits = model(batch)
                if logits.ndim != 2 or logits.shape != (
                    batch.shape[0],
                    expected_output_classes,
                ):
                    raise ValueError("Model logits disagree with the declared class count")
                if not torch.isfinite(logits).all():
                    raise ValueError("Model logits must be finite")
                outputs.append(logits.detach().to(device="cpu", dtype=torch.float32))
    finally:
        model.train(was_training)
    values = torch.cat(outputs, dim=0).numpy().copy()
    values.setflags(write=False)
    return values


def _training_result_binding(
    path: Path | str,
    scope: TrainingRunScope,
) -> Mapping[str, object]:
    training_path = Path(path)
    decoded = json.loads(training_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Training result must be a JSON object")
    required_fields = (
        "run_id",
        "experiment_id",
        "purpose",
        "scientific_result",
        "holdout_accessed",
        "source_revision",
        "model_configuration_sha256",
        "learned_preprocessing_sha256",
        "validation_subjects",
    )
    if any(field not in decoded for field in required_fields):
        raise ValueError("Training result lacks prediction-binding fields")
    if str(decoded["purpose"]) != scope.purpose.value:
        raise ValueError("Training result and prediction purpose differ")
    if decoded["holdout_accessed"] is not False:
        raise PermissionError("Development predictions cannot bind a hold-out result")
    expected_scientific = scope.purpose is TrainingPurpose.DEVELOPMENT_SELECTION
    if bool(decoded["scientific_result"]) is not expected_scientific:
        raise ValueError("Training-result scientific status disagrees with its purpose")
    if tuple(map(str, decoded["validation_subjects"])) != scope.validation_subjects:
        raise ValueError("Training result and prediction validation subjects differ")
    for field_name in (
        "model_configuration_sha256",
        "learned_preprocessing_sha256",
    ):
        if SHA256_PATTERN.fullmatch(str(decoded[field_name])) is None:
            raise ValueError(f"Training result has an invalid {field_name}")
    return decoded


def build_prediction_artifact_record(
    *,
    run_id: str,
    created_at_utc: str,
    logits: NDArray[np.floating],
    metadata: Sequence[WindowMetadata],
    class_labels: Sequence[str],
    model: nn.Module,
    training_result_path: Path | str,
    scope: TrainingRunScope,
    protocol: ProtocolConfiguration | None = None,
) -> PredictionArtifactRecord:
    """Bind ordered validation logits to training, model, and path-free provenance."""
    scope.validate(protocol)
    records = tuple(metadata)
    labels = tuple(map(str, class_labels))
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or values.shape != (len(records), len(labels)):
        raise ValueError("Logits must align with metadata and the fixed class vocabulary")
    if not np.isfinite(values).all():
        raise ValueError("Prediction logits must be finite")
    if not records:
        raise ValueError("Prediction artifacts cannot be empty")
    observed_subjects = {record.subject_id for record in records}
    if observed_subjects != set(scope.validation_subjects):
        raise ValueError("Prediction metadata must exactly match validation subjects")
    predicted_labels = tuple(labels[index] for index in np.argmax(values, axis=1))
    reference_labels = tuple(record.main_label for record in records)
    confusion_matrix_from_predictions(reference_labels, predicted_labels, labels)
    training_binding = _training_result_binding(training_result_path, scope)

    rows = tuple(
        PredictionWindowRecord(
            window_index=index,
            participant_id=record.subject_id,
            recording_key_sha256=sha256_canonical_json(
                {
                    "participant_id": record.subject_id,
                    "recording_id": record.recording_id,
                }
            ),
            sensor_stream_key_sha256=sha256_canonical_json(
                {
                    "participant_id": record.subject_id,
                    "recording_id": record.recording_id,
                    "sensor_stream_id": record.sensor_stream_id,
                }
            ),
            main_label=record.main_label,
            sub_activity_label=record.sub_activity_label,
            sensor_side=record.sensor_side,
            start_sample=record.start_sample,
            end_sample_exclusive=record.end_sample_exclusive,
            preprocessing_status=record.preprocessing_status,
            quality_status=record.quality_status,
            predicted_label=predicted_labels[index],
            logits=tuple(float(value) for value in values[index]),
        )
        for index, record in enumerate(records)
    )
    prediction_payload_sha256 = sha256_canonical_json(
        [row.as_dict() for row in rows]
    )
    record = PredictionArtifactRecord(
        schema_version=1,
        run_id=run_id,
        created_at_utc=created_at_utc,
        experiment_id=str(training_binding["experiment_id"]),
        purpose=scope.purpose.value,
        scientific_result=(
            scope.purpose is TrainingPurpose.DEVELOPMENT_SELECTION
        ),
        holdout_accessed=False,
        source_revision=str(training_binding["source_revision"]),
        training_run_id=str(training_binding["run_id"]),
        training_result_sha256=sha256_file(training_result_path),
        model_configuration_sha256=str(
            training_binding["model_configuration_sha256"]
        ),
        learned_preprocessing_sha256=str(
            training_binding["learned_preprocessing_sha256"]
        ),
        model_state_sha256=sha256_model_state(model),
        class_labels=labels,
        validation_subjects=scope.validation_subjects,
        logit_dtype="float32",
        window_count=len(rows),
        prediction_payload_sha256=prediction_payload_sha256,
        windows=rows,
    )
    record.validate()
    return record


def write_prediction_artifact_record(
    path: Path | str,
    record: PredictionArtifactRecord,
) -> None:
    """Write a new prediction artifact exclusively without overwriting evidence."""
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


def load_prediction_artifact_record(path: Path | str) -> PredictionArtifactRecord:
    """Load, reconstruct, and validate an immutable prediction artifact."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Prediction artifact must be a JSON object")
    windows_value = decoded.get("windows")
    if not isinstance(windows_value, list):
        raise TypeError("Prediction artifact windows must be an array")
    windows = tuple(
        PredictionWindowRecord(
            window_index=int(row["window_index"]),
            participant_id=str(row["participant_id"]),
            recording_key_sha256=str(row["recording_key_sha256"]),
            sensor_stream_key_sha256=str(row["sensor_stream_key_sha256"]),
            main_label=str(row["main_label"]),
            sub_activity_label=str(row["sub_activity_label"]),
            sensor_side=str(row["sensor_side"]),
            start_sample=int(row["start_sample"]),
            end_sample_exclusive=int(row["end_sample_exclusive"]),
            preprocessing_status=str(row["preprocessing_status"]),
            quality_status=str(row["quality_status"]),
            predicted_label=str(row["predicted_label"]),
            logits=tuple(float(value) for value in row["logits"]),
        )
        for row in windows_value
    )
    record = PredictionArtifactRecord(
        schema_version=int(decoded["schema_version"]),
        run_id=str(decoded["run_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        experiment_id=str(decoded["experiment_id"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        source_revision=str(decoded["source_revision"]),
        training_run_id=str(decoded["training_run_id"]),
        training_result_sha256=str(decoded["training_result_sha256"]),
        model_configuration_sha256=str(decoded["model_configuration_sha256"]),
        learned_preprocessing_sha256=str(decoded["learned_preprocessing_sha256"]),
        model_state_sha256=str(decoded["model_state_sha256"]),
        class_labels=tuple(map(str, decoded["class_labels"])),
        validation_subjects=tuple(map(str, decoded["validation_subjects"])),
        logit_dtype=str(decoded["logit_dtype"]),
        window_count=int(decoded["window_count"]),
        prediction_payload_sha256=str(decoded["prediction_payload_sha256"]),
        windows=windows,
    )
    record.validate()
    return record
