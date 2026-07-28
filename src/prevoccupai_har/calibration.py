"""Validation-fitted confidence calibration with explicit hold-out exclusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .protocol import ProtocolConfiguration


class CalibrationFitPurpose(str, Enum):
    """Permitted purposes for fitting a probability-calibration transform."""

    SYNTHETIC_VALIDATION = "synthetic_validation"
    DEVELOPMENT_VALIDATION = "development_validation"


@dataclass(frozen=True)
class CalibrationFitScope:
    """Validation and external hold-out cohorts supplied independently of logits."""

    purpose: CalibrationFitPurpose
    validation_subjects: tuple[str, ...]
    holdout_subjects: tuple[str, ...]

    def validate(self, protocol: ProtocolConfiguration | None = None) -> None:
        """Reject hold-out fitting and ungoverned scientific calibration."""
        validation = set(self.validation_subjects)
        holdout = set(self.holdout_subjects)
        if not validation or not holdout:
            raise ValueError("Validation and hold-out subject sets must be non-empty")
        if len(validation) != len(self.validation_subjects):
            raise ValueError("Validation subject identifiers contain duplicates")
        if len(holdout) != len(self.holdout_subjects):
            raise ValueError("Hold-out subject identifiers contain duplicates")
        if validation & holdout:
            raise PermissionError("Calibration fitting cannot include hold-out subjects")

        if self.purpose is CalibrationFitPurpose.SYNTHETIC_VALIDATION:
            if protocol is not None:
                raise ValueError("Synthetic calibration must not receive a data protocol")
            if any(
                not subject.startswith("SYNTHETIC_")
                for subject in validation | holdout
            ):
                raise ValueError("Synthetic calibration requires synthetic subject identifiers")
            return

        if self.purpose is not CalibrationFitPurpose.DEVELOPMENT_VALIDATION:
            raise TypeError("Calibration purpose must be a CalibrationFitPurpose value")
        if protocol is None:
            raise ValueError("Development calibration requires a validated protocol")
        if not protocol.training_authorized:
            raise PermissionError("The protocol does not authorize scientific calibration")
        if holdout != set(protocol.holdout_participants):
            raise PermissionError("Calibration and protocol hold-out cohorts differ")
        if not validation.issubset(set(protocol.development_participants)):
            raise PermissionError("Calibration validation subjects must be developmental")


@dataclass(frozen=True)
class CalibrationBin:
    """One non-empty equal-width confidence bin."""

    lower_bound: float
    upper_bound: float
    sample_count: int
    accuracy: float
    mean_confidence: float
    absolute_gap: float


@dataclass(frozen=True)
class CalibrationEvaluation:
    """Discrimination and confidence diagnostics for fixed class probabilities."""

    class_labels: tuple[str, ...]
    sample_count: int
    requested_bin_count: int
    negative_log_likelihood: float
    multiclass_brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    accuracy: float
    mean_confidence: float
    nonempty_bins: tuple[CalibrationBin, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["class_labels"] = list(self.class_labels)
        return value


@dataclass(frozen=True)
class TemperatureScalingModel:
    """Scalar temperature fitted only from an approved validation partition."""

    temperature: float
    class_labels: tuple[str, ...]
    fit_sample_count: int
    fit_subject_count: int
    fit_purpose: CalibrationFitPurpose

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        if len(self.class_labels) < 2 or len(set(self.class_labels)) != len(
            self.class_labels
        ):
            raise ValueError("At least two unique class labels are required")
        if self.fit_sample_count < 1 or self.fit_subject_count < 1:
            raise ValueError("Calibration fit counts must be positive")
        if not isinstance(self.fit_purpose, CalibrationFitPurpose):
            raise TypeError("Fit purpose must be a CalibrationFitPurpose value")

    def predict_probabilities(
        self,
        logits: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Apply the fixed temperature without learning from evaluation labels."""
        return probabilities_from_logits(
            logits,
            expected_class_count=len(self.class_labels),
            temperature=self.temperature,
        )

    def state_dict(self) -> Mapping[str, object]:
        """Expose an immutable, identifier-free state for provenance hashing."""
        return MappingProxyType(
            {
                "schema_version": 1,
                "transform": "scalar_temperature_scaling",
                "temperature": self.temperature,
                "class_labels": self.class_labels,
                "fit_sample_count": self.fit_sample_count,
                "fit_subject_count": self.fit_subject_count,
                "fit_purpose": self.fit_purpose.value,
                "holdout_accessed_during_fit": False,
            }
        )


def _validate_class_labels(class_labels: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(map(str, class_labels))
    if (
        len(labels) < 2
        or len(labels) != len(set(labels))
        or any(not label.strip() for label in labels)
    ):
        raise ValueError("Class labels must contain at least two unique values")
    return labels


def _validate_logits(
    logits: NDArray[np.floating],
    *,
    expected_class_count: int,
) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Logits must have shape (samples, classes)")
    if values.shape[1] != expected_class_count:
        raise ValueError("Logit columns do not match the declared class count")
    if not np.isfinite(values).all():
        raise ValueError("Logits must be finite")
    return values


def _encode_true_labels(
    true_labels: Sequence[str],
    class_labels: tuple[str, ...],
    *,
    expected_count: int,
) -> NDArray[np.int64]:
    truth = tuple(map(str, true_labels))
    if len(truth) != expected_count:
        raise ValueError("One true label is required per prediction")
    label_indices = {label: index for index, label in enumerate(class_labels)}
    unknown = sorted(set(truth) - set(class_labels))
    if unknown:
        raise ValueError(f"True labels fall outside the declared class set: {unknown}")
    return np.asarray([label_indices[label] for label in truth], dtype=np.int64)


def probabilities_from_logits(
    logits: NDArray[np.floating],
    *,
    expected_class_count: int,
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """Convert logits to immutable probabilities using a positive temperature."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values = _validate_logits(logits, expected_class_count=expected_class_count)
    scaled = values / float(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    probabilities.setflags(write=False)
    return probabilities


def evaluate_calibration(
    probabilities: NDArray[np.floating],
    true_labels: Sequence[str],
    class_labels: Sequence[str],
    *,
    bin_count: int = 15,
) -> CalibrationEvaluation:
    """Evaluate fixed probabilities with equal-width top-label calibration bins.

    Expected calibration error is the sample-weighted absolute difference between
    accuracy and mean top-label confidence in each non-empty bin. The multiclass
    Brier score is the mean per-sample sum of squared class-probability errors.
    """
    labels = _validate_class_labels(class_labels)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Probabilities must have shape (samples, classes)")
    if values.shape[1] != len(labels):
        raise ValueError("Probability columns do not match the declared class count")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Every probability row must sum to one")
    if bin_count < 2:
        raise ValueError("At least two calibration bins are required")
    targets = _encode_true_labels(
        true_labels,
        labels,
        expected_count=values.shape[0],
    )

    predicted = values.argmax(axis=1)
    confidence = values[np.arange(values.shape[0]), predicted]
    correct = predicted == targets
    one_hot = np.eye(len(labels), dtype=np.float64)[targets]
    clipped_true_probabilities = np.clip(
        values[np.arange(values.shape[0]), targets],
        np.finfo(np.float64).tiny,
        1.0,
    )
    negative_log_likelihood = float(-np.log(clipped_true_probabilities).mean())
    brier_score = float(np.square(values - one_hot).sum(axis=1).mean())

    bin_indices = np.minimum(
        np.floor(confidence * bin_count).astype(np.int64),
        bin_count - 1,
    )
    bins: list[CalibrationBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for bin_index in range(bin_count):
        selected = bin_indices == bin_index
        sample_count = int(selected.sum())
        if sample_count == 0:
            continue
        bin_accuracy = float(correct[selected].mean())
        bin_confidence = float(confidence[selected].mean())
        gap = abs(bin_accuracy - bin_confidence)
        weighted_gap += sample_count * gap
        maximum_gap = max(maximum_gap, gap)
        bins.append(
            CalibrationBin(
                lower_bound=bin_index / bin_count,
                upper_bound=(bin_index + 1) / bin_count,
                sample_count=sample_count,
                accuracy=bin_accuracy,
                mean_confidence=bin_confidence,
                absolute_gap=gap,
            )
        )
    return CalibrationEvaluation(
        class_labels=labels,
        sample_count=values.shape[0],
        requested_bin_count=bin_count,
        negative_log_likelihood=negative_log_likelihood,
        multiclass_brier_score=brier_score,
        expected_calibration_error=weighted_gap / values.shape[0],
        maximum_calibration_error=maximum_gap,
        accuracy=float(correct.mean()),
        mean_confidence=float(confidence.mean()),
        nonempty_bins=tuple(bins),
    )


def fit_temperature_scaling(
    validation_logits: NDArray[np.floating],
    true_labels: Sequence[str],
    participant_ids: Sequence[str],
    class_labels: Sequence[str],
    *,
    scope: CalibrationFitScope,
    protocol: ProtocolConfiguration | None = None,
    minimum_temperature: float = 0.05,
    maximum_temperature: float = 10.0,
) -> TemperatureScalingModel:
    """Fit one temperature by validation NLL with no external hold-out mode."""
    scope.validate(protocol)
    labels = _validate_class_labels(class_labels)
    logits = _validate_logits(
        validation_logits,
        expected_class_count=len(labels),
    )
    targets = _encode_true_labels(
        true_labels,
        labels,
        expected_count=logits.shape[0],
    )
    participants = tuple(map(str, participant_ids))
    if len(participants) != logits.shape[0] or any(not item for item in participants):
        raise ValueError("One non-empty participant identifier is required per logit row")
    if set(participants) != set(scope.validation_subjects):
        raise PermissionError("Observed calibration subjects must equal the validation scope")
    if (
        not np.isfinite(minimum_temperature)
        or not np.isfinite(maximum_temperature)
        or minimum_temperature <= 0
        or maximum_temperature <= minimum_temperature
    ):
        raise ValueError("Temperature bounds must be finite, positive, and ordered")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled = logits / temperature
        log_normalizers = logsumexp(scaled, axis=1)
        target_logits = scaled[np.arange(scaled.shape[0]), targets]
        return float(np.mean(log_normalizers - target_logits))

    result = minimize_scalar(
        objective,
        bounds=(np.log(minimum_temperature), np.log(maximum_temperature)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature optimization did not converge")
    return TemperatureScalingModel(
        temperature=float(np.exp(result.x)),
        class_labels=labels,
        fit_sample_count=logits.shape[0],
        fit_subject_count=len(set(participants)),
        fit_purpose=scope.purpose,
    )
