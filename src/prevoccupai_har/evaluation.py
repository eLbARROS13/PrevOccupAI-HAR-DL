"""Transparent aggregate and participant-level classification evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def metrics_from_confusion_matrix(
    confusion_matrix: NDArray[np.integer],
    class_labels: Sequence[str],
) -> dict[str, object]:
    """Compute accuracy, balanced accuracy, macro F1, and class-wise metrics.

    Rows are true classes and columns are predicted classes. Undefined precision,
    recall, or F1 values are reported as zero, matching a conservative zero-division
    convention.
    """
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    labels = tuple(map(str, class_labels))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if matrix.shape[0] != len(labels):
        raise ValueError("One class label is required per matrix row and column")
    if len(set(labels)) != len(labels):
        raise ValueError("Class labels must be unique")
    if np.any(matrix < 0):
        raise ValueError("Confusion-matrix counts cannot be negative")
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Confusion matrix cannot be empty")

    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )

    classwise = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    weighted_f1 = float(np.sum(f1 * support) / total)
    return {
        "accuracy": float(true_positive.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": weighted_f1,
        "support": total,
        "classwise": classwise,
    }


def confusion_matrix_from_predictions(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_labels: Sequence[str],
) -> NDArray[np.int64]:
    """Build a fixed-label confusion matrix from aligned prediction vectors."""
    truth = tuple(map(str, true_labels))
    predictions = tuple(map(str, predicted_labels))
    labels = tuple(map(str, class_labels))
    if not truth or len(truth) != len(predictions):
        raise ValueError("Truth and prediction vectors must be non-empty and aligned")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Class labels must be non-empty and unique")
    label_indices = {label: index for index, label in enumerate(labels)}
    unknown_truth = sorted(set(truth) - set(labels))
    unknown_predictions = sorted(set(predictions) - set(labels))
    if unknown_truth or unknown_predictions:
        raise ValueError(
            "Predictions contain labels outside the declared class set: "
            f"truth={unknown_truth}, predictions={unknown_predictions}"
        )
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, predicted_label in zip(truth, predictions, strict=True):
        matrix[label_indices[true_label], label_indices[predicted_label]] += 1
    return matrix


@dataclass(frozen=True)
class PredictionEvaluation:
    """Overall and participant-level metrics from one frozen prediction vector."""

    class_labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    overall_metrics: dict[str, object]
    per_participant_metrics: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["class_labels"] = list(self.class_labels)
        value["confusion_matrix"] = [list(row) for row in self.confusion_matrix]
        return value


def evaluate_predictions(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    participant_ids: Sequence[str],
    class_labels: Sequence[str],
) -> PredictionEvaluation:
    """Compute fixed-label overall and per-participant classification metrics."""
    truth = tuple(map(str, true_labels))
    predictions = tuple(map(str, predicted_labels))
    participants = tuple(map(str, participant_ids))
    labels = tuple(map(str, class_labels))
    if len(participants) != len(truth):
        raise ValueError("One participant identifier is required per prediction")
    if any(not participant for participant in participants):
        raise ValueError("Participant identifiers cannot be empty")
    matrix = confusion_matrix_from_predictions(truth, predictions, labels)
    per_participant: dict[str, dict[str, object]] = {}
    participant_array = np.asarray(participants, dtype=object)
    truth_array = np.asarray(truth, dtype=object)
    prediction_array = np.asarray(predictions, dtype=object)
    for participant in sorted(set(participants)):
        participant_mask = participant_array == participant
        participant_matrix = confusion_matrix_from_predictions(
            truth_array[participant_mask].tolist(),
            prediction_array[participant_mask].tolist(),
            labels,
        )
        metrics = metrics_from_confusion_matrix(participant_matrix, labels)
        metrics["confusion_matrix"] = participant_matrix.tolist()
        per_participant[participant] = metrics
    return PredictionEvaluation(
        class_labels=labels,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        overall_metrics=metrics_from_confusion_matrix(matrix, labels),
        per_participant_metrics=per_participant,
    )
