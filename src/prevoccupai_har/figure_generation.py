"""Deterministic manuscript figures from one governed prediction analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np

from .analysis_records import load_prediction_analysis_record
from .provenance import sha256_file


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


FIGURE_FILENAMES = {
    "confusion_matrix": "confusion_matrix.pdf",
    "participant_macro_f1": "participant_macro_f1.pdf",
    "calibration_reliability": "calibration_reliability.pdf",
}


@dataclass(frozen=True)
class FigurePackageRecord:
    """Identifier-free manifest for figures derived from one analysis record."""

    schema_version: int
    analysis_id: str
    analysis_record_sha256: str
    analysis_payload_sha256: str
    prediction_artifact_sha256: str
    scientific_result: bool
    holdout_accessed: bool
    participant_count: int
    figure_format: str
    participant_labels: str
    figures: dict[str, dict[str, str]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "analysis_record_sha256": self.analysis_record_sha256,
            "analysis_payload_sha256": self.analysis_payload_sha256,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            "participant_count": self.participant_count,
            "figure_format": self.figure_format,
            "participant_labels": self.participant_labels,
            "figures": self.figures,
        }


def _save_pdf(figure: plt.Figure, path: Path) -> None:
    """Save a vector PDF without volatile timestamps."""
    figure.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "prevoccupai-har deterministic figure generator",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _confusion_matrix_figure(
    matrix: np.ndarray,
    class_labels: tuple[str, ...],
) -> plt.Figure:
    """Create a count-annotated, fixed-label confusion matrix."""
    figure, axis = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Windows")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Reference class")
    positions = np.arange(len(class_labels))
    axis.set_xticks(positions, class_labels, rotation=35, ha="right")
    axis.set_yticks(positions, class_labels)
    threshold = float(matrix.max()) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = int(matrix[row, column])
            axis.text(
                column,
                row,
                f"{count:,}",
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
            )
    return figure


def _participant_macro_f1_figure(
    participant_metrics: Mapping[str, Mapping[str, object]],
) -> plt.Figure:
    """Create an ordinal-labelled participant macro-F1 bar chart."""
    ordered_participants = sorted(participant_metrics)
    values = np.asarray(
        [float(participant_metrics[key]["macro_f1"]) for key in ordered_participants],
        dtype=np.float64,
    )
    width = max(4.2, 1.5 + 0.42 * len(ordered_participants))
    figure, axis = plt.subplots(figsize=(width, 3.4), constrained_layout=True)
    positions = np.arange(len(ordered_participants))
    axis.bar(
        positions,
        values,
        color="#4472C4",
        edgecolor="black",
        linewidth=0.6,
        hatch="//",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Macro F1-score")
    axis.set_xlabel("Participant")
    axis.set_xticks(
        positions,
        [str(index + 1) for index in range(len(ordered_participants))],
    )
    axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    return figure


def _calibration_figure(calibration: Mapping[str, object]) -> plt.Figure:
    """Create a sample-size-encoded reliability diagram."""
    bins = calibration["nonempty_bins"]
    if not isinstance(bins, list) or not bins:
        raise ValueError("Calibration analysis must contain non-empty bins")
    confidence = np.asarray(
        [float(bin_record["mean_confidence"]) for bin_record in bins],
        dtype=np.float64,
    )
    accuracy = np.asarray(
        [float(bin_record["accuracy"]) for bin_record in bins],
        dtype=np.float64,
    )
    counts = np.asarray(
        [int(bin_record["sample_count"]) for bin_record in bins],
        dtype=np.float64,
    )
    marker_sizes = 28.0 + 92.0 * np.sqrt(counts / counts.max())

    figure, axis = plt.subplots(figsize=(4.0, 3.6), constrained_layout=True)
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
    )
    axis.vlines(
        confidence,
        np.minimum(confidence, accuracy),
        np.maximum(confidence, accuracy),
        color="#4472C4",
        linewidth=0.8,
        alpha=0.65,
    )
    axis.plot(confidence, accuracy, color="#4472C4", linewidth=1.0)
    axis.scatter(
        confidence,
        accuracy,
        s=marker_sizes,
        color="#4472C4",
        edgecolor="black",
        linewidth=0.6,
        label="Observed non-empty bins",
        zorder=3,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    return figure


def generate_prediction_figure_package(
    *,
    analysis_record_path: Path | str,
    output_directory: Path | str,
) -> FigurePackageRecord:
    """Generate vector figures and a digest manifest without overwriting outputs."""
    analysis_path = Path(analysis_record_path)
    record = load_prediction_analysis_record(analysis_path)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=False)

    classification = record.analysis_payload["classification"]
    calibration = record.analysis_payload["calibration"]
    if not isinstance(classification, Mapping) or not isinstance(calibration, Mapping):
        raise TypeError("Classification and calibration payloads must be mappings")
    matrix = np.asarray(classification["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (len(record.class_labels), len(record.class_labels)):
        raise ValueError("Confusion matrix does not match the fixed class vocabulary")
    participant_metrics = classification["per_participant_metrics"]
    if not isinstance(participant_metrics, Mapping):
        raise TypeError("Participant metrics must be a mapping")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    ):
        figures = {
            "confusion_matrix": _confusion_matrix_figure(
                matrix,
                record.class_labels,
            ),
            "participant_macro_f1": _participant_macro_f1_figure(
                participant_metrics,
            ),
            "calibration_reliability": _calibration_figure(calibration),
        }
        for role, figure in figures.items():
            _save_pdf(figure, output_path / FIGURE_FILENAMES[role])

    figure_records = {
        role: {
            "filename": filename,
            "sha256": sha256_file(output_path / filename),
        }
        for role, filename in FIGURE_FILENAMES.items()
    }
    manifest = FigurePackageRecord(
        schema_version=1,
        analysis_id=record.analysis_id,
        analysis_record_sha256=sha256_file(analysis_path),
        analysis_payload_sha256=record.analysis_payload_sha256,
        prediction_artifact_sha256=record.prediction_artifact_sha256,
        scientific_result=record.scientific_result,
        holdout_accessed=False,
        participant_count=record.participant_count,
        figure_format="vector_pdf",
        participant_labels="ordinal_only",
        figures=figure_records,
    )
    (output_path / "FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest.as_dict(), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest
