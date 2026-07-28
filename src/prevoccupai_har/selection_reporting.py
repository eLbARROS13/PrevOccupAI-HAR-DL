"""Deterministic figures and tables from one validated model-selection bundle."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np

from .model_selection import load_model_selection_bundle
from .provenance import sha256_file


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


OUTPUT_FILENAMES = {
    "paired_seed_macro_f1": "paired_seed_macro_f1.pdf",
    "learning_curves": "learning_curves.pdf",
    "complexity_performance": "complexity_performance.pdf",
    "selection_summary": "selection_summary.csv",
    "learning_curve_summary": "learning_curve_summary.csv",
}


@dataclass(frozen=True)
class SelectionReportPackageRecord:
    """Identifier-free output manifest bound to one selection bundle."""

    schema_version: int
    bundle_id: str
    selection_bundle_sha256: str
    bundle_payload_sha256: str
    selection_plan_sha256: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    run_count: int
    participant_labels: str
    candidate_labels: str
    outputs: dict[str, dict[str, str]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "selection_bundle_sha256": self.selection_bundle_sha256,
            "bundle_payload_sha256": self.bundle_payload_sha256,
            "selection_plan_sha256": self.selection_plan_sha256,
            "purpose": self.purpose,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            "run_count": self.run_count,
            "participant_labels": self.participant_labels,
            "candidate_labels": self.candidate_labels,
            "outputs": self.outputs,
        }


def _candidate_by_role(
    summaries: Mapping[str, object],
) -> tuple[tuple[str, Mapping[str, object]], tuple[str, Mapping[str, object]]]:
    values: dict[str, tuple[str, Mapping[str, object]]] = {}
    for candidate_id, summary in summaries.items():
        if not isinstance(summary, Mapping):
            raise TypeError("Candidate summary must be an object")
        role = str(summary["role"])
        if role in values:
            raise ValueError("Candidate roles must be unique")
        values[role] = (candidate_id, summary)
    if set(values) != {"reference", "challenger"}:
        raise ValueError("Reporting requires one reference and one challenger")
    return values["reference"], values["challenger"]


def _save_pdf(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "prevoccupai-har deterministic selection reporter",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _seed_values(summary: Mapping[str, object], metric: str) -> np.ndarray:
    per_seed = summary["per_seed"]
    if not isinstance(per_seed, list) or not per_seed:
        raise ValueError("Candidate summary requires non-empty per-seed values")
    return np.asarray([float(value[metric]) for value in per_seed], dtype=np.float64)


def _paired_seed_figure(
    reference: Mapping[str, object],
    challenger: Mapping[str, object],
) -> plt.Figure:
    reference_values = _seed_values(reference, "participant_macro_f1_mean")
    challenger_values = _seed_values(challenger, "participant_macro_f1_mean")
    if reference_values.shape != challenger_values.shape:
        raise ValueError("Candidate seed summaries must be paired")
    figure, axis = plt.subplots(figsize=(4.1, 3.6), constrained_layout=True)
    for index, (reference_value, challenger_value) in enumerate(
        zip(reference_values, challenger_values, strict=True)
    ):
        axis.plot(
            (0, 1),
            (reference_value, challenger_value),
            color="#7F7F7F",
            linewidth=0.8,
            marker="o",
            markersize=4,
            markerfacecolor="white" if index % 2 else "#7F7F7F",
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
    axis.scatter(
        (0, 1),
        (reference_values.mean(), challenger_values.mean()),
        marker="D",
        s=45,
        color="#C00000",
        edgecolor="black",
        linewidth=0.6,
        label="Mean across seeds",
        zorder=3,
    )
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks((0, 1), ("Reference", "Challenger"))
    axis.set_ylabel("Mean participant macro F1-score")
    axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    return figure


def _curve_array(summary: Mapping[str, object], field: str) -> tuple[np.ndarray, np.ndarray]:
    curve = summary["learning_curve"]
    if not isinstance(curve, list) or not curve:
        raise ValueError("Candidate summary requires a non-empty learning curve")
    epochs = np.asarray([int(value["epoch"]) for value in curve], dtype=np.int64)
    values = np.asarray([float(value[field]) for value in curve], dtype=np.float64)
    if not np.array_equal(epochs, np.arange(1, len(curve) + 1)):
        raise ValueError("Learning-curve epochs must be contiguous from one")
    return epochs, values


def _learning_curve_figure(
    reference: Mapping[str, object],
    challenger: Mapping[str, object],
) -> plt.Figure:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(6.2, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    candidates = (
        (reference, "Reference", "#4472C4", "o"),
        (challenger, "Challenger", "#ED7D31", "s"),
    )
    for summary, label, color, marker in candidates:
        epochs, training_loss = _curve_array(summary, "training_loss_mean")
        _, validation_loss = _curve_array(summary, "validation_loss_mean")
        _, macro_f1 = _curve_array(summary, "validation_macro_f1_mean")
        _, balanced_accuracy = _curve_array(
            summary,
            "validation_balanced_accuracy_mean",
        )
        _, contribution = _curve_array(summary, "contributing_run_fraction")
        axes[0].plot(
            epochs,
            training_loss,
            color=color,
            marker=marker,
            markevery=max(1, len(epochs) // 10),
            markersize=3,
            linewidth=1.0,
            label=f"{label}: train",
        )
        axes[0].plot(
            epochs,
            validation_loss,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"{label}: validation",
        )
        axes[1].plot(
            epochs,
            macro_f1,
            color=color,
            marker=marker,
            markevery=max(1, len(epochs) // 10),
            markersize=3,
            linewidth=1.0,
            label=f"{label}: macro F1",
        )
        axes[1].plot(
            epochs,
            balanced_accuracy,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"{label}: balanced accuracy",
        )
        axes[2].step(
            epochs,
            contribution,
            where="post",
            color=color,
            linewidth=1.1,
            label=label,
        )
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Validation metric")
    axes[1].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Contributing runs")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("Epoch")
    for axis in axes:
        axis.grid(linestyle=":", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=7, ncol=2)
    return figure


def _complexity_performance_figure(
    reference: Mapping[str, object],
    challenger: Mapping[str, object],
) -> plt.Figure:
    summaries = (reference, challenger)
    x_values = np.asarray(
        [int(summary["trainable_parameter_count"]) for summary in summaries],
        dtype=np.float64,
    )
    y_values = np.asarray(
        [float(summary["participant_macro_f1_seed_mean"]) for summary in summaries],
        dtype=np.float64,
    )
    y_errors = np.asarray(
        [
            float(summary["participant_macro_f1_seed_population_sd"])
            for summary in summaries
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    axis.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=0.8,
        label="Population SD across seed means",
    )
    axis.scatter(
        x_values,
        y_values,
        s=(55, 70),
        marker="o",
        color=("#4472C4", "#ED7D31"),
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for x_value, y_value, label in zip(
        x_values,
        y_values,
        ("Reference", "Challenger"),
        strict=True,
    ):
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(-5 if y_value > 0.9 else 5, -12 if y_value > 0.9 else 5),
            textcoords="offset points",
            fontsize=8,
            ha="right" if y_value > 0.9 else "left",
        )
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Mean participant macro F1-score")
    axis.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    return figure


def _format_number(value: object) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _write_selection_summary(
    path: Path,
    summaries: Mapping[str, object],
    selected_candidate_id: str,
) -> None:
    fieldnames = (
        "candidate_id",
        "role",
        "selected",
        "trainable_parameter_count",
        "run_count",
        "fold_count",
        "seed_count",
        "development_participant_count",
        "participant_macro_f1_seed_mean",
        "participant_macro_f1_seed_population_sd",
        "participant_balanced_accuracy_seed_mean",
        "participant_balanced_accuracy_seed_population_sd",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for candidate_id, summary in sorted(summaries.items()):
            if not isinstance(summary, Mapping):
                raise TypeError("Candidate summary must be an object")
            row = {
                "candidate_id": candidate_id,
                "role": summary["role"],
                "selected": candidate_id == selected_candidate_id,
                **{field: summary[field] for field in fieldnames[3:]},
            }
            writer.writerow({key: _format_number(value) for key, value in row.items()})


def _write_learning_curve_summary(
    path: Path,
    summaries: Mapping[str, object],
) -> None:
    fieldnames = (
        "candidate_id",
        "role",
        "epoch",
        "contributing_run_count",
        "contributing_run_fraction",
        "training_loss_mean",
        "validation_loss_mean",
        "validation_macro_f1_mean",
        "validation_balanced_accuracy_mean",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for candidate_id, summary in sorted(summaries.items()):
            if not isinstance(summary, Mapping):
                raise TypeError("Candidate summary must be an object")
            curve = summary["learning_curve"]
            if not isinstance(curve, list):
                raise TypeError("Learning curve must be an array")
            for point in curve:
                row = {
                    "candidate_id": candidate_id,
                    "role": summary["role"],
                    **{field: point[field] for field in fieldnames[2:]},
                }
                writer.writerow(
                    {key: _format_number(value) for key, value in row.items()}
                )


def generate_selection_report_package(
    *,
    selection_bundle_path: Path | str,
    output_directory: Path | str,
) -> SelectionReportPackageRecord:
    """Generate deterministic vector figures and CSV tables from one bundle."""
    bundle_path = Path(selection_bundle_path)
    bundle = load_model_selection_bundle(bundle_path)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=False)
    reference, challenger = _candidate_by_role(bundle.candidate_summaries)
    reference_summary = reference[1]
    challenger_summary = challenger[1]

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
            "paired_seed_macro_f1": _paired_seed_figure(
                reference_summary,
                challenger_summary,
            ),
            "learning_curves": _learning_curve_figure(
                reference_summary,
                challenger_summary,
            ),
            "complexity_performance": _complexity_performance_figure(
                reference_summary,
                challenger_summary,
            ),
        }
        for role, figure in figures.items():
            _save_pdf(figure, output_path / OUTPUT_FILENAMES[role])

    _write_selection_summary(
        output_path / OUTPUT_FILENAMES["selection_summary"],
        bundle.candidate_summaries,
        str(bundle.decision["selected_candidate_id"]),
    )
    _write_learning_curve_summary(
        output_path / OUTPUT_FILENAMES["learning_curve_summary"],
        bundle.candidate_summaries,
    )
    outputs = {
        role: {
            "filename": filename,
            "sha256": sha256_file(output_path / filename),
        }
        for role, filename in OUTPUT_FILENAMES.items()
    }
    manifest = SelectionReportPackageRecord(
        schema_version=1,
        bundle_id=bundle.bundle_id,
        selection_bundle_sha256=sha256_file(bundle_path),
        bundle_payload_sha256=bundle.bundle_payload_sha256,
        selection_plan_sha256=bundle.selection_plan_sha256,
        purpose=bundle.purpose,
        scientific_result=bundle.scientific_result,
        holdout_accessed=False,
        run_count=bundle.run_count,
        participant_labels="none",
        candidate_labels="role_only_in_figures",
        outputs=outputs,
    )
    (output_path / "REPORT_MANIFEST.json").write_text(
        json.dumps(manifest.as_dict(), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest
