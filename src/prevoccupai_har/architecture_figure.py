"""Deterministic model-architecture figure derived from frozen configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

from .modeling import (
    CompactCnnExperimentConfiguration,
    CompactTcnExperimentConfiguration,
    build_compact_cnn_1d,
    build_compact_tcn,
    load_compact_cnn_experiment_configuration,
    load_compact_tcn_experiment_configuration,
)
from .provenance import sha256_file


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


FIGURE_FILENAME = "model_architectures.pdf"
MANIFEST_FILENAME = "MODEL_ARCHITECTURE_FIGURE_MANIFEST.json"


@dataclass(frozen=True)
class ArchitectureFigureRecord:
    """Path-free provenance for the configuration-derived methods figure."""

    schema_version: int
    cnn_experiment_id: str
    cnn_configuration_sha256: str
    cnn_parameter_count: int
    tcn_experiment_id: str
    tcn_configuration_sha256: str
    tcn_parameter_count: int
    tcn_receptive_field_samples: int
    protocol_configuration_sha256: str
    input_shape_channels_first: tuple[int, int]
    class_labels: tuple[str, ...]
    participant_data_used: bool
    performance_values_used: bool
    figure_filename: str
    figure_sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible manifest representation."""
        return {
            "schema_version": self.schema_version,
            "cnn_experiment_id": self.cnn_experiment_id,
            "cnn_configuration_sha256": self.cnn_configuration_sha256,
            "cnn_parameter_count": self.cnn_parameter_count,
            "tcn_experiment_id": self.tcn_experiment_id,
            "tcn_configuration_sha256": self.tcn_configuration_sha256,
            "tcn_parameter_count": self.tcn_parameter_count,
            "tcn_receptive_field_samples": self.tcn_receptive_field_samples,
            "protocol_configuration_sha256": self.protocol_configuration_sha256,
            "input_shape_channels_first": list(self.input_shape_channels_first),
            "class_labels": list(self.class_labels),
            "participant_data_used": self.participant_data_used,
            "performance_values_used": self.performance_values_used,
            "figure_filename": self.figure_filename,
            "figure_sha256": self.figure_sha256,
        }


@dataclass(frozen=True)
class _Stage:
    """One labelled box in a model pipeline."""

    heading: str
    detail: str
    color: str
    hatch: str


def _validate_shared_contract(
    cnn: CompactCnnExperimentConfiguration,
    tcn: CompactTcnExperimentConfiguration,
) -> None:
    """Require both rows to describe the same governed prediction task."""
    cnn_shape = (cnn.architecture.input_channels, cnn.architecture.expected_samples)
    tcn_shape = (tcn.architecture.input_channels, tcn.architecture.expected_samples)
    if cnn_shape != tcn_shape:
        raise ValueError("CNN and TCN input shapes differ")
    if cnn.class_labels != tcn.class_labels:
        raise ValueError("CNN and TCN class vocabularies differ")
    if cnn.protocol_config != tcn.protocol_config:
        raise ValueError("CNN and TCN protocol configuration paths differ")
    if not cnn.protocol_config.is_file():
        raise FileNotFoundError(
            f"Shared protocol configuration is missing: {cnn.protocol_config}"
        )


def _draw_stage(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    stage: _Stage,
) -> None:
    """Draw one accessible, text-labelled architecture stage."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=stage.color,
        edgecolor="black",
        linewidth=0.8,
        hatch=stage.hatch,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.62,
        stage.heading,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
    )
    axis.text(
        x + width / 2,
        y + height * 0.29,
        stage.detail,
        ha="center",
        va="center",
        fontsize=7.1,
        linespacing=1.12,
    )


def _draw_pipeline(
    axis: plt.Axes,
    *,
    y: float,
    label: str,
    stages: tuple[_Stage, ...],
) -> None:
    """Draw one left-to-right model row with explicit arrow direction."""
    left = 0.025
    right = 0.985
    gap = 0.022
    height = 0.245
    width = (right - left - gap * (len(stages) - 1)) / len(stages)
    axis.text(
        left,
        y + height + 0.035,
        label,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
    )
    for index, stage in enumerate(stages):
        x = left + index * (width + gap)
        _draw_stage(
            axis,
            x=x,
            y=y,
            width=width,
            height=height,
            stage=stage,
        )
        if index < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.002, y + height / 2),
                (x + width + gap - 0.002, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=9,
                color="black",
                linewidth=0.8,
            )
            axis.add_patch(arrow)


def _create_figure(
    cnn: CompactCnnExperimentConfiguration,
    tcn: CompactTcnExperimentConfiguration,
    *,
    cnn_parameter_count: int,
    tcn_parameter_count: int,
) -> plt.Figure:
    """Create the two-panel architecture schematic without performance data."""
    cnn_architecture = cnn.architecture
    tcn_architecture = tcn.architecture
    input_stage = _Stage(
        "ACC window",
        f"{cnn_architecture.input_channels} channels\n{cnn_architecture.expected_samples} samples",
        "#E6E6E6",
        "..",
    )
    cnn_stages = (
        input_stage,
        _Stage(
            "Conv block 1",
            f"{cnn_architecture.convolution_channels[0]} ch.; k={cnn_architecture.kernel_sizes[0]}, "
            f"s={cnn_architecture.strides[0]}\npool={cnn_architecture.max_pool_sizes[0]}",
            "#D9E8FB",
            "///",
        ),
        _Stage(
            "Conv block 2",
            f"{cnn_architecture.convolution_channels[1]} ch.; k={cnn_architecture.kernel_sizes[1]}, "
            f"s={cnn_architecture.strides[1]}\npool={cnn_architecture.max_pool_sizes[1]}",
            "#D9E8FB",
            "///",
        ),
        _Stage(
            "Conv block 3",
            f"{cnn_architecture.convolution_channels[2]} ch.; k={cnn_architecture.kernel_sizes[2]}, "
            f"s={cnn_architecture.strides[2]}\nno temporal pool",
            "#D9E8FB",
            "///",
        ),
        _Stage(
            "Global pooling",
            f"adaptive average\ndropout={cnn_architecture.dropout_probability:g}",
            "#E3F1DF",
            "xx",
        ),
        _Stage(
            "Classifier",
            f"{cnn_architecture.output_classes} logits\n{cnn_parameter_count:,} parameters",
            "#F1E5F5",
            "++",
        ),
    )
    dilation_groups = tuple(
        tcn_architecture.dilations[index : index + 2]
        for index in range(0, len(tcn_architecture.dilations), 2)
    )
    if not dilation_groups or any(len(group) > 2 for group in dilation_groups):
        raise ValueError("Architecture figure requires non-empty TCN dilation groups")
    tcn_stages = (
        input_stage,
        _Stage(
            "Temporal stem",
            f"{tcn_architecture.channels} ch.; k={tcn_architecture.stem_kernel_size}\n"
            f"stride={tcn_architecture.stem_stride}",
            "#FCE6CC",
            "\\\\\\",
        ),
        *(
            _Stage(
                f"Blocks {2 * index + 1}--{2 * index + 2}",
                f"2 conv./block; k={tcn_architecture.kernel_size}\n"
                "d=" + ", ".join(map(str, group)),
                "#FCE6CC",
                "\\\\\\",
            )
            for index, group in enumerate(dilation_groups)
        ),
        _Stage(
            "Pooling + head",
            f"global average; {tcn_architecture.output_classes} logits\n"
            f"{tcn_parameter_count:,} parameters",
            "#E3F1DF",
            "xx",
        ),
    )

    figure, axis = plt.subplots(figsize=(7.35, 4.15), constrained_layout=True)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    _draw_pipeline(
        axis,
        y=0.58,
        label="a  Compact one-dimensional CNN",
        stages=cnn_stages,
    )
    _draw_pipeline(
        axis,
        y=0.095,
        label=(
            "b  Residual dilated TCN "
            f"(deepest receptive field: {tcn_architecture.receptive_field_samples} samples)"
        ),
        stages=tcn_stages,
    )
    return figure


def generate_model_architecture_figure(
    *,
    cnn_configuration_path: Path | str,
    tcn_configuration_path: Path | str,
    output_directory: Path | str,
) -> ArchitectureFigureRecord:
    """Generate a configuration-bound vector figure and immutable manifest."""
    cnn_path = Path(cnn_configuration_path)
    tcn_path = Path(tcn_configuration_path)
    cnn = load_compact_cnn_experiment_configuration(cnn_path)
    tcn = load_compact_tcn_experiment_configuration(tcn_path)
    _validate_shared_contract(cnn, tcn)

    cnn_parameter_count = build_compact_cnn_1d(cnn).trainable_parameter_count
    tcn_parameter_count = build_compact_tcn(tcn).trainable_parameter_count
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=False)
    figure_path = output_path / FIGURE_FILENAME

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
        }
    ):
        figure = _create_figure(
            cnn,
            tcn,
            cnn_parameter_count=cnn_parameter_count,
            tcn_parameter_count=tcn_parameter_count,
        )
        figure.savefig(
            figure_path,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Creator": "prevoccupai-har deterministic methods-figure generator",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)

    record = ArchitectureFigureRecord(
        schema_version=1,
        cnn_experiment_id=cnn.experiment_id,
        cnn_configuration_sha256=sha256_file(cnn_path),
        cnn_parameter_count=cnn_parameter_count,
        tcn_experiment_id=tcn.experiment_id,
        tcn_configuration_sha256=sha256_file(tcn_path),
        tcn_parameter_count=tcn_parameter_count,
        tcn_receptive_field_samples=tcn.architecture.receptive_field_samples,
        protocol_configuration_sha256=sha256_file(cnn.protocol_config),
        input_shape_channels_first=(
            cnn.architecture.input_channels,
            cnn.architecture.expected_samples,
        ),
        class_labels=cnn.class_labels,
        participant_data_used=False,
        performance_values_used=False,
        figure_filename=FIGURE_FILENAME,
        figure_sha256=sha256_file(figure_path),
    )
    (output_path / MANIFEST_FILENAME).write_text(
        json.dumps(record.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record
