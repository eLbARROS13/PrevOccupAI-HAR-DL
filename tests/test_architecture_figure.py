"""Tests for the configuration-derived model-architecture figure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip("matplotlib")
pytest.importorskip("torch")

from prevoccupai_har.architecture_figure import (  # noqa: E402
    FIGURE_FILENAME,
    MANIFEST_FILENAME,
    generate_model_architecture_figure,
)
from prevoccupai_har.provenance import sha256_file  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_CONFIGURATION = PROJECT_ROOT / "configs" / "cnn_1d.json"
TCN_CONFIGURATION = PROJECT_ROOT / "configs" / "tcn_1d.json"


def test_architecture_figure_is_configuration_bound_and_data_free(
    tmp_path: Path,
) -> None:
    output = tmp_path / "figure"
    record = generate_model_architecture_figure(
        cnn_configuration_path=CNN_CONFIGURATION,
        tcn_configuration_path=TCN_CONFIGURATION,
        output_directory=output,
    )
    decoded = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert record.cnn_configuration_sha256 == sha256_file(CNN_CONFIGURATION)
    assert record.tcn_configuration_sha256 == sha256_file(TCN_CONFIGURATION)
    assert record.cnn_parameter_count == 15_987
    assert record.tcn_parameter_count == 2_307
    assert record.tcn_receptive_field_samples == 369
    assert record.input_shape_channels_first == (3, 500)
    assert record.participant_data_used is False
    assert record.performance_values_used is False
    assert not any("path" in key.lower() for key in decoded)
    figure_path = output / FIGURE_FILENAME
    assert figure_path.read_bytes().startswith(b"%PDF-")
    assert record.figure_sha256 == sha256_file(figure_path)

    with pytest.raises(FileExistsError):
        generate_model_architecture_figure(
            cnn_configuration_path=CNN_CONFIGURATION,
            tcn_configuration_path=TCN_CONFIGURATION,
            output_directory=output,
        )


def test_architecture_figure_pdf_is_byte_deterministic(tmp_path: Path) -> None:
    first = generate_model_architecture_figure(
        cnn_configuration_path=CNN_CONFIGURATION,
        tcn_configuration_path=TCN_CONFIGURATION,
        output_directory=tmp_path / "first",
    )
    second = generate_model_architecture_figure(
        cnn_configuration_path=CNN_CONFIGURATION,
        tcn_configuration_path=TCN_CONFIGURATION,
        output_directory=tmp_path / "second",
    )

    assert first.figure_sha256 == second.figure_sha256


def test_architecture_figure_rejects_mismatched_class_vocabulary(
    tmp_path: Path,
) -> None:
    decoded = json.loads(TCN_CONFIGURATION.read_text(encoding="utf-8"))
    decoded["input"]["class_labels"] = ["sitting", "walking", "standing"]
    altered = tmp_path / "altered_tcn.json"
    altered.write_text(json.dumps(decoded), encoding="utf-8")

    with pytest.raises(ValueError, match="class vocabularies differ"):
        generate_model_architecture_figure(
            cnn_configuration_path=CNN_CONFIGURATION,
            tcn_configuration_path=altered,
            output_directory=tmp_path / "rejected",
        )
