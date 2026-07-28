"""Tests for deterministic figures derived from one analysis record."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("matplotlib")

from prevoccupai_har.analysis_records import (  # noqa: E402
    PredictionAnalysisRecord,
    write_prediction_analysis_record,
)
from prevoccupai_har.calibration import (  # noqa: E402
    evaluate_calibration,
    probabilities_from_logits,
)
from prevoccupai_har.evaluation import evaluate_predictions  # noqa: E402
from prevoccupai_har.figure_generation import (  # noqa: E402
    generate_prediction_figure_package,
)
from prevoccupai_har.provenance import sha256_canonical_json, sha256_file  # noqa: E402
from prevoccupai_har.temporal_evaluation import (  # noqa: E402
    evaluate_temporal_predictions,
)
from prevoccupai_har.windowing import WindowMetadata  # noqa: E402


def _write_analysis_record(path: Path) -> None:
    labels = ("sitting", "standing", "walking")
    truth = ("sitting", "sitting", "walking")
    predictions = ("sitting", "standing", "walking")
    participants = ("SYNTHETIC_VALIDATION_A",) * 3
    logits = np.asarray(
        ((3.0, 1.0, 0.0), (1.0, 3.0, 0.0), (0.0, 1.0, 4.0)),
        dtype=np.float64,
    )
    metadata = tuple(
        WindowMetadata(
            subject_id=participants[index],
            recording_id="synthetic-recording",
            main_label=truth[index],
            sub_activity_label=f"synthetic_{truth[index]}",
            sensor_stream_id="synthetic-stream",
            sensor_side="synthetic",
            start_sample=index * 2500,
            end_sample_exclusive=index * 2500 + 5000,
            preprocessing_status="synthetic",
            quality_status="synthetic",
        )
        for index in range(3)
    )
    classification = evaluate_predictions(
        truth,
        predictions,
        participants,
        labels,
    )
    calibration = evaluate_calibration(
        probabilities_from_logits(logits, expected_class_count=3),
        truth,
        labels,
        bin_count=5,
    )
    temporal = evaluate_temporal_predictions(
        predictions,
        metadata,
        labels,
        expected_step_size_samples=2500,
    )
    payload: dict[str, object] = {
        "classification": classification.as_dict(),
        "calibration": calibration.as_dict(),
        "temporal": temporal.as_dict(),
    }
    record = PredictionAnalysisRecord(
        schema_version=1,
        analysis_id="synthetic-analysis-figures",
        created_at_utc="2026-07-15T19:20:00Z",
        purpose="synthetic_validation",
        scientific_result=False,
        holdout_accessed=False,
        source_revision="unversioned_workspace_software_test",
        prediction_run_id="synthetic-predictions-figures",
        prediction_artifact_sha256="a" * 64,
        model_state_sha256="b" * 64,
        class_labels=labels,
        participant_count=1,
        window_count=3,
        analysis_settings={
            "probability_transform": {
                "method": "softmax",
                "temperature": 1.0,
                "fitted": False,
            },
            "calibration_bin_count": 5,
            "expected_step_size_samples": 2500,
            "short_run_max_windows": 1,
        },
        analysis_payload=payload,
        analysis_payload_sha256=sha256_canonical_json(payload),
    )
    write_prediction_analysis_record(path, record)


def test_figure_package_is_vector_bound_and_identifier_free(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    output = tmp_path / "figures"
    _write_analysis_record(analysis)

    manifest = generate_prediction_figure_package(
        analysis_record_path=analysis,
        output_directory=output,
    )
    decoded = json.loads((output / "FIGURE_MANIFEST.json").read_text())

    assert manifest.analysis_record_sha256 == sha256_file(analysis)
    assert manifest.scientific_result is False
    assert manifest.holdout_accessed is False
    assert manifest.participant_labels == "ordinal_only"
    assert "SYNTHETIC_VALIDATION_A" not in json.dumps(decoded)
    assert set(manifest.figures) == {
        "confusion_matrix",
        "participant_macro_f1",
        "calibration_reliability",
    }
    for figure_record in manifest.figures.values():
        figure_path = output / figure_record["filename"]
        assert figure_path.read_bytes().startswith(b"%PDF-")
        assert figure_record["sha256"] == sha256_file(figure_path)

    with pytest.raises(FileExistsError):
        generate_prediction_figure_package(
            analysis_record_path=analysis,
            output_directory=output,
        )


def test_figure_pdf_digests_are_deterministic(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.json"
    _write_analysis_record(analysis)
    first = generate_prediction_figure_package(
        analysis_record_path=analysis,
        output_directory=tmp_path / "first",
    )
    second = generate_prediction_figure_package(
        analysis_record_path=analysis,
        output_directory=tmp_path / "second",
    )

    assert first.figures == second.figures
