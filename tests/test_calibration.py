"""Synthetic tests for validation-only confidence calibration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prevoccupai_har.calibration import (
    CalibrationFitPurpose,
    CalibrationFitScope,
    evaluate_calibration,
    fit_temperature_scaling,
    probabilities_from_logits,
)
from prevoccupai_har.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
CLASS_LABELS = ("sitting", "standing")


def _synthetic_scope() -> CalibrationFitScope:
    return CalibrationFitScope(
        purpose=CalibrationFitPurpose.SYNTHETIC_VALIDATION,
        validation_subjects=("SYNTHETIC_VALIDATION_A", "SYNTHETIC_VALIDATION_B"),
        holdout_subjects=("SYNTHETIC_HOLDOUT_A",),
    )


def test_perfect_probabilities_have_zero_calibration_error() -> None:
    evaluation = evaluate_calibration(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ["sitting", "standing"],
        CLASS_LABELS,
        bin_count=10,
    )

    assert evaluation.accuracy == 1.0
    assert evaluation.negative_log_likelihood == 0.0
    assert evaluation.multiclass_brier_score == 0.0
    assert evaluation.expected_calibration_error == 0.0
    assert evaluation.maximum_calibration_error == 0.0
    assert evaluation.nonempty_bins[0].sample_count == 2


def test_expected_calibration_error_uses_sample_weighted_top_label_bins() -> None:
    evaluation = evaluate_calibration(
        np.asarray([[0.9, 0.1], [0.9, 0.1]]),
        ["sitting", "standing"],
        CLASS_LABELS,
        bin_count=10,
    )

    assert evaluation.accuracy == 0.5
    assert evaluation.mean_confidence == pytest.approx(0.9)
    assert evaluation.expected_calibration_error == pytest.approx(0.4)
    assert evaluation.maximum_calibration_error == pytest.approx(0.4)
    assert evaluation.multiclass_brier_score == pytest.approx(0.82)


def test_probability_validation_rejects_non_normalized_or_unknown_labels() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_calibration(
            np.asarray([[0.8, 0.3]]),
            ["sitting"],
            CLASS_LABELS,
        )
    with pytest.raises(ValueError, match="outside the declared class set"):
        evaluate_calibration(
            np.asarray([[0.8, 0.2]]),
            ["walking"],
            CLASS_LABELS,
        )


def test_temperature_scaling_is_deterministic_and_does_not_change_argmax() -> None:
    logits = np.asarray(
        [
            [5.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [0.0, 5.0],
        ]
    )
    truth = ["sitting", "standing", "standing", "sitting"]
    participants = [
        "SYNTHETIC_VALIDATION_A",
        "SYNTHETIC_VALIDATION_A",
        "SYNTHETIC_VALIDATION_B",
        "SYNTHETIC_VALIDATION_B",
    ]

    first = fit_temperature_scaling(
        logits,
        truth,
        participants,
        CLASS_LABELS,
        scope=_synthetic_scope(),
    )
    second = fit_temperature_scaling(
        logits,
        truth,
        participants,
        CLASS_LABELS,
        scope=_synthetic_scope(),
    )
    unscaled = probabilities_from_logits(logits, expected_class_count=2)
    scaled = first.predict_probabilities(logits)

    assert first == second
    assert first.temperature > 1.0
    np.testing.assert_array_equal(unscaled.argmax(axis=1), scaled.argmax(axis=1))
    assert scaled.flags.writeable is False
    assert evaluate_calibration(
        scaled, truth, CLASS_LABELS
    ).negative_log_likelihood < evaluate_calibration(
        unscaled, truth, CLASS_LABELS
    ).negative_log_likelihood


def test_temperature_state_is_immutable_and_omits_subject_identifiers() -> None:
    model = fit_temperature_scaling(
        np.asarray([[2.0, 0.0], [0.0, 2.0]]),
        ["sitting", "standing"],
        ["SYNTHETIC_VALIDATION_A", "SYNTHETIC_VALIDATION_B"],
        CLASS_LABELS,
        scope=_synthetic_scope(),
    )

    state = model.state_dict()
    assert state["holdout_accessed_during_fit"] is False
    assert state["fit_subject_count"] == 2
    assert "SYNTHETIC_VALIDATION_A" not in repr(dict(state))
    with pytest.raises(TypeError):
        state["temperature"] = 1.0


def test_calibration_fit_rejects_observed_holdout_or_incomplete_scope() -> None:
    with pytest.raises(PermissionError, match="must equal the validation scope"):
        fit_temperature_scaling(
            np.asarray([[2.0, 0.0], [0.0, 2.0]]),
            ["sitting", "standing"],
            ["SYNTHETIC_VALIDATION_A", "SYNTHETIC_HOLDOUT_A"],
            CLASS_LABELS,
            scope=_synthetic_scope(),
        )


def test_synthetic_scope_rejects_real_identifiers() -> None:
    scope = CalibrationFitScope(
        purpose=CalibrationFitPurpose.SYNTHETIC_VALIDATION,
        validation_subjects=("P003",),
        holdout_subjects=("SYNTHETIC_HOLDOUT_A",),
    )

    with pytest.raises(ValueError, match="synthetic subject identifiers"):
        scope.validate()


def test_development_calibration_is_denied_while_protocol_gate_is_closed() -> None:
    protocol = replace(
        load_protocol(ROOT / "configs/mban_protocol.json"),
        training_authorized=False,
    )
    scope = CalibrationFitScope(
        purpose=CalibrationFitPurpose.DEVELOPMENT_VALIDATION,
        validation_subjects=(protocol.development_participants[0],),
        holdout_subjects=protocol.holdout_participants,
    )

    with pytest.raises(PermissionError, match="does not authorize scientific calibration"):
        scope.validate(protocol)
