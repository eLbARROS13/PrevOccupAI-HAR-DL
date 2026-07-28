"""Tests for metric derivation from the published RF confusion matrix."""

import numpy as np

import pytest

from prevoccupai_har.evaluation import (
    confusion_matrix_from_predictions,
    evaluate_predictions,
    metrics_from_confusion_matrix,
)
from prevoccupai_har.statistical_evaluation import (
    compare_paired_participant_metrics,
    exact_paired_sign_flip_p_value,
    holm_adjust,
    participant_bootstrap_mean_interval,
)


def test_published_rf_confusion_matrix_reproduces_reported_metrics() -> None:
    matrix = np.array(
        [
            [703, 275, 2],
            [71, 895, 10],
            [0, 36, 944],
        ]
    )

    metrics = metrics_from_confusion_matrix(matrix, ["sitting", "standing", "walking"])

    np.testing.assert_allclose(
        metrics["accuracy"],
        0.8658038147138964,
    )
    assert metrics["support"] == 2936
    np.testing.assert_allclose(metrics["balanced_accuracy"], 0.865873480539757)
    np.testing.assert_allclose(metrics["macro_f1"], 0.8657170890251619)
    np.testing.assert_allclose(metrics["weighted_f1"], 0.8657788993585427)
    np.testing.assert_allclose(metrics["classwise"]["sitting"]["recall"], 0.7173469387755103)


def test_prediction_evaluation_preserves_participant_pairing() -> None:
    truth = ["sitting", "standing", "walking"] * 2
    predictions = ["sitting", "sitting", "walking", "standing", "standing", "walking"]
    participants = ["P001"] * 3 + ["P002"] * 3

    evaluation = evaluate_predictions(
        truth,
        predictions,
        participants,
        ["sitting", "standing", "walking"],
    )

    assert evaluation.confusion_matrix == ((1, 1, 0), (1, 1, 0), (0, 0, 2))
    assert set(evaluation.per_participant_metrics) == {"P001", "P002"}
    assert evaluation.per_participant_metrics["P001"]["support"] == 3
    assert evaluation.per_participant_metrics["P002"]["support"] == 3


def test_prediction_confusion_rejects_undeclared_labels() -> None:
    with pytest.raises(ValueError, match="outside the declared class set"):
        confusion_matrix_from_predictions(
            ["sitting"],
            ["unknown"],
            ["sitting", "standing", "walking"],
        )


def test_participant_bootstrap_is_deterministic_and_flags_four_subjects() -> None:
    values = {"P001": 0.80, "P002": 0.84, "P016": 0.88, "P018": 0.92}

    first = participant_bootstrap_mean_interval(values, resample_count=2_000, random_seed=7)
    second = participant_bootstrap_mean_interval(values, resample_count=2_000, random_seed=7)

    assert first == second
    assert first.estimate == pytest.approx(0.86)
    assert first.lower <= first.estimate <= first.upper
    assert first.interpretation == "highly_unstable_descriptive_interval"


def test_paired_comparison_uses_exact_same_participants() -> None:
    candidate = {"P001": 0.82, "P002": 0.86, "P016": 0.90, "P018": 0.94}
    reference = {"P001": 0.80, "P002": 0.84, "P016": 0.88, "P018": 0.92}

    comparison = compare_paired_participant_metrics(
        candidate,
        reference,
        resample_count=1_000,
        random_seed=11,
    )

    assert comparison.mean_difference == pytest.approx(0.02)
    assert comparison.positive_participant_fraction == 1.0
    assert comparison.exact_sign_flip_p_value == 0.125
    with pytest.raises(ValueError, match="must match exactly"):
        compare_paired_participant_metrics(candidate, {"P001": 0.80})


def test_sign_flip_and_holm_edge_cases() -> None:
    assert exact_paired_sign_flip_p_value([0.0, 0.0, 0.0, 0.0]) == 1.0
    assert holm_adjust([0.03, 0.04]) == pytest.approx((0.06, 0.06))
