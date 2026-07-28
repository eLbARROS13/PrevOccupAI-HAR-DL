"""Tests that preprocessing statistics cannot be fitted on held-out subjects."""

import numpy as np
import pytest

from prevoccupai_har.preprocessing import TrainOnlyChannelStandardizer


def test_standardizer_uses_training_statistics_for_validation_data() -> None:
    training = np.array(
        [
            [[0.0, 2.0], [2.0, 4.0]],
            [[4.0, 6.0], [6.0, 8.0]],
        ]
    )
    validation = np.array([[[10.0, 12.0], [12.0, 14.0]]])
    standardizer = TrainOnlyChannelStandardizer.for_subjects(["P003", "P004"])

    transformed_training = standardizer.fit_transform(training, ["P003", "P004"])
    transformed_validation = standardizer.transform(validation)

    np.testing.assert_allclose(transformed_training.mean(axis=(0, 1)), [0.0, 0.0])
    np.testing.assert_allclose(transformed_training.std(axis=(0, 1)), [1.0, 1.0])
    assert np.all(transformed_validation > 1.0)
    np.testing.assert_allclose(standardizer.mean_, [3.0, 5.0])


def test_standardizer_rejects_holdout_subject_during_fit() -> None:
    windows = np.zeros((2, 5, 3))
    standardizer = TrainOnlyChannelStandardizer.for_subjects(["P003"])

    with pytest.raises(ValueError, match="non-training subjects"):
        standardizer.fit(windows, ["P003", "P001"])


def test_standardizer_rejects_window_subject_count_mismatch() -> None:
    windows = np.zeros((2, 5, 3))
    standardizer = TrainOnlyChannelStandardizer.for_subjects(["P003"])

    with pytest.raises(ValueError, match="each window"):
        standardizer.fit(windows, ["P003"])

