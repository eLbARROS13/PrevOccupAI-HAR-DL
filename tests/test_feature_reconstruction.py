"""Tests for the non-authoritative 45-feature TSFEL reconstruction."""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path

import numpy as np
import pytest

tsfel = pytest.importorskip("tsfel")

from prevoccupai_har.feature_reconstruction import (  # noqa: E402
    FeatureExtractionPurpose,
    extract_reconstructed_tsfel_features,
    load_tsfel_feature_reconstruction_configuration,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_PATH = ROOT / "configs/mban_tsfel_feature_reconstruction.json"
def _generated_window() -> np.ndarray:
    time_seconds = np.arange(5_000, dtype=np.float64) / 1_000.0
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * frequency_hz * time_seconds)
            + 0.01 * time_seconds
            + 0.001 * time_seconds**2
            for frequency_hz in (1.0, 2.0, 3.0)
        ]
    )


def test_configuration_freezes_recovered_feature_and_runtime_evidence() -> None:
    configuration = load_tsfel_feature_reconstruction_configuration(
        CONFIGURATION_PATH
    )

    assert configuration.authoritative is False
    assert configuration.controls_scientific_feature_generation is False
    assert configuration.sampling_rate_hz == 1_000
    assert configuration.window_samples == 5_000
    assert configuration.source_axis_names == ("x_ACC", "y_ACC", "z_ACC")
    assert configuration.axis_names == ("acc_0", "acc_1", "acc_2")
    assert configuration.axis_order_verified is False
    assert configuration.window_normalization == "none"
    assert configuration.required_tsfel_version == "0.1.9"
    assert configuration.expected_feature_count_per_axis == 15
    assert configuration.expected_total_feature_count == 45
    assert configuration.validation_reference_id == "three_sine_trend_window_v1"
    assert (
        configuration.exact_runtime_output_sha256
        == "41163f9f9898df509b359ca30e2200181416eb691fe0a4c0c892c87a9b7e343b"
    )
    assert tuple(definition.name for definition in configuration.ordered_features) == (
        "Human range energy",
        "Max power spectrum",
        "Median frequency",
        "Power bandwidth",
        "Spectral entropy",
        "Interquartile range",
        "Max",
        "Mean",
        "Median",
        "Min",
        "Root mean square",
        "Skewness",
        "Standard deviation",
        "Variance",
        "Mean absolute diff",
    )
    assert len(configuration.feature_names) == 45
    assert configuration.feature_names == tuple(sorted(configuration.feature_names))
    assert configuration.feature_names[:3] == (
        "acc_0_Human range energy",
        "acc_0_Interquartile range",
        "acc_0_Max",
    )


def test_direct_reconstruction_exactly_matches_tsfel_public_extractor() -> None:
    window = _generated_window()
    record = extract_reconstructed_tsfel_features(
        window[np.newaxis, :, :],
        configuration_path=CONFIGURATION_PATH,
        purpose=FeatureExtractionPurpose.SYNTHETIC_VALIDATION,
    )
    source_configuration = copy.deepcopy(tsfel.get_features_by_domain())
    selected_features = {
        definition.name
        for definition in load_tsfel_feature_reconstruction_configuration(
            CONFIGURATION_PATH
        ).ordered_features
    }
    for domain_features in source_configuration.values():
        for feature_name, feature_definition in domain_features.items():
            feature_definition["use"] = (
                "yes" if feature_name in selected_features else "no"
            )
    reference = tsfel.time_series_features_extractor(
        source_configuration,
        window[np.newaxis, :, :],
        fs=1_000,
        header_names=["acc_0", "acc_1", "acc_2"],
        verbose=0,
        n_jobs=None,
    )

    assert record.values.shape == (1, 45)
    assert record.values.flags.writeable is False
    assert record.scientific_result is False
    assert record.runtime_versions["tsfel"] == "0.1.9"
    with pytest.raises(TypeError):
        record.runtime_versions["tsfel"] = "tampered"  # type: ignore[index]
    assert record.feature_names == tuple(reference.columns)
    np.testing.assert_allclose(
        record.values,
        reference.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    mean_index = record.feature_names.index("acc_0_Mean")
    median_frequency_index = record.feature_names.index("acc_1_Median frequency")
    assert record.values[0, mean_index] == pytest.approx(np.mean(window[:, 0]))
    assert record.values[0, median_frequency_index] == pytest.approx(2.0)

    recovered_runtime = {
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "pandas": "2.1.4",
        "tsfel": "0.1.9",
    }
    if dict(record.runtime_versions) == recovered_runtime:
        canonical_values = record.values.astype("<f8", copy=False)
        assert hashlib.sha256(canonical_values.tobytes(order="C")).hexdigest() == (
            load_tsfel_feature_reconstruction_configuration(
                CONFIGURATION_PATH
            ).exact_runtime_output_sha256
        )


def test_scientific_extraction_is_refused_while_reconstruction_is_non_authoritative() -> None:
    with pytest.raises(PermissionError, match="cannot generate scientific features"):
        extract_reconstructed_tsfel_features(
            _generated_window()[np.newaxis, :, :],
            configuration_path=CONFIGURATION_PATH,
            purpose=FeatureExtractionPurpose.SCIENTIFIC_DATASET,
        )


@pytest.mark.parametrize(
    "windows",
    (
        np.zeros((1, 4_999, 3), dtype=np.float64),
        np.zeros((1, 5_000, 2), dtype=np.float64),
        np.zeros((0, 5_000, 3), dtype=np.float64),
        np.zeros((5_000, 3), dtype=np.float64),
    ),
)
def test_input_shape_must_be_exact(windows: np.ndarray) -> None:
    with pytest.raises(ValueError, match="5,000 samples x 3 axes"):
        extract_reconstructed_tsfel_features(
            windows,
            configuration_path=CONFIGURATION_PATH,
            purpose=FeatureExtractionPurpose.SYNTHETIC_VALIDATION,
        )


def test_input_and_output_must_be_finite() -> None:
    nonfinite = _generated_window()[np.newaxis, :, :]
    nonfinite[0, 10, 1] = np.nan
    with pytest.raises(ValueError, match="input windows must be finite"):
        extract_reconstructed_tsfel_features(
            nonfinite,
            configuration_path=CONFIGURATION_PATH,
            purpose=FeatureExtractionPurpose.SYNTHETIC_VALIDATION,
        )

    with pytest.raises(ValueError, match="produced non-finite features"):
        extract_reconstructed_tsfel_features(
            np.zeros((1, 5_000, 3), dtype=np.float64),
            configuration_path=CONFIGURATION_PATH,
            purpose=FeatureExtractionPurpose.SYNTHETIC_VALIDATION,
        )
