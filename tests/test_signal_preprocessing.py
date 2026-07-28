from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from prevoccupai_har.signal_preprocessing import (
    SignalPreprocessingConfiguration,
    load_signal_preprocessing_configuration,
    preprocess_accelerometer_segment,
)


CONFIGURATION_PATH = Path("configs/mban_signal_preprocessing.json")


@pytest.fixture
def configuration() -> SignalPreprocessingConfiguration:
    return load_signal_preprocessing_configuration(CONFIGURATION_PATH)


def _reference_preprocessing(
    samples: np.ndarray,
    configuration: SignalPreprocessingConfiguration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median_filtered = np.column_stack(
        [
            signal.medfilt(
                samples[:, channel],
                kernel_size=configuration.median_kernel_samples,
            )
            for channel in range(samples.shape[1])
        ]
    )
    motion_filter = signal.butter(
        configuration.butterworth_order,
        configuration.motion_lowpass_cutoff_hz,
        fs=configuration.sampling_rate_hz,
        output="sos",
    )
    motion_lowpass = np.column_stack(
        [
            signal.sosfilt(motion_filter, median_filtered[:, channel])
            for channel in range(samples.shape[1])
        ]
    )
    gravity_filter = signal.butter(
        configuration.butterworth_order,
        configuration.gravity_lowpass_cutoff_hz,
        fs=configuration.sampling_rate_hz,
        output="sos",
    )
    gravity = np.column_stack(
        [
            signal.sosfilt(gravity_filter, motion_lowpass[:, channel])
            for channel in range(samples.shape[1])
        ]
    )
    return median_filtered, motion_lowpass, gravity, motion_lowpass - gravity


def test_configuration_records_recovered_causal_semantics(
    configuration: SignalPreprocessingConfiguration,
) -> None:
    assert configuration.authoritative is True
    assert configuration.controls_dataset_generation is True
    assert configuration.sampling_rate_hz == 1000
    assert configuration.median_kernel_samples == 11
    assert configuration.butterworth_order == 3
    assert configuration.motion_lowpass_cutoff_hz == 20.0
    assert configuration.gravity_lowpass_cutoff_hz == 0.3
    assert configuration.filter_application == "causal_sos"
    assert configuration.median_boundary == "zero_padding"
    assert configuration.sos_initial_state == "zeros"
    assert configuration.normalization == "none"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("median_kernel_samples", 10, "positive odd"),
        ("butterworth_order", 0, "positive"),
        ("motion_lowpass_cutoff_hz", 500.0, "Nyquist"),
        ("gravity_lowpass_cutoff_hz", 25.0, "must be below"),
        ("filter_application", "zero_phase", "causal_sos"),
        ("normalization", "global_max", "normalization is disabled"),
    ],
)
def test_configuration_rejects_unsupported_variants(
    configuration: SignalPreprocessingConfiguration,
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(configuration, **{field: value})


def test_configuration_cannot_be_made_non_authoritative_while_controlling_generation(
    configuration: SignalPreprocessingConfiguration,
) -> None:
    with pytest.raises(ValueError, match="cannot control dataset generation"):
        replace(configuration, authoritative=False)


def test_transform_matches_direct_scipy_reference_exactly(
    configuration: SignalPreprocessingConfiguration,
) -> None:
    sample_index = np.arange(301, dtype=np.float64)
    samples = np.column_stack(
        (
            np.sin(sample_index / 11.0),
            np.cos(sample_index / 17.0) + 0.02 * sample_index,
            np.where(sample_index == 120, 5.0, sample_index / 100.0),
        )
    )
    expected = _reference_preprocessing(samples, configuration)

    result = preprocess_accelerometer_segment(samples, configuration)

    np.testing.assert_array_equal(result.median_filtered_m_s2, expected[0])
    np.testing.assert_array_equal(result.motion_lowpass_m_s2, expected[1])
    np.testing.assert_array_equal(result.gravity_component_m_s2, expected[2])
    np.testing.assert_array_equal(result.dynamic_acceleration_m_s2, expected[3])


def test_transform_does_not_modify_input_and_returns_read_only_arrays(
    configuration: SignalPreprocessingConfiguration,
) -> None:
    samples = np.arange(99, dtype=np.float64).reshape(33, 3)
    original = samples.copy()

    result = preprocess_accelerometer_segment(samples, configuration)

    np.testing.assert_array_equal(samples, original)
    assert result.dynamic_acceleration_m_s2.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        result.dynamic_acceleration_m_s2[0, 0] = 0.0


@pytest.mark.parametrize(
    "samples",
    [
        np.ones(30),
        np.ones((30, 2)),
        np.ones((10, 3)),
        np.array([[1.0, 2.0, np.nan]] * 11),
    ],
)
def test_transform_rejects_invalid_accelerometer_arrays(
    configuration: SignalPreprocessingConfiguration,
    samples: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        preprocess_accelerometer_segment(samples, configuration)


def test_gravity_subtraction_removes_steady_constant_component(
    configuration: SignalPreprocessingConfiguration,
) -> None:
    samples = np.full((30_000, 3), 9.80665)

    result = preprocess_accelerometer_segment(samples, configuration)

    steady_state = result.dynamic_acceleration_m_s2[20_000:29_000]
    assert np.max(np.abs(steady_state)) < 1e-7


def test_motion_lowpass_attenuates_high_frequency_more_than_low_frequency(
    configuration: SignalPreprocessingConfiguration,
) -> None:
    time_seconds = np.arange(30_000) / configuration.sampling_rate_hz
    low_frequency = np.sin(2.0 * np.pi * 5.0 * time_seconds)
    high_frequency = np.sin(2.0 * np.pi * 100.0 * time_seconds)
    samples = np.column_stack((low_frequency, high_frequency, low_frequency))

    result = preprocess_accelerometer_segment(samples, configuration)
    stable = result.motion_lowpass_m_s2[10_000:29_000]
    low_rms = float(np.sqrt(np.mean(np.square(stable[:, 0]))))
    high_rms = float(np.sqrt(np.mean(np.square(stable[:, 1]))))

    assert low_rms > 0.6
    assert high_rms < 0.1 * low_rms
