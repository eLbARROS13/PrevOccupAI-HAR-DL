from pathlib import Path

import numpy as np
import pytest

from prevoccupai_har.quality import (
    QualityStatus,
    QualityThresholds,
    assess_accelerometer_quality,
    assess_axis_quality,
    detect_flatline_mask,
    detect_impulsive_spikes,
    detect_saturation_mask,
    detect_sequence_gaps,
    load_quality_configuration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reconstructed_configuration_is_fail_closed_and_uses_1000_hz() -> None:
    configuration = load_quality_configuration(
        ROOT / "configs" / "mban_qa_reconstruction.json"
    )

    assert configuration.authoritative is False
    assert configuration.controls_inclusion is False
    assert configuration.thresholds.sampling_rate_hz == 1000
    assert configuration.thresholds.sequence_counter_modulus == 32768
    assert configuration.thresholds.minimum_variance_m_s2_squared is None


def test_sequence_gap_fraction_uses_expected_sample_count() -> None:
    report = detect_sequence_gaps(np.array([0, 1, 4]))

    assert report.missing_samples == 2
    assert report.expected_samples == 5
    assert report.missing_fraction == pytest.approx(0.4)
    assert report.gap_after_indices == (1,)
    assert report.gap_sizes == (2,)


def test_sequence_gap_detection_accepts_32768_counter_wrap() -> None:
    report = detect_sequence_gaps(
        np.array([32_766, 32_767, 0, 1]),
        modulus=32_768,
    )

    assert report.missing_samples == 0
    assert report.expected_samples == 4
    assert report.missing_fraction == 0.0


@pytest.mark.parametrize("sequence", ([0, 1, 1], [0, 2, 1], [0, 1.5, 2]))
def test_sequence_counter_must_be_strictly_increasing_integers(sequence: list[float]) -> None:
    with pytest.raises(ValueError):
        detect_sequence_gaps(sequence)


def test_flatline_uses_full_1000_hz_duration_and_marks_complete_run() -> None:
    full_run = np.concatenate((np.array([0.0]), np.ones(10_000), np.array([2.0])))
    full_mask = detect_flatline_mask(
        full_run,
        sampling_rate_hz=1000,
        minimum_duration_seconds=10.0,
        absolute_tolerance_m_s2=1e-6,
    )
    short_run = np.concatenate((np.array([0.0]), np.ones(1_000), np.array([2.0])))
    short_mask = detect_flatline_mask(
        short_run,
        sampling_rate_hz=1000,
        minimum_duration_seconds=10.0,
        absolute_tolerance_m_s2=1e-6,
    )

    assert int(full_mask.sum()) == 10_000
    assert not short_mask.any()


def test_spike_detector_accepts_narrow_returning_impulse_and_rejects_wide_plateau() -> None:
    narrow = np.zeros(2_000)
    narrow[500] = 10.0
    narrow_report = detect_impulsive_spikes(
        narrow,
        sampling_rate_hz=1000,
        amplitude_threshold_m_s2=4.0,
        maximum_width_samples=3,
        local_std_max_m_s2=2.0,
        local_window_milliseconds=100.0,
        startup_sample_count=30,
        startup_amplitude_threshold_m_s2=10.0,
    )
    wide = np.zeros(2_000)
    wide[500:504] = 10.0
    wide_report = detect_impulsive_spikes(
        wide,
        sampling_rate_hz=1000,
        amplitude_threshold_m_s2=4.0,
        maximum_width_samples=3,
        local_std_max_m_s2=2.0,
        local_window_milliseconds=100.0,
        startup_sample_count=30,
        startup_amplitude_threshold_m_s2=10.0,
    )

    assert narrow_report.event_count == 1
    assert narrow_report.event_start_indices == (500,)
    assert narrow_report.sample_mask[500]
    assert wide_report.event_count == 0


def test_saturation_uses_physical_endpoints() -> None:
    limit = 8.0 * 9.80665
    mask = detect_saturation_mask(
        np.array([-limit, 0.0, limit, limit + 0.1]),
        physical_limit_m_s2=limit,
        absolute_tolerance_m_s2=1e-4,
    )

    assert mask.tolist() == [True, False, True, False]


def test_unset_variance_threshold_is_explicitly_undetermined() -> None:
    sequence = np.arange(2_000)
    signal = np.sin(np.linspace(0.0, 20.0, sequence.size))
    report = assess_axis_quality(signal, sequence, QualityThresholds())

    assert report.status == QualityStatus.UNDETERMINED
    assert report.variance_below_threshold is None
    assert report.undetermined_criteria == ("minimum_variance_not_calibrated",)
    serialized = report.to_dict()
    assert "variance_below_threshold" in serialized
    assert "variance_bellow_threshold" not in serialized


def test_configured_variance_allows_good_status_and_low_variance_fails() -> None:
    sequence = np.arange(2_000)
    healthy_signal = np.sin(np.linspace(0.0, 20.0, sequence.size))
    thresholds = QualityThresholds(minimum_variance_m_s2_squared=1e-4)
    healthy = assess_axis_quality(healthy_signal, sequence, thresholds)
    constant = assess_axis_quality(np.zeros(sequence.size), sequence, thresholds)

    assert healthy.status == QualityStatus.GOOD
    assert constant.status == QualityStatus.BAD
    assert "variance_below_threshold" in constant.failed_criteria


def test_nonfinite_samples_contribute_to_missingness_and_bad_status() -> None:
    sequence = np.arange(100)
    signal = np.linspace(-1.0, 1.0, 100)
    signal[:2] = np.nan
    thresholds = QualityThresholds(minimum_variance_m_s2_squared=0.0)
    report = assess_axis_quality(signal, sequence, thresholds)

    assert report.nonfinite_fraction == pytest.approx(0.02)
    assert report.missing_fraction == pytest.approx(0.02)
    assert report.status == QualityStatus.BAD
    assert "missing_fraction" in report.failed_criteria


def test_three_axis_status_is_bad_if_any_axis_fails() -> None:
    sequence = np.arange(2_000)
    base = np.sin(np.linspace(0.0, 20.0, sequence.size))
    samples = np.column_stack((base, np.zeros_like(base), base))
    thresholds = QualityThresholds(minimum_variance_m_s2_squared=1e-4)
    report = assess_accelerometer_quality(samples, sequence, thresholds)

    assert report.status == QualityStatus.BAD
    assert report.axes["x"].status == QualityStatus.GOOD
    assert report.axes["y"].status == QualityStatus.BAD
    assert report.axes["z"].status == QualityStatus.GOOD
