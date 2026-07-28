"""Versioned signal preprocessing for muscleBAN accelerometer segments.

The primary transform intentionally reproduces the recovered conference code:
an 11-sample median filter, a causal third-order 20 Hz Butterworth low-pass,
a causal third-order 0.3 Hz gravity estimate, and gravity subtraction.  Median
zero padding and zero-state causal filter transients are part of the declared
method rather than hidden implementation details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal


FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class SignalPreprocessingConfiguration:
    """Versioned parameters and governance state for ACC preprocessing."""

    schema_version: int
    name: str
    authoritative: bool
    controls_dataset_generation: bool
    source: Mapping[str, Any]
    notes: tuple[str, ...]
    sampling_rate_hz: int
    median_kernel_samples: int
    butterworth_order: int
    motion_lowpass_cutoff_hz: float
    gravity_lowpass_cutoff_hz: float
    filter_application: str
    median_boundary: str
    sos_initial_state: str
    normalization: str

    def __post_init__(self) -> None:
        """Reject unsupported or internally inconsistent configurations."""
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported preprocessing schema version: {self.schema_version}"
            )
        if self.controls_dataset_generation and not self.authoritative:
            raise ValueError(
                "a non-authoritative configuration cannot control dataset generation"
            )
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.median_kernel_samples < 1 or self.median_kernel_samples % 2 == 0:
            raise ValueError("median_kernel_samples must be a positive odd integer")
        if self.butterworth_order < 1:
            raise ValueError("butterworth_order must be positive")
        nyquist_hz = self.sampling_rate_hz / 2.0
        for name in (
            "motion_lowpass_cutoff_hz",
            "gravity_lowpass_cutoff_hz",
        ):
            cutoff_hz = float(getattr(self, name))
            if not 0.0 < cutoff_hz < nyquist_hz:
                raise ValueError(f"{name} must lie strictly between zero and Nyquist")
        if self.gravity_lowpass_cutoff_hz >= self.motion_lowpass_cutoff_hz:
            raise ValueError(
                "gravity_lowpass_cutoff_hz must be below "
                "motion_lowpass_cutoff_hz"
            )
        if self.filter_application != "causal_sos":
            raise ValueError("only causal_sos filtering is currently supported")
        if self.median_boundary != "zero_padding":
            raise ValueError("only the recovered zero-padding median boundary is supported")
        if self.sos_initial_state != "zeros":
            raise ValueError("only the recovered zero SOS initial state is supported")
        if self.normalization != "none":
            raise ValueError(
                "signal-level normalization is disabled; learned scaling must be fit "
                "on training participants only"
            )


@dataclass(frozen=True)
class PreprocessedAccelerometerSegment:
    """Intermediate and final arrays from one continuous ACC segment."""

    median_filtered_m_s2: FloatMatrix
    motion_lowpass_m_s2: FloatMatrix
    gravity_component_m_s2: FloatMatrix
    dynamic_acceleration_m_s2: FloatMatrix
    configuration_name: str
    sampling_rate_hz: int


def load_signal_preprocessing_configuration(
    path: Path | str,
) -> SignalPreprocessingConfiguration:
    """Load and validate the versioned preprocessing JSON configuration."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("preprocessing configuration must be a JSON object")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("preprocessing configuration must contain a parameters object")
    return SignalPreprocessingConfiguration(
        schema_version=int(payload["schema_version"]),
        name=str(payload["name"]),
        authoritative=bool(payload["authoritative"]),
        controls_dataset_generation=bool(payload["controls_dataset_generation"]),
        source=dict(payload["source"]),
        notes=tuple(str(note) for note in payload.get("notes", ())),
        sampling_rate_hz=int(parameters["sampling_rate_hz"]),
        median_kernel_samples=int(parameters["median_kernel_samples"]),
        butterworth_order=int(parameters["butterworth_order"]),
        motion_lowpass_cutoff_hz=float(
            parameters["motion_lowpass_cutoff_hz"]
        ),
        gravity_lowpass_cutoff_hz=float(
            parameters["gravity_lowpass_cutoff_hz"]
        ),
        filter_application=str(parameters["filter_application"]),
        median_boundary=str(parameters["median_boundary"]),
        sos_initial_state=str(parameters["sos_initial_state"]),
        normalization=str(parameters["normalization"]),
    )


def _accelerometer_matrix(values: ArrayLike, *, minimum_samples: int) -> FloatMatrix:
    """Return a copied, finite sample-by-three-channel matrix."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("accelerometer data must have shape (samples, 3)")
    if samples.shape[0] < minimum_samples:
        raise ValueError(
            "accelerometer segment is shorter than the median-filter kernel"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("accelerometer data must be finite before preprocessing")
    return np.array(samples, dtype=np.float64, copy=True)


def _median_filter_channels(samples: FloatMatrix, kernel_size: int) -> FloatMatrix:
    """Apply SciPy's zero-padded median filter independently by channel."""
    filtered = np.empty_like(samples)
    for channel_index in range(samples.shape[1]):
        filtered[:, channel_index] = signal.medfilt(
            samples[:, channel_index], kernel_size=kernel_size
        )
    return filtered


def _causal_lowpass_channels(
    samples: FloatMatrix,
    *,
    sampling_rate_hz: int,
    cutoff_hz: float,
    order: int,
) -> FloatMatrix:
    """Apply a zero-state causal Butterworth SOS filter by channel."""
    second_order_sections = signal.butter(
        order,
        cutoff_hz,
        fs=sampling_rate_hz,
        output="sos",
    )
    filtered = np.empty_like(samples)
    for channel_index in range(samples.shape[1]):
        filtered[:, channel_index] = signal.sosfilt(
            second_order_sections, samples[:, channel_index]
        )
    return filtered


def _read_only(values: FloatMatrix) -> FloatMatrix:
    """Return a read-only float copy suitable for an immutable result record."""
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def preprocess_accelerometer_segment(
    samples_m_s2: ArrayLike,
    configuration: SignalPreprocessingConfiguration,
) -> PreprocessedAccelerometerSegment:
    """Preprocess one continuous tri-axial segment before window extraction.

    Applying this transform independently to each overlapping window would
    recreate causal startup transients in every window and is therefore not a
    valid reproduction of the recovered segment-level pipeline.
    """
    samples = _accelerometer_matrix(
        samples_m_s2,
        minimum_samples=configuration.median_kernel_samples,
    )
    median_filtered = _median_filter_channels(
        samples,
        configuration.median_kernel_samples,
    )
    motion_lowpass = _causal_lowpass_channels(
        median_filtered,
        sampling_rate_hz=configuration.sampling_rate_hz,
        cutoff_hz=configuration.motion_lowpass_cutoff_hz,
        order=configuration.butterworth_order,
    )
    gravity_component = _causal_lowpass_channels(
        motion_lowpass,
        sampling_rate_hz=configuration.sampling_rate_hz,
        cutoff_hz=configuration.gravity_lowpass_cutoff_hz,
        order=configuration.butterworth_order,
    )
    dynamic_acceleration = motion_lowpass - gravity_component

    return PreprocessedAccelerometerSegment(
        median_filtered_m_s2=_read_only(median_filtered),
        motion_lowpass_m_s2=_read_only(motion_lowpass),
        gravity_component_m_s2=_read_only(gravity_component),
        dynamic_acceleration_m_s2=_read_only(dynamic_acceleration),
        configuration_name=configuration.name,
        sampling_rate_hz=configuration.sampling_rate_hz,
    )
