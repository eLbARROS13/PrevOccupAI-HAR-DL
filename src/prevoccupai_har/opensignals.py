"""Privacy-conscious ingestion for raw muscleBAN OpenSignals text files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatMatrix = NDArray[np.float64]
IntegerVector = NDArray[np.int64]


@dataclass(frozen=True)
class OpenSignalsAccelerometerHeader:
    """Non-identifying header fields required for ACC ingestion."""

    device_type: str
    sampling_rate_hz: int
    columns: tuple[str, ...]
    accelerometer_column_indices: tuple[int, int, int]
    accelerometer_resolution_bits: tuple[int, int, int]
    axis_names: tuple[str, str, str] = ("acc_0", "acc_1", "acc_2")
    axis_order_verified: bool = False


@dataclass(frozen=True)
class OpenSignalsAccelerometerRecording:
    """One decoded ACC stream without path, device, or timestamp metadata."""

    sequence_numbers: IntegerVector
    samples_m_s2: FloatMatrix
    sampling_rate_hz: int
    axis_names: tuple[str, str, str]
    axis_order_verified: bool
    acceleration_range_g: float
    adc_resolution_bits: int
    sequence_counter_modulus: int


def _parse_header_payload(lines: list[str]) -> Mapping[str, Any]:
    """Parse the unique device payload without returning its identifier."""
    if len(lines) < 3:
        raise ValueError("OpenSignals header must contain at least three lines")
    if lines[0].strip() != "# OpenSignals Text File Format":
        raise ValueError("unrecognized OpenSignals file signature")
    if lines[2].strip() != "# EndOfHeader":
        raise ValueError("OpenSignals header terminator is missing")
    metadata_line = lines[1].strip()
    if not metadata_line.startswith("#"):
        raise ValueError("OpenSignals metadata line must be a comment")
    metadata = json.loads(metadata_line[1:].strip())
    if not isinstance(metadata, dict) or len(metadata) != 1:
        raise ValueError("expected exactly one device payload in OpenSignals header")
    payload = next(iter(metadata.values()))
    if not isinstance(payload, dict):
        raise ValueError("OpenSignals device payload must be a mapping")
    return payload


def read_opensignals_accelerometer_header(
    path: Path,
) -> OpenSignalsAccelerometerHeader:
    """Read only the non-identifying fields needed to locate ACC channels."""
    with path.open("r", encoding="utf-8") as stream:
        header_lines = [stream.readline() for _ in range(3)]
    payload = _parse_header_payload(header_lines)

    device_type = str(payload.get("device", "")).lower()
    if device_type != "musclebanplux":
        raise ValueError(f"expected musclebanplux device, found {device_type!r}")
    sampling_rate_hz = int(payload["sampling rate"])
    columns = tuple(str(column) for column in payload["column"])
    accelerometer_indices = tuple(
        index for index, label in enumerate(columns) if label == "gACC"
    )
    if len(accelerometer_indices) != 3:
        raise ValueError(
            "expected exactly three gACC columns, "
            f"found {len(accelerometer_indices)}"
        )
    if any(index < 2 for index in accelerometer_indices):
        raise ValueError("gACC columns must follow nSeq and DI")

    resolutions = tuple(int(value) for value in payload["resolution"])
    signal_column_offset = 2  # nSeq and DI precede the sensor channels.
    try:
        accelerometer_resolutions = tuple(
            resolutions[index - signal_column_offset]
            for index in accelerometer_indices
        )
    except IndexError as exc:
        raise ValueError("resolution metadata does not cover the gACC columns") from exc
    if len(set(accelerometer_resolutions)) != 1:
        raise ValueError("gACC channels do not share one ADC resolution")

    return OpenSignalsAccelerometerHeader(
        device_type=device_type,
        sampling_rate_hz=sampling_rate_hz,
        columns=columns,
        accelerometer_column_indices=cast(
            tuple[int, int, int], accelerometer_indices
        ),
        accelerometer_resolution_bits=cast(
            tuple[int, int, int], accelerometer_resolutions
        ),
    )


def unsigned_adc_to_acceleration_m_s2(
    values: ArrayLike,
    *,
    resolution_bits: int,
    acceleration_range_g: float,
    gravity_m_s2: float = 9.80665,
) -> FloatMatrix:
    """Convert unsigned ADC codes spanning +/-range to acceleration in SI units."""
    if resolution_bits < 2:
        raise ValueError("resolution_bits must be at least two")
    if acceleration_range_g <= 0 or gravity_m_s2 <= 0:
        raise ValueError("acceleration range and gravity must be positive")
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim not in (1, 2):
        raise ValueError("values must be a vector or matrix")
    if raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("ADC values must be non-empty and finite")
    if not np.array_equal(raw, np.rint(raw)):
        raise ValueError("ADC values must be integer-valued")

    level_count = 2**resolution_bits
    if np.any(raw < 0) or np.any(raw > level_count - 1):
        raise ValueError("ADC values fall outside the declared unsigned range")
    midpoint = level_count / 2.0
    acceleration_g = (raw - midpoint) * (2.0 * acceleration_range_g / level_count)
    return np.asarray(acceleration_g * gravity_m_s2, dtype=np.float64)


def load_opensignals_accelerometer(
    path: Path,
    *,
    expected_sampling_rate_hz: int = 1000,
    acceleration_range_g: float = 8.0,
    sequence_counter_modulus: int = 32768,
) -> OpenSignalsAccelerometerRecording:
    """Load sequence and ACC columns while discarding identifying metadata."""
    header = read_opensignals_accelerometer_header(path)
    if header.sampling_rate_hz != expected_sampling_rate_hz:
        raise ValueError(
            f"expected {expected_sampling_rate_hz} Hz, "
            f"found {header.sampling_rate_hz} Hz"
        )
    resolution_bits = header.accelerometer_resolution_bits[0]
    use_columns = (0, *header.accelerometer_column_indices)
    raw = np.loadtxt(
        path,
        dtype=np.float64,
        comments="#",
        delimiter="\t",
        usecols=use_columns,
        ndmin=2,
    )
    if raw.shape[0] == 0:
        raise ValueError("OpenSignals file contains no samples")

    sequence_float = raw[:, 0]
    if not np.all(np.isfinite(sequence_float)) or not np.array_equal(
        sequence_float, np.rint(sequence_float)
    ):
        raise ValueError("sequence counter must be finite and integer-valued")
    sequence_numbers = np.asarray(np.rint(sequence_float), dtype=np.int64)
    if np.any(sequence_numbers < 0) or np.any(
        sequence_numbers >= sequence_counter_modulus
    ):
        raise ValueError("sequence counter falls outside the declared modulus")

    samples_m_s2 = unsigned_adc_to_acceleration_m_s2(
        raw[:, 1:],
        resolution_bits=resolution_bits,
        acceleration_range_g=acceleration_range_g,
    )
    return OpenSignalsAccelerometerRecording(
        sequence_numbers=sequence_numbers,
        samples_m_s2=samples_m_s2,
        sampling_rate_hz=header.sampling_rate_hz,
        axis_names=header.axis_names,
        axis_order_verified=header.axis_order_verified,
        acceleration_range_g=acceleration_range_g,
        adc_resolution_bits=resolution_bits,
        sequence_counter_modulus=sequence_counter_modulus,
    )
