import json
from pathlib import Path

import numpy as np
import pytest

from prevoccupai_har.opensignals import (
    load_opensignals_accelerometer,
    read_opensignals_accelerometer_header,
    unsigned_adc_to_acceleration_m_s2,
)


def _write_opensignals_file(
    path: Path,
    rows: list[tuple[int, int, int, int]],
    *,
    sampling_rate_hz: int = 1000,
    device_type: str = "musclebanplux",
) -> None:
    columns = ["nSeq", "DI", "gEMG", "gACC", "gACC", "gACC", "pMAG"]
    payload = {
        "TEST_DEVICE": {
            "device": device_type,
            "sampling rate": sampling_rate_hz,
            "column": columns,
            "resolution": [16, 16, 16, 16, 16],
        }
    }
    lines = [
        "# OpenSignals Text File Format",
        f"# {json.dumps(payload, separators=(',', ':'))}",
        "# EndOfHeader",
    ]
    for sequence, acc_0, acc_1, acc_2 in rows:
        lines.append(
            f"{sequence}\t0\t32768\t{acc_0}\t{acc_1}\t{acc_2}\t32768"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_header_extracts_only_non_identifying_ingestion_fields(tmp_path: Path) -> None:
    path = tmp_path / "recording.txt"
    _write_opensignals_file(path, [(0, 0, 32768, 65535)])

    header = read_opensignals_accelerometer_header(path)

    assert header.device_type == "musclebanplux"
    assert header.sampling_rate_hz == 1000
    assert header.accelerometer_column_indices == (3, 4, 5)
    assert header.accelerometer_resolution_bits == (16, 16, 16)
    assert header.axis_names == ("acc_0", "acc_1", "acc_2")
    assert header.axis_order_verified is False
    assert not hasattr(header, "device_identifier")
    assert not hasattr(header, "timestamp")


def test_unsigned_adc_conversion_has_expected_zero_and_endpoints() -> None:
    gravity = 9.80665
    converted = unsigned_adc_to_acceleration_m_s2(
        np.array([0, 32768, 65535]),
        resolution_bits=16,
        acceleration_range_g=8.0,
        gravity_m_s2=gravity,
    )

    assert converted[0] == pytest.approx(-8.0 * gravity)
    assert converted[1] == pytest.approx(0.0)
    assert converted[2] == pytest.approx((8.0 - 16.0 / 65536.0) * gravity)


def test_loader_returns_acceleration_matrix_without_source_identity(tmp_path: Path) -> None:
    path = tmp_path / "recording.txt"
    _write_opensignals_file(
        path,
        [
            (32767, 0, 32768, 65535),
            (0, 32768, 32768, 32768),
        ],
    )

    recording = load_opensignals_accelerometer(path)

    assert recording.sequence_numbers.tolist() == [32767, 0]
    assert recording.samples_m_s2.shape == (2, 3)
    assert recording.samples_m_s2[1].tolist() == [0.0, 0.0, 0.0]
    assert recording.axis_order_verified is False
    assert recording.sequence_counter_modulus == 32768
    assert not hasattr(recording, "path")
    assert not hasattr(recording, "device_identifier")


def test_loader_rejects_sampling_rate_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "recording.txt"
    _write_opensignals_file(path, [(0, 32768, 32768, 32768)], sampling_rate_hz=100)

    with pytest.raises(ValueError, match="expected 1000 Hz"):
        load_opensignals_accelerometer(path)


def test_header_rejects_non_muscleban_device(tmp_path: Path) -> None:
    path = tmp_path / "recording.txt"
    _write_opensignals_file(
        path,
        [(0, 32768, 32768, 32768)],
        device_type="smartphone",
    )

    with pytest.raises(ValueError, match="expected musclebanplux"):
        read_opensignals_accelerometer_header(path)


@pytest.mark.parametrize("bad_value", [-1, 65536, 1.5, np.nan])
def test_adc_conversion_rejects_invalid_codes(bad_value: float) -> None:
    with pytest.raises(ValueError):
        unsigned_adc_to_acceleration_m_s2(
            np.array([bad_value]),
            resolution_bits=16,
            acceleration_range_g=8.0,
        )
