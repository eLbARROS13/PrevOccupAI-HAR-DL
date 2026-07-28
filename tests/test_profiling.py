"""Synthetic tests for model-complexity and latency profiling."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from prevoccupai_har.modeling import (  # noqa: E402
    CompactCnn1D,
    CompactCnnArchitecture,
    build_compact_cnn_1d,
    build_compact_tcn,
    load_compact_cnn_experiment_configuration,
    load_compact_tcn_experiment_configuration,
)
from prevoccupai_har.profiling import (  # noqa: E402
    build_synthetic_complexity_profile_record,
    profile_model_complexity,
    write_synthetic_complexity_profile_record,
)
from prevoccupai_har.provenance import sha256_file  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _small_model() -> CompactCnn1D:
    return CompactCnn1D(
        CompactCnnArchitecture(
            input_channels=3,
            expected_samples=64,
            convolution_channels=(4, 8),
            kernel_sizes=(7, 5),
            strides=(2, 2),
            max_pool_sizes=(2, 2),
            batch_normalization=True,
            dropout_probability=0.1,
            initialization="pytorch_module_defaults_seeded",
            output_classes=3,
        )
    )


def test_profile_measurement_is_finite_and_restores_training_mode() -> None:
    model = _small_model()
    model.train()

    measurement = profile_model_complexity(
        model,
        input_shape=(1, 3, 64),
        warmup_iterations=1,
        timed_iterations=4,
    )

    assert model.training is True
    assert measurement.parameter_count == measurement.trainable_parameter_count
    assert measurement.parameter_count > 0
    assert measurement.serialized_state_dict_bytes > 0
    assert measurement.latency.minimum_ms <= measurement.latency.median_ms
    assert measurement.latency.median_ms <= measurement.latency.maximum_ms


def test_profile_rejects_invalid_iteration_counts() -> None:
    with pytest.raises(ValueError, match="timed count must be positive"):
        profile_model_complexity(
            _small_model(),
            input_shape=(1, 3, 64),
            timed_iterations=0,
        )


def test_profile_record_is_non_scientific_and_exclusive(tmp_path: Path) -> None:
    measurement = profile_model_complexity(
        _small_model(),
        input_shape=(1, 3, 64),
        warmup_iterations=0,
        timed_iterations=2,
    )
    record = build_synthetic_complexity_profile_record(
        created_at_utc="2026-07-15T18:30:00Z",
        experiment_id="synthetic-small-model",
        source_revision="unversioned_workspace_software_test",
        model_configuration_path=ROOT / "configs/cnn_1d.json",
        random_seed=1103,
        device="cpu",
        measurement=measurement,
    )
    output_path = tmp_path / "profile.json"

    write_synthetic_complexity_profile_record(output_path, record)

    saved = output_path.read_text(encoding="utf-8")
    assert '"scientific_result": false' in saved
    assert '"holdout_accessed": false' in saved
    assert '"input_kind": "synthetic_zeros"' in saved
    with pytest.raises(FileExistsError):
        write_synthetic_complexity_profile_record(output_path, record)


def test_checked_in_profile_matches_current_model_contract() -> None:
    config_path = ROOT / "configs/cnn_1d.json"
    artifact_path = (
        ROOT / "artifacts/software_validation/cnn_1d_100hz_development_complexity_cpu.json"
    )
    configuration = load_compact_cnn_experiment_configuration(config_path)
    model = build_compact_cnn_1d(configuration)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["scientific_result"] is False
    assert artifact["holdout_accessed"] is False
    assert artifact["input_kind"] == "synthetic_zeros"
    assert artifact["model_configuration_sha256"] == sha256_file(config_path)
    assert artifact["measurement"]["input_shape"] == [1, 3, 500]
    assert artifact["measurement"]["trainable_parameter_count"] == (
        model.trainable_parameter_count
    )


def test_checked_in_tcn_profile_matches_current_model_contract() -> None:
    config_path = ROOT / "configs/tcn_1d.json"
    artifact_path = (
        ROOT / "artifacts/software_validation/tcn_1d_100hz_compact_development_complexity_cpu.json"
    )
    configuration = load_compact_tcn_experiment_configuration(config_path)
    model = build_compact_tcn(configuration)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["scientific_result"] is False
    assert artifact["holdout_accessed"] is False
    assert artifact["input_kind"] == "synthetic_zeros"
    assert artifact["model_configuration_sha256"] == sha256_file(config_path)
    assert artifact["measurement"]["input_shape"] == [1, 3, 500]
    assert artifact["measurement"]["trainable_parameter_count"] == (
        model.trainable_parameter_count
    )
