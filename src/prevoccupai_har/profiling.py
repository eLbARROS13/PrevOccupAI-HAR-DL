"""Synthetic-only model-complexity and inference-latency profiling."""

from __future__ import annotations

import io
import json
import math
import platform
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised only without the DL extra
    raise ImportError(
        "PyTorch is required for prevoccupai_har.profiling; install the 'dl' extra"
    ) from error

from .provenance import sha256_file


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class LatencySummary:
    """Distribution summary for synchronized single-batch forward passes."""

    median_ms: float
    first_quartile_ms: float
    third_quartile_ms: float
    minimum_ms: float
    maximum_ms: float

    def validate(self) -> None:
        """Reject non-finite, negative, or misordered timing summaries."""
        values = (
            self.minimum_ms,
            self.first_quartile_ms,
            self.median_ms,
            self.third_quartile_ms,
            self.maximum_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Latency values must be finite and non-negative")
        if tuple(sorted(values)) != values:
            raise ValueError("Latency quantiles are not monotonically ordered")


@dataclass(frozen=True)
class ModelComplexityMeasurement:
    """Architecture size and device-specific synthetic forward-pass measurement."""

    input_shape: tuple[int, int, int]
    warmup_iterations: int
    timed_iterations: int
    parameter_count: int
    trainable_parameter_count: int
    serialized_state_dict_bytes: int
    latency: LatencySummary

    def validate(self) -> None:
        """Validate model-size and timing measurement invariants."""
        if len(self.input_shape) != 3 or any(value <= 0 for value in self.input_shape):
            raise ValueError("Profile input shape must contain positive batch, channel, and sample counts")
        if self.warmup_iterations < 0 or self.timed_iterations <= 0:
            raise ValueError("Warm-up count cannot be negative and timed count must be positive")
        if self.parameter_count <= 0 or self.trainable_parameter_count <= 0:
            raise ValueError("Model parameter counts must be positive")
        if self.trainable_parameter_count > self.parameter_count:
            raise ValueError("Trainable parameters cannot exceed all parameters")
        if self.serialized_state_dict_bytes <= 0:
            raise ValueError("Serialized state dictionary must be non-empty")
        self.latency.validate()


@dataclass(frozen=True)
class SyntheticComplexityProfileRecord:
    """Immutable record that cannot be confused with participant evaluation."""

    schema_version: int
    created_at_utc: str
    experiment_id: str
    scientific_result: bool
    holdout_accessed: bool
    input_kind: str
    source_revision: str
    model_configuration_sha256: str
    random_seed: int
    device: str
    platform_system: str
    platform_release: str
    platform_machine: str
    python_version: str
    torch_version: str
    torch_num_threads: int
    measurement: ModelComplexityMeasurement

    def validate(self) -> None:
        """Enforce non-scientific status and complete execution provenance."""
        if self.schema_version != 1:
            raise ValueError("Unsupported complexity-profile schema version")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Creation time must use second-resolution UTC with a Z suffix")
        if not self.experiment_id or not self.source_revision:
            raise ValueError("Experiment identifier and source revision are required")
        if self.scientific_result or self.holdout_accessed:
            raise ValueError("Synthetic complexity profiles cannot be scientific or access hold-out data")
        if self.input_kind != "synthetic_zeros":
            raise ValueError("Only synthetic-zero profiling input is authorized")
        if SHA256_PATTERN.fullmatch(self.model_configuration_sha256) is None:
            raise ValueError("Model configuration digest is not a SHA-256 value")
        if self.random_seed < 0 or self.torch_num_threads <= 0:
            raise ValueError("Random seed and PyTorch thread count are invalid")
        if not all(
            (
                self.device,
                self.platform_system,
                self.platform_release,
                self.platform_machine,
                self.python_version,
                self.torch_version,
            )
        ):
            raise ValueError("Software, platform, and device provenance are required")
        self.measurement.validate()

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        value = asdict(self)
        value["measurement"]["input_shape"] = list(self.measurement.input_shape)
        return value


def _synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading a wall-clock timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _serialized_state_dict_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def profile_model_complexity(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    device: str = "cpu",
    warmup_iterations: int = 20,
    timed_iterations: int = 100,
) -> ModelComplexityMeasurement:
    """Measure a model using generated zeros without loading participant data."""
    if len(input_shape) != 3 or any(value <= 0 for value in input_shape):
        raise ValueError("Profile input shape must contain positive batch, channel, and sample counts")
    if warmup_iterations < 0 or timed_iterations <= 0:
        raise ValueError("Warm-up count cannot be negative and timed count must be positive")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling was requested but CUDA is unavailable")
    if selected_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS profiling was requested but MPS is unavailable")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    serialized_bytes = _serialized_state_dict_size(model)
    was_training = model.training
    model.to(selected_device)
    model.eval()
    inputs = torch.zeros(input_shape, dtype=torch.float32, device=selected_device)
    timings_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(inputs)
        _synchronize(selected_device)
        for _ in range(timed_iterations):
            _synchronize(selected_device)
            start_ns = time.perf_counter_ns()
            model(inputs)
            _synchronize(selected_device)
            timings_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)
    model.train(was_training)

    quartiles = np.quantile(np.asarray(timings_ms), (0.25, 0.75))
    measurement = ModelComplexityMeasurement(
        input_shape=input_shape,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        serialized_state_dict_bytes=serialized_bytes,
        latency=LatencySummary(
            median_ms=statistics.median(timings_ms),
            first_quartile_ms=float(quartiles[0]),
            third_quartile_ms=float(quartiles[1]),
            minimum_ms=min(timings_ms),
            maximum_ms=max(timings_ms),
        ),
    )
    measurement.validate()
    return measurement


def build_synthetic_complexity_profile_record(
    *,
    created_at_utc: str,
    experiment_id: str,
    source_revision: str,
    model_configuration_path: Path | str,
    random_seed: int,
    device: str,
    measurement: ModelComplexityMeasurement,
) -> SyntheticComplexityProfileRecord:
    """Build a non-scientific profile with machine and software provenance."""
    record = SyntheticComplexityProfileRecord(
        schema_version=1,
        created_at_utc=created_at_utc,
        experiment_id=experiment_id,
        scientific_result=False,
        holdout_accessed=False,
        input_kind="synthetic_zeros",
        source_revision=source_revision,
        model_configuration_sha256=sha256_file(model_configuration_path),
        random_seed=random_seed,
        device=str(torch.device(device)),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_num_threads=torch.get_num_threads(),
        measurement=measurement,
    )
    record.validate()
    return record


def write_synthetic_complexity_profile_record(
    path: Path | str,
    record: SyntheticComplexityProfileRecord,
) -> None:
    """Write a new profile exclusively; existing records are never overwritten."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(record.as_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
