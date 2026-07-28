"""Non-authoritative reconstruction of the 45 camera-ready TSFEL features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import scipy
from numpy.typing import NDArray

try:
    from tsfel.feature_extraction import features as tsfel_features
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "TSFEL is required for prevoccupai_har.feature_reconstruction; "
        "install the isolated feature runtime"
    ) from error

from .provenance import sha256_file


ALLOWED_FUNCTIONS = {
    "human_range_energy",
    "max_power_spectrum",
    "median_frequency",
    "power_bandwidth",
    "spectral_entropy",
    "interq_range",
    "calc_max",
    "calc_mean",
    "calc_median",
    "calc_min",
    "rms",
    "skewness",
    "calc_std",
    "calc_var",
    "mean_abs_diff",
}


class FeatureExtractionPurpose(StrEnum):
    """Permitted purposes for the reconstructed feature functions."""

    SYNTHETIC_VALIDATION = "synthetic_validation"
    SCIENTIFIC_DATASET = "scientific_dataset"


@dataclass(frozen=True)
class TsfelFeatureDefinition:
    """One ordered scalar TSFEL feature definition."""

    domain: str
    name: str
    function: str
    uses_sampling_rate: bool

    def validate(self) -> None:
        """Require one supported, named scalar feature."""
        if self.domain not in {"spectral", "statistical", "temporal"}:
            raise ValueError("Unsupported TSFEL feature domain")
        if not self.name or self.function not in ALLOWED_FUNCTIONS:
            raise ValueError("Unsupported TSFEL feature definition")
        if self.uses_sampling_rate != (self.domain == "spectral"):
            raise ValueError("TSFEL sampling-rate declaration disagrees with domain")


@dataclass(frozen=True)
class TsfelFeatureReconstructionConfiguration:
    """Frozen feature semantics and recovered-runtime boundary."""

    schema_version: int
    feature_set_id: str
    status: str
    authoritative: bool
    controls_scientific_feature_generation: bool
    source_commit: str
    source_file_sha256: str
    feature_caller_file_sha256: str
    invocation_file_sha256: str
    requirements_file_sha256: str
    sampling_rate_hz: int
    window_samples: int
    source_axis_names: tuple[str, ...]
    axis_names: tuple[str, ...]
    axis_order_verified: bool
    window_normalization: str
    output_column_order: str
    required_tsfel_version: str
    recovered_numpy_version: str
    recovered_scipy_version: str
    recovered_pandas_version: str
    scientific_runtime_must_match_recovered_versions: bool
    validation_with_other_numerical_versions_allowed: bool
    validation_reference_id: str
    validation_canonical_output_dtype: str
    exact_runtime_output_sha256: str
    expected_feature_count_per_axis: int
    expected_total_feature_count: int
    ordered_features: tuple[TsfelFeatureDefinition, ...]

    def validate(self) -> None:
        """Validate feature count, ordering, runtime, and authority invariants."""
        if self.schema_version != 1:
            raise ValueError("Unsupported TSFEL reconstruction schema version")
        if self.feature_set_id != "mban_acc_tsfel_45_reconstruction_v1":
            raise ValueError("Unexpected TSFEL feature-set identifier")
        if self.status != "non_authoritative_reconstruction_only":
            raise ValueError("TSFEL reconstruction status must remain non-authoritative")
        if self.authoritative or self.controls_scientific_feature_generation:
            raise ValueError("Current TSFEL reconstruction cannot control scientific data")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("Recovered TSFEL source commit must be a full Git hash")
        source_hashes = (
            self.source_file_sha256,
            self.feature_caller_file_sha256,
            self.invocation_file_sha256,
            self.requirements_file_sha256,
        )
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in source_hashes
        ):
            raise ValueError("Recovered TSFEL source-file hash is invalid")
        if self.sampling_rate_hz != 1000 or self.window_samples != 5000:
            raise ValueError("TSFEL reconstruction requires 5 s at 1,000 Hz")
        if self.source_axis_names != ("x_ACC", "y_ACC", "z_ACC"):
            raise ValueError("Recovered caller axis names changed")
        if self.axis_names != ("acc_0", "acc_1", "acc_2"):
            raise ValueError("TSFEL reconstruction requires neutral ACC axis names")
        if self.axis_order_verified:
            raise ValueError("Raw ACC axis order has not been verified")
        if self.window_normalization != "none":
            raise ValueError("Published RF target uses no raw-window normalization")
        if self.output_column_order != "alphabetical_tsfel_0.1.9":
            raise ValueError("TSFEL reconstruction requires alphabetical columns")
        if self.required_tsfel_version != "0.1.9":
            raise ValueError("Recovered feature runtime requires TSFEL 0.1.9")
        if (
            self.recovered_numpy_version != "1.26.4"
            or self.recovered_scipy_version != "1.11.4"
            or self.recovered_pandas_version != "2.1.4"
        ):
            raise ValueError("Recovered numerical runtime versions changed")
        if not self.scientific_runtime_must_match_recovered_versions:
            raise ValueError("Scientific feature runtime must match recovered versions")
        if not self.validation_with_other_numerical_versions_allowed:
            raise ValueError("Software validation must declare its version tolerance")
        if self.validation_reference_id != "three_sine_trend_window_v1":
            raise ValueError("Unexpected TSFEL validation reference")
        if self.validation_canonical_output_dtype != "little_endian_float64":
            raise ValueError("TSFEL validation reference requires canonical float64")
        if len(self.exact_runtime_output_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.exact_runtime_output_sha256
        ):
            raise ValueError("TSFEL validation-reference hash is invalid")
        if self.expected_feature_count_per_axis != 15:
            raise ValueError("Expected exactly 15 TSFEL features per ACC axis")
        if self.expected_total_feature_count != 45:
            raise ValueError("Expected exactly 45 TSFEL candidate features")
        if len(self.ordered_features) != self.expected_feature_count_per_axis:
            raise ValueError("Ordered TSFEL feature count is inconsistent")
        for definition in self.ordered_features:
            definition.validate()
        if len({definition.name for definition in self.ordered_features}) != len(
            self.ordered_features
        ):
            raise ValueError("TSFEL feature names must be unique")
        if {definition.function for definition in self.ordered_features} != ALLOWED_FUNCTIONS:
            raise ValueError("TSFEL feature-function set is incomplete")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return deterministic TSFEL-compatible axis/feature names."""
        return tuple(
            sorted(
                f"{axis}_{definition.name}"
                for axis in self.axis_names
                for definition in self.ordered_features
            )
        )


def load_tsfel_feature_reconstruction_configuration(
    path: Path | str,
) -> TsfelFeatureReconstructionConfiguration:
    """Load the checked-in non-authoritative TSFEL reconstruction."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("TSFEL reconstruction configuration must be an object")
    source = decoded.get("source")
    input_value = decoded.get("input")
    runtime = decoded.get("runtime")
    validation_reference = decoded.get("validation_reference")
    if not all(
        isinstance(value, Mapping)
        for value in (source, input_value, runtime, validation_reference)
    ):
        raise TypeError(
            "TSFEL source, input, runtime, and validation reference must be objects"
        )
    feature_values = decoded.get("ordered_features")
    if not isinstance(feature_values, list):
        raise TypeError("TSFEL ordered features must be a list")
    configuration = TsfelFeatureReconstructionConfiguration(
        schema_version=int(decoded["schema_version"]),
        feature_set_id=str(decoded["feature_set_id"]),
        status=str(decoded["status"]),
        authoritative=decoded.get("authoritative") is True,
        controls_scientific_feature_generation=(
            decoded.get("controls_scientific_feature_generation") is True
        ),
        source_commit=str(source["commit"]),
        source_file_sha256=str(source["source_file_sha256"]),
        feature_caller_file_sha256=str(source["feature_caller_file_sha256"]),
        invocation_file_sha256=str(source["invocation_file_sha256"]),
        requirements_file_sha256=str(source["requirements_file_sha256"]),
        sampling_rate_hz=int(input_value["sampling_rate_hz"]),
        window_samples=int(input_value["window_samples"]),
        source_axis_names=tuple(map(str, input_value["source_axis_names"])),
        axis_names=tuple(map(str, input_value["axis_names"])),
        axis_order_verified=input_value.get("axis_order_verified") is True,
        window_normalization=str(input_value["window_normalization"]),
        output_column_order=str(input_value["output_column_order"]),
        required_tsfel_version=str(runtime["required_tsfel_version"]),
        recovered_numpy_version=str(runtime["recovered_numpy_version"]),
        recovered_scipy_version=str(runtime["recovered_scipy_version"]),
        recovered_pandas_version=str(runtime["recovered_pandas_version"]),
        scientific_runtime_must_match_recovered_versions=(
            runtime.get("scientific_runtime_must_match_recovered_versions") is True
        ),
        validation_with_other_numerical_versions_allowed=(
            runtime.get("validation_with_other_numerical_versions_allowed") is True
        ),
        validation_reference_id=str(validation_reference["reference_id"]),
        validation_canonical_output_dtype=str(
            validation_reference["canonical_output_dtype"]
        ),
        exact_runtime_output_sha256=str(
            validation_reference["exact_runtime_output_sha256"]
        ),
        expected_feature_count_per_axis=int(
            decoded["expected_feature_count_per_axis"]
        ),
        expected_total_feature_count=int(decoded["expected_total_feature_count"]),
        ordered_features=tuple(
            TsfelFeatureDefinition(
                domain=str(value["domain"]),
                name=str(value["name"]),
                function=str(value["function"]),
                uses_sampling_rate=value.get("uses_sampling_rate") is True,
            )
            for value in feature_values
        ),
    )
    configuration.validate()
    return configuration


def _runtime_versions() -> Mapping[str, str]:
    return MappingProxyType(
        {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": version("pandas"),
            "tsfel": version("tsfel"),
        }
    )


@dataclass(frozen=True)
class ReconstructedFeatureMatrix:
    """Immutable feature values with exact configuration/runtime evidence."""

    values: NDArray[np.float64]
    feature_names: tuple[str, ...]
    purpose: FeatureExtractionPurpose
    scientific_result: bool
    configuration_sha256: str
    runtime_versions: Mapping[str, str]

    def validate(self) -> None:
        """Validate shape, finiteness, immutability, names, and scope."""
        if self.values.ndim != 2 or self.values.shape[0] == 0:
            raise ValueError("Reconstructed feature matrix must be non-empty and 2D")
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError("Reconstructed feature names do not align with values")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("Reconstructed feature names must be unique")
        if not np.isfinite(self.values).all():
            raise ValueError("Reconstructed features contain non-finite values")
        if self.values.flags.writeable:
            raise ValueError("Reconstructed feature values must be immutable")
        if self.purpose is FeatureExtractionPurpose.SYNTHETIC_VALIDATION:
            if self.scientific_result:
                raise ValueError("Synthetic reconstructed features cannot be scientific")
        elif self.purpose is FeatureExtractionPurpose.SCIENTIFIC_DATASET:
            if not self.scientific_result:
                raise ValueError("Scientific reconstructed features require scientific scope")
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError("Unsupported reconstructed-feature purpose")
        if len(self.configuration_sha256) != 64:
            raise ValueError("Feature configuration digest is invalid")
        if set(self.runtime_versions) != {"numpy", "scipy", "pandas", "tsfel"}:
            raise ValueError("Feature runtime versions are incomplete")
        if any(not value for value in self.runtime_versions.values()):
            raise ValueError("Feature runtime versions must be non-empty")


def extract_reconstructed_tsfel_features(
    windows: NDArray[np.floating],
    *,
    configuration_path: Path | str,
    purpose: FeatureExtractionPurpose,
) -> ReconstructedFeatureMatrix:
    """Extract the ordered 45-feature set without TSFEL multiprocessing."""
    configuration_path = Path(configuration_path)
    configuration = load_tsfel_feature_reconstruction_configuration(
        configuration_path
    )
    runtime_versions = _runtime_versions()
    if runtime_versions["tsfel"] != configuration.required_tsfel_version:
        raise RuntimeError("TSFEL runtime version disagrees with the reconstruction")
    if purpose is FeatureExtractionPurpose.SCIENTIFIC_DATASET:
        if (
            not configuration.authoritative
            or not configuration.controls_scientific_feature_generation
        ):
            raise PermissionError(
                "The current TSFEL reconstruction cannot generate scientific features"
            )
        expected_runtime = {
            "numpy": configuration.recovered_numpy_version,
            "scipy": configuration.recovered_scipy_version,
            "pandas": configuration.recovered_pandas_version,
            "tsfel": configuration.required_tsfel_version,
        }
        if runtime_versions != expected_runtime:
            raise RuntimeError("Scientific TSFEL runtime differs from recovered versions")
    elif purpose is not FeatureExtractionPurpose.SYNTHETIC_VALIDATION:
        raise ValueError("Unsupported TSFEL extraction purpose")

    values = np.asarray(windows, dtype=np.float64)
    expected_shape = (
        values.shape[0] if values.ndim == 3 else 0,
        configuration.window_samples,
        len(configuration.axis_names),
    )
    if values.ndim != 3 or values.shape != expected_shape or values.shape[0] == 0:
        raise ValueError(
            "TSFEL input must be non-empty windows x 5,000 samples x 3 axes"
        )
    if not np.isfinite(values).all():
        raise ValueError("TSFEL input windows must be finite")

    extraction_plan = sorted(
        (
            f"{axis_name}_{definition.name}",
            axis_index,
            definition,
        )
        for axis_index, axis_name in enumerate(configuration.axis_names)
        for definition in configuration.ordered_features
    )
    if tuple(name for name, _, _ in extraction_plan) != configuration.feature_names:
        raise RuntimeError("TSFEL extraction plan disagrees with configured columns")
    output = np.empty(
        (values.shape[0], len(extraction_plan)),
        dtype=np.float64,
    )
    for window_index, window in enumerate(values):
        for feature_index, (_, axis_index, definition) in enumerate(extraction_plan):
            signal = window[:, axis_index]
            function = getattr(tsfel_features, definition.function)
            result = (
                function(signal, configuration.sampling_rate_hz)
                if definition.uses_sampling_rate
                else function(signal)
            )
            output[window_index, feature_index] = float(result)
    if not np.isfinite(output).all():
        raise ValueError(
            "TSFEL reconstruction produced non-finite features; inspect the input"
        )
    output.setflags(write=False)
    record = ReconstructedFeatureMatrix(
        values=output,
        feature_names=configuration.feature_names,
        purpose=purpose,
        scientific_result=purpose is FeatureExtractionPurpose.SCIENTIFIC_DATASET,
        configuration_sha256=sha256_file(configuration_path),
        runtime_versions=runtime_versions,
    )
    record.validate()
    return record
