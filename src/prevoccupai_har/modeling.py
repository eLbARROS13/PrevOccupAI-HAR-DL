"""Conservative PyTorch models for auditable raw-signal HAR baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised only without the DL extra
    raise ImportError(
        "PyTorch is required for prevoccupai_har.modeling; install the 'dl' extra"
    ) from error


@dataclass(frozen=True)
class CompactCnnArchitecture:
    """Validated architecture parameters for a compact raw-signal 1D CNN."""

    input_channels: int
    expected_samples: int
    convolution_channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    strides: tuple[int, ...]
    max_pool_sizes: tuple[int, ...]
    batch_normalization: bool
    dropout_probability: float
    initialization: str
    output_classes: int

    def validate(self) -> None:
        """Reject inconsistent or needlessly unconstrained architecture values."""
        if self.input_channels <= 0 or self.expected_samples <= 0:
            raise ValueError("Input channels and samples must be positive")
        stage_lengths = {
            len(self.convolution_channels),
            len(self.kernel_sizes),
            len(self.strides),
            len(self.max_pool_sizes),
        }
        if stage_lengths != {len(self.convolution_channels)} or not self.convolution_channels:
            raise ValueError("Every convolutional stage requires channels, kernel, stride, and pool")
        if any(value <= 0 for value in self.convolution_channels):
            raise ValueError("Convolutional channel counts must be positive")
        if any(value <= 0 or value % 2 == 0 for value in self.kernel_sizes):
            raise ValueError("Kernel sizes must be positive odd integers")
        if any(value <= 0 for value in self.strides + self.max_pool_sizes):
            raise ValueError("Strides and pool sizes must be positive")
        if not 0 <= self.dropout_probability < 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        if self.initialization != "pytorch_module_defaults_seeded":
            raise ValueError("Unsupported parameter-initialization contract")
        if self.output_classes < 2:
            raise ValueError("At least two output classes are required")


@dataclass(frozen=True)
class OptimizationConfiguration:
    """Pre-specified optimization and validation-only selection settings."""

    learning_rate: float
    weight_decay: float
    batch_size: int
    maximum_epochs: int
    early_stopping_patience: int
    early_stopping_minimum_delta: float

    def validate(self) -> None:
        """Validate bounded optimization settings."""
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative")
        if self.batch_size <= 0 or self.maximum_epochs <= 0:
            raise ValueError("Batch size and maximum epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("Early-stopping patience must be positive")
        if self.early_stopping_minimum_delta < 0:
            raise ValueError("Early-stopping minimum delta cannot be negative")


@dataclass(frozen=True)
class CompactCnnExperimentConfiguration:
    """Executable software contract that remains non-authoritative scientifically."""

    schema_version: int
    experiment_id: str
    status: str
    protocol_config: Path
    scientific_training_authorized_by_this_config: bool
    class_labels: tuple[str, ...]
    architecture: CompactCnnArchitecture
    optimization: OptimizationConfiguration
    random_seeds: tuple[int, ...]

    def validate(self) -> None:
        """Enforce configuration and governance invariants."""
        if self.schema_version != 1:
            raise ValueError(f"Unsupported CNN configuration schema: {self.schema_version}")
        if self.status not in {"synthetic_validation_only", "frozen_for_development"}:
            raise ValueError(f"Unsupported CNN configuration status: {self.status}")
        if self.scientific_training_authorized_by_this_config:
            raise ValueError("A model configuration cannot authorize scientific training")
        if not self.experiment_id:
            raise ValueError("Experiment identifier cannot be empty")
        if (
            not self.class_labels
            or len(set(self.class_labels)) != len(self.class_labels)
            or any(not label.strip() for label in self.class_labels)
        ):
            raise ValueError("Class labels must be non-empty and unique")
        if len(self.class_labels) != self.architecture.output_classes:
            raise ValueError("Class-label count must equal the model output count")
        if not self.random_seeds or len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError("Random seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.random_seeds):
            raise ValueError("Random seeds cannot be negative")
        self.architecture.validate()
        self.optimization.validate()


@dataclass(frozen=True)
class CompactTcnArchitecture:
    """Residual dilated temporal-convolution architecture for one 5-s window."""

    input_channels: int
    expected_samples: int
    channels: int
    stem_kernel_size: int
    stem_stride: int
    kernel_size: int
    dilations: tuple[int, ...]
    batch_normalization: bool
    dropout_probability: float
    padding: str
    initialization: str
    output_classes: int
    depthwise_separable: bool = False

    @property
    def receptive_field_samples(self) -> int:
        """Return the exact raw-sample receptive field of the deepest features."""
        residual_positions = 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)
        return self.stem_kernel_size + (residual_positions - 1) * self.stem_stride

    def validate(self) -> None:
        """Reject over-flexible or internally inconsistent TCN settings."""
        if self.input_channels <= 0 or self.expected_samples <= 0:
            raise ValueError("Input channels and samples must be positive")
        if self.channels <= 0:
            raise ValueError("TCN channel count must be positive")
        if self.stem_kernel_size <= 0 or self.stem_kernel_size % 2 == 0:
            raise ValueError("TCN stem kernel must be a positive odd integer")
        if self.stem_stride <= 0:
            raise ValueError("TCN stem stride must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("TCN residual kernel must be a positive odd integer")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("TCN dilations must be non-empty and positive")
        if self.dilations != tuple(2**index for index in range(len(self.dilations))):
            raise ValueError("TCN dilations must be the ordered sequence 1, 2, 4, ...")
        if not 0 <= self.dropout_probability < 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        if self.padding != "symmetric_same":
            raise ValueError("Only symmetric same padding is supported")
        if self.initialization != "pytorch_module_defaults_seeded":
            raise ValueError("Unsupported parameter-initialization contract")
        if self.output_classes < 2:
            raise ValueError("At least two output classes are required")
        if self.receptive_field_samples > self.expected_samples:
            raise ValueError("TCN receptive field cannot exceed the declared window")


@dataclass(frozen=True)
class CompactTcnExperimentConfiguration:
    """Synthetic-only stronger temporal baseline configuration."""

    schema_version: int
    experiment_id: str
    status: str
    protocol_config: Path
    scientific_training_authorized_by_this_config: bool
    class_labels: tuple[str, ...]
    architecture: CompactTcnArchitecture
    optimization: OptimizationConfiguration
    random_seeds: tuple[int, ...]

    def validate(self) -> None:
        """Enforce the same governance boundary as the compact CNN."""
        if self.schema_version != 1:
            raise ValueError(f"Unsupported TCN configuration schema: {self.schema_version}")
        if self.status not in {"synthetic_validation_only", "frozen_for_development"}:
            raise ValueError(f"Unsupported TCN configuration status: {self.status}")
        if self.scientific_training_authorized_by_this_config:
            raise ValueError("A model configuration cannot authorize scientific training")
        if not self.experiment_id:
            raise ValueError("Experiment identifier cannot be empty")
        if (
            not self.class_labels
            or len(set(self.class_labels)) != len(self.class_labels)
            or any(not label.strip() for label in self.class_labels)
        ):
            raise ValueError("Class labels must be non-empty and unique")
        if len(self.class_labels) != self.architecture.output_classes:
            raise ValueError("Class-label count must equal the model output count")
        if not self.random_seeds or len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError("Random seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.random_seeds):
            raise ValueError("Random seeds cannot be negative")
        self.architecture.validate()
        self.optimization.validate()


def _integer_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    """Decode a JSON sequence as a non-string tuple of integers."""
    if not isinstance(value, list):
        raise TypeError(f"Configuration field '{field_name}' must be an array")
    return tuple(int(item) for item in value)


def load_compact_cnn_experiment_configuration(
    path: Path | str,
) -> CompactCnnExperimentConfiguration:
    """Load and validate the compact-CNN JSON configuration."""
    config_path = Path(path).resolve()
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError(f"Expected a JSON object in {config_path}")
    input_value = decoded.get("input")
    architecture_value = decoded.get("architecture")
    optimization_value = decoded.get("optimization")
    governance_value = decoded.get("governance")
    if not all(
        isinstance(value, Mapping)
        for value in (input_value, architecture_value, optimization_value, governance_value)
    ):
        raise TypeError("Input, architecture, optimization, and governance must be objects")

    input_mapping = input_value
    architecture_mapping = architecture_value
    optimization_mapping = optimization_value
    governance_mapping = governance_value
    input_shape = _integer_tuple(
        input_mapping["shape_channels_first"],
        field_name="input.shape_channels_first",
    )
    if len(input_shape) != 2:
        raise ValueError("Channels-first input shape must contain channels and samples")
    if architecture_mapping.get("type") != "compact_1d_cnn":
        raise ValueError("Only the compact_1d_cnn architecture is supported")
    if architecture_mapping.get("global_pooling") is not True:
        raise ValueError("The compact baseline requires global pooling")
    if optimization_mapping.get("loss") != "cross_entropy":
        raise ValueError("The compact baseline requires cross-entropy loss")
    if optimization_mapping.get("optimizer") != "adamw":
        raise ValueError("The compact baseline requires AdamW")
    if optimization_mapping.get("model_selection_metric") != "macro_f1":
        raise ValueError("Model selection must use validation macro F1")
    if optimization_mapping.get("model_selection_mode") != "maximize":
        raise ValueError("Validation macro F1 must be maximized")

    configuration = CompactCnnExperimentConfiguration(
        schema_version=int(decoded["schema_version"]),
        experiment_id=str(decoded["experiment_id"]),
        status=str(decoded["status"]),
        protocol_config=(config_path.parent.parent / str(decoded["protocol_config"])).resolve(),
        scientific_training_authorized_by_this_config=(
            governance_mapping.get("scientific_training_authorized_by_this_config") is True
        ),
        class_labels=tuple(map(str, input_mapping["class_labels"])),
        architecture=CompactCnnArchitecture(
            input_channels=input_shape[0],
            expected_samples=input_shape[1],
            convolution_channels=_integer_tuple(
                architecture_mapping["convolution_channels"],
                field_name="architecture.convolution_channels",
            ),
            kernel_sizes=_integer_tuple(
                architecture_mapping["kernel_sizes"],
                field_name="architecture.kernel_sizes",
            ),
            strides=_integer_tuple(
                architecture_mapping["strides"],
                field_name="architecture.strides",
            ),
            max_pool_sizes=_integer_tuple(
                architecture_mapping["max_pool_sizes"],
                field_name="architecture.max_pool_sizes",
            ),
            batch_normalization=architecture_mapping.get("batch_normalization") is True,
            dropout_probability=float(architecture_mapping["dropout_probability"]),
            initialization=str(architecture_mapping["initialization"]),
            output_classes=int(architecture_mapping["output_classes"]),
        ),
        optimization=OptimizationConfiguration(
            learning_rate=float(optimization_mapping["learning_rate"]),
            weight_decay=float(optimization_mapping["weight_decay"]),
            batch_size=int(optimization_mapping["batch_size"]),
            maximum_epochs=int(optimization_mapping["maximum_epochs"]),
            early_stopping_patience=int(
                optimization_mapping["early_stopping_patience"]
            ),
            early_stopping_minimum_delta=float(
                optimization_mapping["early_stopping_minimum_delta"]
            ),
        ),
        random_seeds=_integer_tuple(decoded["random_seeds"], field_name="random_seeds"),
    )
    configuration.validate()
    return configuration


def load_compact_tcn_experiment_configuration(
    path: Path | str,
) -> CompactTcnExperimentConfiguration:
    """Load and validate the compact residual-TCN JSON configuration."""
    config_path = Path(path).resolve()
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError(f"Expected a JSON object in {config_path}")
    input_value = decoded.get("input")
    architecture_value = decoded.get("architecture")
    optimization_value = decoded.get("optimization")
    governance_value = decoded.get("governance")
    if not all(
        isinstance(value, Mapping)
        for value in (input_value, architecture_value, optimization_value, governance_value)
    ):
        raise TypeError("Input, architecture, optimization, and governance must be objects")

    input_mapping = input_value
    architecture_mapping = architecture_value
    optimization_mapping = optimization_value
    governance_mapping = governance_value
    input_shape = _integer_tuple(
        input_mapping["shape_channels_first"],
        field_name="input.shape_channels_first",
    )
    if len(input_shape) != 2:
        raise ValueError("Channels-first input shape must contain channels and samples")
    if architecture_mapping.get("type") != "compact_dilated_tcn":
        raise ValueError("Only the compact_dilated_tcn architecture is supported")
    if architecture_mapping.get("residual_connections") is not True:
        raise ValueError("The compact TCN requires residual connections")
    if int(architecture_mapping.get("convolutions_per_block", 0)) != 2:
        raise ValueError("The compact TCN requires two convolutions per block")
    if architecture_mapping.get("global_pooling") is not True:
        raise ValueError("The compact TCN requires global pooling")
    if optimization_mapping.get("loss") != "cross_entropy":
        raise ValueError("The compact TCN requires cross-entropy loss")
    if optimization_mapping.get("optimizer") != "adamw":
        raise ValueError("The compact TCN requires AdamW")
    if optimization_mapping.get("model_selection_metric") != "macro_f1":
        raise ValueError("Model selection must use validation macro F1")
    if optimization_mapping.get("model_selection_mode") != "maximize":
        raise ValueError("Validation macro F1 must be maximized")

    configuration = CompactTcnExperimentConfiguration(
        schema_version=int(decoded["schema_version"]),
        experiment_id=str(decoded["experiment_id"]),
        status=str(decoded["status"]),
        protocol_config=(config_path.parent.parent / str(decoded["protocol_config"])).resolve(),
        scientific_training_authorized_by_this_config=(
            governance_mapping.get("scientific_training_authorized_by_this_config") is True
        ),
        class_labels=tuple(map(str, input_mapping["class_labels"])),
        architecture=CompactTcnArchitecture(
            input_channels=input_shape[0],
            expected_samples=input_shape[1],
            channels=int(architecture_mapping["channels"]),
            stem_kernel_size=int(architecture_mapping["stem_kernel_size"]),
            stem_stride=int(architecture_mapping["stem_stride"]),
            kernel_size=int(architecture_mapping["kernel_size"]),
            dilations=_integer_tuple(
                architecture_mapping["dilations"],
                field_name="architecture.dilations",
            ),
            batch_normalization=architecture_mapping.get("batch_normalization") is True,
            dropout_probability=float(architecture_mapping["dropout_probability"]),
            padding=str(architecture_mapping["padding"]),
            initialization=str(architecture_mapping["initialization"]),
            output_classes=int(architecture_mapping["output_classes"]),
            depthwise_separable=(
                architecture_mapping.get("depthwise_separable") is True
            ),
        ),
        optimization=OptimizationConfiguration(
            learning_rate=float(optimization_mapping["learning_rate"]),
            weight_decay=float(optimization_mapping["weight_decay"]),
            batch_size=int(optimization_mapping["batch_size"]),
            maximum_epochs=int(optimization_mapping["maximum_epochs"]),
            early_stopping_patience=int(
                optimization_mapping["early_stopping_patience"]
            ),
            early_stopping_minimum_delta=float(
                optimization_mapping["early_stopping_minimum_delta"]
            ),
        ),
        random_seeds=_integer_tuple(decoded["random_seeds"], field_name="random_seeds"),
    )
    configuration.validate()
    return configuration


TimeSeriesExperimentConfiguration = (
    CompactCnnExperimentConfiguration | CompactTcnExperimentConfiguration
)


def load_time_series_experiment_configuration(
    path: Path | str,
) -> TimeSeriesExperimentConfiguration:
    """Dispatch to the validated loader identified by the architecture type."""
    config_path = Path(path)
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping) or not isinstance(decoded.get("architecture"), Mapping):
        raise TypeError("Model configuration and architecture must be JSON objects")
    architecture_type = decoded["architecture"].get("type")
    if architecture_type == "compact_1d_cnn":
        return load_compact_cnn_experiment_configuration(config_path)
    if architecture_type == "compact_dilated_tcn":
        return load_compact_tcn_experiment_configuration(config_path)
    raise ValueError(f"Unsupported time-series architecture type: {architecture_type}")


def load_model_input_sample_stride(path: Path | str) -> int:
    """Load and validate deterministic post-low-pass temporal decimation."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping) or not isinstance(decoded.get("input"), Mapping):
        raise TypeError("Model input configuration must be a JSON object")
    input_value = decoded["input"]
    downsampling = input_value.get("temporal_downsampling")
    if downsampling is None:
        return 1
    if not isinstance(downsampling, Mapping):
        raise TypeError("Temporal downsampling must be an object")
    source_rate = int(downsampling["source_sampling_rate_hz"])
    model_rate = int(downsampling["model_sampling_rate_hz"])
    factor = int(downsampling["factor"])
    if downsampling.get("method") != "stride_after_20_hz_lowpass":
        raise ValueError("Unsupported temporal-downsampling method")
    if source_rate != 1000 or model_rate != 100 or factor != 10:
        raise ValueError("The frozen DL representation requires 1,000-to-100 Hz decimation")
    if source_rate != model_rate * factor:
        raise ValueError("Temporal-downsampling rates and factor disagree")
    shape = input_value.get("shape_channels_first")
    if not isinstance(shape, list) or len(shape) != 2 or int(shape[1]) * factor != 5000:
        raise ValueError("Downsampled model shape does not preserve a 5-second window")
    return factor


class CompactCnn1D(nn.Module):
    """Conservative temporal CNN with adaptive pooling and a small classifier head."""

    def __init__(self, configuration: CompactCnnArchitecture) -> None:
        super().__init__()
        configuration.validate()
        self.configuration = configuration
        stages: list[nn.Module] = []
        input_channels = configuration.input_channels
        for output_channels, kernel_size, stride, pool_size in zip(
            configuration.convolution_channels,
            configuration.kernel_sizes,
            configuration.strides,
            configuration.max_pool_sizes,
            strict=True,
        ):
            stages.append(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=not configuration.batch_normalization,
                )
            )
            if configuration.batch_normalization:
                stages.append(nn.BatchNorm1d(output_channels))
            stages.append(nn.ReLU())
            if pool_size > 1:
                stages.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            input_channels = output_channels

        self.feature_extractor = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(configuration.dropout_probability),
            nn.Linear(configuration.convolution_channels[-1], configuration.output_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits for channels-first windows."""
        if inputs.ndim != 3:
            raise ValueError("Model input must have shape (batch, channels, samples)")
        if inputs.shape[1] != self.configuration.input_channels:
            raise ValueError("Model input has the wrong accelerometer-channel count")
        if inputs.shape[2] != self.configuration.expected_samples:
            raise ValueError("Model input has the wrong window length")
        features = self.feature_extractor(inputs)
        return self.classifier(self.global_pool(features))

    @property
    def trainable_parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_compact_cnn_1d(
    configuration: CompactCnnExperimentConfiguration,
) -> CompactCnn1D:
    """Build the compact model without authorizing any scientific data use."""
    configuration.validate()
    return CompactCnn1D(configuration.architecture)


class ResidualDilatedTcnBlock(nn.Module):
    """Two same-padded dilated convolutions with an identity residual path."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dilation: int,
        batch_normalization: bool,
        dropout_probability: float,
        depthwise_separable: bool,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        def convolution() -> nn.Module:
            if depthwise_separable:
                return nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=1,
                        bias=not batch_normalization,
                    ),
                )
            return nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=not batch_normalization,
            )

        self.first_convolution = convolution()
        self.first_normalization: nn.Module = (
            nn.BatchNorm1d(channels) if batch_normalization else nn.Identity()
        )
        self.second_convolution = convolution()
        self.second_normalization: nn.Module = (
            nn.BatchNorm1d(channels) if batch_normalization else nn.Identity()
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return residual block features with unchanged temporal geometry."""
        features = self.first_convolution(inputs)
        features = self.first_normalization(features)
        features = self.dropout(self.activation(features))
        features = self.second_normalization(self.second_convolution(features))
        return self.dropout(self.activation(features + inputs))


class CompactDilatedTcn(nn.Module):
    """Small same-padded residual TCN for one independently labelled window."""

    def __init__(self, configuration: CompactTcnArchitecture) -> None:
        super().__init__()
        configuration.validate()
        self.configuration = configuration
        self.stem = nn.Sequential(
            nn.Conv1d(
                configuration.input_channels,
                configuration.channels,
                kernel_size=configuration.stem_kernel_size,
                stride=configuration.stem_stride,
                padding=configuration.stem_kernel_size // 2,
                bias=not configuration.batch_normalization,
            ),
            (
                nn.BatchNorm1d(configuration.channels)
                if configuration.batch_normalization
                else nn.Identity()
            ),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *(
                ResidualDilatedTcnBlock(
                    channels=configuration.channels,
                    kernel_size=configuration.kernel_size,
                    dilation=dilation,
                    batch_normalization=configuration.batch_normalization,
                    dropout_probability=configuration.dropout_probability,
                    depthwise_separable=configuration.depthwise_separable,
                )
                for dilation in configuration.dilations
            )
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configuration.channels, configuration.output_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits for channels-first windows."""
        if inputs.ndim != 3:
            raise ValueError("Model input must have shape (batch, channels, samples)")
        if inputs.shape[1] != self.configuration.input_channels:
            raise ValueError("Model input has the wrong accelerometer-channel count")
        if inputs.shape[2] != self.configuration.expected_samples:
            raise ValueError("Model input has the wrong window length")
        features = self.residual_blocks(self.stem(inputs))
        return self.classifier(self.global_pool(features))

    @property
    def trainable_parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_compact_tcn(
    configuration: CompactTcnExperimentConfiguration,
) -> CompactDilatedTcn:
    """Build the compact TCN without authorizing any scientific data use."""
    configuration.validate()
    return CompactDilatedTcn(configuration.architecture)


def build_time_series_classifier(
    configuration: TimeSeriesExperimentConfiguration,
) -> CompactCnn1D | CompactDilatedTcn:
    """Build the configured conservative classifier under a shared interface."""
    if isinstance(configuration, CompactCnnExperimentConfiguration):
        return build_compact_cnn_1d(configuration)
    if isinstance(configuration, CompactTcnExperimentConfiguration):
        return build_compact_tcn(configuration)
    raise TypeError("Unsupported time-series experiment configuration")
