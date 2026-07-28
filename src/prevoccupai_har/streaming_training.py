"""Memory-bounded training directly from the approved development window store."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover - exercised only without the DL extra
    raise ImportError(
        "PyTorch is required for prevoccupai_har.streaming_training; install the 'dl' extra"
    ) from error

from .evaluation import metrics_from_confusion_matrix
from .modeling import OptimizationConfiguration
from .preprocessing import TrainOnlyChannelStandardizer
from .protocol import ProtocolConfiguration
from .training import (
    TrainingHistoryEntry,
    TrainingOutcome,
    TrainingRunScope,
    set_reproducible_seed,
)
from .window_store import DevelopmentWindowStore
from .windowing import WindowMetadata


@dataclass(frozen=True)
class StreamingPrediction:
    """Ordered validation logits aligned with immutable window-store indices."""

    row_indices: NDArray[np.int64]
    logits: NDArray[np.float32]


@dataclass(frozen=True)
class TrainingThroughputBenchmark:
    """Compute-only timing with no predictions or performance metrics."""

    measured_batches: int
    measured_examples: int
    elapsed_seconds: float
    estimated_epoch_seconds: float


class _IndexDataset(Dataset[int]):
    """Expose only integer row references; signal data remain memory mapped."""

    def __init__(self, indices: NDArray[np.int64]) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, position: int) -> int:
        return int(self.indices[position])


class _MemmapBatchCollator:
    """Load, standardize, and transpose exactly one batch at a time."""

    def __init__(
        self,
        store: DevelopmentWindowStore,
        standardizer: TrainOnlyChannelStandardizer,
        *,
        sample_stride: int,
    ) -> None:
        if standardizer.mean_ is None or standardizer.scale_ is None:
            raise RuntimeError("Streaming collator requires fitted preprocessing")
        self.windows = store.windows
        self.labels = store.labels
        self.mean = np.asarray(standardizer.mean_, dtype=np.float32)
        self.scale = np.asarray(standardizer.scale_, dtype=np.float32)
        if sample_stride <= 0:
            raise ValueError("Input sample stride must be positive")
        self.sample_stride = sample_stride

    def __call__(
        self,
        indices: Sequence[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        row_indices = np.asarray(indices, dtype=np.int64)
        batch = np.array(
            self.windows[row_indices, :: self.sample_stride, :],
            dtype=np.float32,
            copy=True,
        )
        if batch.ndim != 3 or batch.shape[-1] != self.mean.shape[0]:
            raise ValueError("Window-store batch has an unexpected shape")
        if not np.isfinite(batch).all():
            raise ValueError("Window-store batch contains non-finite values")
        batch -= self.mean
        batch /= self.scale
        channels_first = np.ascontiguousarray(np.transpose(batch, (0, 2, 1)))
        targets = np.array(self.labels[row_indices], dtype=np.int64, copy=True)
        return (
            torch.from_numpy(channels_first),
            torch.from_numpy(targets),
            torch.from_numpy(row_indices.copy()),
        )


def indices_for_subjects(
    store: DevelopmentWindowStore,
    subjects: Iterable[str],
) -> NDArray[np.int64]:
    """Return ordered window rows for exactly the requested participant set."""
    requested = frozenset(map(str, subjects))
    if not requested:
        raise ValueError("At least one participant is required")
    available = frozenset(map(str, store.index["development_participants"]))
    unknown = requested - available
    if unknown:
        raise PermissionError(f"Requested participants are outside the store: {sorted(unknown)}")
    participant_ids = np.asarray(store.metadata["participant_id"])
    indices = np.flatnonzero(np.isin(participant_ids, tuple(sorted(requested)))).astype(
        np.int64
    )
    observed = frozenset(map(str, participant_ids[indices]))
    if observed != requested:
        raise ValueError("Window rows do not cover the requested participant set")
    indices.setflags(write=False)
    return indices


def fit_streaming_channel_standardizer(
    store: DevelopmentWindowStore,
    training_indices: NDArray[np.integer],
    *,
    allowed_training_subjects: Iterable[str],
    chunk_windows: int = 64,
    sample_stride: int = 1,
) -> TrainOnlyChannelStandardizer:
    """Fit channel mean and variance in bounded chunks using batch combination."""
    if chunk_windows <= 0 or sample_stride <= 0:
        raise ValueError("Standardizer chunk size and sample stride must be positive")
    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Training indices must be a non-empty vector")
    if len(np.unique(indices)) != indices.size:
        raise ValueError("Training indices contain duplicates")
    standardizer = TrainOnlyChannelStandardizer.for_subjects(
        allowed_training_subjects
    )
    participants = frozenset(map(str, store.metadata["participant_id"][indices]))
    if participants != standardizer.allowed_training_subjects:
        raise ValueError("Standardizer rows do not exactly match the training subjects")

    channel_count = int(store.windows.shape[-1])
    total_count = 0
    mean = np.zeros(channel_count, dtype=np.float64)
    second_moment = np.zeros(channel_count, dtype=np.float64)
    for start in range(0, indices.size, chunk_windows):
        chunk_indices = indices[start : start + chunk_windows]
        chunk = np.asarray(
            store.windows[chunk_indices, ::sample_stride, :], dtype=np.float64
        )
        if not np.isfinite(chunk).all():
            raise ValueError("Standardizer input contains non-finite values")
        flat = chunk.reshape(-1, channel_count)
        chunk_count = int(flat.shape[0])
        chunk_mean = flat.mean(axis=0)
        centered = flat - chunk_mean
        chunk_second_moment = np.einsum("ij,ij->j", centered, centered)
        if total_count == 0:
            mean = chunk_mean
            second_moment = chunk_second_moment
            total_count = chunk_count
            continue
        delta = chunk_mean - mean
        combined_count = total_count + chunk_count
        second_moment += (
            chunk_second_moment
            + delta * delta * total_count * chunk_count / combined_count
        )
        mean += delta * chunk_count / combined_count
        total_count = combined_count
    variance = second_moment / total_count
    scale = np.sqrt(np.maximum(variance, 0.0))
    standardizer.mean_ = mean
    standardizer.scale_ = np.where(scale == 0.0, 1.0, scale)
    return standardizer


def _loader(
    store: DevelopmentWindowStore,
    indices: NDArray[np.int64],
    standardizer: TrainOnlyChannelStandardizer,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    sample_stride: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _IndexDataset(indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        collate_fn=_MemmapBatchCollator(
            store,
            standardizer,
            sample_stride=sample_stride,
        ),
    )


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    criterion: nn.Module,
    *,
    output_classes: int,
    device: torch.device,
    retain_logits: bool,
) -> tuple[float, dict[str, object], StreamingPrediction | None]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = np.zeros((output_classes, output_classes), dtype=np.int64)
    row_chunks: list[NDArray[np.int64]] = []
    logit_chunks: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for inputs, targets, row_indices in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            if logits.shape != (targets.shape[0], output_classes):
                raise ValueError("Model output disagrees with the declared class count")
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits")
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.shape[0]
            total_examples += int(targets.shape[0])
            predictions = logits.argmax(dim=1)
            truth = targets.detach().cpu().numpy()
            predicted = predictions.detach().cpu().numpy()
            np.add.at(confusion, (truth, predicted), 1)
            if retain_logits:
                row_chunks.append(row_indices.numpy().astype(np.int64, copy=True))
                logit_chunks.append(
                    logits.detach().cpu().numpy().astype(np.float32, copy=True)
                )
    if total_examples == 0:
        raise ValueError("Evaluation loader contains no examples")
    labels = tuple(f"class_{index}" for index in range(output_classes))
    retained: StreamingPrediction | None = None
    if retain_logits:
        rows = np.concatenate(row_chunks)
        logits = np.concatenate(logit_chunks)
        rows.setflags(write=False)
        logits.setflags(write=False)
        retained = StreamingPrediction(row_indices=rows, logits=logits)
    return (
        total_loss / total_examples,
        metrics_from_confusion_matrix(confusion, labels),
        retained,
    )


def fit_classifier_streaming(
    model: nn.Module,
    store: DevelopmentWindowStore,
    training_indices: NDArray[np.integer],
    validation_indices: NDArray[np.integer],
    standardizer: TrainOnlyChannelStandardizer,
    *,
    output_classes: int,
    optimization: OptimizationConfiguration,
    seed: int,
    scope: TrainingRunScope,
    protocol: ProtocolConfiguration,
    device: str = "cpu",
    sample_stride: int = 1,
) -> TrainingOutcome:
    """Train and early-stop without materializing a full signal partition."""
    scope.validate(protocol)
    optimization.validate()
    if output_classes < 2:
        raise ValueError("At least two output classes are required")
    training = np.asarray(training_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if training.ndim != 1 or validation.ndim != 1 or not training.size or not validation.size:
        raise ValueError("Training and validation indices must be non-empty vectors")
    if np.intersect1d(training, validation).size:
        raise ValueError("Training and validation window rows overlap")
    observed_training = frozenset(map(str, store.metadata["participant_id"][training]))
    observed_validation = frozenset(map(str, store.metadata["participant_id"][validation]))
    if observed_training != frozenset(scope.training_subjects) or observed_validation != frozenset(
        scope.validation_subjects
    ):
        raise ValueError("Window partitions disagree with the validated run scope")
    if standardizer.allowed_training_subjects != frozenset(scope.training_subjects):
        raise ValueError("Preprocessing authorization disagrees with the training scope")

    set_reproducible_seed(seed)
    torch_device = torch.device(device)
    training_loader = _loader(
        store,
        training,
        standardizer,
        batch_size=optimization.batch_size,
        shuffle=True,
        seed=seed,
        sample_stride=sample_stride,
    )
    validation_loader = _loader(
        store,
        validation,
        standardizer,
        batch_size=optimization.batch_size,
        shuffle=False,
        seed=seed,
        sample_stride=sample_stride,
    )
    model.to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    best_score = -np.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[TrainingHistoryEntry] = []
    for epoch in range(1, optimization.maximum_epochs + 1):
        model.train()
        total_training_loss = 0.0
        total_training_examples = 0
        for inputs, targets, _ in training_loader:
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_training_loss += float(loss.item()) * targets.shape[0]
            total_training_examples += int(targets.shape[0])
        validation_loss, metrics, _ = _evaluate_loader(
            model,
            validation_loader,
            criterion,
            output_classes=output_classes,
            device=torch_device,
            retain_logits=False,
        )
        score = float(metrics["macro_f1"])
        history.append(
            TrainingHistoryEntry(
                epoch=epoch,
                training_loss=total_training_loss / total_training_examples,
                validation_loss=validation_loss,
                validation_macro_f1=score,
                validation_balanced_accuracy=float(metrics["balanced_accuracy"]),
            )
        )
        if score > best_score + optimization.early_stopping_minimum_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= optimization.early_stopping_patience:
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("Training completed without a validation-selected state")
    model.load_state_dict(best_state)
    return TrainingOutcome(
        seed=seed,
        best_epoch=best_epoch,
        stopped_early=len(history) < optimization.maximum_epochs,
        history=tuple(history),
    )


def benchmark_training_throughput(
    model: nn.Module,
    store: DevelopmentWindowStore,
    training_indices: NDArray[np.integer],
    standardizer: TrainOnlyChannelStandardizer,
    *,
    optimization: OptimizationConfiguration,
    seed: int,
    maximum_batches: int,
    device: str = "cpu",
    sample_stride: int = 1,
) -> TrainingThroughputBenchmark:
    """Measure bounded train-step throughput without evaluating predictions."""
    if maximum_batches <= 0:
        raise ValueError("Benchmark batch count must be positive")
    optimization.validate()
    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or not indices.size:
        raise ValueError("Benchmark training indices must be non-empty")
    if standardizer.allowed_training_subjects != frozenset(
        map(str, store.metadata["participant_id"][indices])
    ):
        raise ValueError("Benchmark preprocessing and training rows disagree")
    set_reproducible_seed(seed)
    loader = _loader(
        store,
        indices,
        standardizer,
        batch_size=optimization.batch_size,
        shuffle=True,
        seed=seed,
        sample_stride=sample_stride,
    )
    selected_device = torch.device(device)
    model.to(selected_device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    measured_batches = 0
    measured_examples = 0
    start = time.perf_counter()
    for inputs, targets, _ in loader:
        inputs = inputs.to(selected_device)
        targets = targets.to(selected_device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        measured_batches += 1
        measured_examples += int(targets.shape[0])
        if measured_batches >= maximum_batches:
            break
    elapsed = time.perf_counter() - start
    if measured_batches == 0 or elapsed <= 0:
        raise RuntimeError("Training throughput benchmark produced no timing")
    total_batches = int(np.ceil(indices.size / optimization.batch_size))
    return TrainingThroughputBenchmark(
        measured_batches=measured_batches,
        measured_examples=measured_examples,
        elapsed_seconds=elapsed,
        estimated_epoch_seconds=elapsed * total_batches / measured_batches,
    )


def predict_classifier_streaming(
    model: nn.Module,
    store: DevelopmentWindowStore,
    validation_indices: NDArray[np.integer],
    standardizer: TrainOnlyChannelStandardizer,
    *,
    output_classes: int,
    batch_size: int,
    seed: int,
    device: str = "cpu",
    sample_stride: int = 1,
) -> StreamingPrediction:
    """Return ordered logits from the restored validation-selected state."""
    indices = np.asarray(validation_indices, dtype=np.int64)
    loader = _loader(
        store,
        indices,
        standardizer,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        sample_stride=sample_stride,
    )
    _, _, prediction = _evaluate_loader(
        model,
        loader,
        nn.CrossEntropyLoss(),
        output_classes=output_classes,
        device=torch.device(device),
        retain_logits=True,
    )
    if prediction is None or not np.array_equal(prediction.row_indices, indices):
        raise RuntimeError("Streaming prediction order changed")
    return prediction


def metadata_for_indices(
    store: DevelopmentWindowStore,
    indices: NDArray[np.integer],
) -> tuple[WindowMetadata, ...]:
    """Decode path-free structured metadata for ordered result construction."""
    return tuple(
        WindowMetadata(
            subject_id=str(row["participant_id"]),
            recording_id=str(row["recording_id"]),
            main_label=str(row["main_label"]),
            sub_activity_label=str(row["sub_activity_label"]),
            sensor_stream_id=str(row["device_stream_id"]),
            sensor_side=str(row["sensor_side"]),
            start_sample=int(row["start_sample"]),
            end_sample_exclusive=int(row["end_sample_exclusive"]),
            preprocessing_status=str(row["preprocessing_status"]),
            quality_status=str(row["quality_status"]),
        )
        for row in store.metadata[np.asarray(indices, dtype=np.int64)]
    )
