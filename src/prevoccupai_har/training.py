"""Validation-selected PyTorch training restricted by explicit run purpose."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover - exercised only without the DL extra
    raise ImportError(
        "PyTorch is required for prevoccupai_har.training; install the 'dl' extra"
    ) from error

from .evaluation import metrics_from_confusion_matrix
from .modeling import OptimizationConfiguration
from .protocol import ProtocolConfiguration


class TrainingPurpose(str, Enum):
    """Permitted purposes for the development-only trainer."""

    SYNTHETIC_VALIDATION = "synthetic_validation"
    DEVELOPMENT_SELECTION = "development_selection"


@dataclass(frozen=True)
class TrainingRunScope:
    """Participant scope and purpose supplied independently of model settings."""

    purpose: TrainingPurpose
    training_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]

    def validate(self, protocol: ProtocolConfiguration | None = None) -> None:
        """Prevent hold-out access and ungoverned scientific training."""
        training = set(self.training_subjects)
        validation = set(self.validation_subjects)
        if not training or not validation:
            raise ValueError("Training and validation subject sets must be non-empty")
        if len(training) != len(self.training_subjects):
            raise ValueError("Training subject identifiers contain duplicates")
        if len(validation) != len(self.validation_subjects):
            raise ValueError("Validation subject identifiers contain duplicates")
        if training & validation:
            raise ValueError("Training and validation subjects must be disjoint")

        if self.purpose is TrainingPurpose.SYNTHETIC_VALIDATION:
            if protocol is not None:
                raise ValueError("Synthetic validation must not be coupled to a data protocol")
            if any(
                not subject.startswith("SYNTHETIC_")
                for subject in training | validation
            ):
                raise ValueError("Synthetic validation requires synthetic subject identifiers")
            return

        if self.purpose is not TrainingPurpose.DEVELOPMENT_SELECTION:
            raise TypeError("Training purpose must be a TrainingPurpose value")
        if protocol is None:
            raise ValueError("Development selection requires a validated protocol")
        if not protocol.training_authorized:
            raise PermissionError("The protocol does not authorize scientific training")
        if (training | validation) & set(protocol.holdout_participants):
            raise PermissionError("The trainer cannot receive external hold-out participants")
        if training | validation != set(protocol.development_participants):
            raise ValueError("Training and validation must partition the development cohort")


@dataclass(frozen=True)
class TrainingHistoryEntry:
    """One epoch of development-only optimization diagnostics."""

    epoch: int
    training_loss: float
    validation_loss: float
    validation_macro_f1: float
    validation_balanced_accuracy: float


@dataclass(frozen=True)
class TrainingOutcome:
    """In-memory training result with the best validation-selected model restored."""

    seed: int
    best_epoch: int
    stopped_early: bool
    history: tuple[TrainingHistoryEntry, ...]


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible software run."""
    if seed < 0:
        raise ValueError("Random seed cannot be negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_validated_tensors(
    inputs: Tensor | NDArray[np.floating],
    targets: Tensor | NDArray[np.integer],
    *,
    output_classes: int,
) -> tuple[Tensor, Tensor]:
    if isinstance(inputs, Tensor):
        input_tensor = inputs.detach().to(dtype=torch.float32).clone()
    else:
        input_tensor = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    if isinstance(targets, Tensor):
        target_tensor = targets.detach().to(dtype=torch.long).clone()
    else:
        target_tensor = torch.tensor(np.asarray(targets), dtype=torch.long)
    if input_tensor.ndim != 3:
        raise ValueError("Inputs must have shape (windows, channels, samples)")
    if target_tensor.ndim != 1 or target_tensor.shape[0] != input_tensor.shape[0]:
        raise ValueError("Targets must contain one class index per input window")
    if input_tensor.shape[0] == 0:
        raise ValueError("Training and validation arrays cannot be empty")
    if not torch.isfinite(input_tensor).all():
        raise ValueError("Model inputs must be finite")
    if torch.any(target_tensor < 0) or torch.any(target_tensor >= output_classes):
        raise ValueError("Target indices fall outside the model output range")
    return input_tensor, target_tensor


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    *,
    output_classes: int,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = np.zeros((output_classes, output_classes), dtype=np.int64)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.shape[0]
            total_examples += targets.shape[0]
            predictions = logits.argmax(dim=1)
            for target, prediction in zip(
                targets.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
                strict=True,
            ):
                confusion[int(target), int(prediction)] += 1
    labels = tuple(f"class_{index}" for index in range(output_classes))
    return total_loss / total_examples, metrics_from_confusion_matrix(confusion, labels)


def fit_classifier(
    model: nn.Module,
    training_inputs: Tensor | NDArray[np.floating],
    training_targets: Tensor | NDArray[np.integer],
    validation_inputs: Tensor | NDArray[np.floating],
    validation_targets: Tensor | NDArray[np.integer],
    *,
    output_classes: int,
    optimization: OptimizationConfiguration,
    seed: int,
    scope: TrainingRunScope,
    protocol: ProtocolConfiguration | None = None,
    device: str = "cpu",
) -> TrainingOutcome:
    """Fit on training data and select epochs from validation macro F1 only.

    This entry point deliberately has no hold-out mode. Final external evaluation
    requires a separate, future, audited command after model selection is frozen.
    """
    scope.validate(protocol)
    optimization.validate()
    if output_classes < 2:
        raise ValueError("At least two output classes are required")
    set_reproducible_seed(seed)
    torch_device = torch.device(device)
    training_dataset = TensorDataset(
        *_as_validated_tensors(
            training_inputs,
            training_targets,
            output_classes=output_classes,
        )
    )
    validation_dataset = TensorDataset(
        *_as_validated_tensors(
            validation_inputs,
            validation_targets,
            output_classes=output_classes,
        )
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=optimization.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=optimization.batch_size,
        shuffle=False,
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
        for inputs, targets in training_loader:
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_training_loss += float(loss.item()) * targets.shape[0]
            total_training_examples += targets.shape[0]

        validation_loss, validation_metrics = _evaluate(
            model,
            validation_loader,
            criterion,
            output_classes=output_classes,
            device=torch_device,
        )
        score = float(validation_metrics["macro_f1"])
        history.append(
            TrainingHistoryEntry(
                epoch=epoch,
                training_loss=total_training_loss / total_training_examples,
                validation_loss=validation_loss,
                validation_macro_f1=score,
                validation_balanced_accuracy=float(
                    validation_metrics["balanced_accuracy"]
                ),
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
        raise RuntimeError("Training completed without a validation-selected model state")
    model.load_state_dict(best_state)
    return TrainingOutcome(
        seed=seed,
        best_epoch=best_epoch,
        stopped_early=len(history) < optimization.maximum_epochs,
        history=tuple(history),
    )
