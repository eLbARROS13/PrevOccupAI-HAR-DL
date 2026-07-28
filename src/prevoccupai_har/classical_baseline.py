"""Leakage-safe, participant-grouped Random Forest baseline reconstruction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import sklearn
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold, ParameterGrid
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_array, check_is_fitted
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "scikit-learn is required for prevoccupai_har.classical_baseline; "
        "install the 'classical' extra"
    ) from error

from .evaluation import evaluate_predictions
from .model_selection import SelectionFold
from .protocol import ProtocolConfiguration, load_protocol
from .provenance import (
    is_reproducible_source_revision,
    sha256_canonical_json,
    sha256_file,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SYNTHETIC_VALIDATION = "synthetic_validation"
DEVELOPMENT_SELECTION = "development_selection"


def _require_sha256(value: str, field_name: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA-256 value for {field_name}")


@dataclass(frozen=True)
class RandomForestReconstructionConfiguration:
    """Executable fold-local feature-selection and RF settings."""

    schema_version: int
    experiment_id: str
    status: str
    expected_candidate_feature_count: int
    variance_threshold: float
    absolute_correlation_threshold: float
    selected_feature_count: int
    balancing_strategy: str
    criteria: tuple[str, ...]
    estimator_counts: tuple[int, ...]
    maximum_depths: tuple[int | None, ...]
    inner_group_folds: int
    inner_selection_metric: str
    n_jobs: int
    random_seed: int

    def validate(self) -> None:
        """Validate the published-target and deterministic execution settings."""
        if self.schema_version != 1:
            raise ValueError("Unsupported Random Forest configuration schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.experiment_id) is None:
            raise ValueError("Random Forest experiment identifier is invalid")
        if self.status not in {"synthetic_validation_only", "frozen_for_development"}:
            raise ValueError("The RF configuration status is unsupported")
        if self.expected_candidate_feature_count < 2:
            raise ValueError("Expected feature count is too small")
        if self.variance_threshold < 0.0:
            raise ValueError("Variance threshold cannot be negative")
        if not 0.0 < self.absolute_correlation_threshold < 1.0:
            raise ValueError("Correlation threshold must be in (0, 1)")
        if not 1 <= self.selected_feature_count <= self.expected_candidate_feature_count:
            raise ValueError("Selected feature count is invalid")
        if (
            self.balancing_strategy
            != "per_participant_per_subactivity_training_fold_only"
        ):
            raise ValueError("RF balancing strategy is invalid")
        if not self.criteria or any(
            value not in {"gini", "entropy", "log_loss"}
            for value in self.criteria
        ):
            raise ValueError("Random Forest criteria are invalid")
        if not self.estimator_counts or any(value <= 0 for value in self.estimator_counts):
            raise ValueError("Random Forest estimator counts must be positive")
        if not self.maximum_depths or any(
            value is not None and value <= 0 for value in self.maximum_depths
        ):
            raise ValueError("Random Forest maximum depths are invalid")
        if self.inner_group_folds < 2:
            raise ValueError("At least two inner participant folds are required")
        if self.inner_selection_metric != "accuracy":
            raise ValueError("Published-target inner selection must use accuracy")
        if self.n_jobs != 1:
            raise ValueError("The governed RF reconstruction requires one job")
        if self.random_seed < 0:
            raise ValueError("Random seed cannot be negative")


def load_random_forest_reconstruction_configuration(
    path: Path | str,
) -> RandomForestReconstructionConfiguration:
    """Load the checked-in fold-local RF reconstruction configuration."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("Random Forest configuration must be a JSON object")
    input_value = decoded.get("input")
    pipeline_value = decoded.get("fold_local_pipeline")
    balancing_value = decoded.get("fold_local_balancing")
    grid_value = pipeline_value.get("hyperparameter_grid") if isinstance(
        pipeline_value, Mapping
    ) else None
    if not all(
        isinstance(value, Mapping)
        for value in (input_value, pipeline_value, balancing_value, grid_value)
    ):
        raise TypeError("RF input, pipeline, and hyperparameter grid must be objects")
    if str(pipeline_value["classifier"]) != "RandomForestClassifier":
        raise ValueError("The baseline classifier must be RandomForestClassifier")
    configuration = RandomForestReconstructionConfiguration(
        schema_version=int(decoded["schema_version"]),
        experiment_id=str(decoded["experiment_id"]),
        status=str(decoded["status"]),
        expected_candidate_feature_count=int(
            input_value["expected_candidate_feature_count"]
        ),
        variance_threshold=float(pipeline_value["variance_threshold"]),
        absolute_correlation_threshold=float(
            pipeline_value["absolute_correlation_threshold"]
        ),
        selected_feature_count=int(
            pipeline_value["anova_selected_feature_count"]
        ),
        balancing_strategy=str(balancing_value["strategy"]),
        criteria=tuple(map(str, grid_value["criterion"])),
        estimator_counts=tuple(int(value) for value in grid_value["n_estimators"]),
        maximum_depths=tuple(
            None if value is None else int(value) for value in grid_value["max_depth"]
        ),
        inner_group_folds=int(pipeline_value["inner_group_folds"]),
        inner_selection_metric=str(pipeline_value["inner_selection_metric"]),
        n_jobs=int(pipeline_value["n_jobs"]),
        random_seed=int(decoded["random_seed"]),
    )
    configuration.validate()
    return configuration


class AbsoluteCorrelationFilter(TransformerMixin, BaseEstimator):
    """Drop later features whose absolute training-fold correlation is too high."""

    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold

    def fit(
        self,
        features: NDArray[np.floating],
        labels: NDArray[np.integer] | None = None,
    ) -> "AbsoluteCorrelationFilter":
        """Fit a deterministic ordered support mask on training rows only."""
        del labels
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("Correlation threshold must be in (0, 1)")
        values = check_array(features, dtype=np.float64, ensure_min_features=1)
        self.n_features_in_ = values.shape[1]
        if self.n_features_in_ == 1:
            self.support_mask_ = np.asarray([True], dtype=bool)
            return self
        correlation = np.corrcoef(values, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=-1.0)
        support = np.ones(self.n_features_in_, dtype=bool)
        for feature_index in range(1, self.n_features_in_):
            retained_indices = np.flatnonzero(support[:feature_index])
            if retained_indices.size and np.any(
                np.abs(correlation[feature_index, retained_indices]) > self.threshold
            ):
                support[feature_index] = False
        if not support.any():
            raise ValueError("Correlation filter removed every feature")
        self.support_mask_ = support
        return self

    def transform(
        self,
        features: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Apply the train-fitted feature mask unchanged."""
        check_is_fitted(self, attributes=("support_mask_", "n_features_in_"))
        values = check_array(features, dtype=np.float64, ensure_min_features=1)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Correlation-filter feature count changed after fitting")
        return values[:, self.support_mask_]

    def get_support(self) -> NDArray[np.bool_]:
        """Return a copy of the train-fitted support mask."""
        check_is_fitted(self, attributes=("support_mask_",))
        return self.support_mask_.copy()


def build_leakage_safe_random_forest_pipeline(
    configuration: RandomForestReconstructionConfiguration,
    *,
    random_seed: int,
) -> Pipeline:
    """Place every learned feature transform inside the estimator pipeline."""
    configuration.validate()
    if random_seed < 0:
        raise ValueError("Random seed cannot be negative")
    return Pipeline(
        steps=(
            (
                "variance_filter",
                VarianceThreshold(threshold=configuration.variance_threshold),
            ),
            (
                "correlation_filter",
                AbsoluteCorrelationFilter(
                    threshold=configuration.absolute_correlation_threshold
                ),
            ),
            (
                "anova_selector",
                SelectKBest(score_func=f_classif, k=configuration.selected_feature_count),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    random_state=random_seed,
                    n_jobs=configuration.n_jobs,
                ),
            ),
        )
    )


def select_fold_local_balanced_training_rows(
    *,
    labels: Sequence[str],
    subactivity_labels: Sequence[str],
    participant_ids: Sequence[str],
    candidate_indices: Sequence[int],
    class_labels: Sequence[str],
    random_seed: int,
) -> tuple[NDArray[np.int64], dict[str, int], dict[str, int]]:
    """Trim participant/sub-activity cells using training-fold counts only.

    The rule reconstructs the camera-ready intent while moving every count and
    sampling decision inside the current training fold. For each main class, its
    smallest participant-by-sub-activity cell defines a feasible per-participant
    class total. The smallest feasible total across main classes is the common
    target; integer per-sub-activity quotas are then used, matching the recovered
    implementation's conservative floor operation. Validation rows are never
    candidates for selection.
    """
    targets = np.asarray(tuple(map(str, labels)), dtype=object)
    subactivities = np.asarray(tuple(map(str, subactivity_labels)), dtype=object)
    subjects = np.asarray(tuple(map(str, participant_ids)), dtype=object)
    classes = tuple(map(str, class_labels))
    indices = np.asarray(tuple(int(index) for index in candidate_indices), dtype=np.int64)
    if len(targets) != len(subactivities) or len(targets) != len(subjects):
        raise ValueError("Balancing labels and participants must align")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Balancing class labels must be non-empty and unique")
    if indices.ndim != 1 or indices.size == 0 or len(set(indices.tolist())) != len(indices):
        raise ValueError("Balancing candidate indices must be non-empty and unique")
    if indices.min() < 0 or indices.max() >= len(targets):
        raise IndexError("Balancing candidate index is outside the feature matrix")
    if random_seed < 0:
        raise ValueError("Balancing seed cannot be negative")
    if any(not value for value in subjects[indices]) or any(
        not value for value in subactivities[indices]
    ):
        raise ValueError("Balancing participant and sub-activity labels cannot be empty")
    if set(targets[indices].tolist()) != set(classes):
        raise ValueError("Balancing candidates must contain every declared class")

    subactivities_by_class: dict[str, tuple[str, ...]] = {}
    for class_label in classes:
        class_subactivities = tuple(
            sorted(set(subactivities[indices][targets[indices] == class_label].tolist()))
        )
        if not class_subactivities:
            raise ValueError("Every class requires at least one sub-activity")
        subactivities_by_class[class_label] = class_subactivities
    for subactivity in sorted(set(subactivities[indices].tolist())):
        mapped_classes = set(targets[indices][subactivities[indices] == subactivity].tolist())
        if len(mapped_classes) != 1:
            raise ValueError("A sub-activity must map to exactly one main class")

    training_subjects = tuple(sorted(set(subjects[indices].tolist())))
    feasible_totals: dict[str, int] = {}
    for class_label, class_subactivities in subactivities_by_class.items():
        cell_counts = []
        for participant in training_subjects:
            for subactivity in class_subactivities:
                count = int(
                    np.sum(
                        (subjects[indices] == participant)
                        & (subactivities[indices] == subactivity)
                    )
                )
                if count == 0:
                    raise ValueError(
                        "Every training participant must contain every declared sub-activity"
                    )
                cell_counts.append(count)
        feasible_totals[class_label] = min(cell_counts) * len(class_subactivities)
    common_main_class_target = min(feasible_totals.values())
    quotas = {
        class_label: common_main_class_target // len(class_subactivities)
        for class_label, class_subactivities in subactivities_by_class.items()
    }
    if any(quota <= 0 for quota in quotas.values()):
        raise ValueError("Fold-local balancing produced an empty quota")

    generator = np.random.default_rng(random_seed)
    selected: list[int] = []
    for participant in training_subjects:
        for class_label in classes:
            for subactivity in subactivities_by_class[class_label]:
                cell_indices = indices[
                    (subjects[indices] == participant)
                    & (subactivities[indices] == subactivity)
                ]
                selected.extend(
                    map(
                        int,
                        generator.choice(
                            cell_indices,
                            size=quotas[class_label],
                            replace=False,
                        ).tolist(),
                    )
                )
    selected_indices = np.asarray(sorted(selected), dtype=np.int64)
    class_counts = {
        class_label: int(np.sum(targets[selected_indices] == class_label))
        for class_label in classes
    }
    quota_by_subactivity = {
        subactivity: quotas[class_label]
        for class_label, class_subactivities in subactivities_by_class.items()
        for subactivity in class_subactivities
    }
    return selected_indices, class_counts, quota_by_subactivity


@dataclass(frozen=True)
class ScientificFeatureProvenance:
    """Governed input hashes required for a future scientific RF run."""

    raw_recording_manifest_sha256: str
    segmentation_manifest_sha256: str
    quality_manifest_sha256: str
    split_manifest_sha256: str
    signal_preprocessing_configuration_sha256: str
    feature_extraction_configuration_sha256: str
    feature_matrix_file_sha256: str

    def validate(self) -> None:
        """Require a SHA-256 value for every scientific input artifact."""
        for field_name, value in asdict(self).items():
            _require_sha256(str(value), field_name)


@dataclass(frozen=True)
class FeaturePredictionRow:
    """One path-free validation prediction from a governed feature row."""

    source_row_index: int
    participant_id: str
    true_label: str
    predicted_label: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class RandomForestFoldResult:
    """One outer development-fold result after inner grouped selection."""

    fold_index: int
    training_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    training_row_count: int
    balanced_training_row_count: int
    balanced_training_class_counts: dict[str, int]
    balancing_quota_by_subactivity: dict[str, int]
    validation_row_count: int
    inner_best_accuracy: float
    best_hyperparameters: dict[str, object]
    selected_feature_names: tuple[str, ...]
    predictions: tuple[FeaturePredictionRow, ...]
    validation_metrics: dict[str, object]

    def validate(
        self,
        *,
        class_labels: tuple[str, ...],
        expected_selected_feature_count: int,
    ) -> None:
        """Validate scope, predictions, selected features, and regenerated metrics."""
        if self.fold_index < 0:
            raise ValueError("Fold index cannot be negative")
        training = set(self.training_subjects)
        validation = set(self.validation_subjects)
        if not training or not validation or training & validation:
            raise ValueError("RF fold subjects must be non-empty and disjoint")
        if len(training) != len(self.training_subjects) or len(validation) != len(
            self.validation_subjects
        ):
            raise ValueError("RF fold subject identifiers contain duplicates")
        if (
            self.training_row_count <= 0
            or not 0 < self.balanced_training_row_count <= self.training_row_count
            or self.validation_row_count != len(
            self.predictions
            )
        ):
            raise ValueError("RF fold row counts are invalid")
        if set(self.balanced_training_class_counts) != set(class_labels) or any(
            not isinstance(count, int) or count <= 0
            for count in self.balanced_training_class_counts.values()
        ):
            raise ValueError("RF balanced training class counts are invalid")
        if sum(self.balanced_training_class_counts.values()) != self.balanced_training_row_count:
            raise ValueError("RF balanced class counts disagree with the retained row count")
        if not self.balancing_quota_by_subactivity or any(
            not subactivity or not isinstance(quota, int) or quota <= 0
            for subactivity, quota in self.balancing_quota_by_subactivity.items()
        ):
            raise ValueError("RF balancing quotas are invalid")
        if not math.isfinite(self.inner_best_accuracy) or not 0.0 <= self.inner_best_accuracy <= 1.0:
            raise ValueError("RF inner accuracy must be finite and in [0, 1]")
        if set(self.best_hyperparameters) != {
            "criterion",
            "max_depth",
            "n_estimators",
        }:
            raise ValueError("RF best hyperparameters are incomplete")
        if self.best_hyperparameters["criterion"] not in {
            "gini",
            "entropy",
            "log_loss",
        }:
            raise ValueError("RF best criterion is invalid")
        maximum_depth = self.best_hyperparameters["max_depth"]
        if maximum_depth is not None and (
            not isinstance(maximum_depth, int) or maximum_depth <= 0
        ):
            raise ValueError("RF best maximum depth is invalid")
        estimator_count = self.best_hyperparameters["n_estimators"]
        if not isinstance(estimator_count, int) or estimator_count <= 0:
            raise ValueError("RF best estimator count is invalid")
        if len(self.selected_feature_names) != expected_selected_feature_count or len(
            set(self.selected_feature_names)
        ) != expected_selected_feature_count:
            raise ValueError("RF selected feature names are incomplete or duplicated")
        if not self.predictions:
            raise ValueError("RF fold predictions cannot be empty")
        row_indices = tuple(row.source_row_index for row in self.predictions)
        if len(row_indices) != len(set(row_indices)) or any(index < 0 for index in row_indices):
            raise ValueError("RF prediction row indices must be unique and non-negative")
        observed_subjects = {row.participant_id for row in self.predictions}
        if observed_subjects != validation:
            raise ValueError("RF predictions must exactly cover validation subjects")
        if any(
            row.true_label not in class_labels or row.predicted_label not in class_labels
            for row in self.predictions
        ):
            raise ValueError("RF predictions contain an undeclared label")
        regenerated = evaluate_predictions(
            [row.true_label for row in self.predictions],
            [row.predicted_label for row in self.predictions],
            [row.participant_id for row in self.predictions],
            class_labels,
        ).as_dict()
        if sha256_canonical_json(regenerated) != sha256_canonical_json(
            self.validation_metrics
        ):
            raise ValueError("RF validation metrics disagree with retained predictions")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "fold_index": self.fold_index,
            "training_subjects": list(self.training_subjects),
            "validation_subjects": list(self.validation_subjects),
            "training_row_count": self.training_row_count,
            "balanced_training_row_count": self.balanced_training_row_count,
            "balanced_training_class_counts": self.balanced_training_class_counts,
            "balancing_quota_by_subactivity": self.balancing_quota_by_subactivity,
            "validation_row_count": self.validation_row_count,
            "inner_best_accuracy": self.inner_best_accuracy,
            "best_hyperparameters": self.best_hyperparameters,
            "selected_feature_names": list(self.selected_feature_names),
            "predictions": [row.as_dict() for row in self.predictions],
            "validation_metrics": self.validation_metrics,
        }


@dataclass(frozen=True)
class RandomForestDevelopmentRecord:
    """Immutable, no-hold-out RF outer-fold evaluation artifact."""

    schema_version: int
    run_id: str
    created_at_utc: str
    experiment_id: str
    purpose: str
    scientific_result: bool
    holdout_accessed: bool
    source_revision: str
    model_configuration_sha256: str
    feature_matrix_sha256: str
    protocol_configuration_sha256: str | None
    data_provenance: ScientificFeatureProvenance | None
    random_seed: int
    class_labels: tuple[str, ...]
    holdout_subjects: tuple[str, ...]
    expected_selected_feature_count: int
    folds: tuple[RandomForestFoldResult, ...]
    result_payload_sha256: str
    software_versions: dict[str, str]

    def _payload(self) -> list[dict[str, object]]:
        return [fold.as_dict() for fold in self.folds]

    def validate(self) -> None:
        """Reject under-provenanced, incomplete, or hold-out-like RF records."""
        if self.schema_version != 1:
            raise ValueError("Unsupported RF development-result schema version")
        if IDENTIFIER_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("RF run identifier is invalid")
        if UTC_PATTERN.fullmatch(self.created_at_utc) is None:
            raise ValueError("Creation time must use second-resolution UTC with a Z suffix")
        if not self.experiment_id or not self.source_revision:
            raise ValueError("RF experiment and source revision are required")
        for field_name in (
            "model_configuration_sha256",
            "feature_matrix_sha256",
            "result_payload_sha256",
        ):
            _require_sha256(str(getattr(self, field_name)), field_name)
        if self.holdout_accessed:
            raise ValueError("RF development records cannot claim hold-out access")
        if self.random_seed < 0 or self.expected_selected_feature_count <= 0:
            raise ValueError("RF seed and selected feature count are invalid")
        if not self.class_labels or len(self.class_labels) != len(set(self.class_labels)):
            raise ValueError("RF class labels must be non-empty and unique")
        if not self.holdout_subjects or len(self.holdout_subjects) != len(
            set(self.holdout_subjects)
        ):
            raise ValueError("RF hold-out exclusion set must be non-empty and unique")
        if self.purpose == SYNTHETIC_VALIDATION:
            if self.scientific_result:
                raise ValueError("Synthetic RF evaluation cannot be scientific")
            if (
                self.protocol_configuration_sha256 is not None
                or self.data_provenance is not None
            ):
                raise ValueError("Synthetic RF records must omit scientific provenance")
        elif self.purpose == DEVELOPMENT_SELECTION:
            if not self.scientific_result:
                raise ValueError("Development RF evaluation is a scientific result")
            if self.protocol_configuration_sha256 is None or self.data_provenance is None:
                raise ValueError("Scientific RF records require complete provenance")
            _require_sha256(
                self.protocol_configuration_sha256,
                "protocol_configuration_sha256",
            )
            if not is_reproducible_source_revision(self.source_revision):
                raise ValueError(
                    "Scientific RF records require an immutable source revision"
                )
            self.data_provenance.validate()
        else:
            raise ValueError("RF development-result purpose is unsupported")
        if not self.folds or tuple(fold.fold_index for fold in self.folds) != tuple(
            range(len(self.folds))
        ):
            raise ValueError("RF folds must be non-empty, ordered, and contiguous")
        development_sets: set[frozenset[str]] = set()
        validation_occurrences: list[str] = []
        for fold in self.folds:
            fold.validate(
                class_labels=self.class_labels,
                expected_selected_feature_count=self.expected_selected_feature_count,
            )
            development_sets.add(
                frozenset(fold.training_subjects) | frozenset(fold.validation_subjects)
            )
            validation_occurrences.extend(fold.validation_subjects)
        if len(development_sets) != 1 or sorted(validation_occurrences) != sorted(
            next(iter(development_sets))
        ):
            raise ValueError("RF folds must validate every development participant once")
        if next(iter(development_sets)) & frozenset(self.holdout_subjects):
            raise ValueError("RF development and hold-out subjects overlap")
        if self.purpose == SYNTHETIC_VALIDATION and any(
            not subject.startswith("SYNTHETIC_")
            for subject in next(iter(development_sets)) | frozenset(self.holdout_subjects)
        ):
            raise ValueError("Synthetic RF records require synthetic participants")
        if sha256_canonical_json(self._payload()) != self.result_payload_sha256:
            raise ValueError("RF result payload digest does not match")

    def as_dict(self) -> dict[str, object]:
        """Return a validated JSON-compatible representation."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "experiment_id": self.experiment_id,
            "purpose": self.purpose,
            "scientific_result": self.scientific_result,
            "holdout_accessed": self.holdout_accessed,
            "source_revision": self.source_revision,
            "model_configuration_sha256": self.model_configuration_sha256,
            "feature_matrix_sha256": self.feature_matrix_sha256,
            "protocol_configuration_sha256": self.protocol_configuration_sha256,
            "data_provenance": (
                None if self.data_provenance is None else asdict(self.data_provenance)
            ),
            "random_seed": self.random_seed,
            "class_labels": list(self.class_labels),
            "holdout_subjects": list(self.holdout_subjects),
            "expected_selected_feature_count": self.expected_selected_feature_count,
            "folds": self._payload(),
            "result_payload_sha256": self.result_payload_sha256,
            "software_versions": self.software_versions,
        }


def sha256_feature_matrix(
    features: NDArray[np.floating],
    *,
    labels: Sequence[str],
    subactivity_labels: Sequence[str],
    participant_ids: Sequence[str],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
) -> str:
    """Hash exact feature bytes plus aligned semantic metadata without paths."""
    values = np.ascontiguousarray(np.asarray(features, dtype=np.float64))
    metadata = json.dumps(
        {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "labels": list(map(str, labels)),
            "subactivity_labels": list(map(str, subactivity_labels)),
            "participant_ids": list(map(str, participant_ids)),
            "feature_names": list(map(str, feature_names)),
            "class_labels": list(map(str, class_labels)),
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, byteorder="big"))
    digest.update(metadata)
    payload = values.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, byteorder="big"))
    digest.update(payload)
    return digest.hexdigest()


def _selected_feature_names(
    pipeline: Pipeline,
    feature_names: tuple[str, ...],
) -> tuple[str, ...]:
    names = np.asarray(feature_names, dtype=object)
    names = names[pipeline.named_steps["variance_filter"].get_support()]
    names = names[pipeline.named_steps["correlation_filter"].get_support()]
    names = names[pipeline.named_steps["anova_selector"].get_support()]
    return tuple(map(str, names.tolist()))


def evaluate_random_forest_development_folds(
    *,
    run_id: str,
    created_at_utc: str,
    configuration_path: Path | str,
    configuration: RandomForestReconstructionConfiguration,
    features: NDArray[np.floating],
    labels: Sequence[str],
    subactivity_labels: Sequence[str],
    participant_ids: Sequence[str],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    folds: Sequence[SelectionFold],
    purpose: str,
    source_revision: str,
    protocol: ProtocolConfiguration | None = None,
    protocol_configuration_path: Path | str | None = None,
    data_provenance: ScientificFeatureProvenance | None = None,
) -> RandomForestDevelopmentRecord:
    """Run grouped inner selection inside every shared outer development fold."""
    configuration.validate()
    configuration_path = Path(configuration_path)
    if load_random_forest_reconstruction_configuration(
        configuration_path
    ) != configuration:
        raise ValueError("RF configuration object disagrees with its governed file")
    values = np.asarray(features, dtype=np.float64)
    targets = tuple(map(str, labels))
    subactivities = tuple(map(str, subactivity_labels))
    subjects = tuple(map(str, participant_ids))
    names = tuple(map(str, feature_names))
    classes = tuple(map(str, class_labels))
    partitions = tuple(folds)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Feature matrix must be non-empty, two-dimensional, and finite")
    if (
        values.shape[0] != len(targets)
        or len(targets) != len(subactivities)
        or len(targets) != len(subjects)
    ):
        raise ValueError(
            "Features, main/sub-activity labels, and participant identifiers must align"
        )
    if any(not value for value in subactivities):
        raise ValueError("Sub-activity labels cannot be empty")
    if values.shape[1] != len(names) or len(names) != len(set(names)):
        raise ValueError("Feature names must be unique and align with matrix columns")
    if values.shape[1] != configuration.expected_candidate_feature_count:
        raise ValueError("Feature count disagrees with the RF reconstruction config")
    if not classes or len(classes) != len(set(classes)) or set(targets) - set(classes):
        raise ValueError("Class labels are empty, duplicated, or incomplete")
    if not partitions:
        raise ValueError("At least one outer RF fold is required")
    for partition in partitions:
        partition.validate()
    dataset_subjects = set(subjects)
    development_sets = {
        frozenset(partition.training_subjects)
        | frozenset(partition.validation_subjects)
        for partition in partitions
    }
    holdout_sets = {
        frozenset(partition.holdout_subjects) for partition in partitions
    }
    if len(development_sets) != 1 or len(holdout_sets) != 1:
        raise ValueError("RF folds must share one development and hold-out cohort")
    development = next(iter(development_sets))
    holdout = next(iter(holdout_sets))
    if dataset_subjects & set(holdout):
        raise PermissionError("RF feature matrix cannot contain external hold-out subjects")
    if dataset_subjects != set(development):
        raise ValueError("Feature matrix must exactly contain development participants")
    occurrences = [
        subject for partition in partitions for subject in partition.validation_subjects
    ]
    if sorted(occurrences) != sorted(development):
        raise ValueError("RF outer folds must validate each development participant once")
    expected_scientific = purpose == DEVELOPMENT_SELECTION
    if purpose == SYNTHETIC_VALIDATION:
        if (
            protocol is not None
            or protocol_configuration_path is not None
            or data_provenance is not None
        ):
            raise ValueError("Synthetic RF evaluation must omit scientific provenance")
        if any(not subject.startswith("SYNTHETIC_") for subject in dataset_subjects | set(holdout)):
            raise ValueError("Synthetic RF evaluation requires synthetic participants")
    elif expected_scientific:
        if protocol is None or not protocol.training_authorized:
            raise PermissionError("Protocol does not authorize scientific RF evaluation")
        if protocol_configuration_path is None or data_provenance is None:
            raise ValueError("Scientific RF evaluation requires complete provenance")
        protocol_configuration_path = Path(protocol_configuration_path)
        if load_protocol(protocol_configuration_path) != protocol:
            raise ValueError("Protocol object disagrees with its governed file")
        data_provenance.validate()
        if set(protocol.development_participants) != dataset_subjects or set(
            protocol.holdout_participants
        ) != set(holdout):
            raise ValueError("RF folds disagree with the authorized protocol")
        if not is_reproducible_source_revision(source_revision):
            raise ValueError(
                "Scientific RF evaluation requires an immutable source revision"
            )
    else:
        raise ValueError("RF evaluation purpose is unsupported")

    target_array = np.asarray(targets, dtype=object)
    subject_array = np.asarray(subjects, dtype=object)
    fold_results: list[RandomForestFoldResult] = []
    parameter_grid = tuple(
        ParameterGrid(
            {
                "criterion": list(configuration.criteria),
                "n_estimators": list(configuration.estimator_counts),
                "max_depth": list(configuration.maximum_depths),
            }
        )
    )
    for partition in partitions:
        training_mask = np.isin(subject_array, partition.training_subjects)
        validation_mask = np.isin(subject_array, partition.validation_subjects)
        if set(target_array[training_mask]) != set(classes) or set(
            target_array[validation_mask]
        ) != set(classes):
            raise ValueError("Every RF outer partition must contain all declared classes")
        training_groups = subject_array[training_mask]
        if len(set(training_groups.tolist())) < configuration.inner_group_folds:
            raise ValueError("RF training fold has too few participants for inner CV")
        outer_training_indices = np.flatnonzero(training_mask)
        inner_cv = GroupKFold(
            n_splits=configuration.inner_group_folds,
            shuffle=True,
            random_state=configuration.random_seed,
        )
        inner_partitions = tuple(
            inner_cv.split(
                values[outer_training_indices],
                target_array[outer_training_indices],
                groups=training_groups,
            )
        )
        balanced_inner_training_indices = tuple(
            select_fold_local_balanced_training_rows(
                labels=targets,
                subactivity_labels=subactivities,
                participant_ids=subjects,
                candidate_indices=outer_training_indices[inner_training_relative],
                class_labels=classes,
                random_seed=(
                    configuration.random_seed
                    + partition.fold_index * 1000
                    + inner_fold_index
                ),
            )[0]
            for inner_fold_index, (inner_training_relative, _) in enumerate(
                inner_partitions
            )
        )
        best_parameters: dict[str, object] | None = None
        best_inner_accuracy = -math.inf
        for parameters in parameter_grid:
            inner_scores: list[float] = []
            for inner_fold_index, (_, inner_validation_relative) in enumerate(
                inner_partitions
            ):
                pipeline = build_leakage_safe_random_forest_pipeline(
                    configuration,
                    random_seed=configuration.random_seed,
                )
                pipeline.set_params(
                    random_forest__criterion=parameters["criterion"],
                    random_forest__n_estimators=parameters["n_estimators"],
                    random_forest__max_depth=parameters["max_depth"],
                )
                balanced_indices = balanced_inner_training_indices[inner_fold_index]
                inner_validation_indices = outer_training_indices[
                    inner_validation_relative
                ]
                pipeline.fit(values[balanced_indices], target_array[balanced_indices])
                inner_predictions = pipeline.predict(values[inner_validation_indices])
                inner_scores.append(
                    float(
                        accuracy_score(
                            target_array[inner_validation_indices],
                            inner_predictions,
                        )
                    )
                )
            mean_inner_accuracy = float(np.mean(inner_scores))
            if mean_inner_accuracy > best_inner_accuracy:
                best_inner_accuracy = mean_inner_accuracy
                best_parameters = {
                    "criterion": str(parameters["criterion"]),
                    "max_depth": (
                        None
                        if parameters["max_depth"] is None
                        else int(parameters["max_depth"])
                    ),
                    "n_estimators": int(parameters["n_estimators"]),
                }
        if best_parameters is None:
            raise RuntimeError("RF parameter grid produced no candidate")
        (
            balanced_outer_training_indices,
            balanced_training_class_counts,
            balancing_quota_by_subactivity,
        ) = select_fold_local_balanced_training_rows(
            labels=targets,
            subactivity_labels=subactivities,
            participant_ids=subjects,
            candidate_indices=outer_training_indices,
            class_labels=classes,
            random_seed=(
                configuration.random_seed + partition.fold_index * 1000 + 999
            ),
        )
        best_pipeline = build_leakage_safe_random_forest_pipeline(
            configuration,
            random_seed=configuration.random_seed,
        )
        best_pipeline.set_params(
            random_forest__criterion=best_parameters["criterion"],
            random_forest__n_estimators=best_parameters["n_estimators"],
            random_forest__max_depth=best_parameters["max_depth"],
        )
        best_pipeline.fit(
            values[balanced_outer_training_indices],
            target_array[balanced_outer_training_indices],
        )
        predictions = tuple(
            map(str, best_pipeline.predict(values[validation_mask]).tolist())
        )
        validation_indices = np.flatnonzero(validation_mask)
        prediction_rows = tuple(
            FeaturePredictionRow(
                source_row_index=int(source_index),
                participant_id=str(subject_array[source_index]),
                true_label=str(target_array[source_index]),
                predicted_label=prediction,
            )
            for source_index, prediction in zip(
                validation_indices,
                predictions,
                strict=True,
            )
        )
        metrics = evaluate_predictions(
            [row.true_label for row in prediction_rows],
            [row.predicted_label for row in prediction_rows],
            [row.participant_id for row in prediction_rows],
            classes,
        ).as_dict()
        fold_result = RandomForestFoldResult(
            fold_index=partition.fold_index,
            training_subjects=partition.training_subjects,
            validation_subjects=partition.validation_subjects,
            training_row_count=int(training_mask.sum()),
            balanced_training_row_count=len(balanced_outer_training_indices),
            balanced_training_class_counts=balanced_training_class_counts,
            balancing_quota_by_subactivity=balancing_quota_by_subactivity,
            validation_row_count=int(validation_mask.sum()),
            inner_best_accuracy=best_inner_accuracy,
            best_hyperparameters=best_parameters,
            selected_feature_names=_selected_feature_names(
                best_pipeline,
                names,
            ),
            predictions=prediction_rows,
            validation_metrics=metrics,
        )
        fold_result.validate(
            class_labels=classes,
            expected_selected_feature_count=configuration.selected_feature_count,
        )
        fold_results.append(fold_result)
    fold_results_tuple = tuple(sorted(fold_results, key=lambda result: result.fold_index))
    payload = [fold.as_dict() for fold in fold_results_tuple]
    record = RandomForestDevelopmentRecord(
        schema_version=1,
        run_id=run_id,
        created_at_utc=created_at_utc,
        experiment_id=configuration.experiment_id,
        purpose=purpose,
        scientific_result=expected_scientific,
        holdout_accessed=False,
        source_revision=source_revision,
        model_configuration_sha256=sha256_file(configuration_path),
        feature_matrix_sha256=sha256_feature_matrix(
            values,
            labels=targets,
            subactivity_labels=subactivities,
            participant_ids=subjects,
            feature_names=names,
            class_labels=classes,
        ),
        protocol_configuration_sha256=(
            None
            if protocol_configuration_path is None
            else sha256_file(protocol_configuration_path)
        ),
        data_provenance=data_provenance,
        random_seed=configuration.random_seed,
        class_labels=classes,
        holdout_subjects=tuple(sorted(holdout)),
        expected_selected_feature_count=configuration.selected_feature_count,
        folds=fold_results_tuple,
        result_payload_sha256=sha256_canonical_json(payload),
        software_versions={
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    )
    record.validate()
    return record


def write_random_forest_development_record(
    path: Path | str,
    record: RandomForestDevelopmentRecord,
) -> None:
    """Write an RF development artifact exclusively."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(record.as_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_random_forest_development_record(
    path: Path | str,
) -> RandomForestDevelopmentRecord:
    """Load and validate an immutable RF development artifact."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise TypeError("RF development artifact must be a JSON object")
    provenance_value = decoded.get("data_provenance")
    if provenance_value is None:
        provenance = None
    elif isinstance(provenance_value, Mapping):
        provenance = ScientificFeatureProvenance(
            raw_recording_manifest_sha256=str(
                provenance_value["raw_recording_manifest_sha256"]
            ),
            segmentation_manifest_sha256=str(
                provenance_value["segmentation_manifest_sha256"]
            ),
            quality_manifest_sha256=str(
                provenance_value["quality_manifest_sha256"]
            ),
            split_manifest_sha256=str(provenance_value["split_manifest_sha256"]),
            signal_preprocessing_configuration_sha256=str(
                provenance_value["signal_preprocessing_configuration_sha256"]
            ),
            feature_extraction_configuration_sha256=str(
                provenance_value["feature_extraction_configuration_sha256"]
            ),
            feature_matrix_file_sha256=str(
                provenance_value["feature_matrix_file_sha256"]
            ),
        )
    else:
        raise TypeError("RF data provenance must be an object or null")
    folds = tuple(
        RandomForestFoldResult(
            fold_index=int(value["fold_index"]),
            training_subjects=tuple(map(str, value["training_subjects"])),
            validation_subjects=tuple(map(str, value["validation_subjects"])),
            training_row_count=int(value["training_row_count"]),
            balanced_training_row_count=int(
                value["balanced_training_row_count"]
            ),
            balanced_training_class_counts={
                str(label): int(count)
                for label, count in value[
                    "balanced_training_class_counts"
                ].items()
            },
            balancing_quota_by_subactivity={
                str(label): int(count)
                for label, count in value[
                    "balancing_quota_by_subactivity"
                ].items()
            },
            validation_row_count=int(value["validation_row_count"]),
            inner_best_accuracy=float(value["inner_best_accuracy"]),
            best_hyperparameters=dict(value["best_hyperparameters"]),
            selected_feature_names=tuple(map(str, value["selected_feature_names"])),
            predictions=tuple(
                FeaturePredictionRow(
                    source_row_index=int(row["source_row_index"]),
                    participant_id=str(row["participant_id"]),
                    true_label=str(row["true_label"]),
                    predicted_label=str(row["predicted_label"]),
                )
                for row in value["predictions"]
            ),
            validation_metrics=dict(value["validation_metrics"]),
        )
        for value in decoded["folds"]
    )
    record = RandomForestDevelopmentRecord(
        schema_version=int(decoded["schema_version"]),
        run_id=str(decoded["run_id"]),
        created_at_utc=str(decoded["created_at_utc"]),
        experiment_id=str(decoded["experiment_id"]),
        purpose=str(decoded["purpose"]),
        scientific_result=decoded.get("scientific_result") is True,
        holdout_accessed=decoded.get("holdout_accessed") is True,
        source_revision=str(decoded["source_revision"]),
        model_configuration_sha256=str(decoded["model_configuration_sha256"]),
        feature_matrix_sha256=str(decoded["feature_matrix_sha256"]),
        protocol_configuration_sha256=(
            None
            if decoded.get("protocol_configuration_sha256") is None
            else str(decoded["protocol_configuration_sha256"])
        ),
        data_provenance=provenance,
        random_seed=int(decoded["random_seed"]),
        class_labels=tuple(map(str, decoded["class_labels"])),
        holdout_subjects=tuple(map(str, decoded["holdout_subjects"])),
        expected_selected_feature_count=int(
            decoded["expected_selected_feature_count"]
        ),
        folds=folds,
        result_payload_sha256=str(decoded["result_payload_sha256"]),
        software_versions={
            str(key): str(value)
            for key, value in decoded["software_versions"].items()
        },
    )
    record.validate()
    return record
