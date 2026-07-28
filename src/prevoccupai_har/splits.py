"""Subject-disjoint split construction and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _normalise_subjects(values: Iterable[str]) -> tuple[str, ...]:
    subjects = tuple(sorted(map(str, values)))
    if len(subjects) != len(set(subjects)):
        raise ValueError("A split contains duplicate participant identifiers")
    return subjects


@dataclass(frozen=True)
class SubjectPartition:
    """One train/validation partition with an immutable external hold-out set."""

    training: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]
    fold_index: int

    def validate(self, expected_development: Iterable[str] | None = None) -> None:
        """Enforce pairwise disjointness and complete development-set coverage."""
        training = set(self.training)
        validation = set(self.validation)
        holdout = set(self.holdout)
        if not training or not validation or not holdout:
            raise ValueError("Training, validation, and hold-out sets must be non-empty")
        if training & validation or training & holdout or validation & holdout:
            raise ValueError("Training, validation, and hold-out sets must be pairwise disjoint")
        if expected_development is not None:
            expected = set(expected_development)
            if training | validation != expected:
                raise ValueError("Training and validation do not partition the development cohort")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "fold_index": self.fold_index,
            "training_participants": list(self.training),
            "validation_participants": list(self.validation),
            "holdout_participants": list(self.holdout),
        }


def build_validation_folds(
    development_participants: Sequence[str],
    holdout_participants: Sequence[str],
    *,
    n_splits: int = 5,
    random_seed: int = 42,
) -> tuple[SubjectPartition, ...]:
    """Build deterministic subject-disjoint validation folds within development data.

    Each development participant appears in validation exactly once. The external
    hold-out cohort is copied unchanged into every fold and is never used for model
    selection.
    """
    development = _normalise_subjects(development_participants)
    holdout = _normalise_subjects(holdout_participants)
    if set(development) & set(holdout):
        raise ValueError("Development and hold-out cohorts overlap")
    if n_splits < 2:
        raise ValueError("At least two validation folds are required")
    if n_splits > len(development):
        raise ValueError("The number of folds exceeds the number of development participants")

    generator = np.random.default_rng(random_seed)
    shuffled = list(generator.permutation(np.asarray(development, dtype=object)))
    validation_groups: list[list[str]] = [[] for _ in range(n_splits)]
    for participant_index, participant in enumerate(shuffled):
        validation_groups[participant_index % n_splits].append(str(participant))

    folds: list[SubjectPartition] = []
    development_set = set(development)
    for fold_index, validation_group in enumerate(validation_groups):
        validation = tuple(sorted(validation_group))
        training = tuple(sorted(development_set - set(validation)))
        fold = SubjectPartition(
            training=training,
            validation=validation,
            holdout=holdout,
            fold_index=fold_index,
        )
        fold.validate(expected_development=development)
        folds.append(fold)

    validation_occurrences = [
        participant for fold in folds for participant in fold.validation
    ]
    if sorted(validation_occurrences) != list(development):
        raise RuntimeError("Validation folds do not cover each development participant exactly once")
    return tuple(folds)

