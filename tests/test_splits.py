"""Tests for fixed hold-out and development-fold invariants."""

import pytest

from prevoccupai_har.splits import SubjectPartition, build_validation_folds


def test_validation_folds_cover_development_once_and_preserve_holdout() -> None:
    development = tuple(f"P{index:03d}" for index in range(3, 19))
    holdout = ("P001", "P002", "P019", "P020")

    folds = build_validation_folds(development, holdout, n_splits=5, random_seed=42)

    assert len(folds) == 5
    assert sorted(subject for fold in folds for subject in fold.validation) == list(development)
    for fold in folds:
        assert fold.holdout == holdout
        assert set(fold.training).isdisjoint(fold.validation)
        assert set(fold.training).isdisjoint(fold.holdout)
        assert set(fold.validation).isdisjoint(fold.holdout)
        assert set(fold.training) | set(fold.validation) == set(development)


def test_fold_construction_is_deterministic() -> None:
    development = tuple(f"P{index:03d}" for index in range(3, 19))
    holdout = ("P001", "P002")

    assert build_validation_folds(development, holdout) == build_validation_folds(
        development,
        holdout,
    )


def test_partition_rejects_overlap() -> None:
    fold = SubjectPartition(
        training=("P001", "P002"),
        validation=("P002",),
        holdout=("P003",),
        fold_index=0,
    )

    with pytest.raises(ValueError, match="pairwise disjoint"):
        fold.validate()

