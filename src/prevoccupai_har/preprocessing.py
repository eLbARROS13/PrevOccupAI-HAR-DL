"""Preprocessing components that explicitly enforce train-only fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrainOnlyChannelStandardizer:
    """Channel-wise z-score transform fitted only on authorised training subjects."""

    allowed_training_subjects: frozenset[str]
    mean_: NDArray[np.floating] | None = None
    scale_: NDArray[np.floating] | None = None

    @classmethod
    def for_subjects(cls, subjects: Iterable[str]) -> "TrainOnlyChannelStandardizer":
        """Create an unfitted standardizer authorised for a fixed training cohort."""
        allowed = frozenset(map(str, subjects))
        if not allowed:
            raise ValueError("At least one authorised training subject is required")
        return cls(allowed_training_subjects=allowed)

    def fit(
        self,
        windows: NDArray[np.floating],
        subject_ids: Iterable[str],
    ) -> "TrainOnlyChannelStandardizer":
        """Fit channel statistics after verifying every source subject is authorised."""
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("Windows must have shape (windows, samples, channels)")
        subjects = tuple(map(str, subject_ids))
        if len(subjects) != values.shape[0]:
            raise ValueError("One subject identifier is required for each window")
        observed_subjects = set(subjects)
        unauthorised = observed_subjects - self.allowed_training_subjects
        if unauthorised:
            raise ValueError(
                "Preprocessing fit received non-training subjects: "
                f"{sorted(unauthorised)}"
            )
        if values.shape[0] == 0:
            raise ValueError("Cannot fit preprocessing statistics on zero windows")

        self.mean_ = values.mean(axis=(0, 1))
        raw_scale = values.std(axis=(0, 1))
        self.scale_ = np.where(raw_scale == 0, 1.0, raw_scale)
        return self

    def transform(self, windows: NDArray[np.floating]) -> NDArray[np.float64]:
        """Apply previously fitted training statistics to any data partition."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fitted before transform")
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("Windows must have shape (windows, samples, channels)")
        if values.shape[-1] != self.mean_.shape[0]:
            raise ValueError("Window channel count does not match fitted statistics")
        return (values - self.mean_) / self.scale_

    def fit_transform(
        self,
        windows: NDArray[np.floating],
        subject_ids: Iterable[str],
    ) -> NDArray[np.float64]:
        """Fit on authorised training windows and transform those windows."""
        return self.fit(windows, subject_ids).transform(windows)

    def state_dict(self) -> dict[str, object]:
        """Return serialisable preprocessing provenance without participant data."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has no fitted state")
        return {
            "transform": type(self).__name__,
            "fit_subject_count": len(self.allowed_training_subjects),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

