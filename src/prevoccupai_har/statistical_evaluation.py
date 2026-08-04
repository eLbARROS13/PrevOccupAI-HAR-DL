"""Participant-level uncertainty and paired comparisons for HAR models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParticipantBootstrapInterval:
    """Percentile interval obtained by resampling participants, not windows."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    participant_count: int
    resample_count: int
    random_seed: int
    interpretation: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class ExactParticipantBootstrapInterval:
    """Percentile interval from all ordered participant bootstrap resamples."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    participant_count: int
    enumeration_count: int
    unique_resampled_mean_count: int
    interpretation: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class PairedParticipantComparison:
    """Participant-paired model difference with descriptive uncertainty."""

    participant_differences: dict[str, float]
    mean_difference: float
    median_difference: float
    positive_participant_fraction: float
    bootstrap_interval: ParticipantBootstrapInterval
    exact_sign_flip_p_value: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["bootstrap_interval"] = self.bootstrap_interval.as_dict()
        return value


def _finite_participant_values(values: Mapping[str, float]) -> tuple[tuple[str, ...], np.ndarray]:
    if not values:
        raise ValueError("At least one participant value is required")
    participants = tuple(sorted(map(str, values)))
    if any(not participant for participant in participants):
        raise ValueError("Participant identifiers cannot be empty")
    numeric_values = np.asarray([values[participant] for participant in participants], dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Participant values must be finite")
    return participants, numeric_values


def participant_bootstrap_mean_interval(
    values: Mapping[str, float],
    *,
    confidence_level: float = 0.95,
    resample_count: int = 10_000,
    random_seed: int = 1103,
) -> ParticipantBootstrapInterval:
    """Estimate a percentile interval for a participant-level arithmetic mean."""
    _, numeric_values = _finite_participant_values(values)
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")
    if resample_count <= 0 or random_seed < 0:
        raise ValueError("Resample count must be positive and seed non-negative")
    generator = np.random.default_rng(random_seed)
    indices = generator.integers(
        0,
        numeric_values.size,
        size=(resample_count, numeric_values.size),
    )
    resampled_means = numeric_values[indices].mean(axis=1)
    tail_probability = (1 - confidence_level) / 2
    lower, upper = np.quantile(
        resampled_means,
        [tail_probability, 1 - tail_probability],
    )
    interpretation = (
        "highly_unstable_descriptive_interval"
        if numeric_values.size < 5
        else "participant_grouped_descriptive_interval"
    )
    return ParticipantBootstrapInterval(
        estimate=float(numeric_values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        participant_count=int(numeric_values.size),
        resample_count=resample_count,
        random_seed=random_seed,
        interpretation=interpretation,
    )


def exact_participant_bootstrap_mean_interval(
    values: Mapping[str, float],
    *,
    confidence_level: float = 0.95,
    maximum_enumerations: int = 1_000_000,
) -> ExactParticipantBootstrapInterval:
    """Enumerate every ordered participant resample for a mean interval.

    For ``n`` participants, the nonparametric bootstrap has ``n**n`` ordered
    resamples of size ``n``. Exhaustive enumeration avoids Monte Carlo error
    when that finite space is small, as in the four-participant final comparison.
    """
    _, numeric_values = _finite_participant_values(values)
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")
    if maximum_enumerations <= 0:
        raise ValueError("Maximum enumerations must be positive")
    participant_count = int(numeric_values.size)
    enumeration_count = participant_count**participant_count
    if enumeration_count > maximum_enumerations:
        raise ValueError(
            "Exact participant bootstrap would exceed the enumeration limit: "
            f"{enumeration_count} > {maximum_enumerations}"
        )
    resampled_means = np.fromiter(
        (
            float(numeric_values[np.asarray(indices, dtype=int)].mean())
            for indices in product(range(participant_count), repeat=participant_count)
        ),
        dtype=float,
        count=enumeration_count,
    )
    tail_probability = (1 - confidence_level) / 2
    lower, upper = np.quantile(
        resampled_means,
        [tail_probability, 1 - tail_probability],
    )
    return ExactParticipantBootstrapInterval(
        estimate=float(numeric_values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        participant_count=participant_count,
        enumeration_count=enumeration_count,
        unique_resampled_mean_count=int(np.unique(resampled_means).size),
        interpretation="exhaustive_participant_bootstrap_descriptive_interval",
    )


def exact_paired_sign_flip_p_value(differences: Sequence[float]) -> float:
    """Return an exact two-sided sign-flip p-value for a paired mean difference."""
    values = np.asarray(tuple(differences), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Finite paired differences are required")
    if values.size > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 participants")
    observed = abs(float(values.mean()))
    exceedances = 0
    assignment_count = 0
    for signs in product((-1.0, 1.0), repeat=values.size):
        permuted = abs(float(np.mean(values * np.asarray(signs))))
        exceedances += permuted >= observed - 1e-15
        assignment_count += 1
    return exceedances / assignment_count


def compare_paired_participant_metrics(
    candidate_values: Mapping[str, float],
    reference_values: Mapping[str, float],
    *,
    confidence_level: float = 0.95,
    resample_count: int = 10_000,
    random_seed: int = 1103,
) -> PairedParticipantComparison:
    """Compare one candidate and reference metric using participants as pairs."""
    candidate_participants, candidate = _finite_participant_values(candidate_values)
    reference_participants, reference = _finite_participant_values(reference_values)
    if candidate_participants != reference_participants:
        raise ValueError("Candidate and reference participant sets must match exactly")
    differences = candidate - reference
    difference_mapping = {
        participant: float(difference)
        for participant, difference in zip(
            candidate_participants,
            differences,
            strict=True,
        )
    }
    interval = participant_bootstrap_mean_interval(
        difference_mapping,
        confidence_level=confidence_level,
        resample_count=resample_count,
        random_seed=random_seed,
    )
    return PairedParticipantComparison(
        participant_differences=difference_mapping,
        mean_difference=float(differences.mean()),
        median_difference=float(np.median(differences)),
        positive_participant_fraction=float(np.mean(differences > 0)),
        bootstrap_interval=interval,
        exact_sign_flip_p_value=exact_paired_sign_flip_p_value(differences),
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Apply Holm's step-down family-wise-error adjustment."""
    values = np.asarray(tuple(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one p-value is required")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("P-values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running_maximum = 0.0
    family_size = values.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return tuple(float(value) for value in adjusted)
