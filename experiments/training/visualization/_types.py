"""Type definitions for bespoke training visualizations."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import numpy as np


class CEVHistoryEntry(TypedDict):
    """Single snapshot of cumulative explained variance at a training step."""

    step: int
    cumvar: np.ndarray


class CEVHistoryEntryWithMetadata(CEVHistoryEntry, total=False):
    """Optional metadata that may be attached to a CEV history entry."""

    d_eff: NotRequired[float]
    n_samples: NotRequired[int]
    timestamp: NotRequired[float]


CEVHistory = dict[str, list[CEVHistoryEntryWithMetadata]]


class BeliefRegressionHistoryEntry(TypedDict):
    """Single snapshot of belief regression metrics at a training step."""

    step: int
    overall_rmse: float
    factor_rmse_scores: list[float]


class BeliefRegressionHistoryEntryWithMetadata(BeliefRegressionHistoryEntry, total=False):
    """Optional metadata that may be attached to a belief regression history entry."""

    num_factors: NotRequired[int]
    belief_dim_per_factor: NotRequired[int]
    d_model: NotRequired[int]
    num_samples: NotRequired[int]
    timestamp: NotRequired[float]


BeliefRegressionHistory = dict[str, list[BeliefRegressionHistoryEntryWithMetadata]]


class OrthogonalityHistoryEntry(TypedDict):
    """Single snapshot of orthogonality metrics for a factor pair at a training step."""

    step: int
    singular_values: np.ndarray  # Raw SVs = cos(principal angles)
    overlap: float  # mean(sv²) - current metric
    mean_sv: float  # mean(sv) - average alignment
    sv_max: float  # Largest SV = cos(smallest angle)
    sv_min: float  # Smallest SV = cos(largest angle)


class OrthogonalityHistoryEntryWithMetadata(OrthogonalityHistoryEntry, total=False):
    """Optional metadata for orthogonality history entry."""

    p_ratio: NotRequired[float]
    entropy: NotRequired[float]
    eff_rank: NotRequired[float]
    timestamp: NotRequired[float]


# Maps factor pair string "F{i},{j}" to list of history entries
OrthogonalityHistory = dict[str, list[OrthogonalityHistoryEntryWithMetadata]]

