"""Orthogonality figure generation module.

This module provides tools for generating orthogonality over training figures
from MLflow training runs. It follows the belief-grid pattern.

Usage:
    uv run python experiments/figure_generation/orthogonality/run.py run_id=<your_run_id>

Example:
    >>> from experiments.figure_generation.orthogonality import (
    ...     setup_from_mlflow,
    ...     compute_orthogonality_at_checkpoints,
    ...     plot_overlap_over_training,
    ... )
"""

from .data_loader import (
    setup_from_mlflow,
    list_checkpoints,
    select_evenly_spaced_checkpoints,
    get_num_factors_from_config,
    load_random_init_baseline,
    get_baseline_ci,
)
from .analysis import (
    OrthogonalityData,
    compute_factor_pca,
    compute_orthogonality_at_checkpoints,
)
from .plotting import (
    plot_overlap_over_training,
    apply_icml_style,
    ICML_STYLE,
)

__all__ = [
    # Data loading
    "setup_from_mlflow",
    "list_checkpoints",
    "select_evenly_spaced_checkpoints",
    "get_num_factors_from_config",
    "load_random_init_baseline",
    "get_baseline_ci",
    # Analysis
    "OrthogonalityData",
    "compute_factor_pca",
    "compute_orthogonality_at_checkpoints",
    # Plotting
    "plot_overlap_over_training",
    "apply_icml_style",
    "ICML_STYLE",
]
