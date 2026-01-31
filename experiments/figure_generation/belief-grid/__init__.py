"""Figure generation pipeline for reproducing training analysis figures from MLflow runs."""

from .data_loader import (
    setup_from_mlflow,
    fetch_metric_history,
    fetch_all_metrics_for_figure,
    list_checkpoints,
    list_all_metrics,
    discover_layer_names,
    get_num_factors_from_config,
    select_evenly_spaced_checkpoints,
)
from .analysis import (
    prepare_sequences,
    get_activations,
    compute_belief_regression,
    compute_cev,
    compute_belief_regression_at_checkpoint,
    compute_cev_at_checkpoint,
    compute_cev_at_checkpoints,
    PreparedSequences,
    PreparedActivations,
)
from .plotting import (
    plot_belief_simplex_grid,
    plot_cev_curves,
    plot_dims95_and_loss,
    plot_dims95_vs_rmse,
    create_composite_figure,
)

__all__ = [
    # Data loading
    "setup_from_mlflow",
    "fetch_metric_history",
    "fetch_all_metrics_for_figure",
    "list_checkpoints",
    "list_all_metrics",
    "discover_layer_names",
    "get_num_factors_from_config",
    "select_evenly_spaced_checkpoints",
    # Analysis
    "prepare_sequences",
    "get_activations",
    "compute_belief_regression",
    "compute_cev",
    "compute_belief_regression_at_checkpoint",
    "compute_cev_at_checkpoint",
    "compute_cev_at_checkpoints",
    "PreparedSequences",
    "PreparedActivations",
    # Plotting
    "plot_belief_simplex_grid",
    "plot_cev_curves",
    "plot_dims95_and_loss",
    "plot_dims95_vs_rmse",
    "create_composite_figure",
]
