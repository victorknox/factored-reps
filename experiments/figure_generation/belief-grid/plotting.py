"""Matplotlib plotting functions for figure generation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# ICML publication-quality style settings
ICML_STYLE = {
    # Font settings
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    # Line settings
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.5,
    # Grid
    "axes.grid": False,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 450,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_icml_style():
    """Apply ICML publication style to matplotlib."""
    plt.rcParams.update(ICML_STYLE)


def _merge_metrics_by_nearest_step(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_value_name: str,
    right_value_name: str,
) -> pd.DataFrame:
    """Merge two metric histories by nearest training step.

    MLflow metrics and checkpoint-derived metrics are often logged at different
    step granularities. Using an inner merge on exact step equality can yield
    very few (or zero) matched points. This helper aligns rows using the
    nearest step in the right-hand DataFrame.
    """
    if left_df.empty or right_df.empty:
        return pd.DataFrame(columns=["step", left_value_name, right_value_name])

    left = (
        left_df[["step", "value"]]
        .rename(columns={"value": left_value_name})
        .copy()
    )
    right = (
        right_df[["step", "value"]]
        .rename(columns={"value": right_value_name})
        .copy()
    )

    left["step"] = pd.to_numeric(left["step"], errors="coerce")
    right["step"] = pd.to_numeric(right["step"], errors="coerce")
    left = left.dropna(subset=["step"]).sort_values("step")
    right = right.dropna(subset=["step"]).sort_values("step")

    # Ensure merge_asof has a single row per step on each side.
    left = left.drop_duplicates(subset=["step"], keep="last")
    right = right.drop_duplicates(subset=["step"], keep="last")

    merged = pd.merge_asof(left, right, on="step", direction="nearest")
    return merged.dropna(subset=[right_value_name])


def _resolve_layer_name(layer_name: str, available_layers: list[str]) -> str | None:
    """Resolve a layer name against available layers.

    Handles both raw hook names (e.g., 'ln_final.hook_normalized') and
    formatted names (e.g., 'ln_final.normalized'). Returns the matching
    key from available_layers, or None if no match.
    """
    if layer_name in available_layers:
        return layer_name

    # Try to find a match by checking if available layers contain this name
    # or if this name is a formatted version of an available layer
    from fwh_core.analysis.metric_keys import format_layer_spec

    # Check if any available layer matches when formatted
    for avail in available_layers:
        if format_layer_spec(avail) == layer_name or format_layer_spec(layer_name) == avail:
            return avail

    # Also check if the formatted version of layer_name is in available_layers
    formatted = format_layer_spec(layer_name)
    if formatted in available_layers:
        return formatted

    return None

def get_factor_colors(num_factors: int, cmap: str = "plasma") -> list:
    """Get colors for factors using specified colormap.

    Samples colors evenly from the colormap, matching the orthogonality figure style.

    Args:
        num_factors: Number of factors to get colors for.
        cmap: Colormap name (default "plasma" to match orthogonality figures).

    Returns:
        List of colors (as RGBA tuples) for each factor.
    """
    colormap = plt.cm.get_cmap(cmap)
    if num_factors == 1:
        return [colormap(0.5)]
    return [colormap(i / (num_factors - 1)) for i in range(num_factors)]


def _compute_rgb_colors(
    belief_values: np.ndarray,
    dim: int,
    use_cmap_barycentric: bool = False,
    cmap: str = "magma",
    cmap_start: float = 0.0,
    cmap_mid: float = 0.5,
    cmap_end: float = 0.9,
) -> np.ndarray:
    """Compute RGB colors from belief state values.

    Args:
        belief_values: Array of belief state values [n_samples, belief_dim]
        dim: Dimensionality of the belief state
        use_cmap_barycentric: If True, use colormap-based barycentric blending.
                              If False, use teal-coral-gold barycentric blending.
        cmap: Matplotlib colormap name to use for barycentric blending (default: magma).
        cmap_start: Position in colormap for first corner (0-1).
        cmap_mid: Position in colormap for second corner (0-1).
        cmap_end: Position in colormap for third corner (0-1).
    """
    n_samples = len(belief_values)

    # Define color palettes
    teal = np.array([0.12, 0.65, 0.75])
    coral = np.array([0.95, 0.35, 0.25])
    gold = np.array([0.85, 0.55, 0.15])  # More muted, less yellow

    if use_cmap_barycentric and dim >= 3:
        # For triangular simplices (F0, F1, F2): use colormap-based barycentric
        colormap = plt.cm.get_cmap(cmap)
        color_0 = np.array(colormap(cmap_start)[:3])
        color_1 = np.array(colormap(cmap_mid)[:3])
        color_2 = np.array(colormap(cmap_end)[:3])

        # Normalize belief values to sum to 1
        weights = belief_values[:, :3].copy()
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        rgb_values = (
            weights[:, 0:1] * color_0 +
            weights[:, 1:2] * color_1 +
            weights[:, 2:3] * color_2
        )
        return np.clip(rgb_values, 0, 1)
    else:
        # For 2D factors: use colormap-based barycentric on 2D coordinates
        colormap = plt.cm.get_cmap(cmap)
        color_0 = np.array(colormap(cmap_start)[:3])
        color_1 = np.array(colormap(cmap_mid)[:3])
        color_2 = np.array(colormap(cmap_end)[:3])

        x = belief_values[:, 0].copy()
        y = belief_values[:, 1].copy() if belief_values.shape[1] >= 2 else np.zeros(n_samples)

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.full(n_samples, 0.5)

        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)
        else:
            y = np.full(n_samples, 0.5)

        # Barycentric-style blending
        w_0 = (1 - x) * (1 - y)  # bottom-left
        w_1 = x * (1 - y)        # bottom-right
        w_2 = y                  # top

        total = w_0 + w_1 + w_2
        w_0 /= total
        w_1 /= total
        w_2 /= total

        rgb_values = np.zeros((n_samples, 3))
        for i in range(3):
            rgb_values[:, i] = w_0 * color_0[i] + w_1 * color_1[i] + w_2 * color_2[i]
        return rgb_values


def _compute_rgb_colors_legacy(belief_values: np.ndarray, dim: int) -> np.ndarray:
    """Legacy RGB color computation (unused, kept for reference).

    Original RGB mapping:
    """
    n_samples = len(belief_values)
    if dim >= 3:
        # Use first 3 dimensions directly for RGB
        rgb_values = belief_values[:, :3].copy()
    elif dim == 2:
        # For 2D: use the two values for R and G, set B to inverse of R
        rgb_values = np.zeros((n_samples, 3))
        rgb_values[:, 0] = belief_values[:, 0]
        rgb_values[:, 1] = belief_values[:, 1]
        rgb_values[:, 2] = 1.0 - belief_values[:, 0]
    else:
        # For 1D: create gradient from value
        rgb_values = np.zeros((n_samples, 3))
        rgb_values[:, 0] = belief_values[:, 0]
        rgb_values[:, 1] = 0.5 * belief_values[:, 0]
        rgb_values[:, 2] = 1.0 - belief_values[:, 0]

    # Normalize each channel to [0, 1]
    for i in range(3):
        min_val = rgb_values[:, i].min()
        max_val = rgb_values[:, i].max()
        if max_val > min_val:
            rgb_values[:, i] = (rgb_values[:, i] - min_val) / (max_val - min_val)
        else:
            rgb_values[:, i] = 0.5

    return rgb_values


def plot_belief_regression_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_factors: int,
    overall_rmse: float,
    factor_rmse_scores: list[float],
    factor_dims: list[int] | None = None,
    factor_names: list[str] | None = None,
    step: int | None = None,
    layer_name: str | None = None,
    ax_grid: np.ndarray | None = None,
    max_samples: int = 10000,
    marker_size: float = 2.0,
    marker_opacity: float = 0.7,
    pred_marker_size: float = 1.0,
    pred_marker_opacity: float = 0.35,
    title: str | None = None,
    title_suffix: str = "",
    sample_indices: np.ndarray | None = None,
    cmap: str = "magma",
    cmap_start: float = 0.0,
    cmap_mid: float = 0.5,
    cmap_end: float = 0.9,
) -> Figure | None:
    """Plot belief regression as a 2-row grid: ground truth (top) vs predictions (bottom).

    Creates a compact visualization with:
    - Row 1: Ground truth belief states for each factor
    - Row 2: Predicted belief states for each factor
    - Columns: One per factor
    - Factor names as column headers (top row only)
    - Row labels ("Ground Truth", "Prediction") on the left

    All plots are 2D scatter plots. For beliefs with >2 dimensions, PCA is used
    to project to 2D.

    This implementation mirrors the training script's Plotly-based
    plot_belief_regression_grid for consistency across experiments.

    Args:
        y_true: Ground truth belief states, shape [n_samples, belief_dim].
        y_pred: Predicted belief states, shape [n_samples, belief_dim].
        num_factors: Number of factors in the belief state.
        overall_rmse: Overall RMSE across all dimensions.
        factor_rmse_scores: Per-factor RMSE scores.
        factor_dims: Per-factor dimensions (if None, assumes equal dims).
        factor_names: Names for each factor (if None, uses "Factor 0", etc.).
        step: Training step number (for title).
        layer_name: Name of the layer (for title).
        ax_grid: Optional pre-created axes grid (2 x num_factors) for embedding.
        max_samples: Maximum samples to plot (for performance).
        marker_size: Size of scatter markers.
        marker_opacity: Opacity of markers.
        title: Optional custom title override.
        title_suffix: Optional suffix to append to title.
        sample_indices: Optional indices to subsample for visualization.

    Returns:
        Figure if ax_grid was None, else None.
    """
    from sklearn.decomposition import PCA

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    n_samples, belief_dim = y_true.shape

    # Handle factor dimensions
    if factor_dims is None:
        dim_per_factor = belief_dim // num_factors
        factor_dims = [dim_per_factor] * num_factors

    # Handle factor names
    if factor_names is None:
        factor_names = [f"Factor {i}" for i in range(num_factors)]

    # Subsample if needed
    if sample_indices is not None:
        y_true = y_true[sample_indices]
        y_pred = y_pred[sample_indices]
        n_samples = len(sample_indices)
    elif n_samples > max_samples:
        # Always include index 0 (BOS/initial belief if present) plus random samples
        other_indices = np.random.choice(
            np.arange(1, n_samples), max_samples - 1, replace=False
        )
        indices = np.concatenate([[0], other_indices])
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        n_samples = max_samples

    # Build title
    if title is None:
        title_parts = ["Belief State Regression"]
        if layer_name:
            title_parts.append(f"- {layer_name}")
        if step is not None:
            title_parts.append(f"- Step {step}")
        title = " ".join(title_parts)
        per_factor_rmse_text = ", ".join([f"F{i}: {rmse:.3f}" for i, rmse in enumerate(factor_rmse_scores)])
        title += f"\nOverall RMSE: {overall_rmse:.4f} | Per-factor: {per_factor_rmse_text}"
        if title_suffix:
            title += f"\n{title_suffix}"

    # Create figure if not provided
    created_fig = ax_grid is None
    if created_fig:
        fig, axes = plt.subplots(2, num_factors, figsize=(3 * num_factors, 6))
        ax_grid = np.atleast_2d(axes)
    else:
        fig = None

    # Process each factor
    offset = 0
    for factor_idx in range(num_factors):
        factor_dim = factor_dims[factor_idx]
        start_idx = offset
        end_idx = offset + factor_dim
        offset = end_idx

        # Extract true and predicted beliefs for this factor
        true_factor = y_true[:, start_idx:end_idx]
        pred_factor = y_pred[:, start_idx:end_idx]

        # Project to 2D if needed
        if factor_dim > 2:
            pca = PCA(n_components=2)
            combined = np.vstack([true_factor, pred_factor])
            pca.fit(combined)
            true_2d = pca.transform(true_factor)
            pred_2d = pca.transform(pred_factor)
        elif factor_dim == 1:
            # For 1D, use the value as x and zeros as y
            true_2d = np.column_stack([true_factor, np.zeros(n_samples)])
            pred_2d = np.column_stack([pred_factor, np.zeros(n_samples)])
        else:
            true_2d = true_factor
            pred_2d = pred_factor

        # Compute RGB colors:
        # - F0, F1, F2 (triangular simplices): colormap barycentric
        # - F3, F4 (circular): colormap barycentric on 2D projection
        use_cmap = factor_idx < 3 and factor_dim >= 3
        colors = _compute_rgb_colors(
            true_factor, factor_dim, use_cmap_barycentric=use_cmap,
            cmap=cmap, cmap_start=cmap_start, cmap_mid=cmap_mid, cmap_end=cmap_end
        )

        # Compute SHARED axis ranges for GT and Pred (so they're directly comparable)
        def compute_padded_range(data_2d: np.ndarray, pad_frac: float = 0.1) -> tuple[tuple[float, float], tuple[float, float]]:
            """Compute padded axis ranges that maintain aspect ratio."""
            x_min, x_max = float(data_2d[:, 0].min()), float(data_2d[:, 0].max())
            y_min, y_max = float(data_2d[:, 1].min()), float(data_2d[:, 1].max())
            x_span = x_max - x_min
            y_span = y_max - y_min
            max_span = max(x_span, y_span)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            half_span = max_span / 2 * (1 + pad_frac)
            return (
                (x_center - half_span, x_center + half_span),
                (y_center - half_span, y_center + half_span),
            )

        # Use GT ranges for both rows so they're directly comparable
        true_x_range, true_y_range = compute_padded_range(true_2d)
        shared_x_range, shared_y_range = true_x_range, true_y_range

        # Column header
        col_header = f"$F_{factor_idx}$"

        # Plot ground truth (top row) - scaled to fill space
        ax_true: Axes = ax_grid[0, factor_idx]
        ax_true.scatter(
            true_2d[:, 0], true_2d[:, 1], c=colors,
            s=marker_size, alpha=marker_opacity, edgecolors="none", rasterized=True,
        )
        ax_true.set_title(col_header, fontsize=15, fontweight="medium", pad=2)
        ax_true.set_xlim(true_x_range)
        ax_true.set_ylim(true_y_range)
        ax_true.set_xticks([])
        ax_true.set_yticks([])
        ax_true.set_aspect("equal", adjustable="box")
        # No borders for cleaner look
        for spine in ax_true.spines.values():
            spine.set_visible(False)

        # Plot predictions (bottom row) - same colors, SAME scale as GT
        ax_pred: Axes = ax_grid[1, factor_idx]
        ax_pred.scatter(
            pred_2d[:, 0], pred_2d[:, 1], c=colors,
            s=pred_marker_size, alpha=pred_marker_opacity, edgecolors="none", rasterized=True,
        )
        ax_pred.set_xlim(shared_x_range)
        ax_pred.set_ylim(shared_y_range)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_aspect("equal", adjustable="box")
        for spine in ax_pred.spines.values():
            spine.set_visible(False)

    # Row labels
    ax_grid[0, 0].set_ylabel("Theory", fontsize=14, labelpad=8)
    ax_grid[1, 0].set_ylabel("Activations", fontsize=14, labelpad=8)

    if created_fig and fig is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig
    return None


def plot_cev_curves(
    cev_history: dict[str, list[tuple[int, np.ndarray]]],
    layer_name: str,
    ax: Axes | None = None,
    max_step: int | None = None,
    cmap: str = "viridis",
    linewidth: float = 1.0,
    alpha: float = 0.8,
    belief_baselines: dict[str, np.ndarray] | None = None,
    dims95_inset: tuple[pd.DataFrame, float | None, int | None] | None = None,
) -> Figure | None:
    """Plot CEV curves colored by training step.

    Args:
        cev_history: Dictionary mapping layer names to list of (step, cev_array).
        layer_name: Layer to plot.
        ax: Optional pre-created axes.
        max_step: Optional maximum step to include.
        cmap: Colormap name for step coloring.
        linewidth: Line width.
        alpha: Line opacity.
        belief_baselines: Optional dict with "factored" and/or "product" CEV arrays
            to plot as reference baselines.

    Returns:
        Figure if ax was None, else None.
    """
    if layer_name not in cev_history:
        raise ValueError(f"Layer {layer_name} not found in cev_history")

    history = cev_history[layer_name]

    # Filter by max_step if specified
    if max_step is not None:
        history = [(step, cev) for step, cev in history if step <= max_step]

    if not history:
        raise ValueError("No CEV data after filtering")

    # Create figure if not provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Get step range for color normalization (log scale to spread early steps, reversed so early=yellow, late=blue)
    steps = [step for step, _ in history]
    # Use log normalization to spread out early steps (add 1 to handle step 0)
    norm = mcolors.LogNorm(vmin=max(min(steps), 1), vmax=max(steps))
    colormap = plt.cm.get_cmap(cmap + "_r")  # Reverse colormap

    # Plot each CEV curve
    for step, cev in history:
        color = colormap(norm(max(step, 1)))  # Handle step 0 for LogNorm
        n_components = np.arange(1, len(cev) + 1)
        ax.plot(n_components, cev, color=color, linewidth=linewidth, alpha=alpha)

    # Add horizontal colorbar at bottom right
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    # Create inset axes for colorbar at bottom right
    cbar_ax = ax.inset_axes([0.45, 0.07, 0.52, 0.07])  # [x, y, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.outline.set_visible(False)  # Remove outline
    # Add log scale tick marks
    max_step = max(steps)
    min_step = max(min(steps), 1)  # Avoid log(0)
    # Choose tick values on log scale - only show ticks up to 5000
    import math
    all_tick_values = [1, 10, 100, 1000, 5000]
    tick_values = [t for t in all_tick_values if min_step <= t <= max_step]

    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{t}" if t < 1000 else f"{t//1000}k" for t in tick_values])
    cbar.ax.tick_params(labelsize=7, top=True, bottom=False, labeltop=True, labelbottom=False)

    # Label in white on top of colorbar
    cbar.ax.text(0.5, 0.45, "Training Step", transform=cbar.ax.transAxes,
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Plot belief baselines as reference lines with inline labels
    if belief_baselines is not None:
        if "factored" in belief_baselines:
            factored_cev = belief_baselines["factored"]
            n_comp_factored = np.arange(1, len(factored_cev) + 1)
            ax.plot(
                n_comp_factored, factored_cev,
                color="#c44e52", linewidth=2.0, linestyle="--",
                zorder=10,
            )
            # Inline label at right end of line
            label_idx = min(len(factored_cev) - 1, 30)  # Position label around x=30
            ax.text(
                n_comp_factored[label_idx], factored_cev[label_idx] + 0.02,
                "Factored", fontsize=11, color="#c44e52", ha="center", va="bottom",
            )
        if "product" in belief_baselines:
            product_cev = belief_baselines["product"]
            n_comp_product = np.arange(1, len(product_cev) + 1)
            ax.plot(
                n_comp_product, product_cev,
                color="#c44e52", linewidth=2.0, linestyle="--",
                zorder=10,
            )
            # Inline label at right end of line
            label_idx = min(len(product_cev) - 1, 50)  # Position label around x=50
            ax.text(
                n_comp_product[label_idx], product_cev[label_idx] - 0.04,
                "Joint", fontsize=11, color="#c44e52", ha="center", va="top",
            )

    ax.set_xlabel("Dimension", fontsize=14)
    ax.set_ylabel("Cumulative Variance", fontsize=14)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, 64)  # Limit to model dimension
    ax.tick_params(axis="both", labelsize=12)
    # Set x-axis ticks
    ax.set_xticks([20, 40, 60])
    # Shorter y-tick labels: show as .0, .2, .4, etc.
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['.0', '.2', '.4', '.6', '.8', '1'])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Add subtle horizontal grid
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

    # Add dims@95 inset if provided
    if dims95_inset is not None:
        dims95_df, gt_dims95, inset_max_step = dims95_inset

        # Create inset axes flush with top, more to the right
        inset_ax = ax.inset_axes([0.68, 0.62, 0.38, 0.38])  # [x, y, width, height]

        # Filter by max_step
        if inset_max_step is not None:
            dims95_df = dims95_df[dims95_df["step"] <= inset_max_step]

        dims_color = "#1f77b4"
        inset_ax.plot(
            dims95_df["step"], dims95_df["value"],
            color=dims_color, linewidth=1.0,
        )

        if gt_dims95 is not None:
            inset_ax.axhline(y=gt_dims95, color=dims_color, linestyle="--", linewidth=0.8, alpha=0.7)

        inset_ax.set_xlabel("Step", fontsize=11, labelpad=1)
        inset_ax.set_ylabel("Dims for 95%", fontsize=11, labelpad=1)
        inset_ax.tick_params(axis="both", labelsize=7, pad=1)
        inset_ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        ))
        inset_ax.spines["top"].set_visible(False)
        inset_ax.spines["right"].set_visible(False)
        # Transparent background
        inset_ax.patch.set_alpha(0.7)
        inset_ax.set_facecolor("white")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_rmse_over_training(
    loss_df: pd.DataFrame,
    factor_rmse_dfs: dict[str, pd.DataFrame],
    ax: Axes | None = None,
    optimal_loss: float | None = None,
    max_step: int | None = None,
    show_inset: bool = True,
    show_legend: bool = True,
) -> Figure | None:
    """Plot per-factor RMSE over training with optional Loss vs RMSE scatter as inset.

    Args:
        loss_df: DataFrame with 'step' and 'value' columns for loss.
        factor_rmse_dfs: Dict mapping factor names to DataFrames with step/value.
        ax: Optional pre-created axes.
        optimal_loss: Optimal loss (entropy rate) for normalization in inset.
        max_step: Optional maximum step to include.
        show_legend: Whether to show the legend (default True).

    Returns:
        Figure if created, None if axes were provided.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    # Filter by max_step
    if max_step is not None:
        loss_df = loss_df[loss_df["step"] <= max_step]
        factor_rmse_dfs = {k: df[df["step"] <= max_step] for k, df in factor_rmse_dfs.items()}

    # Get factor colors from plasma colormap (matching orthogonality figures)
    num_factors = len(factor_rmse_dfs)
    factor_colors = get_factor_colors(num_factors, cmap="plasma")

    # Main plot: RMSE over training step
    for i, (name, df) in enumerate(sorted(factor_rmse_dfs.items())):
        factor_idx = name.replace("factor_", "").replace("_rmse", "")
        label = f"$F_{factor_idx}$"
        ax.plot(
            df["step"], df["value"],
            color=factor_colors[i],
            linewidth=1.2, alpha=0.9,
            label=label,
        )

    ax.set_xlabel("Training step", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(fontsize=9, loc="upper center", framealpha=0.9, ncol=5)

    # Inset: Loss vs RMSE scatter (top right) - optional
    if show_inset:
        inset_ax = ax.inset_axes([0.58, 0.58, 0.38, 0.38])  # [x, y, width, height]

        for i, (name, df) in enumerate(sorted(factor_rmse_dfs.items())):
            factor_idx = name.replace("factor_", "").replace("_rmse", "")

            merged = _merge_metrics_by_nearest_step(
                df,
                loss_df,
                left_value_name="rmse",
                right_value_name="loss",
            )

            # Normalize loss if optimal provided
            if optimal_loss is not None:
                x_values = merged["loss"] / optimal_loss
            else:
                x_values = merged["loss"]

            inset_ax.scatter(
                x_values, merged["rmse"],
                c=[factor_colors[i]],
                s=1, alpha=0.7,
            )

        # Inset axis formatting
        if optimal_loss is not None:
            inset_ax.set_xlabel("Loss / Opt", fontsize=11, labelpad=1)
            inset_ax.axvline(1.0, color="gray", ls="--", lw=0.5, alpha=0.5)
            inset_ax.invert_xaxis()
        else:
            inset_ax.set_xlabel("Loss", fontsize=11, labelpad=1)
            inset_ax.invert_xaxis()

        inset_ax.set_ylabel("RMSE", fontsize=11, labelpad=1)
        inset_ax.set_yscale("log")
        inset_ax.tick_params(axis="both", labelsize=6, pad=1)
        inset_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}" if x < 0.1 else f"{x:.1f}"))
        inset_ax.spines["top"].set_visible(False)
        inset_ax.spines["right"].set_visible(False)
        inset_ax.set_facecolor("#f8f8f8")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_dims95_vs_loss(
    dims95_df: pd.DataFrame,
    loss_df: pd.DataFrame,
    ax: Axes | None = None,
    gt_dims95: float | None = None,
    max_step: int | None = None,
    optimal_loss: float | None = None,
    initial_loss: float | None = None,
) -> Figure | plt.cm.ScalarMappable | None:
    """Plot Dims@95% vs Loss scatter plot.

    Args:
        dims95_df: DataFrame with 'step' and 'value' columns for dims@95.
        loss_df: DataFrame with 'step' and 'value' columns for loss.
        ax: Optional pre-created axes.
        gt_dims95: Ground truth dims@95 for reference line.
        max_step: Optional maximum step to include.
        optimal_loss: Optimal loss (entropy rate) for normalization.
        initial_loss: Initial loss (first checkpoint) for normalization.

    Returns:
        Figure if created standalone, ScalarMappable if axes provided (for colorbar), None on error.
    """
    import matplotlib.ticker as ticker

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        created_fig = True

    # Filter by max_step
    if max_step is not None:
        dims95_df = dims95_df[dims95_df["step"] <= max_step]
        loss_df = loss_df[loss_df["step"] <= max_step]

    merged = _merge_metrics_by_nearest_step(
        dims95_df,
        loss_df,
        left_value_name="dims95",
        right_value_name="loss",
    )

    # Color by training step (viridis_r: yellow=early, purple=late)
    steps = merged["step"].values
    # Replace 0 with 1 for log scale (step 0 -> step 1 for coloring purposes)
    steps_for_color = np.maximum(steps, 1)
    norm = mcolors.LogNorm(vmin=steps_for_color.min(), vmax=steps_for_color.max())
    colormap = plt.cm.get_cmap("viridis_r")
    colors = colormap(norm(steps_for_color))

    # Normalize loss if both optimal and initial are provided
    if optimal_loss is not None and initial_loss is not None:
        x_values = (merged["loss"] - optimal_loss) / (initial_loss - optimal_loss)
        xlabel = "Normalized Loss"
    else:
        x_values = merged["loss"]
        xlabel = "Loss"

    scatter = ax.scatter(
        x_values, merged["dims95"],
        c=colors,
        s=15, alpha=0.8,
        edgecolors="none",
    )

    # Create ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # Add ground truth reference line
    if gt_dims95 is not None:
        ax.axhline(
            y=gt_dims95,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"GT ({gt_dims95:.0f})",
        )
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Dims for 95%", fontsize=14)
    ax.invert_xaxis()  # Lower loss on right (better performance)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created_fig:
        fig.tight_layout()
        return fig
    return sm  # Return ScalarMappable for colorbar when axes provided


def plot_loss_vs_rmse(
    loss_df: pd.DataFrame,
    factor_rmse_dfs: dict[str, pd.DataFrame],
    ax: Axes | None = None,
    optimal_loss: float | None = None,
    initial_loss: float | None = None,
    max_step: int | None = None,
    show_legend: bool = True,
) -> Figure | None:
    """Plot Loss vs RMSE scatter as standalone panel.

    Args:
        loss_df: DataFrame with 'step' and 'value' columns for loss.
        factor_rmse_dfs: Dict mapping factor names to DataFrames with step/value.
        ax: Optional pre-created axes.
        optimal_loss: Optimal loss (entropy rate) for normalization.
        initial_loss: Initial loss (first checkpoint) for normalization.
        max_step: Optional maximum step to include.
        show_legend: Whether to show the legend (default True).

    Returns:
        Figure if created, None if axes were provided.
    """
    import matplotlib.ticker as ticker

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        created_fig = True

    # Filter by max_step
    if max_step is not None:
        loss_df = loss_df[loss_df["step"] <= max_step]
        factor_rmse_dfs = {k: df[df["step"] <= max_step] for k, df in factor_rmse_dfs.items()}

    # Get factor colors from plasma colormap (matching orthogonality figures)
    num_factors = len(factor_rmse_dfs)
    factor_colors = get_factor_colors(num_factors, cmap="plasma")

    for i, (name, df) in enumerate(sorted(factor_rmse_dfs.items())):
        factor_idx = name.replace("factor_", "").replace("_rmse", "")

        merged = _merge_metrics_by_nearest_step(
            df,
            loss_df,
            left_value_name="rmse",
            right_value_name="loss",
        )

        # Normalize loss if both optimal and initial are provided
        if optimal_loss is not None and initial_loss is not None:
            x_values = (merged["loss"] - optimal_loss) / (initial_loss - optimal_loss)
        else:
            x_values = merged["loss"]

        ax.scatter(
            x_values, merged["rmse"],
            c=[factor_colors[i]],
            s=8, alpha=0.7,
            label=f"$F_{factor_idx}$",
        )

    # Set axis label based on normalization
    if optimal_loss is not None and initial_loss is not None:
        ax.set_xlabel("Normalized Loss", fontsize=14)
    else:
        ax.set_xlabel("Loss", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.invert_xaxis()  # Lower loss on right
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_dims95_over_training(
    dims95_df: pd.DataFrame,
    ax: Axes | None = None,
    gt_dims95: float | None = None,
    max_step: int | None = None,
) -> Figure | None:
    """Plot dims@95 over training steps.

    Args:
        dims95_df: DataFrame with 'step' and 'value' columns for dims@95.
        ax: Optional pre-created axes.
        gt_dims95: Ground truth dims@95 for reference line.
        max_step: Optional maximum step to include.

    Returns:
        Figure if created, None if axes were provided.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    # Filter by max_step
    if max_step is not None:
        dims95_df = dims95_df[dims95_df["step"] <= max_step]

    dims_color = "#1f77b4"

    ax.plot(
        dims95_df["step"],
        dims95_df["value"],
        color=dims_color,
        linewidth=1.5,
        label="Dims for 95%",
    )

    if gt_dims95 is not None:
        ax.axhline(
            y=gt_dims95,
            color=dims_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"GT ({gt_dims95:.0f})",
        )

    ax.set_xlabel("Training step", fontsize=14)
    ax.set_ylabel("Dims for 95%", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Linear scale starting at 0
    if max_step is not None:
        ax.set_xlim(0, max_step)
    else:
        ax.set_xlim(0, dims95_df["step"].max())

    # Format x-axis with k for thousands
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))

    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.set_title("(c) Intrinsic dimension", fontsize=15, fontweight="medium", loc="left")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_dims95_and_loss(
    dims95_df: pd.DataFrame,
    loss_df: pd.DataFrame,
    ax: Axes | None = None,
    gt_dims95: float | None = None,
    max_step: int | None = None,
    smooth_window: int = 100,
    random_loss: float | None = None,
    entropy_rate: float | None = None,
    log_scale: bool = False,
) -> Figure | None:
    """Plot dims@95 and loss over training with dual y-axis.

    Args:
        dims95_df: DataFrame with 'step' and 'value' columns for dims@95.
        loss_df: DataFrame with 'step' and 'value' columns for loss.
        ax: Optional pre-created axes.
        log_scale: If True, use log scale for x-axis.
        gt_dims95: Optional ground truth dims@95 reference line.
        max_step: Optional maximum step to include.
        smooth_window: Window size for smoothing loss.
        random_loss: Optional random guesser loss (log(vocab_size)) reference line.
        entropy_rate: Optional entropy rate (optimal loss) reference line.

    Returns:
        Figure if ax was None, else None.
    """
    # Create figure if not provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Filter by max_step
    if max_step is not None:
        dims95_df = dims95_df[dims95_df["step"] <= max_step]
        loss_df = loss_df[loss_df["step"] <= max_step]

    # Color scheme for clean dual-axis
    dims_color = "#1f77b4"  # Blue
    loss_color = "#d62728"  # Red

    # Plot dims@95 (left y-axis)
    ax.plot(
        dims95_df["step"],
        dims95_df["value"],
        color=dims_color,
        linewidth=1.5,
        label="Dims for 95%",
        drawstyle="steps-post",
    )

    # GT reference line
    if gt_dims95 is not None:
        ax.axhline(
            y=gt_dims95,
            color=dims_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"GT ({gt_dims95:.0f})",
        )

    ax.set_xlabel("Training step", fontsize=14)
    ax.set_ylabel("Dims for 95%", fontsize=14, color=dims_color)
    ax.tick_params(axis="both", labelsize=12)
    ax.tick_params(axis="y", colors=dims_color)
    ax.spines["left"].set_color(dims_color)
    ax.spines["top"].set_visible(False)

    # Format x-axis to use k for thousands
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    # Set log scale if requested
    if log_scale:
        ax.set_xscale("log")

    # Create second y-axis for loss
    ax2 = ax.twinx()

    # Raw loss (light, thin)
    ax2.plot(
        loss_df["step"],
        loss_df["value"],
        color=loss_color,
        linewidth=0.3,
        alpha=0.3,
    )

    # Smoothed loss
    smoothed_loss = None
    if len(loss_df) > smooth_window:
        smoothed_loss = uniform_filter1d(loss_df["value"].values, size=smooth_window)
        ax2.plot(
            loss_df["step"],
            smoothed_loss,
            color=loss_color,
            linewidth=1.5,
            label="Loss",
        )

    # Reference lines for loss (subtle styling)
    if random_loss is not None:
        ax2.axhline(
            y=random_loss,
            color="#888888",
            linestyle=":",
            linewidth=0.8,
            label=f"Random ({random_loss:.2f})",
        )

    if entropy_rate is not None:
        ax2.axhline(
            y=entropy_rate,
            color="#2ca02c",
            linestyle="--",
            linewidth=0.8,
            label=f"Optimal ({entropy_rate:.2f})",
        )

    ax2.set_ylabel("Loss", fontsize=14, color=loss_color)
    ax2.tick_params(axis="y", colors=loss_color, labelsize=12)
    ax2.spines["right"].set_color(loss_color)
    ax2.spines["top"].set_visible(False)

    # Set y-axis limits based on smoothed loss (if available) to avoid early spike artifacts
    if smoothed_loss is not None:
        loss_min = smoothed_loss.min()
        loss_max = smoothed_loss.max()
    else:
        loss_min = loss_df["value"].min()
        loss_max = loss_df["value"].max()

    # Expand range to include reference lines if needed
    if entropy_rate is not None:
        loss_min = min(loss_min, entropy_rate)
    if random_loss is not None:
        loss_max = max(loss_max, random_loss)

    loss_range = loss_max - loss_min
    padding = loss_range * 0.08
    ax2.set_ylim(loss_min - padding, loss_max + padding)

    # Combined legend (positioned to avoid data)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="center right", fontsize=12, framealpha=0.9, edgecolor="none",
    )

    ax.set_title("(c) Intrinsic dimension & loss", fontsize=15, fontweight="medium", loc="left")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_dims95_vs_rmse(
    dims95_df: pd.DataFrame,
    factor_rmse_dfs: dict[str, pd.DataFrame],
    ax: Axes | None = None,
    max_step: int | None = None,
    log_scale: bool = False,
) -> Figure | None:
    """Plot dims@95 vs per-factor RMSE over training.

    Args:
        dims95_df: DataFrame with 'step' and 'value' columns for dims@95.
        factor_rmse_dfs: Dict mapping factor names to DataFrames with step/value.
        ax: Optional pre-created axes.
        max_step: Optional maximum step to include.
        log_scale: If True, use log scale for x-axis.

    Returns:
        Figure if ax was None, else None.
    """
    # Create figure if not provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Filter by max_step
    if max_step is not None:
        dims95_df = dims95_df[dims95_df["step"] <= max_step]
        factor_rmse_dfs = {
            k: df[df["step"] <= max_step] for k, df in factor_rmse_dfs.items()
        }

    # Color scheme
    dims_color = "#1f77b4"  # Blue for dims

    # Plot dims@95 (left y-axis)
    ax.plot(
        dims95_df["step"],
        dims95_df["value"],
        color=dims_color,
        linewidth=1.5,
        label="Dims for 95%",
        drawstyle="steps-post",
    )
    ax.set_xlabel("Training step", fontsize=14)
    ax.set_ylabel("Dims for 95%", fontsize=14, color=dims_color)
    ax.tick_params(axis="both", labelsize=12)
    ax.tick_params(axis="y", colors=dims_color)
    ax.spines["left"].set_color(dims_color)
    ax.spines["top"].set_visible(False)

    # Format x-axis to use k for thousands
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    # Set log scale if requested
    if log_scale:
        ax.set_xscale("log")

    # Create second y-axis for RMSE
    ax2 = ax.twinx()

    # Use a perceptually distinct colormap for factors
    # Viridis-based factor colors (sampled evenly from viridis)
    _viridis = plt.cm.get_cmap("viridis")
    factor_colors = [mcolors.to_hex(_viridis(i)) for i in [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]]

    # Plot per-factor RMSE with different colors
    for i, (factor_name, df) in enumerate(sorted(factor_rmse_dfs.items())):
        # Extract factor index from name like "factor_0_rmse"
        factor_idx = factor_name.replace("factor_", "").replace("_rmse", "")
        label = f"$F_{factor_idx}$"
        ax2.plot(
            df["step"],
            df["value"],
            color=factor_colors[i % len(factor_colors)],
            linewidth=1.0,
            label=label,
            alpha=0.8,
        )

    ax2.set_ylabel("RMSE", fontsize=14)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.spines["top"].set_visible(False)

    # Combined legend (outside plot area)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", fontsize=12, framealpha=0.9, edgecolor="none",
        ncol=2,
    )

    ax.set_title("(d) Dimension & regression accuracy", fontsize=15, fontweight="medium", loc="left")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def create_main_figure(
    belief_regression_data: dict[str, Any] | None = None,
    cev_history: dict[str, list[tuple[int, np.ndarray]]] | None = None,
    metrics: dict[str, pd.DataFrame] | None = None,
    layer_name: str | None = None,
    gt_dims95: float | None = None,
    joint_dims95: float | None = None,
    cev_max_step: int | None = None,
    dims_max_step: int | None = None,
    figsize: tuple[float, float] = (6.75, 4.0),
    belief_baselines: dict[str, np.ndarray] | None = None,
    cev_alpha: float = 0.8,
    cmap: str = "magma",
    cmap_start: float = 0.0,
    cmap_mid: float = 0.5,
    cmap_end: float = 0.9,
) -> Figure:
    """Create the main 2-row figure for the paper.

    Layout:
    - Row 1: Belief regression (Theory vs Activations)
    - Row 2: CEV curves (left), dims@95 over training (right)

    Args:
        belief_regression_data: Data for belief regression plot.
        cev_history: CEV history for CEV curves plot.
        metrics: Dict of metric DataFrames (must include dims95).
        layer_name: Layer name for layer-specific plots.
        gt_dims95: Ground truth dims@95 reference line (factored belief).
        joint_dims95: Joint (product) belief dims@95 reference line.
        cev_max_step: Max step for CEV plot.
        dims_max_step: Max step for dims@95 plot.
        figsize: Figure size.
        belief_baselines: Optional CEV baselines for reference lines.

    Returns:
        Matplotlib Figure with 2 rows.
    """
    apply_icml_style()
    fig = plt.figure(figsize=figsize)


    # Create grid: 2 rows, 2 cols
    # Row 1: belief regression (spans both cols)
    # Row 2: CEV (left), dims@95 over training (right)
    # Use separate gridspecs for each row so they can have different left margins
    # Row 1: Belief regression (smaller left margin - no y-axis ticks)
    gs_row1 = fig.add_gridspec(
        1, 1,
        left=0.08,
        right=0.95,
        top=0.95,
        bottom=0.55,
    )

    # Row 2: CEV and dims (adjusted margins to align with belief regression row)
    gs_row2 = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        wspace=0.3,
        left=0.125,  # Push bottom row right to align labels
        right=0.905,  # Pull right edge inward to match
        top=0.48,
        bottom=0.08,
    )

    # Row 1: Belief regression visualization
    if belief_regression_data is not None and layer_name is not None:
        resolved_belief_layer = _resolve_layer_name(layer_name, list(belief_regression_data.keys()))
        if resolved_belief_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in belief_regression_data. "
                f"Available layers: {list(belief_regression_data.keys())}"
            )
        data = belief_regression_data[resolved_belief_layer]
        num_factors = data["num_factors"]
        gs_belief = gs_row1[0, 0].subgridspec(2, num_factors, hspace=0.08, wspace=0.05)
        ax_belief = np.array(
            [[fig.add_subplot(gs_belief[i, j]) for j in range(num_factors)] for i in range(2)]
        )
        plot_belief_regression_grid(
            data["y_true"],
            data["y_pred"],
            num_factors=data["num_factors"],
            overall_rmse=data.get("overall_rmse", 0.0),
            factor_rmse_scores=data.get("factor_rmse_scores", []),
            factor_dims=data["factor_dims"],
            factor_names=data.get("factor_names"),
            ax_grid=ax_belief,
            cmap=cmap,
            cmap_start=cmap_start,
            cmap_mid=cmap_mid,
            cmap_end=cmap_end,
        )
        # Panel label (a)
        fig.text(0.045, 0.98, "(a)", fontsize=15, fontweight="bold", va="top", ha="left")

    # Row 2, Col 0: CEV curves (without inset)
    if cev_history is not None and layer_name is not None:
        ax_cev = fig.add_subplot(gs_row2[0, 0])
        resolved_cev_layer = _resolve_layer_name(layer_name, list(cev_history.keys()))
        if resolved_cev_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in cev_history. "
                f"Available layers: {list(cev_history.keys())}"
            )
        plot_cev_curves(
            cev_history, resolved_cev_layer,
            ax=ax_cev,
            max_step=cev_max_step,
            belief_baselines=belief_baselines,
            dims95_inset=None,  # No inset - dims@95 is now its own panel
            linewidth=1.5,
            alpha=cev_alpha,
        )
        # Panel label (b)
        fig.text(0.045, 0.51, "(b)", fontsize=15, fontweight="bold", va="top", ha="left")

    # Row 2, Col 1: dims@95 over training with broken y-axis
    # Panel label (c)
    fig.text(0.52, 0.51, "(c)", fontsize=15, fontweight="bold", va="top", ha="left")

    if metrics is not None and "dims95" in metrics:
        from matplotlib.collections import LineCollection

        dims95_df = metrics["dims95"]
        if dims_max_step is not None:
            dims95_df = dims95_df[dims95_df["step"] <= dims_max_step]

        steps = dims95_df["step"].values
        values = dims95_df["value"].values
        x_max = dims95_df["step"].max()

        # Create broken axis if joint_dims95 is much larger than the data
        use_broken_axis = joint_dims95 is not None and joint_dims95 > values.max() * 1.5

        if use_broken_axis:
            # Create two subplots for broken axis
            gs_dims = gs_row2[0, 1].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.08)
            ax_top = fig.add_subplot(gs_dims[0])  # For joint line
            ax_bottom = fig.add_subplot(gs_dims[1])  # For data

            # Set y-limits
            data_max = max(values.max(), gt_dims95 if gt_dims95 else 0) * 1.15
            ax_bottom.set_ylim(0, data_max)
            ax_top.set_ylim(joint_dims95 - 10, joint_dims95 + 10)

            # Plot data on bottom axis
            points = np.array([steps, values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm_max = cev_max_step if cev_max_step is not None else steps.max()
            norm = mcolors.LogNorm(vmin=max(steps.min(), 1), vmax=norm_max)
            lc = LineCollection(segments, cmap="viridis_r", norm=norm, capstyle='round', joinstyle='round')
            lc.set_array(steps[:-1])
            lc.set_linewidth(3.0)
            lc.set_zorder(2)
            ax_bottom.add_collection(lc)
            ax_bottom.set_xlim(steps.min(), steps.max())

            # Factored line on bottom (behind data)
            if gt_dims95 is not None:
                ax_bottom.axhline(y=gt_dims95, color="#c44e52", linewidth=2.0, linestyle="--", zorder=1)
                ax_bottom.text(
                    x_max * 0.95, gt_dims95 + 4,
                    f"Factored ({gt_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="bottom",
                )

            # Joint line on top
            ax_top.axhline(y=joint_dims95, color="#c44e52", linewidth=2.0, linestyle="--")
            ax_top.text(
                x_max * 0.95, joint_dims95 + 2,
                f"Joint ({joint_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="bottom",
            )
            ax_top.set_xlim(steps.min(), steps.max())

            # Hide spines between axes
            ax_top.spines["bottom"].set_visible(False)
            ax_bottom.spines["top"].set_visible(False)
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            ax_bottom.spines["right"].set_visible(False)

            # Add break marks (parallel diagonal lines in figure coordinates)
            import matplotlib.transforms as mtransforms

            # Get the positions in figure coordinates
            top_bottom_left = ax_top.transAxes.transform((0, 0))
            top_bottom_right = ax_top.transAxes.transform((1, 0))
            bottom_top_left = ax_bottom.transAxes.transform((0, 1))
            bottom_top_right = ax_bottom.transAxes.transform((1, 1))

            # Draw break marks in figure coordinates (pixels) for consistent angle
            d_pix = 5  # pixels
            kwargs = dict(color='k', clip_on=False, linewidth=0.8, transform=fig.transFigure)

            # Convert pixel offsets to figure coordinates
            fig_width, fig_height = fig.get_size_inches() * fig.dpi
            dx = d_pix / fig_width
            dy = d_pix / fig_height

            # Get positions in figure coords
            inv = fig.transFigure.inverted()
            tbl = inv.transform(top_bottom_left)
            tbr = inv.transform(top_bottom_right)
            btl = inv.transform(bottom_top_left)
            btr = inv.transform(bottom_top_right)

            # Draw parallel break marks on left side only
            ax_top.plot([tbl[0]-dx, tbl[0]+dx], [tbl[1]-dy, tbl[1]+dy], **kwargs)
            ax_bottom.plot([btl[0]-dx, btl[0]+dx], [btl[1]-dy, btl[1]+dy], **kwargs)

            # Labels
            ax_bottom.set_xlabel("Training step", fontsize=14)
            ax_bottom.set_ylabel("Dimensions for 95%", fontsize=14)
            ax_bottom.tick_params(axis="both", labelsize=12)
            ax_top.tick_params(axis="y", labelsize=12)
            # Dynamic x-ticks: [0, 2500, 5000] + every 5000 after that
            max_tick = int(x_max) if x_max else 5000
            xticks = [0, 2500, 5000] + list(range(10000, max_tick + 1, 5000))
            ax_bottom.set_xticks([t for t in xticks if t <= max_tick])
            ax_bottom.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            ))
            # Add subtle horizontal grid
            ax_bottom.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
            ax_top.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

        else:
            # Regular single axis (no break needed)
            ax_dims = fig.add_subplot(gs_row2[0, 1])

            points = np.array([steps, values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm_max = cev_max_step if cev_max_step is not None else steps.max()
            norm = mcolors.LogNorm(vmin=max(steps.min(), 1), vmax=norm_max)
            lc = LineCollection(segments, cmap="viridis_r", norm=norm, capstyle='round', joinstyle='round')
            lc.set_array(steps[:-1])
            lc.set_linewidth(3.0)
            lc.set_zorder(2)
            ax_dims.add_collection(lc)
            ax_dims.autoscale()

            if gt_dims95 is not None:
                ax_dims.axhline(y=gt_dims95, color="#c44e52", linewidth=2.0, linestyle="--", zorder=1)
                ax_dims.text(
                    x_max * 0.95, gt_dims95 + 4,
                    f"Factored ({gt_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="bottom",
                )
            if joint_dims95 is not None:
                ax_dims.axhline(y=joint_dims95, color="#c44e52", linewidth=2.0, linestyle="--", zorder=1)
                ax_dims.text(
                    x_max * 0.95, joint_dims95 - 1.5,
                    f"Joint ({joint_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="top",
                )

            ax_dims.set_xlabel("Training step", fontsize=14)
            ax_dims.set_ylabel("Dimensions for 95%", fontsize=14)
            ax_dims.set_ylim(bottom=0)
            ax_dims.tick_params(axis="both", labelsize=12)
            # Add subtle horizontal grid
            ax_dims.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
            ax_dims.spines["top"].set_visible(False)
            ax_dims.spines["right"].set_visible(False)
            # Dynamic x-ticks: [0, 2500, 5000] + every 5000 after that
            max_tick = int(x_max) if x_max else 5000
            xticks = [0, 2500, 5000] + list(range(10000, max_tick + 1, 5000))
            ax_dims.set_xticks([t for t in xticks if t <= max_tick])
            ax_dims.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            ))

    return fig


def create_supplemental_figure(
    metrics: dict[str, pd.DataFrame] | None = None,
    gt_dims95: float | None = None,
    rmse_max_step: int | None = None,
    figsize: tuple[float, float] = (6.75, 2.5),
    entropy_rate: float | None = None,
) -> Figure:
    """Create the supplemental 1-row figure with 3 panels.

    Layout:
    - Col 0: RMSE over training
    - Col 1: dims@95 vs Loss
    - Col 2: Loss vs RMSE

    Args:
        metrics: Dict of metric DataFrames.
        gt_dims95: Ground truth dims@95 reference line.
        rmse_max_step: Max step for all plots.
        figsize: Figure size.
        entropy_rate: Optimal loss for reference.

    Returns:
        Matplotlib Figure with 3 panels.
    """
    apply_icml_style()
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.35,
        left=0.08,
        right=0.95,
        top=0.80,  # Leave room for legend above
    )

    # Compute initial_loss from first checkpoint for normalization
    initial_loss = None
    if metrics is not None and "loss" in metrics:
        loss_df = metrics["loss"].sort_values("step")
        if not loss_df.empty:
            initial_loss = float(loss_df["value"].iloc[0])

    # Compute shared normalized loss range for plots with loss on x-axis
    loss_range = None
    if metrics is not None and "loss" in metrics and entropy_rate is not None and initial_loss is not None:
        loss_df = metrics["loss"]
        if rmse_max_step is not None:
            loss_df = loss_df[loss_df["step"] <= rmse_max_step]
        # Normalize: (loss - optimal) / (initial - optimal)
        norm_min = (loss_df["value"].min() - entropy_rate) / (initial_loss - entropy_rate)
        norm_max = (loss_df["value"].max() - entropy_rate) / (initial_loss - entropy_rate)
        padding = (norm_max - norm_min) * 0.05
        loss_range = (norm_max + padding, norm_min - padding)

    # Col 0: RMSE over training (uses step on x-axis, no loss normalization needed)
    ax_rmse = None
    factor_rmse_dfs = None
    if metrics is not None and "loss" in metrics:
        ax_rmse = fig.add_subplot(gs[0, 0])
        factor_rmse_dfs = {
            k: v for k, v in metrics.items() if k.startswith("factor_") and k.endswith("_rmse")
        }
        if factor_rmse_dfs:
            plot_rmse_over_training(
                metrics["loss"],
                factor_rmse_dfs,
                ax=ax_rmse,
                optimal_loss=entropy_rate,
                max_step=rmse_max_step,
                show_inset=False,
                show_legend=False,  # Use figure-level legend instead
            )

    # Col 1: Loss vs RMSE (normalized)
    if metrics is not None and "loss" in metrics:
        ax_scatter = fig.add_subplot(gs[0, 1])
        factor_rmse_dfs = {
            k: v for k, v in metrics.items() if k.startswith("factor_") and k.endswith("_rmse")
        }
        if factor_rmse_dfs:
            plot_loss_vs_rmse(
                metrics["loss"],
                factor_rmse_dfs,
                ax=ax_scatter,
                optimal_loss=entropy_rate,
                initial_loss=initial_loss,
                max_step=rmse_max_step,
                show_legend=False,  # Shared legend from left panel
            )
            if loss_range is not None:
                ax_scatter.set_xlim(loss_range)

    # Col 2: dims@95 vs Loss (normalized)
    sm_steps = None
    if metrics is not None and "dims95" in metrics and "loss" in metrics:
        ax_dims_loss = fig.add_subplot(gs[0, 2])
        sm_steps = plot_dims95_vs_loss(
            metrics["dims95"],
            metrics["loss"],
            ax=ax_dims_loss,
            gt_dims95=gt_dims95,
            max_step=rmse_max_step,
            optimal_loss=entropy_rate,
            initial_loss=initial_loss,
        )
        if loss_range is not None:
            ax_dims_loss.set_xlim(loss_range)

        # Add colorbar for steps above this panel
        if sm_steps is not None:
            cbar_ax = fig.add_axes([0.73, 0.88, 0.156, 0.025])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm_steps, cax=cbar_ax, orientation="horizontal")
            cbar.ax.set_title("Step", fontsize=9, pad=2)
            cbar.set_ticks([1, 10, 100, 1000, 5000])
            cbar.set_ticklabels(["1", "10", "100", "1k", "5k"])
            cbar.ax.tick_params(labelsize=8)

    # Add figure-level legend for factors (centered above left two panels)
    if ax_rmse is not None:
        handles, labels = ax_rmse.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center",
                ncol=len(handles),
                fontsize=9,
                framealpha=0.9,
                bbox_to_anchor=(0.37, 0.98),
            )

    return fig


def create_main_figure_vertical(
    belief_regression_data: dict[str, Any] | None = None,
    cev_history: dict[str, list[tuple[int, np.ndarray]]] | None = None,
    metrics: dict[str, pd.DataFrame] | None = None,
    layer_name: str | None = None,
    gt_dims95: float | None = None,
    joint_dims95: float | None = None,
    cev_max_step: int | None = None,
    dims_max_step: int | None = None,
    figsize: tuple[float, float] = (6.75, 4.5),
    belief_baselines: dict[str, np.ndarray] | None = None,
    cev_alpha: float = 0.8,
    cmap: str = "magma",
    cmap_start: float = 0.0,
    cmap_mid: float = 0.5,
    cmap_end: float = 0.9,
) -> Figure:
    """Create a vertical layout figure with belief regression factors stacked vertically.

    Layout:
    - Column 1: Belief regression factors stacked vertically (Theory | Activations)
    - Column 2: CEV curves (top), dims@95 over training (bottom)

    Args:
        belief_regression_data: Data for belief regression plot.
        cev_history: CEV history for CEV curves plot.
        metrics: Dict of metric DataFrames (must include dims95).
        layer_name: Layer name for layer-specific plots.
        gt_dims95: Ground truth dims@95 reference line (factored belief).
        joint_dims95: Joint (product) belief dims@95 reference line.
        cev_max_step: Max step for CEV plot.
        dims_max_step: Max step for dims@95 plot.
        figsize: Figure size.
        belief_baselines: Optional CEV baselines for reference lines.

    Returns:
        Matplotlib Figure with vertical layout.
    """
    from sklearn.decomposition import PCA
    from matplotlib.collections import LineCollection

    apply_icml_style()

    # Determine number of factors
    num_factors = 5  # Default
    if belief_regression_data is not None and layer_name is not None:
        resolved_belief_layer = _resolve_layer_name(layer_name, list(belief_regression_data.keys()))
        if resolved_belief_layer is not None:
            num_factors = belief_regression_data[resolved_belief_layer]["num_factors"]

    fig = plt.figure(figsize=figsize)

    # Create main grid: num_factors rows, 2 columns
    # Col 0: Belief regression (Theory + Activations side by side)
    # Col 1: CEV (top half) and dims@95 (bottom half)
    gs_main = fig.add_gridspec(
        num_factors, 2,
        width_ratios=[1.8, 1.4],
        height_ratios=[1.0] * num_factors,
        hspace=0.02,  # Minimal vertical space between rows
        wspace=0.12,  # Gap between belief grid and CEV/dims
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.10,
    )

    # For belief regression, we'll create subgridspecs per row with minimal internal spacing
    # This is handled in the loop below

    # Plot belief regression factors vertically
    if belief_regression_data is not None and layer_name is not None:
        resolved_belief_layer = _resolve_layer_name(layer_name, list(belief_regression_data.keys()))
        if resolved_belief_layer is not None:
            data = belief_regression_data[resolved_belief_layer]
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            factor_dims = data["factor_dims"]
            n_samples = len(y_true)

            # Subsample if needed
            max_samples = 10000
            if n_samples > max_samples:
                indices = np.random.choice(n_samples, max_samples, replace=False)
                y_true = y_true[indices]
                y_pred = y_pred[indices]
                n_samples = max_samples

            offset = 0
            for factor_idx in range(num_factors):
                factor_dim = factor_dims[factor_idx]
                start_idx = offset
                end_idx = offset + factor_dim
                offset = end_idx

                # Extract beliefs for this factor
                true_factor = y_true[:, start_idx:end_idx]
                pred_factor = y_pred[:, start_idx:end_idx]

                # Project to 2D if needed
                if factor_dim > 2:
                    pca = PCA(n_components=2)
                    combined = np.vstack([true_factor, pred_factor])
                    pca.fit(combined)
                    true_2d = pca.transform(true_factor)
                    pred_2d = pca.transform(pred_factor)
                elif factor_dim == 1:
                    true_2d = np.column_stack([true_factor, np.zeros(n_samples)])
                    pred_2d = np.column_stack([pred_factor, np.zeros(n_samples)])
                else:
                    true_2d = true_factor
                    pred_2d = pred_factor

                # Compute RGB colors:
                # - F0, F1, F2 (triangular simplices): colormap barycentric
                # - F3, F4 (circular): colormap barycentric on 2D projection
                use_cmap = factor_idx < 3 and factor_dim >= 3
                colors = _compute_rgb_colors(
                    true_factor, factor_dim, use_cmap_barycentric=use_cmap,
                    cmap=cmap, cmap_start=cmap_start, cmap_mid=cmap_mid, cmap_end=cmap_end
                )

                # Compute shared axis ranges
                x_min = min(true_2d[:, 0].min(), pred_2d[:, 0].min())
                x_max = max(true_2d[:, 0].max(), pred_2d[:, 0].max())
                y_min = min(true_2d[:, 1].min(), pred_2d[:, 1].min())
                y_max = max(true_2d[:, 1].max(), pred_2d[:, 1].max())
                x_span = x_max - x_min
                y_span = y_max - y_min
                max_span = max(x_span, y_span)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                pad_frac = 0.02  # Small padding
                half_span = max_span / 2 * (1 + pad_frac)
                shared_x_range = (x_center - half_span, x_center + half_span)
                shared_y_range = (y_center - half_span, y_center + half_span)

                # Create subgridspec for this row's belief plots (Theory + Activations)
                gs_belief_row = gs_main[factor_idx, 0].subgridspec(1, 2, wspace=-0.4)

                # Plot Theory (left)
                ax_theory = fig.add_subplot(gs_belief_row[0, 0])
                ax_theory.scatter(
                    true_2d[:, 0], true_2d[:, 1], c=colors,
                    s=2.5, alpha=0.7, edgecolors="none", rasterized=True,
                )
                ax_theory.set_xlim(shared_x_range)
                ax_theory.set_ylim(shared_y_range)
                ax_theory.set_xticks([])
                ax_theory.set_yticks([])
                ax_theory.set_aspect("equal", adjustable="box")
                for spine in ax_theory.spines.values():
                    spine.set_visible(False)

                # Add factor label on left
                ax_theory.set_ylabel(f"$F_{factor_idx}$", fontsize=15, rotation=0, ha="right", va="center", labelpad=10)

                # Plot Activations (right)
                ax_act = fig.add_subplot(gs_belief_row[0, 1])
                ax_act.scatter(
                    pred_2d[:, 0], pred_2d[:, 1], c=colors,
                    s=1.5, alpha=0.35, edgecolors="none", rasterized=True,
                )
                ax_act.set_xlim(shared_x_range)
                ax_act.set_ylim(shared_y_range)
                ax_act.set_xticks([])
                ax_act.set_yticks([])
                ax_act.set_aspect("equal", adjustable="box")
                for spine in ax_act.spines.values():
                    spine.set_visible(False)

            # Add column headers - aligned with scatter plot columns, moved down
            fig.text(0.18, 0.94, "Theory", ha="center", va="bottom", fontsize=14)
            fig.text(0.36, 0.94, "Activations", ha="center", va="bottom", fontsize=14)

    # Panel label (a) for belief regression - aligned with (b)
    fig.text(0.06, 0.98, "(a)", fontsize=15, fontweight="bold", va="top", ha="left")

    # Determine if we need broken axis for dims@95
    use_broken_axis = False
    if metrics is not None and "dims95" in metrics and joint_dims95 is not None:
        dims95_df = metrics["dims95"]
        if dims_max_step is not None:
            dims95_df = dims95_df[dims95_df["step"] <= dims_max_step]
        values = dims95_df["value"].values
        use_broken_axis = joint_dims95 > values.max() * 1.5

    # Create subplots for CEV and dims@95
    # Use subgridspec for col 1 to control spacing between CEV and dims independently
    gs_right = gs_main[:, 1].subgridspec(2, 1, height_ratios=[1.0, 0.8], hspace=0.35)

    # Create subplot for CEV (top)
    ax_cev = fig.add_subplot(gs_right[0])

    # Panel label (b) for CEV
    fig.text(0.52, 0.98, "(b)", fontsize=15, fontweight="bold", va="top", ha="left")

    # Plot CEV curves
    if cev_history is not None and layer_name is not None:
        resolved_cev_layer = _resolve_layer_name(layer_name, list(cev_history.keys()))
        if resolved_cev_layer is not None:
            plot_cev_curves(
                cev_history, resolved_cev_layer,
                ax=ax_cev,
                max_step=cev_max_step,
                belief_baselines=belief_baselines,
                dims95_inset=None,
                linewidth=2.5,  # Thicker lines
                alpha=cev_alpha,
            )
            ax_cev.set_xlabel("Dimension", fontsize=14)

    # Plot dims@95 over training with broken y-axis if needed (matching figure 2)
    if metrics is not None and "dims95" in metrics:
        dims95_df = metrics["dims95"]
        if dims_max_step is not None:
            dims95_df = dims95_df[dims95_df["step"] <= dims_max_step]

        steps = dims95_df["step"].values
        values = dims95_df["value"].values
        x_max = dims95_df["step"].max()

        # Panel label (c) for dims@95
        fig.text(0.52, 0.44, "(c)", fontsize=15, fontweight="bold", va="top", ha="left")

        if use_broken_axis:
            # Create broken axis using subgridspec
            gs_dims_broken = gs_right[1].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.08)
            ax_top = fig.add_subplot(gs_dims_broken[0])  # For joint line
            ax_bottom = fig.add_subplot(gs_dims_broken[1])  # For data

            # Set y-limits
            data_max = max(values.max(), gt_dims95 if gt_dims95 else 0) * 1.15
            ax_bottom.set_ylim(0, data_max)
            ax_top.set_ylim(joint_dims95 - 10, joint_dims95 + 10)

            # Plot data on bottom axis with color by step
            points = np.array([steps, values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm_max = cev_max_step if cev_max_step is not None else steps.max()
            norm = mcolors.LogNorm(vmin=max(steps.min(), 1), vmax=norm_max)
            lc = LineCollection(segments, cmap="viridis_r", norm=norm, capstyle='round', joinstyle='round')
            lc.set_array(steps[:-1])
            lc.set_linewidth(4.5)  # Thicker data line
            lc.set_zorder(2)
            ax_bottom.add_collection(lc)
            ax_bottom.set_xlim(steps.min(), steps.max())

            # Factored line on bottom (behind data)
            if gt_dims95 is not None:
                ax_bottom.axhline(y=gt_dims95, color="#c44e52", linewidth=2.0, linestyle="--", zorder=1)
                ax_bottom.text(
                    x_max * 0.95, gt_dims95 + 4,
                    f"Factored ({gt_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="bottom",
                )

            # Joint line on top
            ax_top.axhline(y=joint_dims95, color="#c44e52", linewidth=2.0, linestyle="--")
            ax_top.text(
                x_max * 0.95, joint_dims95 + 2,
                f"Joint ({joint_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="bottom",
            )
            ax_top.set_xlim(steps.min(), steps.max())

            # Hide spines between axes
            ax_top.spines["bottom"].set_visible(False)
            ax_bottom.spines["top"].set_visible(False)
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            ax_bottom.spines["right"].set_visible(False)

            # Add break marks
            d_pix = 5
            kwargs = dict(color='k', clip_on=False, linewidth=0.8, transform=fig.transFigure)
            fig_width, fig_height = fig.get_size_inches() * fig.dpi
            dx = d_pix / fig_width
            dy = d_pix / fig_height

            top_bottom_left = ax_top.transAxes.transform((0, 0))
            bottom_top_left = ax_bottom.transAxes.transform((0, 1))
            inv = fig.transFigure.inverted()
            tbl = inv.transform(top_bottom_left)
            btl = inv.transform(bottom_top_left)

            ax_top.plot([tbl[0]-dx, tbl[0]+dx], [tbl[1]-dy, tbl[1]+dy], **kwargs)
            ax_bottom.plot([btl[0]-dx, btl[0]+dx], [btl[1]-dy, btl[1]+dy], **kwargs)

            # Labels
            ax_bottom.set_xlabel("Training step", fontsize=14)
            ax_bottom.set_ylabel("Dimensions for 95%", fontsize=14)
            ax_bottom.tick_params(axis="both", labelsize=12)
            ax_top.tick_params(axis="y", labelsize=12)
            # Dynamic x-ticks: [0, 2500, 5000] + every 5000 after that
            max_tick = int(x_max) if x_max else 5000
            xticks = [0, 2500, 5000] + list(range(10000, max_tick + 1, 5000))
            ax_bottom.set_xticks([t for t in xticks if t <= max_tick])
            ax_bottom.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            ))
            ax_bottom.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
            ax_top.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

        else:
            # Regular single axis (no break needed)
            ax_dims = fig.add_subplot(gs_right[1])

            # Color by training step (matching CEV colormap)
            points = np.array([steps, values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm_max = cev_max_step if cev_max_step is not None else steps.max()
            norm = mcolors.LogNorm(vmin=max(steps.min(), 1), vmax=norm_max)
            lc = LineCollection(segments, cmap="viridis_r", norm=norm, capstyle='round', joinstyle='round')
            lc.set_array(steps[:-1])
            lc.set_linewidth(4.5)  # Thicker data line
            lc.set_zorder(2)
            ax_dims.add_collection(lc)
            ax_dims.autoscale()

            # Reference lines (matching figure 2 style)
            if gt_dims95 is not None:
                ax_dims.axhline(y=gt_dims95, color="#c44e52", linewidth=2.0, linestyle="--", zorder=1)
                ax_dims.text(
                    x_max * 0.95, gt_dims95 + 4,
                    f"Factored ({gt_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="bottom",
                )
            if joint_dims95 is not None:
                ax_dims.axhline(y=joint_dims95, color="#c44e52", linewidth=2.0, linestyle="--", zorder=1)
                ax_dims.text(
                    x_max * 0.95, joint_dims95 - 1.5,
                    f"Joint ({joint_dims95:.0f})", fontsize=12, color="#c44e52", ha="right", va="top",
                )

            ax_dims.set_xlabel("Training step", fontsize=14)
            ax_dims.set_ylabel("Dimensions for 95%", fontsize=14)
            ax_dims.set_ylim(bottom=0)
            ax_dims.tick_params(axis="both", labelsize=12)
            ax_dims.spines["top"].set_visible(False)
            ax_dims.spines["right"].set_visible(False)
            # Dynamic x-ticks: [0, 2500, 5000] + every 5000 after that
            max_tick = int(x_max) if x_max else 5000
            xticks = [0, 2500, 5000] + list(range(10000, max_tick + 1, 5000))
            ax_dims.set_xticks([t for t in xticks if t <= max_tick])
            ax_dims.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            ))
            ax_dims.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

    return fig


def create_composite_figure(
    belief_regression_data: dict[str, Any] | None = None,
    cev_history: dict[str, list[tuple[int, np.ndarray]]] | None = None,
    metrics: dict[str, pd.DataFrame] | None = None,
    layer_name: str | None = None,
    gt_dims95: float | None = None,
    cev_max_step: int | None = None,
    dims_max_step: int | None = None,
    rmse_max_step: int | None = None,
    figsize: tuple[float, float] = (14, 10),
    checkpoint_step: int | None = None,
    random_loss: float | None = None,
    entropy_rate: float | None = None,
    belief_baselines: dict[str, np.ndarray] | None = None,
    log_scale: bool = False,
) -> Figure:
    """Create the composite 5-panel figure.

    Layout:
    - Row 1: (a) Belief regression - full width
    - Row 2: (b) CEV curves, (c) RMSE over training
    - Row 3: (d) Dims@95 vs Loss, (e) Loss vs RMSE

    Args:
        belief_regression_data: Data for belief regression plot (from analysis.py).
        cev_history: CEV history for CEV curves plot.
        metrics: Dict of metric DataFrames (dims95, loss, factor_*_rmse).
        layer_name: Layer name for layer-specific plots.
        gt_dims95: Ground truth dims@95 reference line.
        cev_max_step: Max step for CEV plot.
        dims_max_step: Max step for dims/loss plot (panel c).
        rmse_max_step: Max step for dims/RMSE plot (panel d). If None, uses dims_max_step.
        figsize: Figure size.
        checkpoint_step: Checkpoint step used for belief regression (for labeling).
        random_loss: Random guesser loss (log(vocab_size)) for reference line.
        entropy_rate: Entropy rate (optimal loss) for reference line.
        belief_baselines: Optional dict with "factored" and/or "product" CEV arrays
            to plot as reference baselines on the CEV curves plot.

    Returns:
        Matplotlib Figure with 5 panels.
    """
    # Apply ICML style
    apply_icml_style()

    fig = plt.figure(figsize=figsize)

    # Create grid: 3 rows, 2 cols
    # Row 1: belief regression (spans both cols)
    # Row 2: CEV, RMSE over training
    # Row 3: Dims@95 vs Loss, Loss vs RMSE

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 0.9, 0.9],
        width_ratios=[1.0, 1.0],
        hspace=0.45,
        wspace=0.3,
        left=0.08,
        right=0.95,
    )

    # Row 1: Belief regression visualization (spans both columns)
    if belief_regression_data is not None and layer_name is not None:
        resolved_belief_layer = _resolve_layer_name(layer_name, list(belief_regression_data.keys()))
        if resolved_belief_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in belief_regression_data. "
                f"Available layers: {list(belief_regression_data.keys())}"
            )
        data = belief_regression_data[resolved_belief_layer]
        num_factors = data["num_factors"]
        # Create subgrid spanning all columns: 2 data rows only
        gs_belief = gs[0, :].subgridspec(2, num_factors, hspace=0.08, wspace=0.05)
        ax_belief = np.array(
            [[fig.add_subplot(gs_belief[i, j]) for j in range(num_factors)] for i in range(2)]
        )
        plot_belief_regression_grid(
            data["y_true"],
            data["y_pred"],
            num_factors=data["num_factors"],
            overall_rmse=data.get("overall_rmse", 0.0),
            factor_rmse_scores=data.get("factor_rmse_scores", []),
            factor_dims=data["factor_dims"],
            factor_names=data.get("factor_names"),
            ax_grid=ax_belief,
            cmap=cmap,
            cmap_start=cmap_start,
            cmap_mid=cmap_mid,
            cmap_end=cmap_end,
        )

    # Row 2, Col 0: CEV curves
    if cev_history is not None and layer_name is not None:
        ax_cev = fig.add_subplot(gs[1, 0])
        resolved_cev_layer = _resolve_layer_name(layer_name, list(cev_history.keys()))
        if resolved_cev_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in cev_history. "
                f"Available layers: {list(cev_history.keys())}"
            )
        # Prepare dims@95 inset data
        dims95_inset = None
        if metrics is not None and "dims95" in metrics:
            dims95_inset = (metrics["dims95"], gt_dims95, dims_max_step)

        plot_cev_curves(
            cev_history, resolved_cev_layer,
            ax=ax_cev,
            max_step=cev_max_step,
            belief_baselines=belief_baselines,
            dims95_inset=dims95_inset,
        )

    # Row 2, Col 1: RMSE over training
    if metrics is not None and "loss" in metrics:
        ax_rmse = fig.add_subplot(gs[1, 1])
        factor_rmse_dfs = {
            k: v for k, v in metrics.items() if k.startswith("factor_") and k.endswith("_rmse")
        }
        if factor_rmse_dfs:
            rmse_step_limit = rmse_max_step if rmse_max_step is not None else dims_max_step
            plot_rmse_over_training(
                metrics["loss"],
                factor_rmse_dfs,
                ax=ax_rmse,
                optimal_loss=entropy_rate,
                max_step=rmse_step_limit,
                show_inset=False,
            )

    # Compute shared loss range for bottom row plots
    loss_range = None
    if metrics is not None and "loss" in metrics:
        rmse_step_limit = rmse_max_step if rmse_max_step is not None else dims_max_step
        loss_df = metrics["loss"]
        if rmse_step_limit is not None:
            loss_df = loss_df[loss_df["step"] <= rmse_step_limit]
        loss_min, loss_max = loss_df["value"].min(), loss_df["value"].max()
        loss_padding = (loss_max - loss_min) * 0.05
        loss_range = (loss_max + loss_padding, loss_min - loss_padding)  # Inverted for invert_xaxis

    # Row 3, Col 0: Dims@95 vs Loss scatter
    if metrics is not None and "dims95" in metrics and "loss" in metrics:
        ax_dims_loss = fig.add_subplot(gs[2, 0])
        rmse_step_limit = rmse_max_step if rmse_max_step is not None else dims_max_step
        plot_dims95_vs_loss(
            metrics["dims95"],
            metrics["loss"],
            ax=ax_dims_loss,
            gt_dims95=gt_dims95,
            max_step=rmse_step_limit,
        )
        if loss_range is not None:
            ax_dims_loss.set_xlim(loss_range)

    # Row 3, Col 1: Loss vs RMSE scatter
    if metrics is not None and "loss" in metrics:
        ax_scatter = fig.add_subplot(gs[2, 1])
        factor_rmse_dfs = {
            k: v for k, v in metrics.items() if k.startswith("factor_") and k.endswith("_rmse")
        }
        if factor_rmse_dfs:
            rmse_step_limit = rmse_max_step if rmse_max_step is not None else dims_max_step
            plot_loss_vs_rmse(
                metrics["loss"],
                factor_rmse_dfs,
                ax=ax_scatter,
                optimal_loss=entropy_rate,
                max_step=rmse_step_limit,
            )
            if loss_range is not None:
                ax_scatter.set_xlim(loss_range)

    return fig
