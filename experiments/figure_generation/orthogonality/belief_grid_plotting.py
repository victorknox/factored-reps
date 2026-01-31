"""Standalone belief regression grid plotting for orthogonality figures.

This module contains the belief regression grid visualization, copied from
figure_generation/belief-grid/plotting.py to make orthogonality figure
generation self-contained and extensible.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def _compute_rgb_colors(
    belief_values: np.ndarray,
    dim: int,
    use_cmap_barycentric: bool = False,
    cmap: str = "viridis",
    cmap_start: float = 0.2,
    cmap_mid: float = 0.5,
    cmap_end: float = 0.8,
) -> np.ndarray:
    """Compute RGB colors from belief state values.

    Args:
        belief_values: Array of belief state values [n_samples, belief_dim]
        dim: Dimensionality of the belief state
        use_cmap_barycentric: If True, use colormap-based barycentric blending.
                              If False, use teal-coral-gold barycentric blending.
        cmap: Matplotlib colormap name to use for barycentric blending (default: viridis).
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


def compute_factor_plot_sizes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    factor_dims: list[int],
    scatter_pad_frac: float = 0.1,
) -> list[float]:
    """Compute the plot size (max span) for each factor.

    Used to determine height_ratios for subgridspec so cells match
    actual plot sizes when preserve_aspect=True.

    Returns list of max_span values, one per factor.
    """
    from sklearn.decomposition import PCA

    sizes = []
    offset = 0
    for factor_dim in factor_dims:
        start_idx = offset
        end_idx = offset + factor_dim
        offset = end_idx

        true_factor = y_true[:, start_idx:end_idx]
        pred_factor = y_pred[:, start_idx:end_idx]

        # Project to 2D if needed (same logic as _plot_belief_regression_grid)
        if factor_dim > 2:
            pca = PCA(n_components=2)
            combined = np.vstack([true_factor, pred_factor])
            pca.fit(combined)
            true_2d = pca.transform(true_factor)
        elif factor_dim == 1:
            n_samples = len(true_factor)
            true_2d = np.column_stack([true_factor, np.zeros(n_samples)])
        else:
            true_2d = true_factor

        # Compute range (same logic as compute_padded_range)
        x_min, x_max = float(true_2d[:, 0].min()), float(true_2d[:, 0].max())
        y_min, y_max = float(true_2d[:, 1].min()), float(true_2d[:, 1].max())
        x_span = x_max - x_min
        y_span = y_max - y_min
        max_span = max(x_span, y_span) * (1 + scatter_pad_frac)
        sizes.append(max_span)

    return sizes


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
    marker_size: float = 0.5,
    marker_opacity: float = 0.7,
    pred_marker_size: float = 0.75,
    pred_marker_opacity: float = 0.35,
    scatter_pad_frac: float = 0.1,
    title_pad: float = 2.0,
    preserve_aspect: bool = True,
    title: str | None = None,
    title_suffix: str = "",
    sample_indices: np.ndarray | None = None,
    orientation: str = "horizontal",
    factor_label_x: float = 1.12,
    factor_label_y: float = 0.5,
    factor_label_ha: str = "center",
    anchor_theory: str | None = "N",
    anchor_activations: str | None = "N",
    cmap: str = "viridis",
    cmap_start: float = 0.2,
    cmap_mid: float = 0.5,
    cmap_end: float = 0.8,
) -> Figure | None:
    """Plot belief regression as a grid: ground truth vs predictions.

    Creates a compact visualization comparing theory (ground truth) belief states
    to activations (predictions from linear regression).

    Orientation modes:
    - "horizontal": 2 rows × N cols (rows = GT/Pred, cols = factors)
      - Factor labels as column headers
      - "Theory"/"Activations" as row labels on left
    - "vertical": N rows × 2 cols (rows = factors, cols = GT/Pred)
      - "Theory"/"Activations" as column headers
      - Factor labels centered between the two columns

    All plots are 2D scatter plots. For beliefs with >2 dimensions, PCA is used
    to project to 2D.

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
        ax_grid: Optional pre-created axes grid for embedding.
            - horizontal: shape (2, num_factors)
            - vertical: shape (num_factors, 2)
        max_samples: Maximum samples to plot (for performance).
        marker_size: Size of scatter markers.
        marker_opacity: Opacity of markers.
        pred_marker_size: Size of prediction scatter markers.
        pred_marker_opacity: Opacity of prediction markers.
        title: Optional custom title override.
        title_suffix: Optional suffix to append to title.
        sample_indices: Optional indices to subsample for visualization.
        orientation: Layout orientation - "horizontal" or "vertical".

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
        if orientation == "vertical":
            fig, axes = plt.subplots(num_factors, 2, figsize=(6, 3 * num_factors))
        else:
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

        # Compute RGB colors from original belief values (not 2D-projected)
        # Use cmap barycentric for factors 0-2 (with dim >= 3), teal-coral-gold for others
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
        true_x_range, true_y_range = compute_padded_range(true_2d, pad_frac=scatter_pad_frac)
        shared_x_range, shared_y_range = true_x_range, true_y_range

        # Factor label
        factor_label = f"$F_{factor_idx}$"

        # Get axes based on orientation
        if orientation == "vertical":
            ax_true: Axes = ax_grid[factor_idx, 0]  # Row = factor, Col = 0 (GT)
            ax_pred: Axes = ax_grid[factor_idx, 1]  # Row = factor, Col = 1 (Pred)
        else:
            ax_true: Axes = ax_grid[0, factor_idx]  # Row = 0 (GT), Col = factor
            ax_pred: Axes = ax_grid[1, factor_idx]  # Row = 1 (Pred), Col = factor

        # Plot ground truth
        ax_true.scatter(
            true_2d[:, 0], true_2d[:, 1], c=colors,
            s=marker_size, alpha=marker_opacity, edgecolors="none", rasterized=True,
        )
        ax_true.set_xlim(true_x_range)
        ax_true.set_ylim(true_y_range)
        ax_true.set_xticks([])
        ax_true.set_yticks([])
        if preserve_aspect:
            ax_true.set_aspect("equal", adjustable="box")
            if anchor_theory:
                ax_true.set_anchor(anchor_theory)
        for spine in ax_true.spines.values():
            spine.set_visible(False)

        # Plot predictions - same colors, SAME scale as GT
        ax_pred.scatter(
            pred_2d[:, 0], pred_2d[:, 1], c=colors,
            s=pred_marker_size, alpha=pred_marker_opacity, edgecolors="none", rasterized=True,
        )
        ax_pred.set_xlim(shared_x_range)
        ax_pred.set_ylim(shared_y_range)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        if preserve_aspect:
            ax_pred.set_aspect("equal", adjustable="box")
            if anchor_activations:
                ax_pred.set_anchor(anchor_activations)
        for spine in ax_pred.spines.values():
            spine.set_visible(False)

        # Labels depend on orientation
        if orientation == "vertical":
            # Column headers (only for first factor row)
            if factor_idx == 0:
                ax_true.set_title("Theory", fontsize=8, fontweight="medium", pad=title_pad)
                ax_pred.set_title("Activations", fontsize=8, fontweight="medium", pad=title_pad)
            # Factor label position controlled by factor_label_x, factor_label_y, factor_label_ha
            # Default: x=-0.15, ha="right" puts it to the left of Theory column
            ax_true.text(factor_label_x, factor_label_y, factor_label, transform=ax_true.transAxes,
                         fontsize=9, fontweight="medium", va="center", ha=factor_label_ha)
        else:
            # Column header (factor label) for horizontal orientation
            ax_true.set_title(factor_label, fontsize=9, fontweight="medium", pad=title_pad)

    # Row labels (only for horizontal orientation)
    if orientation == "horizontal":
        ax_grid[0, 0].set_ylabel("Theory", fontsize=7, labelpad=2)
        ax_grid[1, 0].set_ylabel("Activations", fontsize=7, labelpad=2)

    if created_fig and fig is not None:
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig
    return None
