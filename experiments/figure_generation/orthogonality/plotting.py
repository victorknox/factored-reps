"""Plotting functions for orthogonality figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patheffects as patheffects

from analysis import OrthogonalityData, compute_dims_at_threshold

# Custom colormap matching freeze_vary background colors (teal → coral)
TEAL_CORAL_CMAP = LinearSegmentedColormap.from_list(
    "teal_coral",
    [(0.25, 0.70, 0.75), (0.95, 0.45, 0.35)],  # teal → coral
)
# Register so it can be used by name in configs
plt.colormaps.register(cmap=TEAL_CORAL_CMAP)
from belief_regression import ProjectedBeliefRegressionData

# Import from local standalone module
from belief_grid_plotting import plot_belief_regression_grid as _plot_belief_regression_grid


# ICML publication-quality style settings
ICML_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_icml_style():
    """Apply ICML publication style to matplotlib."""
    plt.rcParams.update(ICML_STYLE)


def _auto_crop_whitespace(img: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """Crop whitespace from image edges.

    Args:
        img: Image array [H, W, C] or [H, W]
        threshold: Pixels with all channels > threshold are considered white

    Returns:
        Cropped image array
    """
    if img.ndim == 2:
        # Grayscale
        mask = img < threshold
    else:
        # RGB/RGBA - check if any channel is non-white
        if img.shape[2] == 4:
            # For RGBA, also check alpha
            mask = (img[:, :, :3].max(axis=2) < threshold) | (img[:, :, 3] > 0.01)
        else:
            mask = img.max(axis=2) < threshold

    # Find rows and columns with content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img  # No content found, return original

    # Get bounding box
    row_start, row_end = np.where(rows)[0][[0, -1]]
    col_start, col_end = np.where(cols)[0][[0, -1]]

    # Add small padding (1-2 pixels) to avoid cutting content
    pad = 2
    row_start = max(0, row_start - pad)
    row_end = min(img.shape[0], row_end + pad + 1)
    col_start = max(0, col_start - pad)
    col_end = min(img.shape[1], col_end + pad + 1)

    return img[row_start:row_end, col_start:col_end]


def plot_overlap_over_training(
    orthogonality_data: OrthogonalityData,
    baseline_mean: np.ndarray | None = None,
    baseline_lower: np.ndarray | None = None,
    baseline_upper: np.ndarray | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.75, 4.0),
    cmap: str = "viridis",
    individual_alpha: float = 0.2,
    individual_linewidth: float = 1.0,
    average_linewidth: float = 2.5,
    marker_size: float = 4,
    markevery: int = 3,
    ci_color: str = "orange",
    ci_alpha: float = 0.15,
    baseline_linestyle: str = "--",
    ci_percentile: int = 90,
    title: str | None = None,
    step_filter: list[int] | None = None,
    legend_loc: str = "upper right",
    xlabel_labelpad: float = 4.0,
    show_baseline_mean: bool = True,
) -> Figure | None:
    """Plot normalized overlap over training (single panel).

    Creates a single panel showing:
    - Y-axis (log scale): Overlap values from ~0.0001 to 1
    - X-axis (log scale): Number of components k (1 to max_k)
    - Colored lines for each training step (viridis colormap)
    - Faint lines (alpha=0.2) for individual factor pairs
    - Bold lines with markers for average across off-diagonal pairs
    - Orange shaded region: Random init baseline CI
    - Orange dashed line: Random init mean
    - Legend showing step labels

    Args:
        orthogonality_data: Results from compute_orthogonality_at_checkpoints
        baseline_mean: Random init baseline mean [max_k]
        baseline_lower: Random init baseline lower CI [max_k]
        baseline_upper: Random init baseline upper CI [max_k]
        ax: Pre-created axes (optional)
        figsize: Figure size if creating new figure
        cmap: Colormap name for step coloring
        individual_alpha: Alpha for individual pair lines
        individual_linewidth: Line width for individual pairs
        average_linewidth: Line width for average line
        marker_size: Marker size for average line
        markevery: Plot marker every N points
        ci_color: Color for baseline CI
        ci_alpha: Alpha for baseline CI fill
        baseline_linestyle: Line style for baseline mean
        ci_percentile: CI percentile for label (e.g., 90 for 90% CI)
        title: Optional title override

    Returns:
        Figure if ax was None, else None
    """
    # Apply ICML style
    apply_icml_style()

    # Create figure if not provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Determine which steps to plot
    all_steps = orthogonality_data.steps
    if step_filter is not None:
        # Find indices of requested steps
        step_indices = [i for i, s in enumerate(all_steps) if s in step_filter]
        steps = [all_steps[i] for i in step_indices]
    else:
        step_indices = list(range(len(all_steps)))
        steps = all_steps

    # Color normalization for training steps
    # Reversed: early steps = light (yellow, 0.9), late steps = dark (purple, 0.1)
    step_colors = plt.cm.get_cmap(cmap)(np.linspace(0.9, 0.1, len(steps)))

    k_range = orthogonality_data.k_range
    max_k = orthogonality_data.max_k

    # Plot baseline CI (with high zorder so it's visible above training lines)
    if baseline_mean is not None:
        baseline_k_range = min(max_k, len(baseline_mean))
        baseline_k = np.arange(1, baseline_k_range + 1)

        if baseline_lower is not None and baseline_upper is not None:
            ax.fill_between(
                baseline_k,
                baseline_lower[:baseline_k_range],
                baseline_upper[:baseline_k_range],
                alpha=ci_alpha,
                color=ci_color,
                label="Random Init",
                zorder=10,
            )

        if show_baseline_mean:
            ax.plot(
                baseline_k,
                baseline_mean[:baseline_k_range],
                color=ci_color,
                linestyle=baseline_linestyle,
                linewidth=1.5,
                alpha=0.8,
                label="Random Init Mean",
                zorder=11,
            )

    # Plot each training step (filtered)
    for color_idx, (orig_step_idx, step) in enumerate(zip(step_indices, steps)):
        color = step_colors[color_idx]

        # Individual factor pair lines (faint)
        for pair_key, values in orthogonality_data.per_pair_overlap[orig_step_idx]:
            ax.plot(
                k_range,
                values,
                color=color,
                alpha=individual_alpha,
                linewidth=individual_linewidth,
            )

        # Average line (bold with markers)
        ax.plot(
            k_range,
            orthogonality_data.avg_overlap_per_step[orig_step_idx],
            color=color,
            alpha=1.0,
            linewidth=average_linewidth,
            marker="o",
            markersize=marker_size,
            markevery=markevery,
            label=f"Step {step:,}",
        )

    # Configure axes
    ax.set_xlabel("Components (k)", labelpad=xlabel_labelpad)
    ax.set_ylabel("Overlap")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    ax.legend(loc=legend_loc, fontsize="small")

    # Title (skip if empty string)
    if title is None:
        title = "Factor Subspace Orthogonality Over Training"
    if title:
        ax.set_title(title, fontsize=9, fontweight="medium", loc="left")

    if created_fig and fig is not None:
        fig.tight_layout()
        return fig
    return None


def plot_stacked_dims_over_training(
    steps: list[int],
    dims_per_factor: np.ndarray,  # [n_steps, n_factors]
    combined_dims: np.ndarray | None = None,  # [n_steps] - "Joint" (natural generation)
    union_dims: np.ndarray | None = None,  # [n_steps] - "Union" (concatenated vary-one)
    expected_total: int | None = None,
    threshold_label: str = "95%",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.75, 4.0),
    cmap: str = "tab10",
    combined_linestyle: str = "k-",
    combined_linewidth: float = 2.5,
    combined_marker_size: float = 4.0,
    line_cmap: str | None = None,  # Colormap for lines (None = use combined_linestyle)
    show_joint: bool = True,  # Show "Joint" line (natural generation)
    show_union: bool = True,  # Show "Union" line (concatenated vary-one)
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str | None = None,
    inset_enabled: bool = False,
    inset_bounds: tuple[float, float, float, float] = (0.55, 0.45, 0.42, 0.5),
    inset_xlim: tuple[float, float] | None = None,
    inset_ylim: tuple[float, float] | None = None,
    title: str | None = None,
    legend_loc: str = "upper right",
    xlabel_labelpad: float = 4.0,
    # Theory line parameters
    theory_lines: dict[str, np.ndarray] | None = None,  # {"Mean": [...], "Per-Factor": [...]}
    theory_linestyle: str = ":",
    theory_color: str = "0.5",
    theory_linewidth: float = 1.0,
    theory_marker: str = "s",
    theory_markersize: float = 2.0,
) -> Figure | None:
    """Plot stacked area chart of factor dimensions over training.

    Shows how many dimensions each factor requires at a given variance threshold,
    stacked to show total dimensionality over training.

    Args:
        steps: Training step numbers
        dims_per_factor: Array [n_steps, n_factors] with dims per factor
        combined_dims: Optional array [n_steps] of combined dims
        expected_total: Optional expected total for reference line
        threshold_label: Label for variance threshold (e.g., '95%')
        ax: Pre-created axes (optional)
        figsize: Figure size if creating new figure
        cmap: Colormap for factor colors
        combined_linestyle: Linestyle for combined dims line (e.g., "k-", "k--")
        combined_linewidth: Linewidth for combined dims line
        combined_marker_size: Marker size for combined dims line
        xlim: X-axis limits (min, max) or None for auto
        ylim: Y-axis limits (min, max) or None for auto
        xscale: X-axis scale (e.g., "log") or None for linear
        inset_enabled: Whether to show an inset with zoomed region
        inset_bounds: Inset position/size as (x, y, width, height) in axes coords
        inset_xlim: X-axis limits for inset (step indices), auto if None
        inset_ylim: Y-axis limits for inset, auto if None
        title: Optional title override
        legend_loc: Location for legend (matplotlib location string)

    Returns:
        Figure if ax was None, else None
    """
    apply_icml_style()

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    num_factors = dims_per_factor.shape[1]
    colormap = plt.cm.get_cmap(cmap)
    # Sample colormap at evenly-spaced positions for categorical data
    factor_colors = [colormap(i / (num_factors - 1) if num_factors > 1 else 0.5) for i in range(num_factors)]

    # Use indices for x-axis (equal spacing), label with actual step numbers
    indices = list(range(len(steps)))

    # Stacked area plot (no legend labels - we'll overlay them directly)
    ax.stackplot(
        indices,
        dims_per_factor.T,
        colors=factor_colors,
        alpha=0.4,
    )

    # Overlay factor labels at left edge of each stacked region
    cumulative = 0.0
    for factor_idx in range(num_factors):
        # Calculate vertical center of this factor's band at first step (x=0)
        band_bottom = cumulative
        band_top = cumulative + dims_per_factor[0, factor_idx]
        band_center = (band_bottom + band_top) / 2
        cumulative = band_top

        # Place label with small offset from left edge
        ax.text(
            0.3,  # Slight offset from x=0
            band_center,
            f"$F_{factor_idx}$",
            fontsize=7,
            fontweight="medium",
            va="center",
            ha="left",
            color="black",
        )

    # Joint line (natural generation, all factors varying together) - color-coded by training step
    if combined_dims is not None and show_joint:
        if line_cmap is not None and len(indices) > 1:
            cmap_obj = plt.cm.get_cmap(line_cmap)
            n_points = len(indices)

            # White background line for visibility (subtle)
            ax.plot(indices, combined_dims, 'w-', linewidth=combined_linewidth + 0.8,
                    zorder=4)

            # Create line segments with colors matching training progression
            # Reversed: early=light (0.9), late=dark (0.1)
            points = np.column_stack([indices, combined_dims])
            segments = np.array([[points[i], points[i + 1]] for i in range(n_points - 1)])
            colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points - 1)]

            lc = LineCollection(segments, colors=colors, linewidth=combined_linewidth, zorder=5)
            ax.add_collection(lc)

            # Colored markers (no legend label)
            marker_colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points)]
            ax.scatter(indices, combined_dims, c=marker_colors, s=combined_marker_size**2,
                       marker='s', zorder=6, edgecolors='white', linewidths=0.3)
        else:
            # Fallback to single-color line (no legend label)
            ax.plot(
                indices,
                combined_dims,
                combined_linestyle,
                linewidth=combined_linewidth,
                marker="s",
                markersize=combined_marker_size,
            )

        # Arrow annotation pointing to "Joint" line
        arrow_x_idx = int(len(indices) * 0.6)
        arrow_x = indices[arrow_x_idx]
        arrow_y = combined_dims[arrow_x_idx]
        ax.annotate(
            "Joint",
            xy=(arrow_x, arrow_y),
            xytext=(arrow_x + 0.8, arrow_y + 15),
            fontsize=7,
            fontweight="medium",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            ha="left",
            va="bottom",
        )

    # Union line (concatenated vary-one data, union of factor subspaces) - color-coded dashed
    if union_dims is not None and show_union:
        union_lw = combined_linewidth * 0.8
        union_ms = combined_marker_size * 0.6

        if line_cmap is not None and len(indices) > 1:
            cmap_obj = plt.cm.get_cmap(line_cmap)
            n_points = len(indices)

            # White background line for visibility (dashed)
            ax.plot(indices, union_dims, 'w--', linewidth=union_lw + 0.8, zorder=7)

            # Create dashed line segments with colors matching training progression
            # Reversed: early=light (0.9), late=dark (0.1)
            points = np.column_stack([indices, union_dims])
            segments = np.array([[points[i], points[i + 1]] for i in range(n_points - 1)])
            colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points - 1)]

            lc = LineCollection(segments, colors=colors, linewidth=union_lw,
                               linestyles='dashed', zorder=8)
            ax.add_collection(lc)

            # Colored markers (circle to differentiate from Joint's square)
            marker_colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points)]
            ax.scatter(indices, union_dims, c=marker_colors, s=union_ms**2,
                       marker='o', zorder=9, edgecolors='white', linewidths=0.3)
        else:
            # Fallback to single-color dashed line
            ax.plot(indices, union_dims, "k--", linewidth=union_lw,
                    marker="o", markersize=union_ms, zorder=7)

        # Arrow annotation pointing to "Union" line (on the right side)
        arrow_x_idx = int(len(indices) * 0.7)
        arrow_x = indices[arrow_x_idx]
        arrow_y = union_dims[arrow_x_idx]
        ax.annotate(
            "Union",
            xy=(arrow_x, arrow_y),
            xytext=(arrow_x + 0.8, arrow_y + 18),
            fontsize=7,
            fontweight="medium",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            ha="left",
            va="bottom",
        )

    # Expected total reference line
    if expected_total is not None:
        ax.axhline(
            y=expected_total,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Expected ({expected_total})",
        )

    # Theory lines (prediction based on orthogonality)
    if theory_lines is not None:
        # Define distinct markers for each theory line
        theory_markers = {"Mean": "^", "Per-Factor": "v"}  # Triangle up/down

        for name, values in theory_lines.items():
            marker = theory_markers.get(name, theory_marker)
            ax.plot(
                indices,
                values,
                linestyle=theory_linestyle,
                color=theory_color,
                linewidth=theory_linewidth,
                marker=marker,
                markersize=theory_markersize,
                alpha=0.8,
                zorder=3,
            )

            # Add arrow annotation for theory line
            arrow_x_idx = int(len(indices) * 0.25) if name == "Mean" else int(len(indices) * 0.5)
            arrow_x = indices[arrow_x_idx]
            arrow_y = values[arrow_x_idx]
            # Position annotation below the line
            ax.annotate(
                name,
                xy=(arrow_x, arrow_y),
                xytext=(arrow_x + 0.5, arrow_y - 6),
                fontsize=7,
                fontweight="medium",
                arrowprops=dict(arrowstyle="->", color=theory_color, lw=0.8),
                ha="left",
                va="top",
                color=theory_color,
            )

    # Set x-ticks to indices, labeled with compact step numbers
    def format_step(s: int) -> str:
        if s >= 1000:
            return f"{s // 1000}k"
        return str(s)

    ax.set_xticks(indices)
    ax.set_xticklabels([format_step(s) for s in steps], rotation=45, ha="right")

    ax.set_xlabel("Training Step", labelpad=xlabel_labelpad)
    ax.set_ylabel(f"Dims for {threshold_label}")
    # Legend removed - using direct labels on plot
    ax.grid(True, alpha=0.3)

    # Apply axis limits and scale if specified
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xscale is not None:
        ax.set_xscale(xscale)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add inset if enabled
    if inset_enabled:
        # Create inset axes
        ax_inset = ax.inset_axes(inset_bounds)

        # Re-plot the stacked area in the inset
        ax_inset.stackplot(
            indices,
            dims_per_factor.T,
            colors=factor_colors,
            alpha=0.4,
        )

        # Re-plot combined line in inset (color-coded if line_cmap provided)
        if combined_dims is not None and show_joint:
            if line_cmap is not None and len(indices) > 1:
                cmap_obj = plt.cm.get_cmap(line_cmap)
                n_points = len(indices)
                inset_lw = combined_linewidth * 0.8
                inset_ms = combined_marker_size * 0.8

                # White background line for visibility (subtle)
                ax_inset.plot(indices, combined_dims, 'w-', linewidth=inset_lw + 0.6, zorder=4)

                # Create line segments with colors
                points = np.column_stack([indices, combined_dims])
                segments = np.array([[points[i], points[i + 1]] for i in range(n_points - 1)])
                colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points - 1)]

                lc = LineCollection(segments, colors=colors, linewidth=inset_lw, zorder=5)
                ax_inset.add_collection(lc)

                # Colored markers
                marker_colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points)]
                ax_inset.scatter(indices, combined_dims, c=marker_colors, s=inset_ms**2,
                                 marker='s', zorder=6, edgecolors='white', linewidths=0.2)
            else:
                ax_inset.plot(
                    indices,
                    combined_dims,
                    combined_linestyle,
                    linewidth=combined_linewidth * 0.8,
                    marker="s",
                    markersize=combined_marker_size * 0.8,
                )

        # Re-plot union line in inset (color-coded dashed)
        if union_dims is not None and show_union:
            inset_union_lw = combined_linewidth * 0.6
            inset_union_ms = combined_marker_size * 0.5

            if line_cmap is not None and len(indices) > 1:
                cmap_obj = plt.cm.get_cmap(line_cmap)
                n_points = len(indices)

                # White background line for visibility (dashed)
                ax_inset.plot(indices, union_dims, 'w--', linewidth=inset_union_lw + 0.6, zorder=7)

                # Create dashed line segments with colors
                points = np.column_stack([indices, union_dims])
                segments = np.array([[points[i], points[i + 1]] for i in range(n_points - 1)])
                colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points - 1)]

                lc = LineCollection(segments, colors=colors, linewidth=inset_union_lw,
                                   linestyles='dashed', zorder=8)
                ax_inset.add_collection(lc)

                # Colored markers
                marker_colors = [cmap_obj(0.9 - 0.8 * i / (n_points - 1)) for i in range(n_points)]
                ax_inset.scatter(indices, union_dims, c=marker_colors, s=inset_union_ms**2,
                                marker='o', zorder=9, edgecolors='white', linewidths=0.2)
            else:
                ax_inset.plot(indices, union_dims, "k--", linewidth=inset_union_lw,
                             marker="o", markersize=inset_union_ms, zorder=7)

        # Re-plot expected total in inset
        if expected_total is not None:
            ax_inset.axhline(
                y=expected_total,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

        # Re-plot theory lines in inset (no annotations, too small)
        if theory_lines is not None:
            theory_markers = {"Mean": "^", "Per-Factor": "v"}
            for name, values in theory_lines.items():
                marker = theory_markers.get(name, theory_marker)
                ax_inset.plot(
                    indices,
                    values,
                    linestyle=theory_linestyle,
                    color=theory_color,
                    linewidth=theory_linewidth * 0.8,
                    marker=marker,
                    markersize=theory_markersize * 0.8,
                    alpha=0.8,
                    zorder=3,
                )

        # Set inset limits (default to last ~25% of data if not specified)
        if inset_xlim is not None:
            ax_inset.set_xlim(inset_xlim)
        else:
            # Default: zoom into last quarter of steps
            n_indices = len(indices)
            default_start = max(0, int(n_indices * 0.75) - 1)
            ax_inset.set_xlim(default_start, n_indices - 1)

        if inset_ylim is not None:
            ax_inset.set_ylim(inset_ylim)

        # Set x-ticks to show step labels (not raw indices)
        inset_xlim_actual = ax_inset.get_xlim()
        visible_indices = [i for i in indices if inset_xlim_actual[0] <= i <= inset_xlim_actual[1]]
        ax_inset.set_xticks(visible_indices)
        ax_inset.set_xticklabels([format_step(steps[i]) for i in visible_indices], rotation=0, ha="center")

        # Style the inset
        ax_inset.tick_params(labelsize=6)
        ax_inset.grid(True, alpha=0.3)
        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)

        # Add indicator box and connector lines
        ax.indicate_inset_zoom(ax_inset, edgecolor="gray", linewidth=0.8)

    # Title (skip if empty string)
    if title is None:
        title = f"Factor Dims for {threshold_label} Over Training"
    if title:
        ax.set_title(title, fontsize=9, fontweight="medium", loc="left")

    if created_fig and fig is not None:
        fig.tight_layout()
        return fig
    return None


def plot_stacked_dims_bars(
    steps: list[int],
    dims_per_factor: np.ndarray,  # [n_steps, n_factors]
    combined_dims: np.ndarray | None = None,  # [n_steps] - "Joint" (natural generation)
    union_dims: np.ndarray | None = None,  # [n_steps] - "Union" (concatenated vary-one)
    expected_total: int | None = None,
    threshold_label: str = "95%",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.75, 4.0),
    cmap: str = "tab10",
    bar_width: float = 0.7,
    joint_linestyle: str = "-",
    joint_linewidth: float = 1.5,
    union_linestyle: str = "--",
    union_linewidth: float = 1.5,
    show_joint: bool = True,  # Show "Joint" line (natural generation)
    show_union: bool = True,  # Show "Union" line (concatenated vary-one)
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    inset_enabled: bool = False,
    inset_bounds: tuple[float, float, float, float] = (0.55, 0.45, 0.42, 0.5),
    inset_xlim: tuple[float, float] | None = None,
    inset_ylim: tuple[float, float] | None = None,
    title: str | None = None,
    legend_loc: str = "upper right",
    xlabel_labelpad: float = 4.0,
) -> Figure | None:
    """Plot stacked bar chart of factor dimensions at selected training steps.

    Shows how many dimensions each factor requires at a given variance threshold,
    as stacked bars at discrete steps.

    Key features:
    - Each bar is stacked by factor (colored by factor)
    - "Joint" and "Union" values shown as horizontal lines over each bar
    - Lines only span the width of their respective bar

    Args:
        steps: Training step numbers
        dims_per_factor: Array [n_steps, n_factors] with dims per factor
        combined_dims: Optional array [n_steps] of "Joint" dims (natural generation)
        union_dims: Optional array [n_steps] of "Union" dims (concatenated vary-one)
        expected_total: Optional expected total for reference line
        threshold_label: Label for variance threshold (e.g., '95%')
        ax: Pre-created axes (optional)
        figsize: Figure size if creating new figure
        cmap: Colormap for factor colors
        bar_width: Width of bars relative to spacing (0-1)
        joint_linestyle: Linestyle for "Joint" horizontal lines
        joint_linewidth: Linewidth for "Joint" horizontal lines
        union_linestyle: Linestyle for "Union" horizontal lines
        union_linewidth: Linewidth for "Union" horizontal lines
        show_joint: Whether to show "Joint" line
        show_union: Whether to show "Union" line
        xlim: X-axis limits (min, max) or None for auto
        ylim: Y-axis limits (min, max) or None for auto
        inset_enabled: Whether to show an inset with zoomed region
        inset_bounds: Inset position/size as (x, y, width, height) in axes coords
        inset_xlim: X-axis limits for inset (step indices), auto if None
        inset_ylim: Y-axis limits for inset, auto if None
        title: Optional title override
        legend_loc: Location for legend (matplotlib location string)
        xlabel_labelpad: Label padding for x-axis

    Returns:
        Figure if ax was None, else None
    """
    apply_icml_style()

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    num_factors = dims_per_factor.shape[1]
    colormap = plt.cm.get_cmap(cmap)
    # Sample colormap at evenly-spaced positions for categorical data
    factor_colors = [colormap(i / (num_factors - 1) if num_factors > 1 else 0.5) for i in range(num_factors)]

    # Use indices for x-axis (equal spacing)
    indices = np.arange(len(steps))

    # Stacked bar plot
    bottom = np.zeros(len(steps))
    for factor_idx in range(num_factors):
        ax.bar(
            indices,
            dims_per_factor[:, factor_idx],
            bar_width,
            bottom=bottom,
            color=factor_colors[factor_idx],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += dims_per_factor[:, factor_idx]

    # Overlay factor labels on first bar (at left edge)
    cumulative = 0.0
    for factor_idx in range(num_factors):
        # Calculate vertical center of this factor's bar segment
        band_center = cumulative + dims_per_factor[0, factor_idx] / 2
        cumulative += dims_per_factor[0, factor_idx]

        # Place label inside first bar
        ax.text(
            0,  # First bar position
            band_center,
            f"$F_{factor_idx}$",
            fontsize=7,
            fontweight="medium",
            va="center",
            ha="center",
            color="black",
        )

    # "Joint" horizontal lines (solid, only span bar width)
    if combined_dims is not None and show_joint:
        for i, (idx, joint_val) in enumerate(zip(indices, combined_dims)):
            # Line spans only over the bar width
            x_left = idx - bar_width / 2
            x_right = idx + bar_width / 2
            ax.hlines(
                joint_val,
                x_left,
                x_right,
                colors=["black"],
                linestyles=joint_linestyle,
                linewidths=joint_linewidth,
                zorder=5,
                label="Joint" if i == 0 else None,  # Only add legend entry once
            )

    # "Union" horizontal lines (dashed, only span bar width)
    if union_dims is not None and show_union:
        for i, (idx, union_val) in enumerate(zip(indices, union_dims)):
            # Line spans only over the bar width
            x_left = idx - bar_width / 2
            x_right = idx + bar_width / 2
            ax.hlines(
                union_val,
                x_left,
                x_right,
                colors=["#666666"],  # Dark gray to differentiate from Joint
                linestyles=union_linestyle,
                linewidths=union_linewidth,
                zorder=6,
                label="Union" if i == 0 else None,  # Only add legend entry once
            )

    # Expected total reference line
    if expected_total is not None:
        ax.axhline(
            y=expected_total,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Expected ({expected_total})",
        )

    # Set x-ticks to indices, labeled with compact step numbers
    def format_step(s: int) -> str:
        if s >= 1000:
            return f"{s // 1000}k"
        return str(s)

    ax.set_xticks(indices)
    ax.set_xticklabels([format_step(s) for s in steps], rotation=45, ha="right")

    ax.set_xlabel("Training Step", labelpad=xlabel_labelpad)
    ax.set_ylabel(f"Dims for {threshold_label}")
    ax.grid(True, alpha=0.3, axis="y")  # Only y-axis grid for bar charts

    # Apply axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-0.5, len(indices) - 0.5)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend if there are labeled handles
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=legend_loc, fontsize=7, framealpha=0.9)

    # Add inset if enabled
    if inset_enabled and len(indices) > 2:
        ax_inset = ax.inset_axes(inset_bounds)

        # Re-plot the stacked bars in the inset
        bottom_inset = np.zeros(len(steps))
        for factor_idx in range(num_factors):
            ax_inset.bar(
                indices,
                dims_per_factor[:, factor_idx],
                bar_width,
                bottom=bottom_inset,
                color=factor_colors[factor_idx],
                alpha=0.7,
                edgecolor="white",
                linewidth=0.3,
            )
            bottom_inset += dims_per_factor[:, factor_idx]

        # Re-plot "Joint" lines in inset (no legend needed in inset)
        if combined_dims is not None and show_joint:
            for i, (idx, joint_val) in enumerate(zip(indices, combined_dims)):
                x_left = idx - bar_width / 2
                x_right = idx + bar_width / 2
                ax_inset.hlines(
                    joint_val,
                    x_left,
                    x_right,
                    colors=["black"],
                    linestyles=joint_linestyle,
                    linewidths=joint_linewidth * 0.8,
                    zorder=5,
                )

        # Re-plot "Union" lines in inset (no legend needed in inset)
        if union_dims is not None and show_union:
            for i, (idx, union_val) in enumerate(zip(indices, union_dims)):
                x_left = idx - bar_width / 2
                x_right = idx + bar_width / 2
                ax_inset.hlines(
                    union_val,
                    x_left,
                    x_right,
                    colors=["#666666"],
                    linestyles=union_linestyle,
                    linewidths=union_linewidth * 0.8,
                    zorder=6,
                )

        # Set inset limits
        if inset_xlim is not None:
            ax_inset.set_xlim(inset_xlim)
        else:
            # Default: zoom into last ~third of steps
            n_indices = len(indices)
            default_start = max(0, int(n_indices * 0.6) - 1)
            ax_inset.set_xlim(default_start - 0.5, n_indices - 0.5)

        if inset_ylim is not None:
            ax_inset.set_ylim(inset_ylim)

        # Set x-ticks for inset
        inset_xlim_actual = ax_inset.get_xlim()
        visible_indices = [i for i in indices if inset_xlim_actual[0] <= i <= inset_xlim_actual[1]]
        ax_inset.set_xticks(visible_indices)
        ax_inset.set_xticklabels([format_step(steps[i]) for i in visible_indices], rotation=0, ha="center")

        # Style the inset
        ax_inset.tick_params(labelsize=6)
        ax_inset.grid(True, alpha=0.3, axis="y")
        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)

        # Add indicator box and connector lines
        ax.indicate_inset_zoom(ax_inset, edgecolor="gray", linewidth=0.8)

    # Title (skip if empty string)
    if title is None:
        title = f"Factor Dims for {threshold_label} Over Training"
    if title:
        ax.set_title(title, fontsize=9, fontweight="medium", loc="left")

    if created_fig and fig is not None:
        fig.tight_layout()
        return fig
    return None


def create_composite_figure(
    orthogonality_data: OrthogonalityData,
    baseline_mean: np.ndarray | None = None,
    baseline_lower: np.ndarray | None = None,
    baseline_upper: np.ndarray | None = None,
    top_left_images: dict[str, str] | None = None,
    belief_regression_data: ProjectedBeliefRegressionData | None = None,
    expected_total: int | None = None,
    # Layout selection
    layout: str = "full",  # "full", "regression", or "graphs_only"
    # Figure sizes for each layout
    figsize: tuple[float, float] = (9.0, 4.5),  # "full" layout
    figsize_regression: tuple[float, float] = (3.25, 5.0),  # "regression" layout
    figsize_graphs_only: tuple[float, float] = (6.75, 2.5),  # "graphs_only" layout
    # Width ratios for each layout
    width_ratios: list[float] | None = None,  # "full" (3 columns)
    width_ratios_regression: list[float] | None = None,  # "regression" (2 columns)
    width_ratios_graphs_only: list[float] | None = None,  # "graphs_only" (2 columns)
    panel_aspects: dict[str, float | None] | None = None,
    wspace: float = 0.05,
    hspace: float = 0.05,
    variance_threshold: float = 0.95,
    ci_percentile: int = 90,
    orthogonality_cmap: str = "viridis",
    belief_regression_max_samples: int = 10000,
    belief_regression_marker_size: float = 0.5,
    belief_regression_marker_opacity: float = 0.7,
    belief_regression_pred_marker_size: float = 0.75,
    belief_regression_pred_marker_opacity: float = 0.35,
    belief_regression_scatter_pad_frac: float = 0.1,
    belief_regression_title_pad: float = 2.0,
    belief_regression_preserve_aspect: bool = True,
    belief_regression_cmap: str = "viridis",
    belief_regression_cmap_start: float = 0.2,
    belief_regression_cmap_mid: float = 0.5,
    belief_regression_cmap_end: float = 0.8,
    combined_linestyle: str = "k-",
    combined_linewidth: float = 2.5,
    combined_marker_size: float = 4.0,
    dims_cmap: str = "tab10",
    show_joint: bool = True,  # Show "Joint" line (natural generation)
    show_union: bool = True,  # Show "Union" line (concatenated vary-one)
    dims_xlim: tuple[float, float] | None = None,
    dims_ylim: tuple[float, float] | None = None,
    dims_xscale: str | None = None,
    dims_inset_enabled: bool = False,
    dims_inset_bounds: tuple[float, float, float, float] = (0.55, 0.45, 0.42, 0.5),
    dims_inset_xlim: tuple[float, float] | None = None,
    dims_inset_ylim: tuple[float, float] | None = None,
    orthogonality_average_linewidth: float = 2.5,
    orthogonality_marker_size: float = 4.0,
    orthogonality_show_baseline_mean: bool = True,
    # Theory line parameters
    theory_lines: dict[str, np.ndarray] | None = None,
    theory_linestyle: str = ":",
    theory_color: str = "0.5",
    theory_linewidth: float = 1.0,
    theory_marker: str = "s",
    theory_markersize: float = 2.0,
    dims_steps: list[int] | None = None,
    orthogonality_steps: list[int] | None = None,
    dims_legend_loc: str = "upper right",
    orthogonality_legend_loc: str = "upper right",
    # Bar mode parameters
    dims_bar_mode_enabled: bool = False,
    dims_bar_mode_steps: list[int] | None = None,
    dims_bar_mode_bar_width: float = 0.7,
    dims_bar_mode_inset_enabled: bool = False,
    dims_bar_mode_inset_bounds: tuple[float, float, float, float] = (0.55, 0.45, 0.42, 0.5),
    dims_bar_mode_inset_xlim: tuple[float, float] | None = None,
    dims_bar_mode_inset_ylim: tuple[float, float] | None = None,
    dims_bar_mode_all_linestyle: str = "--",
    dims_bar_mode_all_linewidth: float = 1.5,
    # Layout spacing parameters
    left_grid_wspace: float = 0.02,
    left_grid_hspace: float = 0.02,
    belief_wspace: float = 0.05,
    belief_hspace: float = 0.03,
    belief_factor_label_x: float = 1.12,
    belief_factor_label_y: float = 0.5,
    belief_factor_label_ha: str = "center",
    belief_anchor_theory: str | None = "N",
    belief_anchor_activations: str | None = "N",
    # Panel label positions [x, y]
    label_a_pos: tuple[float, float] = (-0.08, 1.10),
    label_b_pos: tuple[float, float] = (-0.15, 1.12),
    label_c_pos: tuple[float, float] = (-0.15, 1.12),
    label_d_pos: tuple[float, float] = (-0.35, 1.35),
    # Axis label padding
    middle_xlabel_labelpad: float = 2.0,
    bottom_xlabel_labelpad: float = 4.0,
    # Manual positioning for "regression" layout (tighter margins)
    regression_left: float = 0.02,
    regression_right: float = 0.98,
    regression_top: float = 0.92,
    regression_bottom: float = 0.08,
    regression_wspace: float = 0.25,
    regression_hspace: float = 0.12,
) -> Figure:
    """Create composite figure with configurable layout.

    Layout modes:
        "full" (3 columns):
            - Left: 2x2 grid of factor visualization images (panel a)
            - Middle top: stacked dims over training (panel b)
            - Middle bottom: orthogonality plot (panel c)
            - Right: belief regression grid (panel d)

        "regression" (2 columns, single ICML column width):
            - Left: belief regression grid (panel a)
            - Right top: stacked dims over training (panel b)
            - Right bottom: orthogonality plot (panel c)

        "graphs_only" (2 columns side-by-side):
            - Left: stacked dims over training (panel a)
            - Right: orthogonality plot (panel b)

    Args:
        orthogonality_data: Results from compute_orthogonality_at_checkpoints
        baseline_mean: Random init baseline mean [max_k]
        baseline_lower: Random init baseline lower CI [max_k]
        baseline_upper: Random init baseline upper CI [max_k]
        top_left_images: Dict with paths to 4 images (for "full" layout)
        layout: Layout mode - "full", "regression", or "graphs_only"
        figsize: Figure size for "full" layout
        figsize_regression: Figure size for "regression" layout
        figsize_graphs_only: Figure size for "graphs_only" layout
        width_ratios: Column widths for "full" layout [left, middle, right]
        width_ratios_regression: Column widths for "regression" layout
        width_ratios_graphs_only: Column widths for "graphs_only" layout
        variance_threshold: Threshold for stacked dims plot
        ci_percentile: CI percentile for baseline label

    Returns:
        Figure object
    """
    apply_icml_style()

    panel_aspects = panel_aspects or {}
    top_left_images = top_left_images or {}

    # Choose layout based on layout parameter
    if layout == "full":
        # 3-column layout with 2x2 grid on left - use constrained_layout
        actual_figsize = figsize
        actual_width_ratios = width_ratios or [1.0, 1.0, 0.6]
        n_cols = 3

        fig = plt.figure(figsize=actual_figsize, constrained_layout=True)
        fig.get_layout_engine().set(wspace=wspace, hspace=hspace)
        gs = fig.add_gridspec(
            2,
            n_cols,
            height_ratios=[1.0, 1.0],
            width_ratios=actual_width_ratios,
        )
    elif layout == "regression":
        # 2-column layout: belief regression on left, stacked plots on right
        # Use manual positioning for tighter control over whitespace
        actual_figsize = figsize_regression
        actual_width_ratios = width_ratios_regression or [1.2, 1.0]
        n_cols = 2

        fig = plt.figure(figsize=actual_figsize)
        gs = fig.add_gridspec(
            2,
            n_cols,
            height_ratios=[1.0, 1.0],
            width_ratios=actual_width_ratios,
            left=regression_left,
            right=regression_right,
            top=regression_top,
            bottom=regression_bottom,
            wspace=regression_wspace,
            hspace=regression_hspace,
        )
    elif layout == "graphs_only":
        # 2-column layout: dims on left, orthogonality on right (side-by-side)
        actual_figsize = figsize_graphs_only
        actual_width_ratios = width_ratios_graphs_only or [1.0, 1.0]
        n_cols = 2

        fig = plt.figure(figsize=actual_figsize, constrained_layout=True)
        fig.get_layout_engine().set(wspace=wspace, hspace=hspace)
        gs = fig.add_gridspec(
            1,  # Single row for side-by-side
            n_cols,
            width_ratios=actual_width_ratios,
        )
    else:
        raise ValueError(f"Unknown layout: {layout}. Must be 'full', 'regression', or 'graphs_only'")

    # Left: 2x2 grid of source images spanning rows 0-1 (Punnett square style)
    # Only create when layout="full"
    if layout == "full":
        # Column headers and row labels for Punnett square layout
        col_headers = ["Factored", "Joint"]
        row_labels = ["Uncentered", "Centered"]

        image_specs = [
            (0, 0, "factored_2d"),
            (0, 1, "joint_3d"),
            (1, 0, "factored_2d_centered"),
            (1, 1, "joint_3d_centered"),
        ]

        # Pre-pass: Load, crop whitespace, and compute aspect ratios
        # Cropping ensures cells are sized based on actual content, not whitespace
        aspect_matrix = np.ones((2, 2))  # Default to square if image missing
        loaded_images = {}  # Cache cropped images for display
        for row, col, img_key in image_specs:
            img_path = top_left_images.get(img_key)
            if img_path and Path(img_path).exists():
                img = plt.imread(img_path)
                img = _auto_crop_whitespace(img)  # Crop whitespace
                loaded_images[img_key] = img
                h, w = img.shape[:2]
                aspect_matrix[row, col] = w / h

        # Column widths = max aspect in each column (so widest image fits)
        col_max_aspects = np.max(aspect_matrix, axis=0)
        grid_width_ratios = list(col_max_aspects)

        gs_left = gs[0:2, 0].subgridspec(2, 2, width_ratios=grid_width_ratios,
                                          wspace=left_grid_wspace, hspace=left_grid_hspace)

        # Store axes for positioning labels
        axes_grid = [[None, None], [None, None]]
        first_ax = None  # Track first axis for panel label
        for row, col, img_key in image_specs:
            ax = fig.add_subplot(gs_left[row, col])
            axes_grid[row][col] = ax
            if first_ax is None:
                first_ax = ax
            if img_key in loaded_images:
                # Use cached cropped image with equal aspect to preserve proportions
                ax.imshow(loaded_images[img_key], aspect="equal")
            ax.axis("off")

        # Add column headers centered above each column using the top row axes
        # Position them at y=1.02 in axes coordinates (just above the image)
        for col, header in enumerate(col_headers):
            ax = axes_grid[0][col]
            ax.text(0.5, 1.02, header, transform=ax.transAxes,
                    fontsize=8, fontweight="medium", va="bottom", ha="center")

        # Add row labels to the left of each row
        for row, label in enumerate(row_labels):
            ax = axes_grid[row][0]
            ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                    fontsize=8, va="center", ha="right", rotation=90)

        # Panel label (a) for 2x2 grid
        if first_ax is not None:
            first_ax.text(label_a_pos[0], label_a_pos[1], "(a)", transform=first_ax.transAxes,
                          fontsize=10, fontweight="bold", va="top", ha="left")

    # Dims and orthogonality plots
    # Position depends on layout:
    # - "full": middle column (col 1), stacked vertically (rows 0,1)
    # - "regression": right column (col 1), stacked vertically (rows 0,1)
    # - "graphs_only": side-by-side in single row (cols 0,1)
    if layout == "graphs_only":
        ax_tr = fig.add_subplot(gs[0, 0])  # Dims on left
    else:
        ax_tr = fig.add_subplot(gs[0, 1])  # Dims in middle/right column, top row

    # Compute dims per factor from CEV curves (for all steps first)
    n_steps, n_factors, _ = orthogonality_data.cev_per_factor.shape
    all_dims_per_factor = np.zeros((n_steps, n_factors), dtype=int)
    for step_idx in range(n_steps):
        for factor_idx in range(n_factors):
            cev = orthogonality_data.cev_per_factor[step_idx, factor_idx]
            all_dims_per_factor[step_idx, factor_idx] = compute_dims_at_threshold(
                cev, variance_threshold
            )

    # Compute combined dims from cev_combined (for all steps first)
    all_combined_dims = np.zeros(n_steps, dtype=int)
    for step_idx in range(n_steps):
        cev_combined = orthogonality_data.cev_combined[step_idx]
        all_combined_dims[step_idx] = compute_dims_at_threshold(cev_combined, variance_threshold)

    # Compute union dims from cev_union (if available)
    all_union_dims: np.ndarray | None = None
    if orthogonality_data.cev_union is not None:
        all_union_dims = np.zeros(n_steps, dtype=int)
        for step_idx in range(n_steps):
            cev_union = orthogonality_data.cev_union[step_idx]
            all_union_dims[step_idx] = compute_dims_at_threshold(cev_union, variance_threshold)
    else:
        print("  Note: cev_union not available in cached data. Run with cache.force_recompute=true to compute.")

    threshold_label = f"{int(variance_threshold * 100)}%"

    # Determine which steps and data to use based on bar mode
    if dims_bar_mode_enabled:
        # Bar mode: use dims_bar_mode_steps (or fallback to dims_steps)
        bar_steps_to_use = dims_bar_mode_steps or dims_steps or orthogonality_data.steps
        bar_step_indices = [i for i, s in enumerate(orthogonality_data.steps) if s in bar_steps_to_use]
        bar_steps_filtered = [orthogonality_data.steps[i] for i in bar_step_indices]
        bar_dims_per_factor = all_dims_per_factor[bar_step_indices]
        bar_combined_dims = all_combined_dims[bar_step_indices]
        bar_union_dims = all_union_dims[bar_step_indices] if all_union_dims is not None else None

        plot_stacked_dims_bars(
            steps=bar_steps_filtered,
            dims_per_factor=bar_dims_per_factor,
            combined_dims=bar_combined_dims,
            union_dims=bar_union_dims,
            expected_total=expected_total,
            threshold_label=threshold_label,
            ax=ax_tr,
            cmap=dims_cmap,
            bar_width=dims_bar_mode_bar_width,
            joint_linestyle=dims_bar_mode_all_linestyle,
            joint_linewidth=dims_bar_mode_all_linewidth,
            union_linestyle="--",  # Dashed for Union
            union_linewidth=dims_bar_mode_all_linewidth,
            show_joint=show_joint,
            show_union=show_union,
            xlim=dims_xlim,
            ylim=dims_ylim,
            inset_enabled=dims_bar_mode_inset_enabled,
            inset_bounds=dims_bar_mode_inset_bounds,
            inset_xlim=dims_bar_mode_inset_xlim,
            inset_ylim=dims_bar_mode_inset_ylim,
            title="",  # No title - will be explained in caption
            legend_loc=dims_legend_loc,
            xlabel_labelpad=middle_xlabel_labelpad,
        )
    else:
        # Line mode: filter to dims_steps if specified
        if dims_steps is not None:
            # Find indices of requested steps
            dims_step_indices = [i for i, s in enumerate(orthogonality_data.steps) if s in dims_steps]
            dims_steps_filtered = [orthogonality_data.steps[i] for i in dims_step_indices]
            dims_per_factor = all_dims_per_factor[dims_step_indices]
            combined_dims = all_combined_dims[dims_step_indices]
            union_dims = all_union_dims[dims_step_indices] if all_union_dims is not None else None
        else:
            dims_steps_filtered = orthogonality_data.steps
            dims_per_factor = all_dims_per_factor
            combined_dims = all_combined_dims
            union_dims = all_union_dims

        plot_stacked_dims_over_training(
            steps=dims_steps_filtered,
            dims_per_factor=dims_per_factor,
            combined_dims=combined_dims,
            union_dims=union_dims,
            expected_total=expected_total,
            threshold_label=threshold_label,
            ax=ax_tr,
            cmap=dims_cmap,
            combined_linestyle=combined_linestyle,
            combined_linewidth=combined_linewidth,
            combined_marker_size=combined_marker_size,
            line_cmap=orthogonality_cmap,  # Color-code lines by training step
            show_joint=show_joint,
            show_union=show_union,
            xlim=dims_xlim,
            ylim=dims_ylim,
            xscale=dims_xscale,
            inset_enabled=dims_inset_enabled,
            inset_bounds=dims_inset_bounds,
            inset_xlim=dims_inset_xlim,
            inset_ylim=dims_inset_ylim,
            title="",  # No title - will be explained in caption
            legend_loc=dims_legend_loc,
            xlabel_labelpad=middle_xlabel_labelpad,
            # Theory lines
            theory_lines=theory_lines,
            theory_linestyle=theory_linestyle,
            theory_color=theory_color,
            theory_linewidth=theory_linewidth,
            theory_marker=theory_marker,
            theory_markersize=theory_markersize,
        )

    # Panel label for stacked dims
    # In graphs_only layout: (a), otherwise (b)
    dims_label = "(a)" if layout == "graphs_only" else "(b)"
    dims_label_pos = label_a_pos if layout == "graphs_only" else label_b_pos
    ax_tr.text(dims_label_pos[0], dims_label_pos[1], dims_label, transform=ax_tr.transAxes,
               fontsize=10, fontweight="bold", va="top", ha="left")

    # Orthogonality plot
    # Position depends on layout:
    # - "full"/"regression": bottom of middle/right column (row 1, col 1)
    # - "graphs_only": right column in single row (row 0, col 1)
    if layout == "graphs_only":
        ax_br = fig.add_subplot(gs[0, 1])  # Orthogonality on right
    else:
        ax_br = fig.add_subplot(gs[1, 1])  # Orthogonality in middle/right column, bottom row
    plot_overlap_over_training(
        orthogonality_data,
        baseline_mean=baseline_mean,
        baseline_lower=baseline_lower,
        baseline_upper=baseline_upper,
        ax=ax_br,
        cmap=orthogonality_cmap,
        average_linewidth=orthogonality_average_linewidth,
        marker_size=orthogonality_marker_size,
        ci_percentile=ci_percentile,
        title="",  # No title - will be explained in caption
        step_filter=orthogonality_steps,
        legend_loc=orthogonality_legend_loc,
        xlabel_labelpad=bottom_xlabel_labelpad,
        show_baseline_mean=orthogonality_show_baseline_mean,
    )

    # Panel label for orthogonality
    # In graphs_only layout: (b), otherwise (c)
    orth_label = "(b)" if layout == "graphs_only" else "(c)"
    orth_label_pos = label_b_pos if layout == "graphs_only" else label_c_pos
    ax_br.text(orth_label_pos[0], orth_label_pos[1], orth_label, transform=ax_br.transAxes,
               fontsize=10, fontweight="bold", va="top", ha="left")

    # Belief regression grid (vertical orientation, spans both rows)
    # Only rendered for "full" and "regression" layouts, not "graphs_only"
    # Column position: column 0 for "regression", column 2 for "full"
    if belief_regression_data is not None and layout != "graphs_only":
        num_factors = len(belief_regression_data.factor_dims)
        belief_col = 0 if layout == "regression" else 2

        if layout == "regression":
            # Use per-row subgridspecs like fig_2_vertical for tighter layout
            # First create outer subgridspec for all factor rows
            gs_belief_outer = gs[0:2, belief_col].subgridspec(num_factors, 1, hspace=belief_hspace)

            # Import needed functions for plotting
            from sklearn.decomposition import PCA
            from belief_grid_plotting import _compute_rgb_colors

            # Subsample data if needed
            y_true = belief_regression_data.y_true
            y_pred = belief_regression_data.y_pred
            n_samples = len(y_true)
            max_samples = belief_regression_max_samples
            if n_samples > max_samples:
                indices = np.random.choice(n_samples, max_samples, replace=False)
                y_true = y_true[indices]
                y_pred = y_pred[indices]
                n_samples = max_samples

            factor_dims = belief_regression_data.factor_dims
            offset = 0
            first_ax_theory = None
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

                # Compute RGB colors (use cmap for F0-F2, teal-coral-gold for F3+)
                use_cmap = factor_idx < 3 and factor_dim >= 3
                if use_cmap:
                    colors = _compute_rgb_colors(
                        true_factor, factor_dim, use_cmap_barycentric=True,
                        cmap=belief_regression_cmap,
                        cmap_start=belief_regression_cmap_start,
                        cmap_mid=belief_regression_cmap_mid,
                        cmap_end=belief_regression_cmap_end
                    )
                else:
                    colors = _compute_rgb_colors(
                        true_2d, 2, use_cmap_barycentric=False,
                        cmap=belief_regression_cmap,
                        cmap_start=belief_regression_cmap_start,
                        cmap_mid=belief_regression_cmap_mid,
                        cmap_end=belief_regression_cmap_end
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
                pad_frac = belief_regression_scatter_pad_frac
                half_span = max_span / 2 * (1 + pad_frac)
                shared_x_range = (x_center - half_span, x_center + half_span)
                shared_y_range = (y_center - half_span, y_center + half_span)

                # Create per-row subgridspec with wspace=-0.4 (like fig_2_vertical)
                gs_belief_row = gs_belief_outer[factor_idx, 0].subgridspec(1, 2, wspace=-0.4)

                # Plot Theory (left)
                ax_theory = fig.add_subplot(gs_belief_row[0, 0])
                ax_theory.scatter(
                    true_2d[:, 0], true_2d[:, 1], c=colors,
                    s=belief_regression_marker_size, alpha=belief_regression_marker_opacity,
                    edgecolors="none", rasterized=True,
                )
                ax_theory.set_xlim(shared_x_range)
                ax_theory.set_ylim(shared_y_range)
                ax_theory.set_xticks([])
                ax_theory.set_yticks([])
                ax_theory.set_aspect("equal", adjustable="box")
                for spine in ax_theory.spines.values():
                    spine.set_visible(False)

                # Add F label on left using set_ylabel (like fig_2_vertical)
                ax_theory.set_ylabel(f"$F_{factor_idx}$", fontsize=9, rotation=0, ha="right", va="center", labelpad=8)

                if first_ax_theory is None:
                    first_ax_theory = ax_theory

                # Plot Activations (right)
                ax_act = fig.add_subplot(gs_belief_row[0, 1])
                ax_act.scatter(
                    pred_2d[:, 0], pred_2d[:, 1], c=colors,
                    s=belief_regression_pred_marker_size, alpha=belief_regression_pred_marker_opacity,
                    edgecolors="none", rasterized=True,
                )
                ax_act.set_xlim(shared_x_range)
                ax_act.set_ylim(shared_y_range)
                ax_act.set_xticks([])
                ax_act.set_yticks([])
                ax_act.set_aspect("equal", adjustable="box")
                for spine in ax_act.spines.values():
                    spine.set_visible(False)

            # Add column headers using fig.text (like fig_2_vertical)
            # Position based on belief column width ratio
            belief_width_frac = actual_width_ratios[0] / sum(actual_width_ratios)
            theory_x = regression_left + belief_width_frac * (regression_right - regression_left) * 0.28
            act_x = regression_left + belief_width_frac * (regression_right - regression_left) * 0.65
            header_y = regression_top + 0.02
            fig.text(theory_x, header_y, "Theory", ha="center", va="bottom", fontsize=8, fontweight="medium")
            fig.text(act_x, header_y, "Activations", ha="center", va="bottom", fontsize=8, fontweight="medium")

            # Panel label (a) for belief regression
            if first_ax_theory is not None:
                fig.text(0.02, regression_top + 0.06, "(a)", fontsize=10, fontweight="bold", va="top", ha="left")

        else:
            # Original approach for layout="full" (3-column layout)
            gs_belief = gs[0:2, belief_col].subgridspec(num_factors, 2, hspace=belief_hspace, wspace=belief_wspace)
            ax_belief_grid = np.array(
                [[fig.add_subplot(gs_belief[i, j]) for j in range(2)] for i in range(num_factors)]
            )

            # Build title suffix with k values
            k_text = ", ".join([str(k) for k in belief_regression_data.k_values])
            title_suffix = f"k=[{k_text}]"

            _plot_belief_regression_grid(
                y_true=belief_regression_data.y_true,
                y_pred=belief_regression_data.y_pred,
                num_factors=num_factors,
                overall_rmse=belief_regression_data.overall_rmse,
                factor_rmse_scores=belief_regression_data.factor_rmse_scores,
                factor_dims=belief_regression_data.factor_dims,
                step=belief_regression_data.step,
                layer_name=belief_regression_data.layer_hook,
                ax_grid=ax_belief_grid,
                max_samples=belief_regression_max_samples,
                marker_size=belief_regression_marker_size,
                marker_opacity=belief_regression_marker_opacity,
                pred_marker_size=belief_regression_pred_marker_size,
                pred_marker_opacity=belief_regression_pred_marker_opacity,
                scatter_pad_frac=belief_regression_scatter_pad_frac,
                title_pad=belief_regression_title_pad,
                preserve_aspect=belief_regression_preserve_aspect,
                title_suffix=title_suffix,
                orientation="vertical",
                factor_label_x=belief_factor_label_x,
                factor_label_y=belief_factor_label_y,
                factor_label_ha=belief_factor_label_ha,
                anchor_theory=belief_anchor_theory,
                anchor_activations=belief_anchor_activations,
                cmap=belief_regression_cmap,
                cmap_start=belief_regression_cmap_start,
                cmap_mid=belief_regression_cmap_mid,
                cmap_end=belief_regression_cmap_end,
            )

            # Panel label (d) for belief regression in 3-column layout
            ax_belief_grid[0, 0].text(label_d_pos[0], label_d_pos[1], "(d)", transform=ax_belief_grid[0, 0].transAxes,
                                       fontsize=10, fontweight="bold", va="top", ha="left")

    return fig
