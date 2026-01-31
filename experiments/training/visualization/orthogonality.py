"""Orthogonality visualizations for factor subspace analysis.

Visualizes how orthogonal (independent) different factor representations are
in the model's activation space. Based on principal angles between factor subspaces.

Key metrics:
- singular_values: cos(principal angles) between factor subspaces [0=orthogonal, 1=aligned]
- overlap: mean(sv²) - current metric emphasizing large overlaps
- mean_sv: mean(sv) - average alignment (more interpretable)
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Mapping

import numpy as np
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from omegaconf import DictConfig

from visualization._types import OrthogonalityHistory
from visualization.configs import OrthogonalityVizConfig
from visualization.styles import (
    ARCHIVAL_COLORSCALE,
    TITLE_FONT,
    PAPER_BG,
    PLOT_BG,
    DEFAULT_FONT,
    GRID_COLOR,
    AXIS_LINE_COLOR,
    TICK_FONT,
    AXIS_LABEL_FONT,
    apply_default_layout,
)


def _empty_figure(title: str, reason: str) -> go.Figure:
    """Create an empty figure with an explanation message."""
    fig = go.Figure()
    fig.add_annotation(
        text=reason,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(title=title)
    return fig


def _coerce_config(
    config: OrthogonalityVizConfig | Mapping[str, Any] | DictConfig | None,
) -> OrthogonalityVizConfig:
    """Coerce config to OrthogonalityVizConfig."""
    if config is None:
        return OrthogonalityVizConfig()
    if isinstance(config, OrthogonalityVizConfig):
        return config
    return OrthogonalityVizConfig.from_dict(config)


def _parse_factor_pair(pair_key: str) -> tuple[str | None, int, int] | None:
    """Parse factor pair key like 'L0.resid.pre/F0,F1' into (layer, i, j) tuple.

    Also handles simple 'F0,F1' format (returns None for layer).
    """
    # Try "{layer}/F{i},F{j}" format
    match = re.match(r"^(.+)/F(\d+),F(\d+)$", pair_key)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    # Try simple "F0,F1" format
    match = re.match(r"^F?(\d+),\s*F?(\d+)$", pair_key)
    if match:
        return None, int(match.group(1)), int(match.group(2))
    return None


def _get_sorted_pairs(history: OrthogonalityHistory) -> list[str]:
    """Get factor pair keys sorted by (layer, factor indices)."""
    pairs_with_indices = []
    for key in history.keys():
        parsed = _parse_factor_pair(key)
        if parsed:
            layer, i, j = parsed
            # Sort by layer name first, then by factor indices
            sort_key = (layer or "", i, j)
            pairs_with_indices.append((sort_key, key))
    pairs_with_indices.sort(key=lambda x: x[0])
    return [key for _, key in pairs_with_indices]


def _extract_pair_label(pair_key: str) -> str:
    """Extract just the factor pair label (e.g., 'F0,F1') from a key."""
    parsed = _parse_factor_pair(pair_key)
    if parsed:
        _, i, j = parsed
        return f"F{i},F{j}"
    return pair_key


# =============================================================================
# Plot 1: Principal Angle Spectrum
# =============================================================================

def plot_orthogonality_spectrum(
    history: OrthogonalityHistory,
    pair_key: str | None = None,
    *,
    config: OrthogonalityVizConfig | Mapping[str, Any] | DictConfig | None = None,
    title: str | None = None,
    title_suffix: str = "",
    step_range: tuple[int, int] | None = None,
) -> go.Figure:
    """Plot principal angle spectrum over training for factor pair(s).

    Shows cos(θ) for each principal angle, with lines colored by training step.
    Lower values = more orthogonal (better factor separation).

    Args:
        history: Orthogonality history dict mapping pair keys to entries
        pair_key: Specific factor pair to plot (e.g., "F0,F1"). If None, plots all pairs.
        config: Visualization configuration
        title: Optional custom title
        title_suffix: Optional suffix to append to title
        step_range: Optional (min_step, max_step) to filter

    Returns:
        Plotly Figure with principal angle spectrum plot
    """
    cfg = _coerce_config(config)

    if not history:
        return _empty_figure("Orthogonality Spectrum", "No orthogonality history available.")

    # Determine which pairs to plot
    if pair_key is not None:
        if pair_key not in history:
            return _empty_figure(
                f"Orthogonality Spectrum - {pair_key}",
                f"Factor pair '{pair_key}' not found in history.",
            )
        pairs_to_plot = [pair_key]
    else:
        pairs_to_plot = _get_sorted_pairs(history)

    if not pairs_to_plot:
        return _empty_figure("Orthogonality Spectrum", "No valid factor pairs in history.")

    num_pairs = len(pairs_to_plot)

    # Build title
    if title is None:
        if num_pairs == 1:
            title = f"Principal Angle Spectrum - {pairs_to_plot[0]}"
        else:
            title = "Principal Angle Spectrum - All Factor Pairs"
        if title_suffix:
            title = f"{title}<br><sup>{title_suffix}</sup>"

    # Create subplots - one row per factor pair
    subplot_titles = [f"{_extract_pair_label(pair)}" for pair in pairs_to_plot]
    # Adjust vertical spacing based on number of rows
    if num_pairs <= 1:
        v_spacing = 0.1
    else:
        max_spacing = 1.0 / (num_pairs - 1) - 0.01  # Leave small margin
        v_spacing = min(0.08, max_spacing)

    fig = make_subplots(
        rows=num_pairs,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=v_spacing,
        shared_xaxes=True,
    )

    # Track all steps for colorscale normalization
    all_steps = set()
    for pair in pairs_to_plot:
        for entry in history.get(pair, []):
            step = entry.get("step")
            if isinstance(step, int):
                if step_range is None or (step_range[0] <= step <= step_range[1]):
                    all_steps.add(step)

    if not all_steps:
        return _empty_figure(title, "No valid entries in the specified step range.")

    sorted_steps = sorted(all_steps)
    step_min, step_max = min(sorted_steps), max(sorted_steps)
    denom = (step_max - step_min) if step_max > step_min else 1

    # Get colorscale
    cs = ARCHIVAL_COLORSCALE if cfg.spectrum_colorscale == "Archival" else cfg.spectrum_colorscale

    # Plot each factor pair
    for row_idx, pair in enumerate(pairs_to_plot, start=1):
        entries = history.get(pair, [])

        # Filter and sort entries
        valid_entries = []
        for entry in entries:
            step = entry.get("step")
            sv = entry.get("singular_values")
            if not isinstance(step, int) or sv is None:
                continue
            if step_range is not None and not (step_range[0] <= step <= step_range[1]):
                continue
            sv_arr = np.asarray(sv, dtype=float).reshape(-1)
            if sv_arr.size == 0 or not np.all(np.isfinite(sv_arr)):
                continue
            valid_entries.append((step, sv_arr, entry))

        valid_entries.sort(key=lambda x: x[0])

        # Apply history window if configured
        if cfg.history_window is not None and cfg.history_window > 0:
            valid_entries = valid_entries[-cfg.history_window:]

        if not valid_entries:
            # Add empty annotation for this subplot
            fig.add_annotation(
                text="No data",
                xref=f"x{row_idx}" if row_idx > 1 else "x",
                yref=f"y{row_idx}" if row_idx > 1 else "y",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12, color="gray"),
            )
            continue

        # Compute colors for each step
        for step, sv_arr, entry in valid_entries:
            norm = (step - step_min) / denom
            color = plotly.colors.sample_colorscale(cs, [norm])[0]

            x = np.arange(1, len(sv_arr) + 1)

            # Get summary metrics for hover
            mean_sv = entry.get("mean_sv", np.mean(sv_arr))
            overlap = entry.get("overlap", np.mean(sv_arr**2))

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=sv_arr,
                    mode="lines+markers",
                    line=dict(color=color, width=cfg.line_width),
                    marker=dict(size=4, color=color),
                    opacity=cfg.line_opacity,
                    name=f"Step {step}",
                    hovertemplate=(
                        f"Step: {step}<br>"
                        "Angle idx: %{x}<br>"
                        "cos(θ): %{y:.4f}<br>"
                        f"mean(sv): {mean_sv:.4f}<br>"
                        f"overlap: {overlap:.4f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

        # Add reference line if configured
        if cfg.show_reference_line:
            max_angles = max(len(e[1]) for e in valid_entries) if valid_entries else 5
            fig.add_hline(
                y=cfg.reference_line_value,
                line_dash="dash",
                line_color="rgba(150, 150, 150, 0.5)",
                row=row_idx,
                col=1,
            )

        # Update axes for this subplot
        fig.update_yaxes(
            range=[-0.05, 1.05],
            title_text="cos(θ)" if row_idx == (num_pairs + 1) // 2 else None,
            row=row_idx,
            col=1,
        )

    # Update x-axis label on bottom subplot only
    fig.update_xaxes(title_text="Principal Angle Index", row=num_pairs, col=1)

    # Calculate total height
    total_height = cfg.spectrum_height_per_pair * num_pairs + 100

    # Apply layout
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT),
        height=total_height,
        width=cfg.spectrum_width,
        showlegend=False,
        margin=dict(l=70, r=90, t=100, b=70),
        font=DEFAULT_FONT,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
    )

    # Store metadata for step slider
    if len(sorted_steps) > 1:
        fig._orthogonality_slider_meta = {
            "steps": sorted_steps,
            "colorscale": cfg.spectrum_colorscale,
            "step_min": step_min,
            "step_max": step_max,
        }

    return fig


# =============================================================================
# Plot 2: Orthogonality Timeline Heatmap
# =============================================================================

def plot_orthogonality_heatmap(
    history: OrthogonalityHistory,
    *,
    config: OrthogonalityVizConfig | Mapping[str, Any] | DictConfig | None = None,
    title: str | None = None,
    title_suffix: str = "",
    step_range: tuple[int, int] | None = None,
) -> go.Figure:
    """Plot orthogonality timeline heatmap showing all factor pairs over training.

    X-axis: Factor pairs (F0,F1 | F0,F2 | ...)
    Y-axis: Training step
    Color: Overlap metric (blue=orthogonal, red=aligned)

    Args:
        history: Orthogonality history dict
        config: Visualization configuration
        title: Optional custom title
        title_suffix: Optional suffix
        step_range: Optional (min_step, max_step) to filter

    Returns:
        Plotly Figure with timeline heatmap
    """
    cfg = _coerce_config(config)

    if not history:
        return _empty_figure("Orthogonality Timeline", "No orthogonality history available.")

    pairs = _get_sorted_pairs(history)
    if not pairs:
        return _empty_figure("Orthogonality Timeline", "No valid factor pairs in history.")

    # Collect all steps across all pairs
    all_steps = set()
    for pair in pairs:
        for entry in history.get(pair, []):
            step = entry.get("step")
            if isinstance(step, int):
                if step_range is None or (step_range[0] <= step <= step_range[1]):
                    all_steps.add(step)

    if not all_steps:
        return _empty_figure("Orthogonality Timeline", "No valid entries found.")

    sorted_steps = sorted(all_steps)

    # Build 2D array: rows=steps, cols=pairs
    z_matrix = np.full((len(sorted_steps), len(pairs)), np.nan)
    step_to_idx = {s: i for i, s in enumerate(sorted_steps)}

    metric_key = cfg.heatmap_metric  # "overlap" or "mean_sv"

    for col_idx, pair in enumerate(pairs):
        for entry in history.get(pair, []):
            step = entry.get("step")
            if not isinstance(step, int) or step not in step_to_idx:
                continue

            # Get the metric value
            if metric_key == "mean_sv":
                value = entry.get("mean_sv")
                if value is None:
                    sv = entry.get("singular_values")
                    if sv is not None:
                        value = float(np.mean(np.asarray(sv)))
            else:  # "overlap"
                value = entry.get("overlap")
                if value is None:
                    sv = entry.get("singular_values")
                    if sv is not None:
                        value = float(np.mean(np.asarray(sv) ** 2))

            if value is not None:
                z_matrix[step_to_idx[step], col_idx] = value

    # Build title
    if title is None:
        metric_label = "mean(sv)" if metric_key == "mean_sv" else "overlap"
        title = f"Factor Orthogonality Over Training ({metric_label})"
        if title_suffix:
            title = f"{title}<br><sup>{title_suffix}</sup>"

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=pairs,
            y=sorted_steps,
            colorscale=cfg.heatmap_colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="cos(θ)", side="right"),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0 (orthog)", "0.25", "0.5", "0.75", "1 (aligned)"],
                len=0.8,
            ),
            hovertemplate=(
                "Pair: %{x}<br>"
                "Step: %{y}<br>"
                f"{metric_key}: " + "%{z:.4f}"
                "<extra></extra>"
            ),
        )
    )

    # Apply layout
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT),
        xaxis=dict(
            title="Factor Pair",
            tickfont=TICK_FONT,
            title_font=AXIS_LABEL_FONT,
        ),
        yaxis=dict(
            title="Training Step",
            tickfont=TICK_FONT,
            title_font=AXIS_LABEL_FONT,
            autorange="reversed",  # Latest step at bottom
        ),
        width=cfg.heatmap_width,
        height=cfg.heatmap_height,
        margin=dict(l=80, r=100, t=100, b=60),
        font=DEFAULT_FONT,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
    )

    return fig


# =============================================================================
# Plot 3: Orthogonality Matrix (Single-Step Snapshot)
# =============================================================================

def plot_orthogonality_matrix(
    history: OrthogonalityHistory,
    step: int | None = None,
    *,
    config: OrthogonalityVizConfig | Mapping[str, Any] | DictConfig | None = None,
    title: str | None = None,
    title_suffix: str = "",
) -> go.Figure:
    """Plot pairwise orthogonality matrix at a single training step.

    Square heatmap showing overlap between all factor pairs.
    Diagonal = 1.0 (factor with itself), off-diagonal = pairwise overlap.

    Args:
        history: Orthogonality history dict
        step: Training step to visualize. If None, uses the latest step.
        config: Visualization configuration
        title: Optional custom title
        title_suffix: Optional suffix

    Returns:
        Plotly Figure with orthogonality matrix
    """
    cfg = _coerce_config(config)

    if not history:
        return _empty_figure("Orthogonality Matrix", "No orthogonality history available.")

    pairs = _get_sorted_pairs(history)
    if not pairs:
        return _empty_figure("Orthogonality Matrix", "No valid factor pairs in history.")

    # Determine the step to use
    all_steps = set()
    for pair in pairs:
        for entry in history.get(pair, []):
            s = entry.get("step")
            if isinstance(s, int):
                all_steps.add(s)

    if not all_steps:
        return _empty_figure("Orthogonality Matrix", "No valid entries found.")

    if step is None:
        step = max(all_steps)
    elif step not in all_steps:
        # Find closest step
        step = min(all_steps, key=lambda s: abs(s - step))

    # Determine number of factors from pairs
    factor_indices = set()
    for pair in pairs:
        parsed = _parse_factor_pair(pair)
        if parsed:
            _, i, j = parsed
            factor_indices.add(i)
            factor_indices.add(j)

    num_factors = max(factor_indices) + 1 if factor_indices else 0
    factor_labels = [f"F{i}" for i in range(num_factors)]

    # Build symmetric matrix
    z_matrix = np.eye(num_factors)  # Diagonal = 1.0

    metric_key = cfg.heatmap_metric

    for pair in pairs:
        parsed = _parse_factor_pair(pair)
        if not parsed:
            continue
        _, i, j = parsed

        # Find entry at this step
        value = None
        for entry in history.get(pair, []):
            if entry.get("step") == step:
                if metric_key == "mean_sv":
                    value = entry.get("mean_sv")
                    if value is None:
                        sv = entry.get("singular_values")
                        if sv is not None:
                            value = float(np.mean(np.asarray(sv)))
                else:
                    value = entry.get("overlap")
                    if value is None:
                        sv = entry.get("singular_values")
                        if sv is not None:
                            value = float(np.mean(np.asarray(sv) ** 2))
                break

        if value is not None and i < num_factors and j < num_factors:
            z_matrix[i, j] = value
            z_matrix[j, i] = value  # Symmetric

    # Build title
    if title is None:
        metric_label = "mean(sv)" if metric_key == "mean_sv" else "overlap"
        title = f"Factor Orthogonality Matrix - Step {step}"
        title += f"<br><sup>Metric: {metric_label} | 0=orthogonal, 1=aligned</sup>"
        if title_suffix:
            title += f"<br><sup>{title_suffix}</sup>"

    # Create heatmap with text annotations
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=factor_labels,
            y=factor_labels,
            colorscale=cfg.heatmap_colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="cos(θ)", side="right"),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5", "0.75", "1"],
                len=0.8,
            ),
            hovertemplate=(
                "Row: %{y}<br>"
                "Col: %{x}<br>"
                f"{metric_key}: " + "%{z:.4f}"
                "<extra></extra>"
            ),
            text=np.round(z_matrix, 3),
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
        )
    )

    # Apply layout
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT),
        xaxis=dict(
            title="Factor",
            tickfont=TICK_FONT,
            title_font=AXIS_LABEL_FONT,
            side="bottom",
        ),
        yaxis=dict(
            title="Factor",
            tickfont=TICK_FONT,
            title_font=AXIS_LABEL_FONT,
            autorange="reversed",  # F0 at top
        ),
        width=cfg.matrix_width,
        height=cfg.matrix_height,
        margin=dict(l=80, r=100, t=120, b=60),
        font=DEFAULT_FONT,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
    )

    return fig


# =============================================================================
# History Update Helper
# =============================================================================

# Regex to parse orthogonality metric keys like "reg/orth/overlap/L0.resid.pre-F0,1"
# Format: {prefix}/orth/{metric}/{layer}-F{i},{j}
_ORTH_SCALAR_RE = re.compile(
    r"^(?:.*/)?"  # Optional prefix like "reg/" or "acts/"
    r"orth/"
    r"(?P<metric>overlap|sv_max|sv_min|p_ratio|entropy|eff_rank)"
    r"/(?P<layer>[^/]+)-F(?P<i>\d+),(?P<j>\d+)$"
)

_ORTH_SV_ARRAY_RE = re.compile(
    r"^(?:.*/)?"
    r"orth/"
    r"singular_values"
    r"/(?P<layer>[^/]+)-F(?P<i>\d+),(?P<j>\d+)$"
)


def update_orthogonality_history_from_scalars(
    history: OrthogonalityHistory,
    scalars: Mapping[str, float],
    arrays: Mapping[str, Any] | None = None,
    *,
    step: int,
    layer_filter: str | None = None,
) -> None:
    """Update orthogonality history from activation tracker scalars and arrays.

    Parses scalar keys like:
    - 'reg/orth/overlap/L0.resid.pre-F0,1'
    - 'reg/orth/sv_max/L0.resid.pre-F0,1', etc.

    And array keys like:
    - 'reg/orth/singular_values/L0.resid.pre-F0,1'

    Args:
        history: Dict to update in-place, keys are "{layer}/F{i},F{j}"
        scalars: Scalar metrics from activation analysis
        arrays: Array outputs (containing singular_values)
        step: Current training step
        layer_filter: Optional layer name to filter (if None, uses first layer found)
    """
    arrays = arrays or {}

    # Group metrics by (layer, factor_pair)
    by_layer_pair: dict[tuple[str, str], dict[str, Any]] = {}

    # Parse scalar metrics
    for key, value in scalars.items():
        match = _ORTH_SCALAR_RE.match(key)
        if not match:
            continue

        layer = match.group("layer")
        metric = match.group("metric")
        i, j = int(match.group("i")), int(match.group("j"))
        pair_key = f"F{i},F{j}"

        by_layer_pair.setdefault((layer, pair_key), {})[metric] = float(value)

    # Parse array metrics (singular values)
    for key, value in arrays.items():
        match = _ORTH_SV_ARRAY_RE.match(key)
        if not match:
            continue

        layer = match.group("layer")
        i, j = int(match.group("i")), int(match.group("j"))
        pair_key = f"F{i},F{j}"

        sv_arr = np.asarray(value, dtype=float).reshape(-1)
        by_layer_pair.setdefault((layer, pair_key), {})["singular_values"] = sv_arr

    # Determine which layers to process
    layers_found = set(layer for layer, _ in by_layer_pair.keys())
    if layer_filter is not None:
        target_layers = {layer_filter} if layer_filter in layers_found else set()
    else:
        # Process all layers when no filter specified
        target_layers = layers_found

    if not target_layers:
        return  # No data found

    # Build history entries for all target layers
    for (layer, pair_key), metrics in by_layer_pair.items():
        if layer not in target_layers:
            continue

        sv = metrics.get("singular_values")
        if sv is None:
            # Can't compute mean_sv without singular values
            continue

        entry = {
            "step": step,
            "singular_values": sv,
            "overlap": metrics.get("overlap", float(np.mean(sv**2))),
            "mean_sv": float(np.mean(sv)),
            "sv_max": metrics.get("sv_max", float(np.max(sv))),
            "sv_min": metrics.get("sv_min", float(np.min(sv))),
        }

        # Add optional metadata
        if "p_ratio" in metrics:
            entry["p_ratio"] = metrics["p_ratio"]
        if "entropy" in metrics:
            entry["entropy"] = metrics["entropy"]
        if "eff_rank" in metrics:
            entry["eff_rank"] = metrics["eff_rank"]

        # Include layer in the history key for clarity
        history_key = f"{layer}/{pair_key}"
        history.setdefault(history_key, []).append(entry)


# =============================================================================
# HTML Export
# =============================================================================

def write_orthogonality_html(
    fig: go.Figure,
    path: str,
    include_plotlyjs: str = "cdn",
) -> None:
    """Write orthogonality figure to HTML file.

    Args:
        fig: Plotly figure to save
        path: Output file path
        include_plotlyjs: How to include Plotly.js
    """
    html_content = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
    with open(path, "w") as f:
        f.write(html_content)


__all__ = [
    "plot_orthogonality_spectrum",
    "plot_orthogonality_heatmap",
    "plot_orthogonality_matrix",
    "update_orthogonality_history_from_scalars",
    "write_orthogonality_html",
]
