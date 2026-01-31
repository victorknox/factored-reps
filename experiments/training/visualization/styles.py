"""Shared styling defaults for bespoke Plotly visualizations.

Warm, archival research-publication aesthetic inspired by Distill.pub
and Transformer Circuits posts.
"""

from __future__ import annotations

import plotly.graph_objects as go

# =============================================================================
# Color Palette - Warm Archival
# =============================================================================

PAPER_BG = "#faf8f5"  # warm cream (aged paper)
PLOT_BG = "#faf8f5"
PRIMARY_TEXT = "#333333"
SECONDARY_TEXT = "#666666"
MUTED_TEXT = "#999999"
GRID_COLOR = "rgba(180, 170, 160, 0.25)"
AXIS_LINE_COLOR = "#888888"

# Custom colorscale (indigo -> teal -> gold) - elegant contrast on cream
DEFAULT_COLORSCALE = "Archival"

# Archival colorscale definition for Plotly
ARCHIVAL_COLORSCALE = [
    [0.0, "#2e3a5f"],   # deep indigo
    [0.25, "#2d6a7a"],  # dark teal
    [0.5, "#3a9a8c"],   # teal
    [0.75, "#7dbf7d"],  # sage green
    [1.0, "#d4a84b"],   # gold
]

# =============================================================================
# Typography - Serif for scholarly feel
# =============================================================================

DEFAULT_FONT = dict(
    family="Georgia, 'Times New Roman', serif",
    size=12,
    color=PRIMARY_TEXT,
)

TITLE_FONT = dict(
    family="Georgia, 'Times New Roman', serif",
    size=18,
    color=PRIMARY_TEXT,
)

TICK_FONT = dict(
    family="Georgia, 'Times New Roman', serif",
    size=11,
    color=SECONDARY_TEXT,
)

AXIS_LABEL_FONT = dict(
    family="Georgia, 'Times New Roman', serif",
    size=13,
    color=PRIMARY_TEXT,
)

# =============================================================================
# Layout Defaults
# =============================================================================

DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 600
DEFAULT_MARGIN = dict(l=70, r=90, t=100, b=70)  # more breathing room

DEFAULT_LINE_WIDTH = 1.8
DEFAULT_LINE_OPACITY = 0.85

COLORBAR_DEFAULTS = dict(
    thickness=15,
    len=0.7,
    x=1.02,
    tickfont=TICK_FONT,
)

# =============================================================================
# Axis Styling - Subtle, low-contrast
# =============================================================================

XAXIS_DEFAULTS = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor=GRID_COLOR,
    showline=True,
    linewidth=1,
    linecolor=AXIS_LINE_COLOR,
    mirror=False,
    tickfont=TICK_FONT,
    title_font=AXIS_LABEL_FONT,
)

YAXIS_DEFAULTS = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor=GRID_COLOR,
    zeroline=False,  # cleaner without zero line
    showline=True,
    linewidth=1,
    linecolor=AXIS_LINE_COLOR,
    mirror=False,
    tickfont=TICK_FONT,
    title_font=AXIS_LABEL_FONT,
)


def apply_default_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Apply consistent default styling to a figure."""
    fig.update_layout(
        font=DEFAULT_FONT,
        margin=DEFAULT_MARGIN,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        **overrides,
    )
    fig.update_xaxes(**XAXIS_DEFAULTS)
    fig.update_yaxes(**YAXIS_DEFAULTS)
    return fig
