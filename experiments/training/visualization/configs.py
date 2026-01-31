"""Configuration dataclasses for bespoke training visualizations."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from visualization.styles import (
    DEFAULT_COLORSCALE,
    DEFAULT_HEIGHT,
    DEFAULT_LINE_OPACITY,
    DEFAULT_LINE_WIDTH,
    DEFAULT_WIDTH,
)


def _to_dict(cfg: Mapping[str, Any] | DictConfig | None) -> dict[str, Any] | None:
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
        return container if isinstance(container, dict) else None
    if isinstance(cfg, dict):
        return cfg
    return dict(cfg)


@dataclass(frozen=True)
class CEVVizConfig:
    enabled: bool = True
    colorscale: str = DEFAULT_COLORSCALE
    history_window: int | None = None
    max_components: int | None = 15

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    show_rangeslider: bool = False

    line_width: float = DEFAULT_LINE_WIDTH
    line_opacity: float = DEFAULT_LINE_OPACITY

    y_range: tuple[float, float] = (0.0, 1.0)
    x_label: str = "Number of Components"
    y_label: str = "Cumulative Explained Variance"
    colorbar_title: str = "Training Step"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | DictConfig | None) -> "CEVVizConfig":
        raw = _to_dict(d) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


@dataclass(frozen=True)
class BeliefRegressionVizConfig:
    """Configuration for belief regression visualization."""

    enabled: bool = True
    colorscale: str = DEFAULT_COLORSCALE

    # Plot dimensions - height scales with number of factors
    width: int = 1000
    height_per_factor: int = 400

    # Scatter plot styling
    marker_size: int = 4
    marker_opacity: float = 0.7

    # Maximum samples to visualize (for performance)
    max_samples: int = 200

    # Whether to use PCA for high-dimensional belief states (>3D)
    use_pca_for_high_dim: bool = True
    pca_components: int = 3

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | DictConfig | None) -> "BeliefRegressionVizConfig":
        raw = _to_dict(d) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


@dataclass(frozen=True)
class OrthogonalityVizConfig:
    """Configuration for orthogonality visualization."""

    enabled: bool = True

    # Colorscales
    spectrum_colorscale: str = DEFAULT_COLORSCALE  # For spectrum plot lines (by step)
    heatmap_colorscale: str = "RdBu_r"  # Diverging: blue=orthogonal, red=aligned

    # Spectrum plot dimensions
    spectrum_width: int = DEFAULT_WIDTH
    spectrum_height: int = DEFAULT_HEIGHT
    spectrum_height_per_pair: int = 300  # Height per factor pair subplot

    # Heatmap dimensions
    heatmap_width: int = 800
    heatmap_height: int = 500

    # Matrix snapshot dimensions
    matrix_width: int = 500
    matrix_height: int = 450

    # Line styling for spectrum plot
    line_width: float = DEFAULT_LINE_WIDTH
    line_opacity: float = DEFAULT_LINE_OPACITY

    # Reference lines
    show_reference_line: bool = True
    reference_line_value: float = 0.5  # Threshold for "moderate" overlap

    # History window (None = show all)
    history_window: int | None = None

    # Which overlap metric to use in heatmaps: "overlap" (mean sv²) or "mean_sv" (mean sv)
    heatmap_metric: str = "mean_sv"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | DictConfig | None) -> "OrthogonalityVizConfig":
        raw = _to_dict(d) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


@dataclass(frozen=True)
class VisualizationConfig:
    every: int = 100
    cev: CEVVizConfig = field(default_factory=CEVVizConfig)
    belief_regression: BeliefRegressionVizConfig = field(default_factory=BeliefRegressionVizConfig)
    orthogonality: OrthogonalityVizConfig = field(default_factory=OrthogonalityVizConfig)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | DictConfig | None) -> "VisualizationConfig":
        raw = _to_dict(d) or {}
        return cls(
            every=int(raw.get("every", 100)),
            cev=CEVVizConfig.from_dict(raw.get("cev")),
            belief_regression=BeliefRegressionVizConfig.from_dict(raw.get("belief_regression")),
            orthogonality=OrthogonalityVizConfig.from_dict(raw.get("orthogonality")),
        )

