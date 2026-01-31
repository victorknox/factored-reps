"""Main entry point for orthogonality figure generation from MLflow training runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import MISSING

from data_loader import (
    setup_from_mlflow,
    list_checkpoints,
    select_evenly_spaced_checkpoints,
    get_num_factors_from_config,
    load_random_init_baseline,
    get_baseline_ci,
    verify_baseline_compatibility,
)
from baseline_generator import (
    generate_baseline,
    get_default_baseline_path,
    save_baseline,
)
from typing import cast

from analysis import (
    compute_orthogonality_at_checkpoints,
    get_cache_path,
    load_orthogonality_cache,
    save_orthogonality_cache,
)
from vary_one_functions import ComputeDevice
from belief_regression import (
    ProjectedBeliefRegressionData,
    generate_validation_data,
    compute_projected_belief_regression,
    get_default_k_values,
)
from plotting import create_composite_figure

# Suppress Databricks SDK logging
logging.getLogger("databricks.sdk").setLevel(logging.WARNING)


@dataclass
class TopLeftImagesConfig:
    """Configuration for the 4 source images in the top-left 2x2 grid."""

    base_dir: str = "experiments/figure_generation/figure1"
    format: str = "png"  # png, pdf, svg
    factored_2d: str = "freeze_vary_2d"
    joint_3d: str = "freeze_vary_3d"
    factored_2d_centered: str = "freeze_vary_2d_centered"
    joint_3d_centered: str = "freeze_vary_3d_centered"

    def get_paths(self) -> dict[str, str]:
        """Resolve full paths for all 4 images."""
        base = Path(self.base_dir) / self.format
        return {
            "factored_2d": str(base / f"{self.factored_2d}.{self.format}"),
            "joint_3d": str(base / f"{self.joint_3d}.{self.format}"),
            "factored_2d_centered": str(base / f"{self.factored_2d_centered}.{self.format}"),
            "joint_3d_centered": str(base / f"{self.joint_3d_centered}.{self.format}"),
        }


@dataclass
class PanelAspectsConfig:
    """Per-panel aspect ratio configuration."""

    top_left: float | None = None
    top_right: float | None = None
    bottom_left: float | None = None
    bottom_right: float | None = None


@dataclass
class CacheConfig:
    """Configuration for caching orthogonality data."""

    enabled: bool = True
    path: str | None = None  # None = auto-generate from parameters
    force_recompute: bool = False


@dataclass
class BaselineConfig:
    """Configuration for random init baseline."""

    path: str | None = None  # Explicit path, or null for auto-detection based on run_id
    auto_generate: bool = False  # Generate baseline if missing (slow: runs N random models)
    n_models: int = 5  # Number of random initializations (use 2-3 for quick testing)
    verify_compatibility: bool = True  # Check baseline matches architecture
    ci_percentile: int = 90  # CI percentile (90 = 5th-95th)


@dataclass
class BeliefRegressionConfig:
    """Configuration for projected belief regression (bottom row of composite figure)."""

    enabled: bool = True
    batch_size: int = 256  # Batch size for validation data generation
    seed: int = 12345  # Seed for validation data generation
    k_per_factor: list[int] | None = None  # None = use intrinsic dimension (state_dim - 1)
    checkpoint_step: int | None = None  # None = use last checkpoint from n_checkpoints
    max_samples: int = 10000  # Max samples for belief regression plot
    marker_size: float = 0.5  # Scatter marker size
    marker_opacity: float = 0.7  # Scatter marker opacity
    pred_marker_size: float = 0.75  # Prediction marker size
    pred_marker_opacity: float = 0.35  # Prediction marker opacity
    scatter_pad_frac: float = 0.1  # Padding fraction around scatter data
    title_pad: float = 2.0  # Padding for Theory/Activations headers
    preserve_aspect: bool = True  # True = square plots, False = fill space


@dataclass
class TheoryLinesConfig:
    """Configuration for theory lines in stacked dims plot."""

    enabled: bool = True
    show_simple: bool = True  # Theory 1: average-scaled formula
    show_per_factor: bool = True  # Theory 2: per-factor unique contribution
    linestyle: str = ":"  # Dotted
    color: str = "0.5"  # Gray
    linewidth: float = 1.0
    marker: str = "s"  # Square markers
    markersize: float = 2.0


@dataclass
class DimsBarModeConfig:
    """Configuration for stacked bar chart mode in dims plot."""

    enabled: bool = False  # Toggle for bar chart mode (replaces line plot when true)
    steps: list[int] | None = field(default_factory=lambda: [0, 800, 200000])  # init, late, final
    bar_width: float = 0.7  # Bar width relative to spacing (0-1)
    # Inset controls (separate from line plot inset)
    inset_enabled: bool = False
    inset_bounds: list[float] = field(default_factory=lambda: [0.55, 0.45, 0.42, 0.5])
    inset_xlim: list[float] | None = None
    inset_ylim: list[float] | None = None
    # "All" line styling (horizontal lines per bar)
    all_linestyle: str = "--"
    all_linewidth: float = 1.5


@dataclass
class CompositeFigureConfig:
    """Configuration for composite figure layout."""

    # Layout mode: "full" (3-column), "regression" (2-column with regression), "graphs_only" (side-by-side)
    layout: str = "full"

    # Figure sizes for each layout
    figsize: list[float] = field(default_factory=lambda: [9.0, 4.5])  # "full" layout
    figsize_regression: list[float] = field(default_factory=lambda: [3.25, 5.0])  # "regression" layout
    figsize_graphs_only: list[float] = field(default_factory=lambda: [6.75, 2.5])  # "graphs_only" layout

    # Width ratios for each layout
    width_ratios: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.6])  # "full" (3 columns)
    width_ratios_regression: list[float] = field(default_factory=lambda: [1.2, 1.0])  # "regression" (2 columns)
    width_ratios_graphs_only: list[float] = field(default_factory=lambda: [1.0, 1.0])  # "graphs_only" (2 columns)

    # Manual positioning for "regression" layout (tighter margins)
    regression_left: float = 0.02
    regression_right: float = 0.98
    regression_top: float = 0.92
    regression_bottom: float = 0.08
    regression_wspace: float = 0.25  # Gap between belief column and middle plots column
    regression_hspace: float = 0.12  # Gap between top and bottom rows
    panel_aspects: PanelAspectsConfig = field(default_factory=PanelAspectsConfig)
    top_left_images: TopLeftImagesConfig = field(default_factory=TopLeftImagesConfig)
    variance_threshold: float = 0.95
    wspace: float = 0.05  # Horizontal spacing between subplots (fraction)
    hspace: float = 0.05  # Vertical spacing between subplots (fraction)
    orthogonality_cmap: str = "viridis"  # Colormap for orthogonality plot (e.g., viridis, plasma, cividis)
    orthogonality_average_linewidth: float = 2.5  # Linewidth for average line in orthogonality plot
    orthogonality_marker_size: float = 4.0  # Marker size for orthogonality plot
    orthogonality_show_baseline_mean: bool = True  # Show Random Init Mean line (false = only show CI band)
    theory_lines: TheoryLinesConfig = field(default_factory=TheoryLinesConfig)
    dims_bar_mode: DimsBarModeConfig = field(default_factory=DimsBarModeConfig)
    dims_cmap: str = "tab10"  # Colormap for factor colors in dims plot (Set2, Pastel1, tab10, etc.)
    show_joint: bool = True  # Show "Joint" line (natural generation, all factors varying together)
    show_union: bool = True  # Show "Union" line (concatenated vary-one, union of factor subspaces)
    combined_linestyle: str = "k-"  # Linestyle for combined dims line (e.g., "k-", "k--", "k:")
    combined_linewidth: float = 2.5  # Linewidth for combined dims line
    combined_marker_size: float = 4.0  # Marker size for combined dims line
    dims_xlim: list[float] | None = None  # X-axis limits for dims plot [min, max] or null for auto
    dims_ylim: list[float] | None = None  # Y-axis limits for dims plot [min, max] or null for auto
    dims_xscale: str | None = None  # X-axis scale for dims plot (e.g., "log") or null for linear
    dims_inset_enabled: bool = False  # Show inset with zoomed region in dims plot
    dims_inset_bounds: list[float] = field(default_factory=lambda: [0.55, 0.45, 0.42, 0.5])  # Inset position [x, y, w, h]
    dims_inset_xlim: list[float] | None = None  # Inset x-axis limits or null for auto (last 25% of steps)
    dims_inset_ylim: list[float] | None = None  # Inset y-axis limits or null for auto
    dims_legend_loc: str = "upper right"  # Legend location for dims plot (panel b)
    orthogonality_legend_loc: str = "upper right"  # Legend location for orthogonality plot (panel c)
    # Left 2x2 grid spacing
    left_grid_wspace: float = 0.02  # Horizontal spacing between images
    left_grid_hspace: float = 0.02  # Vertical spacing between images
    # Belief regression grid spacing
    belief_wspace: float = 0.05  # Horizontal spacing between Theory/Activations columns
    belief_hspace: float = 0.03  # Vertical spacing between factor rows
    belief_factor_label_x: float = -0.15  # X position of F labels (in axes coords, negative = left of Theory)
    belief_factor_label_y: float = 0.5  # Y position of F labels (0.5 = vertically centered)
    belief_factor_label_ha: str = "right"  # Horizontal alignment of F labels (right, center, left)
    belief_anchor_theory: str | None = "N"  # Anchor for Theory column (N, S, E, W, C, NE, NW, SE, SW, or None)
    belief_anchor_activations: str | None = "N"  # Anchor for Activations column
    # Panel label positions [x, y] in axes coordinates
    label_a_pos: list[float] = field(default_factory=lambda: [-0.08, 1.10])
    label_b_pos: list[float] = field(default_factory=lambda: [-0.15, 1.12])
    label_c_pos: list[float] = field(default_factory=lambda: [-0.15, 1.12])
    label_d_pos: list[float] = field(default_factory=lambda: [-0.35, 1.35])
    # Axis label padding (reduces whitespace between stacked plots)
    middle_xlabel_labelpad: float = 2.0  # Padding for middle plot (b) xlabel
    bottom_xlabel_labelpad: float = 4.0  # Padding for bottom plot (c) xlabel (default matplotlib is 4.0)


@dataclass
class FigureGenerationConfig:
    """Configuration for orthogonality figure generation."""

    # MLflow run settings
    run_id: str = MISSING
    experiment_id: str | None = None
    tracking_uri: str = "databricks"
    registry_uri: str = "databricks"

    # Layer selection
    layer: str | None = None  # None = auto-select last layer (resid_post)

    # Checkpoint selection
    n_checkpoints: int = 4  # Number of evenly-spaced checkpoints to analyze (if checkpoint_steps is None)
    max_step: int | None = None  # Filter checkpoints <= max_step
    checkpoint_steps: list[int] | None = None  # Explicit list of steps (overrides n_checkpoints if provided)

    # Per-panel step filtering (subsets of checkpoint_steps, null = use all)
    dims_steps: list[int] | None = None  # Steps to show in dims@95% plot
    orthogonality_steps: list[int] | None = None  # Steps to show in orthogonality plot

    # Orthogonality analysis parameters
    max_k: int = 20  # Maximum k for orthogonality curves
    num_frozen_points: int = 10  # Frozen configurations per factor
    batch_per_frozen: int = 200  # Batch size per frozen point
    seed: int = 42  # Base seed for reproducibility
    n_seed_iterations: int = 1  # Number of independent seed iterations (1 = current behavior)
    compute_device: str = "auto"  # Device for SVD: "cuda", "cpu", or "auto"

    # Baseline settings
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    # Output settings
    output_path: str = "orthogonality_figure2.png"
    output_format: str = "png"  # png, pdf, svg
    dpi: int = 300

    # Composite figure settings
    composite_figure: CompositeFigureConfig = field(default_factory=CompositeFigureConfig)

    # Belief regression settings
    belief_regression: BeliefRegressionConfig = field(default_factory=BeliefRegressionConfig)

    # Caching settings
    cache: CacheConfig = field(default_factory=CacheConfig)


def compute_theory_lines(
    orthogonality_data,
    dims_per_factor: np.ndarray,  # [n_steps, n_factors]
    step_indices: list[int],  # Indices into orthogonality_data matching dims_per_factor rows
) -> dict[str, np.ndarray]:
    """Compute theory predictions for combined dimensionality.

    Theory 1 (Mean): max(d_i) × (1 - orth) + sum(d_i) × orth
        Where orth = 1 - avg_normalized_overlap at k=2
        - Fully orthogonal (orth=1): sum(d_i) — factors need independent subspaces
        - Fully aligned (orth=0): max(d_i) — factors share the same subspace

    Theory 2 (Per-Factor): Σ d_i × (1 - avg_overlap_i)
        Where avg_overlap_i is factor i's average overlap with all other factors
        - Each factor contributes based on its independence from others

    Args:
        orthogonality_data: OrthogonalityData from compute_orthogonality_at_checkpoints
        dims_per_factor: Array [n_steps, n_factors] of dimensions per factor
        step_indices: List of indices into orthogonality_data for each row of dims_per_factor

    Returns:
        Dict with "Mean" and "Per-Factor" arrays [n_steps]
    """
    n_steps, n_factors = dims_per_factor.shape

    # Theory 1: Interpolate between max and sum based on orthogonality
    # max(d_i) × (1 - orth) + sum(d_i) × orth
    # Where orth = 1 - avg_normalized_overlap at k=2
    mean_theory = np.zeros(n_steps)
    k = 2  # Use first 2 PCA components for orthogonality measure
    for local_idx, orig_step_idx in enumerate(step_indices):
        dims = dims_per_factor[local_idx]
        max_dims = dims.max()
        sum_dims = dims.sum()
        # Get avg overlap at k=2 (clamped to available range)
        max_k = len(orthogonality_data.avg_overlap_per_step[orig_step_idx])
        k_clamped = min(k, max_k)
        avg_overlap = orthogonality_data.avg_overlap_per_step[orig_step_idx][k_clamped - 1]
        orth_factor = 1 - avg_overlap
        mean_theory[local_idx] = max_dims * (1 - orth_factor) + sum_dims * orth_factor

    # Theory 2: Per-factor unique contribution
    per_factor_theory = np.zeros(n_steps)
    k_pf = 2  # Use first 2 PCA components for orthogonality measure
    for local_idx, orig_step_idx in enumerate(step_indices):
        dims = dims_per_factor[local_idx]
        total = 0.0
        for i in range(n_factors):
            # Get overlaps with all other factors at k=2
            overlaps = []
            for j in range(n_factors):
                if i != j:
                    # Pair key is always sorted (smaller index first)
                    pair_key = f"F{min(i, j)},F{max(i, j)}"
                    # Access orthogonality_results: [step_idx][iter_idx][pair_key][metric][k-1]
                    orth_results = orthogonality_data.orthogonality_results[orig_step_idx][0]
                    max_k = len(orth_results[pair_key]["normalized_overlap"])
                    k_clamped = min(k_pf, max_k)
                    overlap = orth_results[pair_key]["normalized_overlap"][k_clamped - 1]
                    overlaps.append(overlap)
            avg_overlap_i = np.mean(overlaps) if overlaps else 0.0
            total += dims[i] * (1 - avg_overlap_i)
        per_factor_theory[local_idx] = total

    return {"Mean": mean_theory, "Per-Factor": per_factor_theory}


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: FigureGenerationConfig) -> None:
    """Generate orthogonality figure from MLflow training run."""
    print(f"Loading run {cfg.run_id}...")

    # Setup from MLflow
    run_cfg, components, persister = setup_from_mlflow(
        run_id=cfg.run_id,
        experiment_id=cfg.experiment_id,
        tracking_uri=cfg.tracking_uri,
        registry_uri=cfg.registry_uri,
    )

    client = persister.client
    run_id = persister.run_id

    # Get components
    gp = components.get_generative_process()
    model = components.get_predictive_model()
    device = next(model.parameters()).device

    # Get config values
    num_factors = get_num_factors_from_config(run_cfg)
    context_len = run_cfg.predictive_model.instance.cfg.n_ctx
    bos_token = getattr(run_cfg.generative_process, "bos_token", None)
    eos_token = getattr(run_cfg.generative_process, "eos_token", None)
    sequence_len = context_len - int(bos_token is not None) - int(eos_token is not None)
    n_layers = model.cfg.n_layers

    print(f"Number of factors: {num_factors}")
    print(f"Sequence length: {sequence_len}")
    print(f"Number of layers: {n_layers}")

    # Determine layer hook
    if cfg.layer is None:
        layer_hook = f"blocks.{n_layers - 1}.hook_resid_post"
    else:
        layer_hook = cfg.layer
    print(f"Using layer: {layer_hook}")

    # Get available checkpoints
    checkpoints = list_checkpoints(client, run_id)
    print(f"Available checkpoints: {len(checkpoints)} (from step {checkpoints[0]} to {checkpoints[-1]})")

    # Determine which checkpoints to use
    if cfg.checkpoint_steps is not None:
        # Use explicit checkpoint steps
        selected_checkpoints = list(cfg.checkpoint_steps)
        # Validate that all requested steps exist
        missing = [s for s in selected_checkpoints if s not in checkpoints]
        if missing:
            print(f"Warning: Requested steps not found in checkpoints: {missing}")
            selected_checkpoints = [s for s in selected_checkpoints if s in checkpoints]
        print(f"Using explicit checkpoints: {selected_checkpoints}")
    else:
        # Auto-select evenly spaced checkpoints
        # Filter by max_step if specified
        if cfg.max_step is not None:
            checkpoints = [s for s in checkpoints if s <= cfg.max_step]
            print(f"Filtered to {len(checkpoints)} checkpoints <= {cfg.max_step}")

        selected_checkpoints = select_evenly_spaced_checkpoints(checkpoints, cfg.n_checkpoints)
        print(f"Auto-selected checkpoints: {selected_checkpoints}")

    # Determine belief regression step (if enabled)
    belief_regression_step = None
    if cfg.belief_regression.enabled:
        if cfg.belief_regression.checkpoint_step is not None:
            belief_regression_step = cfg.belief_regression.checkpoint_step
        else:
            # Use last selected checkpoint by default
            belief_regression_step = selected_checkpoints[-1]
        print(f"Belief regression will use checkpoint: step {belief_regression_step}")

    # Determine cache path
    cache_path = None
    if cfg.cache.enabled:
        if cfg.cache.path is not None:
            cache_path = Path(cfg.cache.path)
        else:
            cache_path = get_cache_path(
                run_id=cfg.run_id,
                layer_hook=layer_hook,
                n_checkpoints=cfg.n_checkpoints,
                max_k=cfg.max_k,
                n_seed_iterations=cfg.n_seed_iterations,
            )

    # Try to load from cache
    orthogonality_data = None
    if cache_path and not cfg.cache.force_recompute:
        orthogonality_data = load_orthogonality_cache(cache_path)

    # Compute if not cached
    if orthogonality_data is None:
        print(f"Computing orthogonality at {len(selected_checkpoints)} checkpoints with {cfg.n_seed_iterations} seed iteration(s)...")
        orthogonality_data = compute_orthogonality_at_checkpoints(
            model=model,
            gp=gp,
            persister=persister,
            checkpoint_steps=selected_checkpoints,
            layer_hook=layer_hook,
            num_factors=num_factors,
            sequence_len=sequence_len,
            bos_token=bos_token,
            device=device,
            num_frozen_points=cfg.num_frozen_points,
            batch_per_frozen=cfg.batch_per_frozen,
            max_k=cfg.max_k,
            seed_base=cfg.seed,
            n_seed_iterations=cfg.n_seed_iterations,
            store_factor_pca_for_step=belief_regression_step,
            compute_device=cast(ComputeDevice, cfg.compute_device),
        )

        # Save to cache
        if cache_path:
            save_orthogonality_cache(orthogonality_data, cache_path)
    else:
        print("Using cached orthogonality data")
    print(f"Computed orthogonality for k=1 to {orthogonality_data.max_k}")

    # Load or generate baseline
    baseline_mean = None
    baseline_lower = None
    baseline_upper = None
    baseline_data = None

    # Resolve baseline path
    # Note: Hydra changes working directory, so we resolve relative paths from script location
    script_dir = Path(__file__).parent
    if cfg.baseline.path is not None:
        baseline_path = Path(cfg.baseline.path)
        # If relative, resolve from script directory (orthogonality folder)
        if not baseline_path.is_absolute():
            # Try relative to script first, then try from repo root
            if (script_dir / baseline_path).exists():
                baseline_path = script_dir / baseline_path
            elif (script_dir.parent.parent.parent / baseline_path).exists():
                # Path like "experiments/figure_generation/orthogonality/baseline_..."
                baseline_path = script_dir.parent.parent.parent / baseline_path
    else:
        baseline_path = get_default_baseline_path(cfg.run_id)

    # Try to load existing baseline
    if baseline_path.exists():
        print(f"Loading baseline from {baseline_path}...")
        baseline_data = load_random_init_baseline(baseline_path)

        if baseline_data is not None and cfg.baseline.verify_compatibility:
            is_compat, warnings = verify_baseline_compatibility(
                baseline_data, layer_hook, model.cfg.d_model, num_factors
            )
            for w in warnings:
                print(f"  Warning: {w}")
            if not is_compat:
                if cfg.baseline.auto_generate:
                    print("  Baseline incompatible, will regenerate")
                else:
                    print("  Baseline incompatible, skipping")
                baseline_data = None
            else:
                print("  Baseline compatibility verified")

    # Generate baseline if needed and enabled
    if baseline_data is None and cfg.baseline.auto_generate:
        print("=" * 60)
        print("BASELINE GENERATION (one-time operation)")
        print("=" * 60)
        print(f"  Generating baseline with {cfg.baseline.n_models} random models...")
        print(f"  This may take several minutes but only needs to run once.")
        print(f"  Future runs will load from: {baseline_path}")
        print("-" * 60)

        # Get state dimensions from generative process
        state_dims = [tm.shape[-1] for tm in gp.transition_matrices]

        baseline_data = generate_baseline(
            gp=gp,
            base_config=model.cfg,
            layer_hooks=[layer_hook],
            num_factors=num_factors,
            state_dims=state_dims,
            sequence_len=sequence_len,
            bos_token=bos_token,
            device=device,
            run_id=cfg.run_id,
            experiment_id=cfg.experiment_id,
            n_models=cfg.baseline.n_models,
            num_frozen_points=cfg.num_frozen_points,
            batch_per_frozen=cfg.batch_per_frozen,
            data_seed=cfg.seed,
            max_k=cfg.max_k,
        )
        save_baseline(baseline_data, baseline_path)
        print("-" * 60)
        print(f"  Baseline saved to: {baseline_path}")
        print("  (Won't regenerate unless you delete this file or set baseline.path to a different location)")
        print("=" * 60)

    # Extract CI if we have baseline
    if baseline_data is not None:
        ci_lower_key = f"percentile_{(100 - cfg.baseline.ci_percentile) // 2}"
        ci_upper_key = f"percentile_{100 - (100 - cfg.baseline.ci_percentile) // 2}"
        baseline_mean, baseline_lower, baseline_upper = get_baseline_ci(
            baseline_data,
            "normalized_overlap",
            ci_lower_key,
            ci_upper_key,
            layer=layer_hook,
        )
        print(f"  Baseline loaded: {len(baseline_mean)} k values")
    elif cfg.baseline.path is not None or cfg.baseline.auto_generate:
        print("  Warning: No baseline available, figure will not include CI band")

    # Compute belief regression if enabled and data is available
    belief_regression_data = None
    if (
        cfg.belief_regression.enabled
        and orthogonality_data.factor_pca_data is not None
    ):
        print("Computing projected belief regression...")

        # Reload the model at the belief regression step (needed if cache was used)
        if orthogonality_data.belief_regression_step is not None:
            print(f"  Loading checkpoint at step {orthogonality_data.belief_regression_step}")
            persister.load_weights(model, step=orthogonality_data.belief_regression_step)
            model.eval()

        # Get state dimensions for default k values
        state_dims = [fpd.state_dim for fpd in orthogonality_data.factor_pca_data]

        # Determine k values
        if cfg.belief_regression.k_per_factor is not None:
            k_values = list(cfg.belief_regression.k_per_factor)
        else:
            k_values = get_default_k_values(state_dims)
        print(f"  Using k values: {k_values}")

        # Generate validation data
        print(f"  Generating validation data (batch_size={cfg.belief_regression.batch_size})...")
        activations_flat, beliefs_flat = generate_validation_data(
            gp=gp,
            model=model,
            batch_size=cfg.belief_regression.batch_size,
            sequence_len=sequence_len,
            layer_hook=layer_hook,
            bos_token=bos_token,
            eos_token=eos_token,
            device=device,
            seed=cfg.belief_regression.seed,
        )
        print(f"  Validation activations shape: {activations_flat.shape}")

        # Compute projected belief regression
        print("  Computing projected belief regression...")
        belief_regression_data = compute_projected_belief_regression(
            factor_pca_data=orthogonality_data.factor_pca_data,
            activations_flat=activations_flat,
            beliefs_flat=beliefs_flat,
            k_values=k_values,
        )

        # Set metadata
        belief_regression_data.step = orthogonality_data.belief_regression_step
        belief_regression_data.layer_hook = layer_hook

        print(f"  Overall RMSE: {belief_regression_data.overall_rmse:.4f}")
        print(f"  Per-factor RMSE: {[f'{r:.4f}' for r in belief_regression_data.factor_rmse_scores]}")

    elif cfg.belief_regression.enabled:
        print("Warning: Belief regression enabled but factor PCA data not available (cached data?)")
        print("  To compute belief regression, set cache.force_recompute=true")

    # Create figure
    print("Creating figure...")

    # Compute theory lines if enabled
    theory_lines_data = None
    if cfg.composite_figure.theory_lines.enabled:
        from analysis import compute_dims_at_threshold

        # Compute dims_per_factor (same logic as in create_composite_figure)
        n_steps, n_factors_cev, _ = orthogonality_data.cev_per_factor.shape
        all_dims_per_factor = np.zeros((n_steps, n_factors_cev), dtype=int)
        for step_idx in range(n_steps):
            for factor_idx in range(n_factors_cev):
                cev = orthogonality_data.cev_per_factor[step_idx, factor_idx]
                all_dims_per_factor[step_idx, factor_idx] = compute_dims_at_threshold(
                    cev, cfg.composite_figure.variance_threshold
                )

        # Filter to dims_steps if specified
        if cfg.dims_steps is not None:
            dims_step_indices = [i for i, s in enumerate(orthogonality_data.steps) if s in cfg.dims_steps]
            dims_per_factor_filtered = all_dims_per_factor[dims_step_indices]
        else:
            dims_step_indices = list(range(n_steps))
            dims_per_factor_filtered = all_dims_per_factor

        # Compute theory lines
        theory_lines_computed = compute_theory_lines(
            orthogonality_data=orthogonality_data,
            dims_per_factor=dims_per_factor_filtered,
            step_indices=dims_step_indices,
        )

        # Filter based on config
        theory_lines_data = {}
        if cfg.composite_figure.theory_lines.show_simple:
            theory_lines_data["Mean"] = theory_lines_computed["Mean"]
        if cfg.composite_figure.theory_lines.show_per_factor:
            theory_lines_data["Per-Factor"] = theory_lines_computed["Per-Factor"]

        if theory_lines_data:
            print(f"  Theory lines computed for {len(dims_step_indices)} steps")
            for name, values in theory_lines_data.items():
                print(f"    {name}: range [{values.min():.1f}, {values.max():.1f}]")
        else:
            theory_lines_data = None

    # Convert panel_aspects to dict
    panel_aspects = {
        "top_left": cfg.composite_figure.panel_aspects.top_left,
        "top_right": cfg.composite_figure.panel_aspects.top_right,
        "bottom_left": cfg.composite_figure.panel_aspects.bottom_left,
        "bottom_right": cfg.composite_figure.panel_aspects.bottom_right,
    }

    # Resolve top-left image paths
    tl_cfg = cfg.composite_figure.top_left_images
    base_path = Path(tl_cfg.base_dir) / tl_cfg.format
    top_left_images = {
        "factored_2d": str(base_path / f"{tl_cfg.factored_2d}.{tl_cfg.format}"),
        "joint_3d": str(base_path / f"{tl_cfg.joint_3d}.{tl_cfg.format}"),
        "factored_2d_centered": str(base_path / f"{tl_cfg.factored_2d_centered}.{tl_cfg.format}"),
        "joint_3d_centered": str(base_path / f"{tl_cfg.joint_3d_centered}.{tl_cfg.format}"),
    }
    fig = create_composite_figure(
        orthogonality_data=orthogonality_data,
        baseline_mean=baseline_mean,
        baseline_lower=baseline_lower,
        baseline_upper=baseline_upper,
        top_left_images=top_left_images,
        belief_regression_data=belief_regression_data,
        expected_total=None,  # TODO: compute from generative process state dims
        layout=cfg.composite_figure.layout,
        figsize=tuple(cfg.composite_figure.figsize),
        figsize_regression=tuple(cfg.composite_figure.figsize_regression),
        figsize_graphs_only=tuple(cfg.composite_figure.figsize_graphs_only),
        width_ratios=list(cfg.composite_figure.width_ratios),
        width_ratios_regression=list(cfg.composite_figure.width_ratios_regression),
        width_ratios_graphs_only=list(cfg.composite_figure.width_ratios_graphs_only),
        panel_aspects=panel_aspects,
        wspace=cfg.composite_figure.wspace,
        hspace=cfg.composite_figure.hspace,
        variance_threshold=cfg.composite_figure.variance_threshold,
        ci_percentile=cfg.baseline.ci_percentile,
        orthogonality_cmap=cfg.composite_figure.orthogonality_cmap,
        orthogonality_average_linewidth=cfg.composite_figure.orthogonality_average_linewidth,
        orthogonality_marker_size=cfg.composite_figure.orthogonality_marker_size,
        orthogonality_show_baseline_mean=cfg.composite_figure.orthogonality_show_baseline_mean,
        belief_regression_max_samples=cfg.belief_regression.max_samples,
        belief_regression_marker_size=cfg.belief_regression.marker_size,
        belief_regression_marker_opacity=cfg.belief_regression.marker_opacity,
        belief_regression_pred_marker_size=cfg.belief_regression.pred_marker_size,
        belief_regression_pred_marker_opacity=cfg.belief_regression.pred_marker_opacity,
        belief_regression_scatter_pad_frac=cfg.belief_regression.scatter_pad_frac,
        belief_regression_title_pad=cfg.belief_regression.title_pad,
        belief_regression_preserve_aspect=cfg.belief_regression.preserve_aspect,
        belief_regression_cmap=cfg.belief_regression.cmap,
        belief_regression_cmap_start=cfg.belief_regression.cmap_start,
        belief_regression_cmap_mid=cfg.belief_regression.cmap_mid,
        belief_regression_cmap_end=cfg.belief_regression.cmap_end,
        combined_linestyle=cfg.composite_figure.combined_linestyle,
        combined_linewidth=cfg.composite_figure.combined_linewidth,
        combined_marker_size=cfg.composite_figure.combined_marker_size,
        dims_cmap=cfg.composite_figure.dims_cmap,
        show_joint=cfg.composite_figure.show_joint,
        show_union=cfg.composite_figure.show_union,
        dims_xlim=tuple(cfg.composite_figure.dims_xlim) if cfg.composite_figure.dims_xlim else None,
        dims_ylim=tuple(cfg.composite_figure.dims_ylim) if cfg.composite_figure.dims_ylim else None,
        dims_xscale=cfg.composite_figure.dims_xscale,
        dims_inset_enabled=cfg.composite_figure.dims_inset_enabled,
        dims_inset_bounds=tuple(cfg.composite_figure.dims_inset_bounds),
        dims_inset_xlim=tuple(cfg.composite_figure.dims_inset_xlim) if cfg.composite_figure.dims_inset_xlim else None,
        dims_inset_ylim=tuple(cfg.composite_figure.dims_inset_ylim) if cfg.composite_figure.dims_inset_ylim else None,
        dims_steps=list(cfg.dims_steps) if cfg.dims_steps else None,
        orthogonality_steps=list(cfg.orthogonality_steps) if cfg.orthogonality_steps else None,
        dims_legend_loc=cfg.composite_figure.dims_legend_loc,
        orthogonality_legend_loc=cfg.composite_figure.orthogonality_legend_loc,
        # Layout spacing parameters
        left_grid_wspace=cfg.composite_figure.left_grid_wspace,
        left_grid_hspace=cfg.composite_figure.left_grid_hspace,
        belief_wspace=cfg.composite_figure.belief_wspace,
        belief_hspace=cfg.composite_figure.belief_hspace,
        belief_factor_label_x=cfg.composite_figure.belief_factor_label_x,
        belief_factor_label_y=cfg.composite_figure.belief_factor_label_y,
        belief_factor_label_ha=cfg.composite_figure.belief_factor_label_ha,
        belief_anchor_theory=cfg.composite_figure.belief_anchor_theory,
        belief_anchor_activations=cfg.composite_figure.belief_anchor_activations,
        # Panel label positions
        label_a_pos=tuple(cfg.composite_figure.label_a_pos),
        label_b_pos=tuple(cfg.composite_figure.label_b_pos),
        label_c_pos=tuple(cfg.composite_figure.label_c_pos),
        label_d_pos=tuple(cfg.composite_figure.label_d_pos),
        # Axis label padding
        middle_xlabel_labelpad=cfg.composite_figure.middle_xlabel_labelpad,
        bottom_xlabel_labelpad=cfg.composite_figure.bottom_xlabel_labelpad,
        # Manual positioning for "regression" layout
        regression_left=cfg.composite_figure.regression_left,
        regression_right=cfg.composite_figure.regression_right,
        regression_top=cfg.composite_figure.regression_top,
        regression_bottom=cfg.composite_figure.regression_bottom,
        regression_wspace=cfg.composite_figure.regression_wspace,
        regression_hspace=cfg.composite_figure.regression_hspace,
        # Theory line parameters
        theory_lines=theory_lines_data,
        theory_linestyle=cfg.composite_figure.theory_lines.linestyle,
        theory_color=cfg.composite_figure.theory_lines.color,
        theory_linewidth=cfg.composite_figure.theory_lines.linewidth,
        theory_marker=cfg.composite_figure.theory_lines.marker,
        theory_markersize=cfg.composite_figure.theory_lines.markersize,
        # Bar mode parameters
        dims_bar_mode_enabled=cfg.composite_figure.dims_bar_mode.enabled,
        dims_bar_mode_steps=list(cfg.composite_figure.dims_bar_mode.steps) if cfg.composite_figure.dims_bar_mode.steps else None,
        dims_bar_mode_bar_width=cfg.composite_figure.dims_bar_mode.bar_width,
        dims_bar_mode_inset_enabled=cfg.composite_figure.dims_bar_mode.inset_enabled,
        dims_bar_mode_inset_bounds=tuple(cfg.composite_figure.dims_bar_mode.inset_bounds),
        dims_bar_mode_inset_xlim=tuple(cfg.composite_figure.dims_bar_mode.inset_xlim) if cfg.composite_figure.dims_bar_mode.inset_xlim else None,
        dims_bar_mode_inset_ylim=tuple(cfg.composite_figure.dims_bar_mode.inset_ylim) if cfg.composite_figure.dims_bar_mode.inset_ylim else None,
        dims_bar_mode_all_linestyle=cfg.composite_figure.dims_bar_mode.all_linestyle,
        dims_bar_mode_all_linewidth=cfg.composite_figure.dims_bar_mode.all_linewidth,
    )

    # Save figure
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=cfg.output_format, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
