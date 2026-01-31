"""Parallel figure generation entry point for belief-grid figures.

This is the parallel version of run.py with optimized checkpoint processing:
1. Combined forward passes (activations + loss in one pass) - ~2x speedup
2. Multi-GPU parallel checkpoint processing - additional ~Nx speedup for N GPUs

Usage (from this directory):
- Sequential (single GPU): `uv run python run_parallel.py --config-name=fig2`
- Parallel with workers: `uv run python run_parallel.py --config-name=fig2 workers=4`
- Multi-GPU: `CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python run_parallel.py --config-name=fig2 workers=4`

The workers parameter controls parallelism:
- workers=1 (default): Sequential processing with combined forward passes (~2x faster than run.py)
- workers=N: Process N checkpoints in parallel across available GPUs (~2Nx faster)
"""

from __future__ import annotations

import os
import pickle
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import MISSING, OmegaConf

from fwh_core.analysis.metric_keys import format_layer_spec

from data_loader import (
    setup_from_mlflow,
    list_all_metrics,
    discover_layer_names,
    fetch_all_metrics_for_figure,
    list_checkpoints,
    get_num_factors_from_config,
    select_evenly_spaced_checkpoints,
)

# Import from parallel analysis module
from analysis_parallel import (
    prepare_sequences,
    get_activations,
    compute_belief_regression,
    compute_cev_at_checkpoints,
    compute_all_metrics_at_checkpoints_parallel,  # Use parallel version
    compute_entropy_rate,
    compute_belief_cev_baselines,
)
from plotting import create_main_figure, create_supplemental_figure, create_main_figure_vertical

# Suppress Databricks SDK logging
logging.getLogger("databricks.sdk").setLevel(logging.WARNING)


def get_preferred_checkpoints(available_checkpoints: list[int], max_step: int | None = None) -> list[int]:
    """Generate preferred checkpoint pattern from available checkpoints.

    Pattern:
    - 0-10: every step
    - 20-500: every 10
    - 600-900: every 100
    - 1000-5000: every 1000
    - 5000+: every 5000 (10000, 15000, 20000, ...)

    Args:
        available_checkpoints: List of available checkpoint steps.
        max_step: Maximum step to include. If None, uses max available.

    Returns:
        Filtered list of checkpoints matching the preferred pattern.
    """
    if not available_checkpoints:
        return []

    available_set = set(available_checkpoints)
    max_available = max(available_checkpoints)
    if max_step is not None:
        max_available = min(max_available, max_step)

    # Build preferred pattern
    preferred = []

    # 0-10: every step
    preferred.extend(range(0, 11))

    # 20-500: every 10
    preferred.extend(range(20, 501, 10))

    # 600-900: every 100
    preferred.extend(range(600, 1000, 100))

    # 1000-5000: every 1000
    preferred.extend(range(1000, 5001, 1000))

    # 5000+: every 5000
    step = 10000
    while step <= max_available:
        preferred.append(step)
        step += 5000

    # Filter to only available checkpoints and respect max_step
    result = [s for s in preferred if s in available_set and s <= max_available]
    return sorted(result)


def filter_cev_to_preferred(cev_history: dict | None, max_step: int | None = None) -> dict | None:
    """Filter CEV history to only include preferred checkpoints.

    Args:
        cev_history: Dict mapping layer names to list of (step, cev_array) tuples.
        max_step: Maximum step to include.

    Returns:
        Filtered CEV history with only preferred checkpoints.
    """
    if cev_history is None:
        return None

    filtered = {}
    for layer_name, step_data in cev_history.items():
        # Get all available steps from the cached data
        available_steps = [step for step, _ in step_data]
        # Get the preferred subset
        preferred_steps = set(get_preferred_checkpoints(available_steps, max_step))
        # Filter to only preferred steps
        filtered[layer_name] = [(step, cev) for step, cev in step_data if step in preferred_steps]

    return filtered


@dataclass
class FigureGenerationConfig:
    """Configuration for figure generation."""

    # MLflow run settings
    run_id: str = MISSING
    experiment_id: str | None = None
    tracking_uri: str = "databricks"
    registry_uri: str = "databricks"

    # Device override (e.g., "cuda", "cpu"). If None, uses the run's config.yaml value.
    device: str | None = None

    # Layer selection
    layer: str | None = None  # None = auto-select last layer

    # Ground truth reference
    gt_dims95: float | None = None

    # Step filtering
    cev_max_step: int | None = None
    dims_max_step: int | None = None  # Max step for panel (c) dims+loss
    rmse_max_step: int | None = None  # Max step for panel (d) dims+RMSE
    log_scale: bool = False  # Use log scale for x-axis on panels (c) and (d)

    # Metric keys
    loss_metric_key: str = "loss/step"

    # Checkpoint selection for belief regression
    checkpoint_step: int | None = None  # None = use latest

    # CEV recomputation settings
    recompute_cev: bool = False
    cev_n_checkpoints: int = 20
    cev_batch_size: int = 1000

    # dims@95 recomputation settings
    recompute_dims95: bool = False
    dims95_n_checkpoints: int = 50  # More checkpoints for smoother curve

    # RMSE recomputation settings
    recompute_rmse: bool = False
    rmse_n_checkpoints: int = 20  # Number of checkpoints to compute RMSE at

    # BOS token handling
    skip_bos: bool = False  # Set to True to skip BOS-only prefixes in analysis

    # Prefix length filtering
    max_prefix_length: int | None = None  # Max prefix length to include

    # Output settings
    output_path: str = "figure.png"
    supplemental_path: str | None = "figure_supplemental.png"
    vertical_path: str | None = None
    output_format: str = "png"
    dpi: int = 150
    figsize: list[float] = field(default_factory=lambda: [6.75, 4.0])
    supplemental_figsize: list[float] = field(default_factory=lambda: [6.75, 2.5])

    # Caching
    use_cache: bool = True
    cache_dir: str = ".cache"

    # Parallel processing settings
    workers: int = 1  # Number of parallel workers for checkpoint processing


def get_cache_key(cfg: FigureGenerationConfig) -> str:
    """Compute a cache key from config options that affect computation.

    Note: max_step values are excluded from the key to allow incremental updates
    when new checkpoints become available. The max_step filtering happens at
    plot time, not computation time.
    """
    cache_relevant = {
        "run_id": cfg.run_id,
        "layer": cfg.layer,
        # Note: cev_max_step excluded to allow incremental checkpoint updates
        "cev_n_checkpoints": cfg.cev_n_checkpoints,
        "cev_batch_size": cfg.cev_batch_size,
        "rmse_n_checkpoints": cfg.rmse_n_checkpoints,
        "skip_bos": cfg.skip_bos,
        "max_prefix_length": cfg.max_prefix_length,
        # Note: checkpoint_step excluded - belief regression uses latest available
        "recompute_cev": cfg.recompute_cev,
        "recompute_rmse": cfg.recompute_rmse,
        "recompute_dims95": cfg.recompute_dims95,
        "device": cfg.get("device", None),
    }
    key_str = str(sorted(cache_relevant.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def load_cache(cfg: FigureGenerationConfig) -> dict | None:
    """Load cached data if available."""
    if not cfg.use_cache:
        return None
    cache_dir = Path(cfg.cache_dir)
    cache_key = get_cache_key(cfg)
    cache_file = cache_dir / f"cache_{cache_key}.pkl"
    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(cfg: FigureGenerationConfig, data: dict) -> None:
    """Save computed data to cache."""
    if not cfg.use_cache:
        return
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_key = get_cache_key(cfg)
    cache_file = cache_dir / f"cache_{cache_key}.pkl"
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


def get_cached_checkpoint_steps(cached_data: dict) -> set[int]:
    """Extract the set of checkpoint steps already in the cache."""
    cached_steps = set()
    if cached_data is None:
        return cached_steps

    metrics = cached_data.get("metrics", {})
    if "dims95" in metrics:
        dims95_df = metrics["dims95"]
        if hasattr(dims95_df, "step"):
            cached_steps = set(dims95_df["step"].tolist())
    return cached_steps


def merge_metrics(cached_metrics: dict, new_metrics: dict) -> dict:
    """Merge new checkpoint metrics into cached metrics."""
    import pandas as pd

    merged = cached_metrics.copy()

    for key in ["dims95", "loss", "overall_rmse"]:
        if key in new_metrics and key in merged:
            # Both have data - concatenate and sort
            cached_df = merged[key]
            new_df = new_metrics[key]
            combined = pd.concat([cached_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["step"], keep="last")
            combined = combined.sort_values("step").reset_index(drop=True)
            merged[key] = combined
        elif key in new_metrics:
            # Only new has data
            merged[key] = new_metrics[key]

    # Handle per-factor RMSE
    for key in cached_metrics.keys():
        if key.startswith("factor_") and key.endswith("_rmse"):
            if key in new_metrics:
                cached_df = merged[key]
                new_df = new_metrics[key]
                combined = pd.concat([cached_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["step"], keep="last")
                combined = combined.sort_values("step").reset_index(drop=True)
                merged[key] = combined

    return merged


def merge_cev_history(cached_cev: dict | None, new_cev: dict | None) -> dict | None:
    """Merge new CEV history into cached CEV history."""
    if cached_cev is None:
        return new_cev
    if new_cev is None:
        return cached_cev

    merged = {}
    all_layers = set(cached_cev.keys()) | set(new_cev.keys())

    for layer in all_layers:
        cached_list = cached_cev.get(layer, [])
        new_list = new_cev.get(layer, [])

        # Create dict from lists (step -> cev_array)
        combined = {step: cev for step, cev in cached_list}
        combined.update({step: cev for step, cev in new_list})

        # Sort by step
        merged[layer] = sorted(combined.items(), key=lambda x: x[0])

    return merged


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: FigureGenerationConfig) -> None:
    """Generate composite figure from MLflow training run (parallel version)."""
    print(f"Loading run {cfg.run_id}...")

    # Get workers setting
    workers = cfg.get("workers", 1)
    if workers > 1:
        print(f"Using {workers} parallel workers for checkpoint processing")

    # Check for cached data first
    cached_data = load_cache(cfg)
    if cached_data is not None:
        # Check for new checkpoints that aren't in the cache
        cached_steps = get_cached_checkpoint_steps(cached_data)
        layer_name = cached_data["layer_name"]

        # Get current available checkpoints from MLflow
        run_cfg, components, persister = setup_from_mlflow(
            run_id=cfg.run_id,
            experiment_id=cfg.experiment_id,
            tracking_uri=cfg.tracking_uri,
            registry_uri=cfg.registry_uri,
            device=cfg.get("device", None),
        )
        current_checkpoints = list_checkpoints(persister.client, persister.run_id)
        max_step = cfg.cev_max_step if cfg.cev_max_step is not None else max(current_checkpoints)
        available_steps = set(s for s in current_checkpoints if s <= max_step)

        new_steps = available_steps - cached_steps
        if new_steps:
            print(f"Found {len(new_steps)} new checkpoints not in cache: {sorted(new_steps)[:10]}...")
            print("Computing metrics for new checkpoints...")

            # Compute only for new checkpoints
            import pandas as pd

            generative_process = components.get_generative_process()
            model = components.get_predictive_model()

            prepared_sequences = prepare_sequences(
                generative_process,
                run_cfg,
                batch_size=cfg.cev_batch_size,
                seed=cfg.checkpoint_step or max(current_checkpoints),
                use_probs_as_weights=False,
                max_prefix_length=cfg.max_prefix_length,
            )

            # Compute metrics at new checkpoints
            new_metrics_result = compute_all_metrics_at_checkpoints_parallel(
                model,
                prepared_sequences,
                sorted(new_steps),
                persister,
                layers=layer_name,
                min_prefix_length=2 if cfg.skip_bos else 1,
                max_prefix_length=None,
                compute_dims95=cfg.recompute_dims95,
                compute_rmse=cfg.recompute_rmse,
                compute_loss=True,
                max_workers=workers,
                show_progress=True,
            )

            # Convert to DataFrames
            new_metrics = {}
            if layer_name in new_metrics_result.dims95:
                new_metrics["dims95"] = pd.DataFrame([
                    {"step": step, "value": value}
                    for step, value in new_metrics_result.dims95[layer_name]
                ])
            if layer_name in new_metrics_result.rmse:
                new_metrics["overall_rmse"] = pd.DataFrame([
                    {"step": step, "value": overall}
                    for step, overall, _ in new_metrics_result.rmse[layer_name]
                ])
            if layer_name in new_metrics_result.loss:
                new_metrics["loss"] = pd.DataFrame([
                    {"step": step, "value": value}
                    for step, value in new_metrics_result.loss[layer_name]
                ])

            # Merge with cached data
            merged_metrics = merge_metrics(cached_data["metrics"], new_metrics)

            # Optionally compute new CEV checkpoints
            new_cev = None
            if cfg.recompute_cev:
                cev_checkpoints = [s for s in new_steps if s <= (cfg.cev_max_step or float('inf'))]
                if cev_checkpoints:
                    new_cev = compute_cev_at_checkpoints(
                        model,
                        prepared_sequences,
                        sorted(cev_checkpoints),
                        persister,
                        layers=layer_name,
                        max_components=64,
                        min_prefix_length=2 if cfg.skip_bos else 1,
                        max_prefix_length=None,
                    )

            merged_cev = merge_cev_history(cached_data["cev_history"], new_cev)

            # Update cached data
            belief_regression_data = cached_data["belief_regression_data"]
            cev_history = merged_cev
            belief_baselines = cached_data["belief_baselines"]
            metrics = merged_metrics
            random_loss = cached_data["random_loss"]
            entropy_rate = cached_data["entropy_rate"]

            print(f"  Updated cache with {len(new_steps)} new checkpoints")

            # Save updated cache
            save_cache(cfg, {
                "belief_regression_data": belief_regression_data,
                "cev_history": cev_history,
                "belief_baselines": belief_baselines,
                "metrics": metrics,
                "layer_name": layer_name,
                "random_loss": random_loss,
                "entropy_rate": entropy_rate,
            })
        else:
            print("Using cached computation results (no new checkpoints)")
            belief_regression_data = cached_data["belief_regression_data"]
            cev_history = cached_data["cev_history"]
            belief_baselines = cached_data["belief_baselines"]
            metrics = cached_data["metrics"]
            random_loss = cached_data["random_loss"]
            entropy_rate = cached_data["entropy_rate"]
    else:
        # Setup from MLflow and compute everything
        belief_regression_data, cev_history, belief_baselines, metrics, layer_name, random_loss, entropy_rate = (
            compute_all_data(cfg, workers=workers)
        )
        # Save to cache
        save_cache(cfg, {
            "belief_regression_data": belief_regression_data,
            "cev_history": cev_history,
            "belief_baselines": belief_baselines,
            "metrics": metrics,
            "layer_name": layer_name,
            "random_loss": random_loss,
            "entropy_rate": entropy_rate,
        })

    # Compute joint_dims95 from product belief CEV if available
    joint_dims95 = None
    if belief_baselines is not None and "product" in belief_baselines:
        product_cev = belief_baselines["product"]
        indices_above = np.where(product_cev >= 0.95)[0]
        if len(indices_above) > 0:
            joint_dims95 = float(indices_above[0] + 1)
        else:
            joint_dims95 = float(len(product_cev))
        print(f"Joint (product) belief dims@95: {joint_dims95:.0f}")

    # Filter CEV history to preferred checkpoints (downsample if cached data has too many)
    if cev_history is not None:
        original_count = len(next(iter(cev_history.values()))) if cev_history else 0
        cev_history = filter_cev_to_preferred(cev_history, cfg.cev_max_step)
        filtered_count = len(next(iter(cev_history.values()))) if cev_history else 0
        if original_count != filtered_count:
            print(f"Filtered CEV history: {original_count} -> {filtered_count} checkpoints")

    # Generate and save main figure
    print("Creating main figure...")
    fig_main = create_main_figure(
        belief_regression_data=belief_regression_data,
        cev_history=cev_history,
        metrics=metrics,
        layer_name=layer_name,
        gt_dims95=cfg.gt_dims95,
        joint_dims95=joint_dims95,
        cev_max_step=cfg.cev_max_step,
        dims_max_step=cfg.dims_max_step,
        figsize=tuple(cfg.figsize),
        belief_baselines=belief_baselines,
        cev_alpha=cfg.get("cev_alpha", 0.8),
        cmap=cfg.get("cmap", "magma"),
        cmap_start=cfg.get("cmap_start", 0.0),
        cmap_mid=cfg.get("cmap_mid", 0.5),
        cmap_end=cfg.get("cmap_end", 0.9),
    )
    fig_main.savefig(cfg.output_path, format=cfg.output_format, dpi=cfg.dpi)
    print(f"Saved main figure to {cfg.output_path}")
    plt.close(fig_main)

    # Generate and save supplemental figure
    if cfg.supplemental_path is not None:
        print("Creating supplemental figure...")
        fig_supp = create_supplemental_figure(
            metrics=metrics,
            gt_dims95=cfg.gt_dims95,
            rmse_max_step=cfg.rmse_max_step,
            figsize=tuple(cfg.supplemental_figsize),
            entropy_rate=entropy_rate,
        )
        fig_supp.savefig(cfg.supplemental_path, format=cfg.output_format, dpi=cfg.dpi)
        print(f"Saved supplemental figure to {cfg.supplemental_path}")
        plt.close(fig_supp)

    # Generate and save vertical layout figure
    if cfg.get("vertical_path") is not None:
        print("Creating vertical layout figure...")
        fig_vert = create_main_figure_vertical(
            belief_regression_data=belief_regression_data,
            cev_history=cev_history,
            metrics=metrics,
            layer_name=layer_name,
            gt_dims95=cfg.gt_dims95,
            joint_dims95=joint_dims95,
            cev_max_step=cfg.cev_max_step,
            dims_max_step=cfg.dims_max_step,
            figsize=(6.75, 6.0),
            belief_baselines=belief_baselines,
            cev_alpha=cfg.get("cev_alpha", 0.8),
            cmap=cfg.get("cmap", "magma"),
            cmap_start=cfg.get("cmap_start", 0.0),
            cmap_mid=cfg.get("cmap_mid", 0.5),
            cmap_end=cfg.get("cmap_end", 0.9),
        )
        fig_vert.savefig(cfg.vertical_path, format=cfg.output_format, dpi=cfg.dpi)
        print(f"Saved vertical figure to {cfg.vertical_path}")
        plt.close(fig_vert)


def compute_all_data(cfg: FigureGenerationConfig, workers: int = 1):
    """Compute all data needed for the figure (parallel version)."""
    # Setup from MLflow
    run_cfg, components, persister = setup_from_mlflow(
        run_id=cfg.run_id,
        experiment_id=cfg.experiment_id,
        tracking_uri=cfg.tracking_uri,
        registry_uri=cfg.registry_uri,
        device=cfg.get("device", None),
    )

    client = persister.client
    run_id = persister.run_id

    # Get components
    generative_process = components.get_generative_process()
    model = components.get_predictive_model()

    # Discover layer names from metrics
    all_metrics = list_all_metrics(client, run_id)
    layer_names = discover_layer_names(all_metrics)
    print(f"Available layers: {layer_names}")

    # Select layer
    if cfg.layer is not None:
        formatted_layer_name = format_layer_spec(cfg.layer)
        if formatted_layer_name not in layer_names:
            raise ValueError(f"Layer {formatted_layer_name} not found in {layer_names}")
        layer_name = formatted_layer_name
    else:
        non_cat_layers = [l for l in layer_names if l != "Lcat"]
        layer_name = non_cat_layers[-1] if non_cat_layers else layer_names[-1]
    print(f"Using layer: {layer_name}")

    # Get number of factors
    num_factors = get_num_factors_from_config(run_cfg)
    print(f"Number of factors: {num_factors}")

    # Fetch metrics from MLflow
    print("Fetching metrics from MLflow...")
    metrics = fetch_all_metrics_for_figure(client, run_id, layer_name, num_factors, cfg.loss_metric_key)
    print(f"Fetched metrics: {list(metrics.keys())}")

    # Get available checkpoints
    checkpoints = list_checkpoints(client, run_id)
    print(f"Available checkpoints: {len(checkpoints)} (from step {checkpoints[0]} to {checkpoints[-1]})")

    # Select checkpoint for belief regression
    if cfg.checkpoint_step is not None:
        belief_checkpoint = cfg.checkpoint_step
    else:
        belief_checkpoint = checkpoints[-1]
    print(f"Using checkpoint step {belief_checkpoint} for belief regression")

    # Compute belief regression data
    print("Computing belief regression data...")
    prepared_sequences = prepare_sequences(
        generative_process,
        run_cfg,
        batch_size=cfg.cev_batch_size,
        seed=belief_checkpoint,
        use_probs_as_weights=False,
        max_prefix_length=cfg.max_prefix_length,
    )
    print(f"  Generated {prepared_sequences.n_samples} unique prefixes (max_len={cfg.max_prefix_length})")

    persister.load_weights(model, step=belief_checkpoint)
    prepared_activations = get_activations(
        model,
        prepared_sequences,
        layers=layer_name,
        min_prefix_length=2 if cfg.skip_bos else 1,
        max_prefix_length=cfg.max_prefix_length,
    )
    belief_regression_data = compute_belief_regression(prepared_activations)
    print(f"  Computed belief regression for layers: {list(belief_regression_data.keys())}")

    # Compute CEV history if requested
    cev_history = None
    if cfg.recompute_cev:
        print(f"Recomputing CEV checkpoints...")

        # Use preferred checkpoint pattern (0-10 every 1, 20-90 every 10, 500-900 every 100,
        # 1000-5000 every 1000, then every 5000 after that)
        selected_checkpoints = get_preferred_checkpoints(checkpoints, cfg.cev_max_step)

        print(f"  Selected checkpoints: {selected_checkpoints}")
        cev_history = compute_cev_at_checkpoints(
            model,
            prepared_sequences,
            selected_checkpoints,
            persister,
            layers=layer_name,
            max_components=64,
            min_prefix_length=2 if cfg.skip_bos else 1,
            max_prefix_length=None,
        )
        print(f"  Computed CEV for layers: {list(cev_history.keys())}")

    # Compute belief CEV baselines
    belief_baselines = None
    if cfg.recompute_cev:
        print("Computing belief CEV baselines...")
        belief_baselines = compute_belief_cev_baselines(
            prepared_sequences,
            max_components=None,
            min_prefix_length=2 if cfg.skip_bos else 1,
            max_prefix_length=None,
        )
        print(f"  Factored baseline: {len(belief_baselines['factored'])} components")
        print(f"  Product baseline: {len(belief_baselines['product'])} components")

    # Recompute dims@95, RMSE, and loss together at the same checkpoints
    # Using PARALLEL version for speedup
    if cfg.recompute_dims95 or cfg.recompute_rmse:
        import pandas as pd

        metric_checkpoints = checkpoints
        if cfg.cev_max_step is not None:
            metric_checkpoints = [s for s in checkpoints if s <= cfg.cev_max_step]

        print(f"Computing metrics at {len(metric_checkpoints)} checkpoints (workers={workers})...")
        print(f"  Checkpoints: {metric_checkpoints[:5]}...{metric_checkpoints[-3:] if len(metric_checkpoints) > 5 else ''}")

        # Use PARALLEL version
        checkpoint_metrics = compute_all_metrics_at_checkpoints_parallel(
            model,
            prepared_sequences,
            metric_checkpoints,
            persister,
            layers=layer_name,
            min_prefix_length=2 if cfg.skip_bos else 1,
            max_prefix_length=None,
            compute_dims95=cfg.recompute_dims95,
            compute_rmse=cfg.recompute_rmse,
            compute_loss=True,
            max_workers=workers,
            show_progress=True,
        )

        # Update dims@95 metrics
        if cfg.recompute_dims95 and layer_name in checkpoint_metrics.dims95:
            dims95_list = checkpoint_metrics.dims95[layer_name]
            metrics["dims95"] = pd.DataFrame([
                {"step": step, "value": value} for step, value in dims95_list
            ])
            print(f"  Computed dims@95 at {len(dims95_list)} checkpoints")

        # Update RMSE metrics
        if cfg.recompute_rmse and layer_name in checkpoint_metrics.rmse:
            rmse_list = checkpoint_metrics.rmse[layer_name]
            metrics["overall_rmse"] = pd.DataFrame([
                {"step": step, "value": overall} for step, overall, _ in rmse_list
            ])
            num_factors = len(rmse_list[0][2]) if rmse_list else 0
            for factor_idx in range(num_factors):
                metrics[f"factor_{factor_idx}_rmse"] = pd.DataFrame([
                    {"step": step, "value": factor_rmses[factor_idx]}
                    for step, _, factor_rmses in rmse_list
                ])
            print(f"  Computed RMSE at {len(rmse_list)} checkpoints")

        # Update loss metrics
        if checkpoint_metrics.loss:
            metrics["loss"] = pd.DataFrame([
                {"step": step, "value": loss} for step, loss in checkpoint_metrics.loss
            ])
            print(f"  Computed loss at {len(checkpoint_metrics.loss)} checkpoints")

    # Compute reference loss values
    vocab_size = generative_process.vocab_size
    random_loss = float(np.log(vocab_size))
    print(f"Random guesser loss: {random_loss:.4f} (log({vocab_size}))")

    entropy_rate = compute_entropy_rate(
        generative_process,
        run_cfg,
        n_sequences=1000,
        skip_first=5,
        seed=0,
    )
    print(f"Entropy rate (optimal loss): {entropy_rate:.4f}")

    return belief_regression_data, cev_history, belief_baselines, metrics, layer_name, random_loss, entropy_rate


if __name__ == "__main__":
    main()
