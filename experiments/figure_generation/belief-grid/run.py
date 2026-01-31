"""Main entry point for figure generation from MLflow training runs.

Usage (from this directory):
- `uv run python run.py --config-name=test`
- GPU override: `CUDA_VISIBLE_DEVICES=0 uv run python run.py --config-name=test +device=cuda`
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
from analysis import (
    prepare_sequences,
    get_activations,
    compute_belief_regression,
    compute_cev_at_checkpoints,
    compute_dims95_at_checkpoints,
    compute_rmse_at_checkpoints,
    compute_all_metrics_at_checkpoints,
    compute_entropy_rate,
    compute_belief_cev_baselines,
)
from plotting import create_composite_figure, create_main_figure, create_supplemental_figure, create_main_figure_vertical

# Suppress Databricks SDK logging
logging.getLogger("databricks.sdk").setLevel(logging.WARNING)


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
    max_prefix_length: int | None = None  # Max prefix length to include (e.g., 10 for first 10 positions)

    # Output settings
    output_path: str = "figure.png"  # Main figure output
    supplemental_path: str | None = "figure_supplemental.png"  # Supplemental figure output (None to skip)
    vertical_path: str | None = None  # Vertical layout figure output (None to skip)
    output_format: str = "png"  # png, pdf, svg
    dpi: int = 150
    figsize: list[float] = field(default_factory=lambda: [6.75, 4.0])  # Main figure size
    supplemental_figsize: list[float] = field(default_factory=lambda: [6.75, 2.5])  # Supplemental figure size

    # Caching
    use_cache: bool = True  # Load/save computed data to cache
    cache_dir: str = ".cache"  # Directory to store cache files


def get_cache_key(cfg: FigureGenerationConfig) -> str:
    """Compute a cache key from config options that affect computation."""
    # Only include options that affect the computed data (not output settings)
    # Note: cfg is typically an OmegaConf DictConfig at runtime (not a dataclass),
    # so use cfg.get(...) for optional keys.
    cache_relevant = {
        "run_id": cfg.run_id,
        "layer": cfg.layer,
        "cev_max_step": cfg.cev_max_step,
        "cev_n_checkpoints": cfg.cev_n_checkpoints,
        "cev_batch_size": cfg.cev_batch_size,
        "rmse_n_checkpoints": cfg.rmse_n_checkpoints,
        "skip_bos": cfg.skip_bos,
        "max_prefix_length": cfg.max_prefix_length,
        "checkpoint_step": cfg.checkpoint_step,
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


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: FigureGenerationConfig) -> None:
    """Generate composite figure from MLflow training run."""
    print(f"Loading run {cfg.run_id}...")

    # Check for cached data first
    cached_data = load_cache(cfg)
    if cached_data is not None:
        print("Using cached computation results")
        belief_regression_data = cached_data["belief_regression_data"]
        cev_history = cached_data["cev_history"]
        belief_baselines = cached_data["belief_baselines"]
        metrics = cached_data["metrics"]
        layer_name = cached_data["layer_name"]
        random_loss = cached_data["random_loss"]
        entropy_rate = cached_data["entropy_rate"]
    else:
        # Setup from MLflow and compute everything
        belief_regression_data, cev_history, belief_baselines, metrics, layer_name, random_loss, entropy_rate = (
            compute_all_data(cfg)
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
        # Find number of components to reach 95% variance
        indices_above = np.where(product_cev >= 0.95)[0]
        if len(indices_above) > 0:
            joint_dims95 = float(indices_above[0] + 1)  # +1 because index is 0-based
        else:
            joint_dims95 = float(len(product_cev))  # All components needed
        print(f"Joint (product) belief dims@95: {joint_dims95:.0f}")

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
            figsize=(6.75, 6.0),  # Taller for vertical layout
            belief_baselines=belief_baselines,
        )
        fig_vert.savefig(cfg.vertical_path, format=cfg.output_format, dpi=cfg.dpi)
        print(f"Saved vertical figure to {cfg.vertical_path}")
        plt.close(fig_vert)


def compute_all_data(cfg: FigureGenerationConfig):
    """Compute all data needed for the figure."""
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
        # Auto-select last non-Lcat layer
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
        belief_checkpoint = checkpoints[-1]  # Latest
    print(f"Using checkpoint step {belief_checkpoint} for belief regression")

    # Compute belief regression data
    print("Computing belief regression data...")
    prepared_sequences = prepare_sequences(
        generative_process,
        run_cfg,
        batch_size=cfg.cev_batch_size,
        seed=belief_checkpoint,  # Use checkpoint as seed for reproducibility
        use_probs_as_weights=False,  # Match training: uniform weights
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
        print(f"Recomputing CEV at {cfg.cev_n_checkpoints} checkpoints...")

        # Filter by cev_max_step first, then select evenly spaced
        cev_checkpoints = checkpoints
        if cfg.cev_max_step is not None:
            cev_checkpoints = [s for s in checkpoints if s <= cfg.cev_max_step]
        selected_checkpoints = select_evenly_spaced_checkpoints(cev_checkpoints, cfg.cev_n_checkpoints)

        print(f"  Selected checkpoints: {selected_checkpoints}")
        cev_history = compute_cev_at_checkpoints(
            model,
            prepared_sequences,
            selected_checkpoints,
            persister,
            layers=layer_name,
            max_components=64,
            min_prefix_length=2 if cfg.skip_bos else 1,  # Skip BOS
            max_prefix_length=None,  # Full context window for analysis
        )
        print(f"  Computed CEV for layers: {list(cev_history.keys())}")

    # Compute belief CEV baselines (factored and product ground truth)
    belief_baselines = None
    if cfg.recompute_cev:
        print("Computing belief CEV baselines...")
        belief_baselines = compute_belief_cev_baselines(
            prepared_sequences,
            max_components=None,  # No truncation - compute all components
            min_prefix_length=2 if cfg.skip_bos else 1,  # Skip BOS
            max_prefix_length=None,  # Full context window
        )
        print(f"  Factored baseline: {len(belief_baselines['factored'])} components")
        print(f"  Product baseline: {len(belief_baselines['product'])} components")

    # Recompute dims@95, RMSE, and loss together at the same checkpoints
    if cfg.recompute_dims95 or cfg.recompute_rmse:
        import pandas as pd

        # Use ALL checkpoints up to cev_max_step for dense early coverage
        metric_checkpoints = checkpoints
        if cfg.cev_max_step is not None:
            metric_checkpoints = [s for s in checkpoints if s <= cfg.cev_max_step]

        print(f"Computing metrics at {len(metric_checkpoints)} checkpoints (all up to step {cfg.cev_max_step})...")
        print(f"  Checkpoints: {metric_checkpoints[:5]}...{metric_checkpoints[-3:] if len(metric_checkpoints) > 5 else ''}")

        checkpoint_metrics = compute_all_metrics_at_checkpoints(
            model,
            prepared_sequences,
            metric_checkpoints,
            persister,
            layers=layer_name,
            min_prefix_length=2 if cfg.skip_bos else 1,
            max_prefix_length=None,
            compute_dims95=cfg.recompute_dims95,
            compute_rmse=cfg.recompute_rmse,
            compute_loss=True,  # Always compute loss at same checkpoints
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

        # Update loss metrics (computed at same checkpoints - no merging needed!)
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
