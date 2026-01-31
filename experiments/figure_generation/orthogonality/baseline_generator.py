"""Baseline generation for orthogonality analysis.

Computes confidence intervals for orthogonality metrics on randomly
initialized (untrained) models to provide a null distribution baseline
for interpreting trained model orthogonality results.

This module is self-contained within the orthogonality folder and reuses
functions from vary_one_functions.py for data generation and analysis.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import jax
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from vary_one_functions import (
    compute_cev,
    compute_subspace_orthogonality_curve,
    generate_vary_one_beliefs,
    get_activations_for_multiple_layers,
    per_position_center,
)


def create_untrained_model(
    base_config: HookedTransformerConfig,
    seed: int,
    device: str | torch.device,
) -> HookedTransformer:
    """Create HookedTransformer with random weights (same architecture).

    Args:
        base_config: Configuration from a trained model to copy architecture from.
        seed: Random seed for weight initialization.
        device: Device to place model on.

    Returns:
        Newly initialized HookedTransformer with random weights.
    """
    new_config = HookedTransformerConfig(
        n_layers=base_config.n_layers,
        d_model=base_config.d_model,
        n_heads=base_config.n_heads,
        d_head=base_config.d_head,
        n_ctx=base_config.n_ctx,
        d_mlp=base_config.d_mlp,
        d_vocab=base_config.d_vocab,
        act_fn=base_config.act_fn,
        normalization_type=base_config.normalization_type,
        device=str(device),
        seed=seed,
    )
    return HookedTransformer(new_config)


def compute_orthogonality_for_model(
    model: HookedTransformer,
    gp,
    num_factors: int,
    layer_hooks: list[str],
    num_frozen_points: int,
    batch_per_frozen: int,
    sequence_len: int,
    bos_token: int | None,
    data_seed: int,
    device: torch.device | str,
    max_k: int = 20,
) -> dict[str, dict]:
    """Compute orthogonality metrics for a single model across specified layers.

    Args:
        model: HookedTransformer to analyze.
        gp: Generative process for data generation.
        num_factors: Number of factors in the generative process.
        layer_hooks: List of hook names to extract activations from.
        num_frozen_points: Number of frozen configurations per factor.
        batch_per_frozen: Batch size per frozen configuration.
        sequence_len: Length of sequences to generate.
        bos_token: BOS token to prepend (if any).
        data_seed: Base seed for data generation.
        device: Device for torch tensors.
        max_k: Maximum k for orthogonality curves.

    Returns:
        Dict mapping layer_hook -> {orthogonality_results, max_k}
    """
    # Initialize per-layer, per-factor activation storage
    per_layer_factor_data: dict[str, list[list[np.ndarray]]] = {
        layer: [[] for _ in range(num_factors)] for layer in layer_hooks
    }

    # Generate data and collect activations for all layers in one forward pass
    for factor_idx in range(num_factors):
        for fp_idx in range(num_frozen_points):
            # Use consistent keys based on data_seed
            frozen_key = jax.random.PRNGKey(data_seed + factor_idx * 100 + fp_idx)
            sample_key = jax.random.PRNGKey(data_seed + 1000 + factor_idx * 100 + fp_idx)

            data = generate_vary_one_beliefs(
                base_gp=gp,
                vary_factor_idx=factor_idx,
                batch_size=batch_per_frozen,
                sequence_len=sequence_len,
                frozen_key=frozen_key,
                sample_key=sample_key,
            )

            # Get activations for ALL layers in one forward pass
            layer_acts = get_activations_for_multiple_layers(
                data.observations, model, layer_hooks, bos_token=bos_token, device=device
            )

            # Process and store activations for each layer
            for layer_hook in layer_hooks:
                acts = layer_acts[layer_hook]
                # Remove BOS position (position 0) and per-position center
                acts = acts[:, 1:, :]
                acts_centered = per_position_center(acts)
                acts_flat = acts_centered.reshape(-1, acts_centered.shape[-1])
                per_layer_factor_data[layer_hook][factor_idx].append(acts_flat)

    # Compute orthogonality for each layer
    results_per_layer = {}

    for layer_hook in layer_hooks:
        per_factor_pcs = []
        per_factor_var_ratios = []

        for factor_idx in range(num_factors):
            # Combine across frozen points
            factor_combined = np.concatenate(
                per_layer_factor_data[layer_hook][factor_idx], axis=0
            )

            # Compute CEV and extract principal components
            cev_i, Vt_i, var_ratios_i = compute_cev(factor_combined, return_components=True)
            per_factor_pcs.append(Vt_i)
            per_factor_var_ratios.append(var_ratios_i)

        # Compute orthogonality curves for all factor pairs
        actual_max_k = min(max_k, min(pc.shape[0] for pc in per_factor_pcs))

        orthogonality_results = {}
        for i in range(num_factors):
            for j in range(i, num_factors):  # Include diagonal
                pair_key = f"F{i},F{j}"
                orthogonality_results[pair_key] = compute_subspace_orthogonality_curve(
                    per_factor_pcs[i], per_factor_var_ratios[i],
                    per_factor_pcs[j], per_factor_var_ratios[j],
                    max_components=actual_max_k,
                )

        results_per_layer[layer_hook] = {
            "orthogonality_results": orthogonality_results,
            "max_k": actual_max_k,
        }

    return results_per_layer


def aggregate_results(
    per_model_results: list[dict],
    num_factors: int,
    layer_hooks: list[str],
) -> dict[str, dict]:
    """Aggregate orthogonality results across models, per layer.

    Args:
        per_model_results: List of results from compute_orthogonality_for_model.
            Each element is {layer_hook: {orthogonality_results, max_k}}
        num_factors: Number of factors.
        layer_hooks: List of layer hooks.

    Returns:
        Dict mapping layer_hook -> aggregated statistics for each factor pair.
    """
    # Match the keys returned by compute_subspace_orthogonality_curve in vary_one_functions.py
    metrics = ["normalized_overlap", "weighted_overlap", "weighted_overlap_svd"]

    aggregated_per_layer = {}

    for layer_hook in layer_hooks:
        # Get sample result to find pair keys
        sample_result = per_model_results[0][layer_hook]["orthogonality_results"]
        pair_keys = list(sample_result.keys())

        aggregated = {}
        for pair_key in pair_keys:
            aggregated[pair_key] = {}

            for metric in metrics:
                # Stack values across all models: [n_models, max_k]
                values = np.array([
                    r[layer_hook]["orthogonality_results"][pair_key][metric]
                    for r in per_model_results
                ])

                aggregated[pair_key][metric] = {
                    "mean": np.mean(values, axis=0),
                    "std": np.std(values, axis=0),
                    "percentile_5": np.percentile(values, 5, axis=0),
                    "percentile_25": np.percentile(values, 25, axis=0),
                    "percentile_50": np.percentile(values, 50, axis=0),
                    "percentile_75": np.percentile(values, 75, axis=0),
                    "percentile_95": np.percentile(values, 95, axis=0),
                }

        aggregated_per_layer[layer_hook] = aggregated

    return aggregated_per_layer


def generate_baseline(
    gp,
    base_config: HookedTransformerConfig,
    layer_hooks: list[str],
    num_factors: int,
    state_dims: list[int],
    sequence_len: int,
    bos_token: int | None,
    device: torch.device | str,
    run_id: str,
    experiment_id: str | None = None,
    n_models: int = 5,
    num_frozen_points: int = 10,
    batch_per_frozen: int = 200,
    data_seed: int = 42,
    max_k: int = 20,
) -> dict:
    """Generate baseline data for the specified layer(s).

    Args:
        gp: Generative process for data generation.
        base_config: HookedTransformerConfig to copy architecture from.
        layer_hooks: List of layers to analyze (can be one or many).
        num_factors: Number of factors in the generative process.
        state_dims: State dimension per factor.
        sequence_len: Length of sequences to generate.
        bos_token: BOS token to prepend (if any).
        device: Device for torch tensors.
        run_id: MLflow run ID (for metadata).
        experiment_id: MLflow experiment ID (for metadata).
        n_models: Number of random initializations.
        num_frozen_points: Number of frozen configurations per factor.
        batch_per_frozen: Batch size per frozen configuration.
        data_seed: Base seed for data generation (same across all models).
        max_k: Maximum k for orthogonality curves.

    Returns:
        Baseline data dict with metadata and per_layer results.
    """
    per_model_results = []
    seeds = list(range(1, n_models + 1))

    for seed in tqdm(seeds, desc="Random models"):
        # Create untrained model with this seed
        model = create_untrained_model(base_config, seed=seed, device=device)
        model.eval()

        # Compute orthogonality metrics for all layers
        result = compute_orthogonality_for_model(
            model=model,
            gp=gp,
            num_factors=num_factors,
            layer_hooks=layer_hooks,
            num_frozen_points=num_frozen_points,
            batch_per_frozen=batch_per_frozen,
            sequence_len=sequence_len,
            bos_token=bos_token,
            data_seed=data_seed,
            device=device,
            max_k=max_k,
        )
        # Add seed to the result
        per_model_results.append({"seed": seed, **result})

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate results per layer
    aggregated_per_layer = aggregate_results(per_model_results, num_factors, layer_hooks)

    # Get max_k from first result, first layer
    actual_max_k = per_model_results[0][layer_hooks[0]]["max_k"]

    # Build output structure (per-layer)
    per_layer_data = {}
    for layer_hook in layer_hooks:
        per_layer_data[layer_hook] = {
            "per_model_results": [
                {
                    "seed": r["seed"],
                    "orthogonality_results": r[layer_hook]["orthogonality_results"],
                    "max_k": r[layer_hook]["max_k"],
                }
                for r in per_model_results
            ],
            "aggregated": aggregated_per_layer[layer_hook],
        }

    results = {
        "metadata": {
            "n_models": n_models,
            "seeds": seeds,
            "architecture": {
                "n_layers": base_config.n_layers,
                "d_model": base_config.d_model,
            },
            "run_id": run_id,
            "experiment_id": experiment_id,
            "layers_analyzed": layer_hooks,
            "num_factors": num_factors,
            "state_dims": state_dims,
            "max_k": actual_max_k,
            "data_seed": data_seed,
            "num_frozen_points": num_frozen_points,
            "num_samples_per_frozen": batch_per_frozen,
            "sequence_len": sequence_len,
            "bos_token": bos_token,
        },
        "per_layer": per_layer_data,
    }

    return results


def get_default_baseline_path(run_id: str) -> Path:
    """Get expected baseline path for a run_id.

    Args:
        run_id: MLflow run ID.

    Returns:
        Path to baseline file (in the orthogonality folder).
    """
    return Path(__file__).parent / f"baseline_{run_id}.pkl"


def save_baseline(data: dict, path: Path) -> None:
    """Save baseline to pickle file.

    Args:
        data: Baseline data dict from generate_baseline().
        path: Path to save to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
