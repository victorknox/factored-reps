"""
Parallel Noise Sweep Analysis Utility Functions

Shadows functions.py and adds parallel processing capabilities for multi-GPU execution.
Re-exports all functions from the sequential version and adds parallel-specific functions.

New functions:
- process_single_checkpoint: Process metrics for one checkpoint
- process_noise_level: Process all checkpoints for one noise level on assigned GPU
- compute_factor_metrics_concat: Compute regression metrics using concatenated beliefs
  (matches activation tracker's concat_belief_states=True approach)
"""

# Re-export all functions from sequential version
from functions import (
    compute_joint_belief_states,
    compute_metrics_from_variance_ratios,
    compute_variance_ratios,
    compute_variance_ratios_torch,
    get_available_checkpoints,
    refresh_databricks_token,
    setup_from_mlflow,
    to_numpy,
    validate_data_consistency,
)

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax.numpy as jnp
import mlflow
import numpy as np
import torch

from fwh_core.analysis.linear_regression import layer_linear_regression
from fwh_core.persistence.mlflow_persister import MLFlowPersister


def compute_factor_metrics_concat(
    activations_flat: np.ndarray,
    belief_states_for_regression: tuple,
    weights: jnp.ndarray,
) -> tuple[float, float, list[float], list[float]]:
    """Compute regression metrics using concatenated belief states.

    This matches the activation tracker's `concat_belief_states=True` approach:
    1. Concatenate all factor beliefs into one target matrix
    2. Run one regression call
    3. Split predictions to compute per-factor metrics

    Args:
        activations_flat: Flattened activations [n_samples, d_model]
        belief_states_for_regression: Tuple of per-factor belief states [n_samples, state_dim_i]
        weights: Sample weights for regression

    Returns:
        Tuple of (r2_overall, rmse_overall, r2_factors, rmse_factors)
    """
    # Flatten belief states if needed and convert to JAX arrays
    belief_states_flat = tuple(
        jnp.array(b.reshape(-1, b.shape[-1]) if b.ndim == 3 else b)
        for b in belief_states_for_regression
    )

    # Use layer_linear_regression with concat_belief_states=True
    # This does concatenated regression and splits results for per-factor metrics
    scalars, arrays = layer_linear_regression(
        layer_activations=jnp.array(activations_flat),
        weights=weights,
        belief_states=belief_states_flat,
        concat_belief_states=True,
        compute_subspace_orthogonality=False,
        use_svd=True,
        fit_intercept=True,
    )

    # Extract overall metrics from concatenated regression (key suffix "Fcat")
    r2_overall = scalars.get("r2/Fcat", 0.0)
    rmse_overall = scalars.get("rmse/Fcat", 0.0)

    # Extract per-factor metrics (key suffix "F0", "F1", etc.)
    num_factors = len(belief_states_for_regression)
    r2_factors = [scalars.get(f"r2/F{i}", 0.0) for i in range(num_factors)]
    rmse_factors = [scalars.get(f"rmse/F{i}", 0.0) for i in range(num_factors)]

    return r2_overall, rmse_overall, r2_factors, rmse_factors


def process_single_checkpoint(
    step: int,
    persister: MLFlowPersister,
    inputs_torch: torch.Tensor,
    labels_torch: torch.Tensor,
    layers_to_analyze: list[str],
    belief_states_for_regression: tuple,
    bos_token: int | None,
    pca_method: str = "full",
    truncated_n_components: int = 100,
) -> dict | None:
    """Process a single checkpoint and return metrics for all layers.

    Uses concatenated belief regression (matching activation tracker's
    concat_belief_states=True approach): one regression on concatenated
    factors, then split for per-factor metrics.

    Args:
        step: Checkpoint step number
        persister: MLFlowPersister for loading model
        inputs_torch: Input tensor [batch, seq]
        labels_torch: Label tensor [batch, seq]
        layers_to_analyze: List of layer names to analyze
        belief_states_for_regression: Tuple of belief state arrays
        bos_token: BOS token ID (or None)
        pca_method: "full" or "truncated"
        truncated_n_components: Number of components for truncated PCA

    Returns:
        Dict with keys: step, loss, layer_data (or None if failed)
    """
    try:
        model = persister.load_model(step=step)
        model.eval()
    except Exception as e:
        return {"step": step, "error": str(e), "skipped": True}

    device = next(model.parameters()).device

    # Ensure inputs are on correct device
    if inputs_torch.device != device:
        inputs_torch = inputs_torch.to(device)
        labels_torch = labels_torch.to(device)

    with torch.no_grad():
        logits, cache = model.run_with_cache(inputs_torch, names_filter=layers_to_analyze)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels_torch.reshape(-1)
    ).item()

    # Check if all layers are available
    all_layers_available = all(layer in cache for layer in layers_to_analyze)
    if not all_layers_available:
        missing = [layer for layer in layers_to_analyze if layer not in cache]
        del model, cache
        return {"step": step, "error": f"missing layers {missing}", "skipped": True}

    # Collect data for all layers
    layer_data = {}
    for layer in layers_to_analyze:
        activations = cache[layer]
        if bos_token is not None:
            activations = activations[:, 1:, :]

        if pca_method == "truncated":
            variance_ratios = compute_variance_ratios_torch(
                activations, n_components=truncated_n_components
            )
        else:
            variance_ratios = compute_variance_ratios(activations.cpu().numpy())

        # Compute belief regression metrics using concatenated approach
        activations_np = activations.cpu().numpy()
        batch_size_act, seq_len_act, d_model = activations_np.shape
        activations_flat = activations_np.reshape(-1, d_model)
        weights = jnp.ones(activations_flat.shape[0]) / activations_flat.shape[0]

        # Use concatenated belief regression (matches activation tracker)
        r2_overall, rmse_overall, r2_factors, rmse_factors = compute_factor_metrics_concat(
            activations_flat, belief_states_for_regression, weights
        )

        layer_data[layer] = {
            "variance_ratios": variance_ratios,
            "r2_overall": r2_overall,
            "rmse_overall": rmse_overall,
            "r2_factors": r2_factors,
            "rmse_factors": rmse_factors,
        }

    del model, cache

    return {
        "step": step,
        "loss": loss,
        "layer_data": layer_data,
        "skipped": False,
    }


def process_noise_level(args: tuple) -> tuple[str, dict, dict | None]:
    """Process all checkpoints for one noise level on assigned GPU.

    This function is designed to be called from ProcessPoolExecutor.
    It sets up its own GPU device and MLflow client.

    IMPORTANT: This function uses lazy imports to ensure CUDA_VISIBLE_DEVICES
    is set before any CUDA libraries are loaded.

    Args:
        args: Tuple of (noise_level, gpu_id, config_dict) where config_dict contains:
            - run_id: MLflow run ID
            - experiment_id: MLflow experiment ID
            - layers_to_analyze: List of layer names
            - belief_states_for_regression: Tuple of belief arrays
            - inputs: Input array (will be converted to torch)
            - labels: Label array (will be converted to torch)
            - bos_token: BOS token ID or None
            - pca_method: "full" or "truncated"
            - truncated_n_components: int
            - num_factors: int
            - cached_steps: Set of already processed steps
            - parallel_checkpoints: Number of checkpoints to process in parallel (0=sequential)

    Returns:
        Tuple of (noise_level, results_dict, optimal_loss_or_None)
        results_dict maps layer -> {steps, losses, variance_ratios, r2_overall, ...}
    """
    import os
    noise, gpu_id, config = args

    # CRITICAL: Set GPU BEFORE importing any CUDA libraries
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now do lazy imports of CUDA-dependent libraries
    import mlflow
    import numpy as np
    import torch

    from functions import get_available_checkpoints, refresh_databricks_token
    from fwh_core.persistence.mlflow_persister import MLFlowPersister

    # Refresh token for this worker process (no-op for local MLflow)
    refresh_databricks_token()

    # Setup MLflow client
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # Extract config
    run_id = config["run_id"]
    experiment_id = config["experiment_id"]
    layers_to_analyze = config["layers_to_analyze"]
    belief_states_for_regression = config["belief_states_for_regression"]
    inputs = config["inputs"]
    labels = config["labels"]
    bos_token = config["bos_token"]
    pca_method = config.get("pca_method", "full")
    truncated_n_components = config.get("truncated_n_components", 100)
    num_factors = config["num_factors"]
    cached_steps = config.get("cached_steps", set())
    parallel_checkpoints = config.get("parallel_checkpoints", 0)

    # Create persister
    persister = MLFlowPersister(
        experiment_id=experiment_id,
        run_id=run_id,
        tracking_uri=tracking_uri,
        registry_uri=os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri),
    )

    # Get available checkpoints
    all_checkpoints = set(get_available_checkpoints(persister))
    new_checkpoints = sorted(all_checkpoints - cached_steps)

    if not new_checkpoints:
        return noise, {}, None

    # Get optimal loss
    run_data = client.get_run(run_id)
    optimal_loss = float(run_data.data.params.get("optimal_loss/average", 0))

    # Initialize results structure
    results = {
        layer: {
            "steps": [],
            "losses": [],
            "variance_ratios": [],
            "r2_overall": [],
            "rmse_overall": [],
            "r2_factors": [[] for _ in range(num_factors)],
            "rmse_factors": [[] for _ in range(num_factors)],
        }
        for layer in layers_to_analyze
    }

    # Convert inputs to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs_torch = (
        torch.tensor(np.array(inputs), dtype=torch.long, device=device)
        if not isinstance(inputs, torch.Tensor)
        else inputs.to(device=device, dtype=torch.long)
    )
    labels_torch = (
        torch.tensor(np.array(labels), dtype=torch.long, device=device)
        if not isinstance(labels, torch.Tensor)
        else labels.to(device=device, dtype=torch.long)
    )

    # Process checkpoints
    if parallel_checkpoints > 1:
        # Process checkpoints in parallel batches on same GPU
        # Note: This requires careful memory management
        def process_step(step):
            return process_single_checkpoint(
                step=step,
                persister=persister,
                inputs_torch=inputs_torch,
                labels_torch=labels_torch,
                layers_to_analyze=layers_to_analyze,
                belief_states_for_regression=belief_states_for_regression,
                bos_token=bos_token,
                pca_method=pca_method,
                truncated_n_components=truncated_n_components,
            )

        with ThreadPoolExecutor(max_workers=parallel_checkpoints) as executor:
            checkpoint_results = list(executor.map(process_step, new_checkpoints))
    else:
        # Process checkpoints sequentially
        checkpoint_results = []
        for step in new_checkpoints:
            result = process_single_checkpoint(
                step=step,
                persister=persister,
                inputs_torch=inputs_torch,
                labels_torch=labels_torch,
                layers_to_analyze=layers_to_analyze,
                belief_states_for_regression=belief_states_for_regression,
                bos_token=bos_token,
                pca_method=pca_method,
                truncated_n_components=truncated_n_components,
            )
            checkpoint_results.append(result)

    # Aggregate results
    for result in checkpoint_results:
        if result is None or result.get("skipped", False):
            continue

        step = result["step"]
        loss = result["loss"]
        layer_data = result["layer_data"]

        for layer in layers_to_analyze:
            results[layer]["steps"].append(step)
            results[layer]["losses"].append(loss)
            results[layer]["variance_ratios"].append(layer_data[layer]["variance_ratios"])
            results[layer]["r2_overall"].append(layer_data[layer]["r2_overall"])
            results[layer]["rmse_overall"].append(layer_data[layer]["rmse_overall"])
            for factor_idx in range(num_factors):
                results[layer]["r2_factors"][factor_idx].append(
                    layer_data[layer]["r2_factors"][factor_idx]
                )
                results[layer]["rmse_factors"][factor_idx].append(
                    layer_data[layer]["rmse_factors"][factor_idx]
                )

    # Sort results by step
    for layer in layers_to_analyze:
        if results[layer]["steps"]:
            sorted_indices = np.argsort(results[layer]["steps"])
            results[layer]["steps"] = [results[layer]["steps"][i] for i in sorted_indices]
            results[layer]["losses"] = [results[layer]["losses"][i] for i in sorted_indices]
            results[layer]["variance_ratios"] = [
                results[layer]["variance_ratios"][i] for i in sorted_indices
            ]
            results[layer]["r2_overall"] = [results[layer]["r2_overall"][i] for i in sorted_indices]
            results[layer]["rmse_overall"] = [
                results[layer]["rmse_overall"][i] for i in sorted_indices
            ]
            for factor_idx in range(num_factors):
                results[layer]["r2_factors"][factor_idx] = [
                    results[layer]["r2_factors"][factor_idx][i] for i in sorted_indices
                ]
                results[layer]["rmse_factors"][factor_idx] = [
                    results[layer]["rmse_factors"][factor_idx][i] for i in sorted_indices
                ]

    return noise, results, optimal_loss


def merge_results(
    existing_results: dict,
    new_results: dict,
    layers_to_analyze: list[str],
    num_factors: int,
) -> dict:
    """Merge new results into existing results dictionary.

    Args:
        existing_results: Existing raw_results dict (may be empty)
        new_results: Dict mapping noise -> layer -> {steps, losses, ...}
        layers_to_analyze: List of layer names
        num_factors: Number of factors

    Returns:
        Merged results dictionary
    """
    for noise, noise_results in new_results.items():
        if not noise_results:
            continue

        if noise not in existing_results:
            existing_results[noise] = {
                layer: {
                    "steps": [],
                    "losses": [],
                    "variance_ratios": [],
                    "r2_overall": [],
                    "rmse_overall": [],
                    "r2_factors": [[] for _ in range(num_factors)],
                    "rmse_factors": [[] for _ in range(num_factors)],
                }
                for layer in layers_to_analyze
            }

        for layer in layers_to_analyze:
            if layer not in noise_results:
                continue
            layer_results = noise_results[layer]
            if not layer_results["steps"]:
                continue

            # Extend existing lists
            existing_results[noise][layer]["steps"].extend(layer_results["steps"])
            existing_results[noise][layer]["losses"].extend(layer_results["losses"])
            existing_results[noise][layer]["variance_ratios"].extend(
                layer_results["variance_ratios"]
            )
            existing_results[noise][layer]["r2_overall"].extend(layer_results["r2_overall"])
            existing_results[noise][layer]["rmse_overall"].extend(layer_results["rmse_overall"])
            for factor_idx in range(num_factors):
                existing_results[noise][layer]["r2_factors"][factor_idx].extend(
                    layer_results["r2_factors"][factor_idx]
                )
                existing_results[noise][layer]["rmse_factors"][factor_idx].extend(
                    layer_results["rmse_factors"][factor_idx]
                )

        # Re-sort by step after merging
        for layer in layers_to_analyze:
            if existing_results[noise][layer]["steps"]:
                sorted_indices = np.argsort(existing_results[noise][layer]["steps"])
                existing_results[noise][layer]["steps"] = [
                    existing_results[noise][layer]["steps"][i] for i in sorted_indices
                ]
                existing_results[noise][layer]["losses"] = [
                    existing_results[noise][layer]["losses"][i] for i in sorted_indices
                ]
                existing_results[noise][layer]["variance_ratios"] = [
                    existing_results[noise][layer]["variance_ratios"][i] for i in sorted_indices
                ]
                existing_results[noise][layer]["r2_overall"] = [
                    existing_results[noise][layer]["r2_overall"][i] for i in sorted_indices
                ]
                existing_results[noise][layer]["rmse_overall"] = [
                    existing_results[noise][layer]["rmse_overall"][i] for i in sorted_indices
                ]
                for factor_idx in range(num_factors):
                    existing_results[noise][layer]["r2_factors"][factor_idx] = [
                        existing_results[noise][layer]["r2_factors"][factor_idx][i]
                        for i in sorted_indices
                    ]
                    existing_results[noise][layer]["rmse_factors"][factor_idx] = [
                        existing_results[noise][layer]["rmse_factors"][factor_idx][i]
                        for i in sorted_indices
                    ]

    return existing_results


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs.

    Respects CUDA_VISIBLE_DEVICES if set, otherwise detects all GPUs.

    Returns:
        List of GPU IDs
    """
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_gpus:
        return [int(g.strip()) for g in visible_gpus.split(",") if g.strip()]
    else:
        try:
            import torch
            return list(range(torch.cuda.device_count()))
        except Exception:
            return [0]


# Export all functions
__all__ = [
    # From functions.py
    "compute_joint_belief_states",
    "compute_metrics_from_variance_ratios",
    "compute_variance_ratios",
    "compute_variance_ratios_torch",
    "get_available_checkpoints",
    "refresh_databricks_token",
    "setup_from_mlflow",
    "to_numpy",
    "validate_data_consistency",
    # New parallel functions
    "compute_factor_metrics_concat",
    "process_single_checkpoint",
    "process_noise_level",
    "merge_results",
    "get_available_gpus",
]
