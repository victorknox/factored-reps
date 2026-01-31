"""Parallel analysis utilities for belief-grid figure generation.

This module shadows analysis.py and adds optimized parallel processing:
1. Combined forward passes (get activations and loss in one pass) - ~2x speedup
2. Multi-GPU parallel checkpoint processing via ProcessPoolExecutor - ~Nx speedup

Re-exports all functions from the sequential version for backwards compatibility.
New/overridden functions are optimized for parallel execution.

Usage:
    # In run_parallel.py, import from this module instead of analysis
    from analysis_parallel import (
        prepare_sequences,  # Re-exported from analysis.py
        compute_all_metrics_at_checkpoints_parallel,  # New optimized version
        ...
    )

Multi-GPU parallelization:
    When workers > 1 and multiple GPUs are available, checkpoints are distributed
    across GPUs using ProcessPoolExecutor with spawn context. Each worker process
    loads the model independently and processes its assigned checkpoints.

    Example:
        CUDA_VISIBLE_DEVICES=0,1,2,3 python run_parallel.py workers=4
"""

from __future__ import annotations

# Re-export everything from sequential version
from analysis import (
    # Dataclasses
    PreparedSequences,
    PreparedActivations,
    CheckpointMetrics,
    # Core functions
    prepare_sequences,
    get_activations,
    compute_belief_regression,
    compute_cev,
    compute_dims95,
    compute_loss,
    compute_entropy_rate,
    compute_belief_cev_baselines,
    compute_product_beliefs,
    # Checkpoint-level functions (kept for backwards compat)
    compute_belief_regression_at_checkpoint,
    compute_cev_at_checkpoint,
    compute_cev_at_checkpoints,
    compute_dims95_at_checkpoint,
    compute_dims95_at_checkpoints,
    compute_rmse_at_checkpoints,
    compute_loss_at_checkpoint,
    compute_all_metrics_at_checkpoints,
    # Constants
    DEFAULT_RCOND_VALUES,
)

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from fwh_core.analysis.metric_keys import format_layer_spec
from fwh_core.analysis.pca import layer_pca_analysis
from fwh_core.analysis.linear_regression import layer_linear_regression
from fwh_core.persistence.mlflow_persister import MLFlowPersister


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs.

    Respects CUDA_VISIBLE_DEVICES if set, otherwise detects all GPUs.

    Returns:
        List of GPU IDs (indices within CUDA_VISIBLE_DEVICES, not physical IDs).
    """
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_gpus:
        return list(range(len([g.strip() for g in visible_gpus.split(",") if g.strip()])))
    else:
        try:
            return list(range(torch.cuda.device_count()))
        except Exception:
            return [0]


@dataclass
class SingleCheckpointResult:
    """Result from processing a single checkpoint."""
    step: int
    loss: float | None = None
    dims95: dict[str, int] | None = None
    rmse_overall: dict[str, float] | None = None
    rmse_factors: dict[str, list[float]] | None = None
    error: str | None = None


def get_activations_with_loss(
    model: Any,
    prepared_sequences: PreparedSequences,
    layers: str | list[str],
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> tuple[PreparedActivations, float]:
    """Run model once and extract both activations and loss.

    This is the key optimization - instead of running two separate forward passes
    (one for activations, one for loss), we combine them into a single pass.

    Args:
        model: The predictive model.
        prepared_sequences: Output from prepare_sequences.
        layers: Layer name(s) to extract.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Tuple of (PreparedActivations, loss_value).
    """
    model.eval()

    # Normalize to list
    if isinstance(layers, str):
        layers = [layers]

    # Convert JAX array to PyTorch tensor on the model's device
    inputs_torch = torch.tensor(
        np.array(prepared_sequences.inputs),
        device=model.cfg.device,
        dtype=torch.long,
    )

    with torch.no_grad():
        # Single forward pass with cache
        logits, cache = model.run_with_cache(inputs_torch, return_type="logits")

    # --- Compute loss from logits ---
    # Filter by prefix length bounds for loss computation
    valid_indices_for_loss = [
        i
        for i, length in enumerate(prepared_sequences.prefix_lengths)
        if length >= min_prefix_length and (max_prefix_length is None or length <= max_prefix_length)
    ]

    losses = []
    weights = []

    for i in valid_indices_for_loss:
        length = prepared_sequences.prefix_lengths[i]
        if length < 2:
            continue  # Need at least 2 tokens for loss computation

        # For positions 0..L-2, predict tokens 1..L-1
        prefix_logits = logits[i, :length-1, :]  # [L-1, vocab]
        prefix_labels = inputs_torch[i, 1:length]  # [L-1]

        # Cross-entropy loss for this prefix
        loss = F.cross_entropy(prefix_logits, prefix_labels, reduction='mean')
        losses.append(loss.item())
        weights.append(float(prepared_sequences.weights[i]))

    if losses:
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()
        weighted_loss = float(np.sum(np.array(losses) * weights_arr))
    else:
        weighted_loss = float('nan')

    # --- Extract activations from cache ---
    raw_activations = {}
    for name, acts in cache.items():
        formatted_name = format_layer_spec(name)
        if name in layers or formatted_name in layers:
            raw_activations[name] = acts

    if not raw_activations:
        available = [format_layer_spec(name) for name in cache.keys()]
        raise ValueError(
            f"None of the requested layers {layers} found in cache. "
            f"Available layers: {available}"
        )

    # Filter by prefix length bounds for activations
    valid_indices = [
        i
        for i, length in enumerate(prepared_sequences.prefix_lengths)
        if length >= min_prefix_length and (max_prefix_length is None or length <= max_prefix_length)
    ]

    # Extract activation at the last real token position for each valid sequence
    activations = {}
    for name, acts in raw_activations.items():
        extracted = []
        for i in valid_indices:
            length = prepared_sequences.prefix_lengths[i]
            extracted.append(acts[i, length - 1, :])  # last real token
        stacked = torch.stack(extracted, dim=0)
        formatted_name = format_layer_spec(name)
        activations[formatted_name] = jnp.array(stacked.cpu().numpy())

    # Filter belief states and weights to match
    valid_indices_array = jnp.array(valid_indices)
    filtered_belief_states = tuple(
        bs[valid_indices_array] for bs in prepared_sequences.belief_states
    )
    filtered_weights = prepared_sequences.weights[valid_indices_array]
    filtered_weights = filtered_weights / filtered_weights.sum()

    prepared_activations = PreparedActivations(
        activations=activations,
        belief_states=filtered_belief_states,
        weights=filtered_weights,
        n_samples=len(valid_indices),
        num_factors=prepared_sequences.num_factors,
        factor_dims=prepared_sequences.factor_dims,
    )

    return prepared_activations, weighted_loss


def process_single_checkpoint(
    step: int,
    model: Any,
    persister: MLFlowPersister,
    prepared_sequences: PreparedSequences,
    layers: str | list[str],
    min_prefix_length: int,
    max_prefix_length: int | None,
    compute_dims95_flag: bool,
    compute_rmse_flag: bool,
    compute_loss_flag: bool,
) -> SingleCheckpointResult:
    """Process a single checkpoint and return all requested metrics.

    This function:
    1. Loads model weights for the checkpoint
    2. Runs a single combined forward pass for activations + loss
    3. Computes requested metrics from activations

    Args:
        step: Checkpoint step number.
        model: The predictive model (weights will be overwritten).
        persister: MLFlowPersister for loading checkpoints.
        prepared_sequences: Pre-computed PreparedSequences.
        layers: Layer name(s) to analyze.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.
        compute_dims95_flag: Whether to compute dims@95.
        compute_rmse_flag: Whether to compute RMSE.
        compute_loss_flag: Whether to compute loss.

    Returns:
        SingleCheckpointResult with computed metrics.
    """
    try:
        # Load checkpoint weights
        persister.load_weights(model, step=step)

        # Combined forward pass for activations and loss
        prepared_activations, loss_val = get_activations_with_loss(
            model,
            prepared_sequences,
            layers,
            min_prefix_length=min_prefix_length,
            max_prefix_length=max_prefix_length,
        )

        result = SingleCheckpointResult(step=step)

        # Compute loss
        if compute_loss_flag:
            result.loss = loss_val

        # Compute dims@95
        if compute_dims95_flag:
            dims95_data: dict[str, int] = {}
            for layer_name, acts in prepared_activations.activations.items():
                scalars, arrays = layer_pca_analysis(
                    layer_activations=acts,
                    weights=prepared_activations.weights,
                    n_components=None,
                    variance_thresholds=(0.95,),
                )
                if "nc_95" in scalars:
                    dims95_data[layer_name] = int(scalars["nc_95"])
            result.dims95 = dims95_data

        # Compute RMSE
        if compute_rmse_flag:
            rmse_overall: dict[str, float] = {}
            rmse_factors: dict[str, list[float]] = {}

            for layer_name, acts in prepared_activations.activations.items():
                scalars, arrays = layer_linear_regression(
                    layer_activations=acts,
                    weights=prepared_activations.weights,
                    belief_states=prepared_activations.belief_states,
                    concat_belief_states=True,
                    compute_subspace_orthogonality=False,
                    use_svd=True,
                    fit_intercept=True,
                    rcond_values=DEFAULT_RCOND_VALUES,
                )

                # Extract predictions and targets for each factor
                y_pred_factors = []
                y_true_factors = []

                for factor_idx in range(prepared_activations.num_factors):
                    proj_key = f"projected/F{factor_idx}"
                    target_key = f"targets/F{factor_idx}"
                    if proj_key in arrays and target_key in arrays:
                        y_pred_factors.append(np.array(arrays[proj_key]))
                        y_true_factors.append(np.array(arrays[target_key]))

                if len(y_pred_factors) == prepared_activations.num_factors:
                    y_pred = np.concatenate(y_pred_factors, axis=-1)
                    y_true = np.concatenate(y_true_factors, axis=-1)

                    # Compute per-factor RMSE scores
                    factor_rmse_scores = []
                    offset = 0
                    for factor_idx in range(prepared_activations.num_factors):
                        factor_dim = prepared_activations.factor_dims[factor_idx]
                        end = offset + factor_dim
                        factor_rmse = float(np.sqrt(np.mean((y_pred[:, offset:end] - y_true[:, offset:end]) ** 2)))
                        factor_rmse_scores.append(factor_rmse)
                        offset = end

                    # Compute overall RMSE
                    overall_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

                    rmse_overall[layer_name] = overall_rmse
                    rmse_factors[layer_name] = factor_rmse_scores

            result.rmse_overall = rmse_overall
            result.rmse_factors = rmse_factors

        return result

    except Exception as e:
        return SingleCheckpointResult(step=step, error=str(e))


@dataclass
class WorkerConfig:
    """Configuration passed to worker processes for multi-GPU processing."""
    run_id: str
    experiment_id: str
    tracking_uri: str
    registry_uri: str
    layers: list[str]
    min_prefix_length: int
    max_prefix_length: int | None
    compute_dims95: bool
    compute_rmse: bool
    compute_loss: bool
    # PreparedSequences data (serializable)
    inputs: np.ndarray
    belief_states: tuple[np.ndarray, ...]
    weights: np.ndarray
    prefix_lengths: list[int]
    n_samples: int
    num_factors: int
    factor_dims: list[int]


def _process_checkpoint_batch_worker(args: tuple) -> list[dict]:
    """Worker function for processing a batch of checkpoints on a specific GPU.

    This function is designed to be called from ProcessPoolExecutor.
    It sets up its own GPU device and loads the model independently.

    IMPORTANT: Uses lazy imports to ensure CUDA_VISIBLE_DEVICES is respected.

    Args:
        args: Tuple of (checkpoint_steps, gpu_index, worker_config)

    Returns:
        List of result dicts with step, loss, dims95, rmse_overall, rmse_factors, error
    """
    checkpoint_steps, gpu_index, config = args

    # Set GPU for this worker BEFORE importing CUDA libraries
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Note: gpu_index is already relative to CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Lazy imports after GPU is set
    import jax
    import jax.numpy as jnp
    import torch
    import torch.nn.functional as F
    import numpy as np

    from fwh_core.persistence.mlflow_persister import MLFlowPersister
    from fwh_core.analysis.metric_keys import format_layer_spec
    from fwh_core.analysis.pca import layer_pca_analysis
    from fwh_core.analysis.linear_regression import layer_linear_regression

    # Reconstruct PreparedSequences from config
    from analysis import PreparedSequences, DEFAULT_RCOND_VALUES

    prepared_sequences = PreparedSequences(
        inputs=jnp.array(config.inputs),
        belief_states=tuple(jnp.array(b) for b in config.belief_states),
        weights=jnp.array(config.weights),
        prefix_lengths=config.prefix_lengths,
        n_samples=config.n_samples,
        num_factors=config.num_factors,
        factor_dims=config.factor_dims,
    )

    # Create persister for this worker
    persister = MLFlowPersister(
        experiment_id=config.experiment_id,
        run_id=config.run_id,
        tracking_uri=config.tracking_uri,
        registry_uri=config.registry_uri,
    )

    # Load model once for this batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to PyTorch
    inputs_torch = torch.tensor(
        np.array(prepared_sequences.inputs),
        device=device,
        dtype=torch.long,
    )

    results = []

    for step in checkpoint_steps:
        try:
            # Load model weights
            model = persister.load_model(step=step)
            model.eval()
            model.to(device)

            with torch.no_grad():
                logits, cache = model.run_with_cache(inputs_torch, names_filter=config.layers)

            result = {
                "step": step,
                "loss": None,
                "dims95": None,
                "rmse_overall": None,
                "rmse_factors": None,
                "error": None,
            }

            # Compute loss
            if config.compute_loss:
                valid_indices = [
                    i for i, length in enumerate(prepared_sequences.prefix_lengths)
                    if length >= config.min_prefix_length and
                    (config.max_prefix_length is None or length <= config.max_prefix_length)
                ]

                losses = []
                weights_list = []

                for i in valid_indices:
                    length = prepared_sequences.prefix_lengths[i]
                    if length < 2:
                        continue

                    prefix_logits = logits[i, :length-1, :]
                    prefix_labels = inputs_torch[i, 1:length]
                    loss = F.cross_entropy(prefix_logits, prefix_labels, reduction='mean')
                    losses.append(loss.item())
                    weights_list.append(float(prepared_sequences.weights[i]))

                if losses:
                    weights_arr = np.array(weights_list)
                    weights_arr = weights_arr / weights_arr.sum()
                    result["loss"] = float(np.sum(np.array(losses) * weights_arr))

            # Extract activations
            valid_indices = [
                i for i, length in enumerate(prepared_sequences.prefix_lengths)
                if length >= config.min_prefix_length and
                (config.max_prefix_length is None or length <= config.max_prefix_length)
            ]

            activations = {}
            for name, acts in cache.items():
                formatted_name = format_layer_spec(name)
                if name in config.layers or formatted_name in config.layers:
                    extracted = []
                    for i in valid_indices:
                        length = prepared_sequences.prefix_lengths[i]
                        extracted.append(acts[i, length - 1, :])
                    stacked = torch.stack(extracted, dim=0)
                    activations[formatted_name] = jnp.array(stacked.cpu().numpy())

            valid_indices_array = jnp.array(valid_indices)
            filtered_belief_states = tuple(
                bs[valid_indices_array] for bs in prepared_sequences.belief_states
            )
            filtered_weights = prepared_sequences.weights[valid_indices_array]
            filtered_weights = filtered_weights / filtered_weights.sum()

            # Compute dims@95
            if config.compute_dims95:
                dims95_data = {}
                for layer_name, acts in activations.items():
                    scalars, arrays = layer_pca_analysis(
                        layer_activations=acts,
                        weights=filtered_weights,
                        n_components=None,
                        variance_thresholds=(0.95,),
                    )
                    if "nc_95" in scalars:
                        dims95_data[layer_name] = int(scalars["nc_95"])
                result["dims95"] = dims95_data

            # Compute RMSE
            if config.compute_rmse:
                rmse_overall = {}
                rmse_factors = {}

                for layer_name, acts in activations.items():
                    scalars, arrays = layer_linear_regression(
                        layer_activations=acts,
                        weights=filtered_weights,
                        belief_states=filtered_belief_states,
                        concat_belief_states=True,
                        compute_subspace_orthogonality=False,
                        use_svd=True,
                        fit_intercept=True,
                        rcond_values=DEFAULT_RCOND_VALUES,
                    )

                    y_pred_factors = []
                    y_true_factors = []

                    for factor_idx in range(config.num_factors):
                        proj_key = f"projected/F{factor_idx}"
                        target_key = f"targets/F{factor_idx}"
                        if proj_key in arrays and target_key in arrays:
                            y_pred_factors.append(np.array(arrays[proj_key]))
                            y_true_factors.append(np.array(arrays[target_key]))

                    if len(y_pred_factors) == config.num_factors:
                        y_pred = np.concatenate(y_pred_factors, axis=-1)
                        y_true = np.concatenate(y_true_factors, axis=-1)

                        factor_rmse_scores = []
                        offset = 0
                        for factor_idx in range(config.num_factors):
                            factor_dim = config.factor_dims[factor_idx]
                            end = offset + factor_dim
                            factor_rmse = float(np.sqrt(np.mean((y_pred[:, offset:end] - y_true[:, offset:end]) ** 2)))
                            factor_rmse_scores.append(factor_rmse)
                            offset = end

                        rmse_overall[layer_name] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                        rmse_factors[layer_name] = factor_rmse_scores

                result["rmse_overall"] = rmse_overall
                result["rmse_factors"] = rmse_factors

            # Clean up
            del model, cache

            results.append(result)

        except Exception as e:
            results.append({
                "step": step,
                "loss": None,
                "dims95": None,
                "rmse_overall": None,
                "rmse_factors": None,
                "error": str(e),
            })

    return results


def compute_all_metrics_at_checkpoints_parallel(
    model: Any,
    prepared_sequences: PreparedSequences,
    steps: list[int],
    persister: MLFlowPersister,
    layers: str | list[str],
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
    compute_dims95: bool = True,
    compute_rmse: bool = True,
    compute_loss: bool = True,
    max_workers: int = 1,
    show_progress: bool = True,
) -> CheckpointMetrics:
    """Compute dims@95, RMSE, and loss at multiple checkpoints in parallel.

    This is the main parallel optimization function. Key improvements:
    1. Combined forward passes (activations + loss in one pass) - ~2x speedup
    2. Multi-GPU parallelization via ProcessPoolExecutor - ~Nx speedup for N GPUs

    Parallelization strategy:
    - max_workers=1: Sequential processing with combined forward passes
    - max_workers>1: Distribute checkpoints across available GPUs using
      ProcessPoolExecutor with spawn context

    Args:
        model: The predictive model (used for sequential mode only).
        prepared_sequences: Pre-computed PreparedSequences.
        steps: List of checkpoint steps.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.
        compute_dims95: Whether to compute dims@95.
        compute_rmse: Whether to compute RMSE.
        compute_loss: Whether to compute loss.
        max_workers: Number of parallel workers (default=1 for sequential).
        show_progress: Whether to show progress bar.

    Returns:
        CheckpointMetrics containing all computed metrics.
    """
    # Normalize layers to list
    if isinstance(layers, str):
        layers = [layers]

    dims95_history: dict[str, list[tuple[int, int]]] = {}
    rmse_history: dict[str, list[tuple[int, float, list[float]]]] = {}
    loss_history: list[tuple[int, float]] = []

    if max_workers <= 1:
        # Sequential processing with combined forward pass
        def process_step(step: int) -> SingleCheckpointResult:
            return process_single_checkpoint(
                step=step,
                model=model,
                persister=persister,
                prepared_sequences=prepared_sequences,
                layers=layers,
                min_prefix_length=min_prefix_length,
                max_prefix_length=max_prefix_length,
                compute_dims95_flag=compute_dims95,
                compute_rmse_flag=compute_rmse,
                compute_loss_flag=compute_loss,
            )

        if show_progress:
            results = [process_step(step) for step in tqdm(steps, desc="Processing checkpoints")]
        else:
            results = [process_step(step) for step in steps]

        # Convert to unified format
        result_dicts = []
        for r in results:
            result_dicts.append({
                "step": r.step,
                "loss": r.loss,
                "dims95": r.dims95,
                "rmse_overall": r.rmse_overall,
                "rmse_factors": r.rmse_factors,
                "error": r.error,
            })

    else:
        # Multi-GPU parallel processing
        available_gpus = get_available_gpus()
        num_gpus = min(len(available_gpus), max_workers)

        if num_gpus == 0:
            print("  Warning: No GPUs available, falling back to sequential processing")
            return compute_all_metrics_at_checkpoints_parallel(
                model=model,
                prepared_sequences=prepared_sequences,
                steps=steps,
                persister=persister,
                layers=layers,
                min_prefix_length=min_prefix_length,
                max_prefix_length=max_prefix_length,
                compute_dims95=compute_dims95,
                compute_rmse=compute_rmse,
                compute_loss=compute_loss,
                max_workers=1,
                show_progress=show_progress,
            )

        print(f"  Using {num_gpus} GPUs for parallel processing")

        # Create worker config with serializable data
        worker_config = WorkerConfig(
            run_id=persister.run_id,
            experiment_id=persister.experiment_id,
            tracking_uri=persister.tracking_uri,
            registry_uri=persister.registry_uri,
            layers=layers,
            min_prefix_length=min_prefix_length,
            max_prefix_length=max_prefix_length,
            compute_dims95=compute_dims95,
            compute_rmse=compute_rmse,
            compute_loss=compute_loss,
            inputs=np.array(prepared_sequences.inputs),
            belief_states=tuple(np.array(b) for b in prepared_sequences.belief_states),
            weights=np.array(prepared_sequences.weights),
            prefix_lengths=prepared_sequences.prefix_lengths,
            n_samples=prepared_sequences.n_samples,
            num_factors=prepared_sequences.num_factors,
            factor_dims=prepared_sequences.factor_dims,
        )

        # Distribute checkpoints across GPUs
        checkpoint_batches = [[] for _ in range(num_gpus)]
        for i, step in enumerate(steps):
            checkpoint_batches[i % num_gpus].append(step)

        # Build worker arguments
        worker_args = []
        for gpu_idx in range(num_gpus):
            if checkpoint_batches[gpu_idx]:
                worker_args.append((checkpoint_batches[gpu_idx], available_gpus[gpu_idx], worker_config))

        print(f"  Distributed {len(steps)} checkpoints across {len(worker_args)} workers")

        # Process in parallel using spawn context for CUDA safety
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            futures = [executor.submit(_process_checkpoint_batch_worker, args) for args in worker_args]

            result_dicts = []
            if show_progress:
                with tqdm(total=len(steps), desc="Processing checkpoints") as pbar:
                    for future in as_completed(futures):
                        batch_results = future.result()
                        result_dicts.extend(batch_results)
                        pbar.update(len(batch_results))
            else:
                for future in as_completed(futures):
                    result_dicts.extend(future.result())

    # Aggregate results
    for result in result_dicts:
        if result.get("error"):
            print(f"  Warning: Step {result['step']} failed: {result['error']}")
            continue

        # Aggregate loss
        if compute_loss and result.get("loss") is not None:
            loss_history.append((result["step"], result["loss"]))

        # Aggregate dims@95
        if compute_dims95 and result.get("dims95"):
            for layer_name, dims95_value in result["dims95"].items():
                if layer_name not in dims95_history:
                    dims95_history[layer_name] = []
                dims95_history[layer_name].append((result["step"], dims95_value))

        # Aggregate RMSE
        if compute_rmse and result.get("rmse_overall"):
            for layer_name in result["rmse_overall"]:
                if layer_name not in rmse_history:
                    rmse_history[layer_name] = []
                rmse_history[layer_name].append((
                    result["step"],
                    result["rmse_overall"][layer_name],
                    result.get("rmse_factors", {}).get(layer_name, []),
                ))

    # Sort all results by step
    loss_history.sort(key=lambda x: x[0])
    for layer_name in dims95_history:
        dims95_history[layer_name].sort(key=lambda x: x[0])
    for layer_name in rmse_history:
        rmse_history[layer_name].sort(key=lambda x: x[0])

    return CheckpointMetrics(
        dims95=dims95_history,
        rmse=rmse_history,
        loss=loss_history,
    )


# Export all functions
__all__ = [
    # Re-exported from analysis.py
    "PreparedSequences",
    "PreparedActivations",
    "CheckpointMetrics",
    "prepare_sequences",
    "get_activations",
    "compute_belief_regression",
    "compute_cev",
    "compute_dims95",
    "compute_loss",
    "compute_entropy_rate",
    "compute_belief_cev_baselines",
    "compute_product_beliefs",
    "compute_belief_regression_at_checkpoint",
    "compute_cev_at_checkpoint",
    "compute_cev_at_checkpoints",
    "compute_dims95_at_checkpoint",
    "compute_dims95_at_checkpoints",
    "compute_rmse_at_checkpoints",
    "compute_loss_at_checkpoint",
    "compute_all_metrics_at_checkpoints",
    "DEFAULT_RCOND_VALUES",
    # New parallel functions
    "SingleCheckpointResult",
    "WorkerConfig",
    "get_available_gpus",
    "get_activations_with_loss",
    "process_single_checkpoint",
    "compute_all_metrics_at_checkpoints_parallel",
]
