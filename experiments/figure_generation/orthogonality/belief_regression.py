"""Belief regression analysis for orthogonality figure generation.

This module provides functions for computing projected belief regression,
where activations are projected onto factor-specific vary-one PCA subspaces
before regressing to predict belief states.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch

from fwh_core.analysis.linear_regression import layer_linear_regression
from fwh_core.generative_processes.torch_generator import generate_data_batch_with_full_history

from vary_one_functions import (
    ComputeDevice,
    expand_state_by_batch_size,
    generate_vary_one_beliefs,
    get_activations_for_inputs,
    per_position_center,
    compute_cev,
)


@dataclass
class FactorPCAData:
    """Container for a factor's vary-one PCA results with data for regression.

    Attributes:
        factor_idx: Index of the factor this data corresponds to.
        Vt: Principal component directions [n_components, d_model].
        var_ratios: Individual variance ratios [n_components].
        cev: Cumulative explained variance [n_components].
        activations_uncentered: Uncentered activations [n_samples, d_model] for regression.
        beliefs: Belief targets [n_samples, state_dim].
        state_dim: State dimension for this factor.
    """

    factor_idx: int
    Vt: np.ndarray
    var_ratios: np.ndarray
    cev: np.ndarray
    activations_uncentered: np.ndarray
    beliefs: np.ndarray
    state_dim: int


@dataclass
class ProjectedBeliefRegressionData:
    """Container for projected belief regression results.

    Attributes:
        y_true: Ground truth beliefs [n_samples, total_belief_dim].
        y_pred: Predicted beliefs [n_samples, total_belief_dim].
        overall_rmse: Overall RMSE across all factors.
        factor_rmse_scores: Per-factor RMSE scores.
        factor_dims: State dimension per factor.
        k_values: Number of PCA components used per factor.
        step: Checkpoint step used for the analysis.
        layer_hook: Layer hook name analyzed.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    overall_rmse: float
    factor_rmse_scores: list[float]
    factor_dims: list[int]
    k_values: list[int]
    step: int
    layer_hook: str


def get_default_k_values(state_dims: list[int]) -> list[int]:
    """Compute default k values from state dimensions.

    Default is intrinsic dimension = state_dim - 1 for each factor.

    Args:
        state_dims: State dimension per factor.

    Returns:
        List of k values (at least 1 per factor).
    """
    return [max(1, d - 1) for d in state_dims]


def collect_factor_pca_with_data(
    gp,
    factor_idx: int,
    num_frozen_points: int,
    batch_per_frozen: int,
    sequence_len: int,
    model: torch.nn.Module,
    hook_name: str,
    bos_token: int | None,
    device: torch.device | str,
    seed_base: int,
    compute_device: ComputeDevice = "auto",
) -> FactorPCAData:
    """Collect vary-one activations, beliefs, and PCA for a single factor.

    This extended version returns all data needed for belief regression:
    - PCA components (Vt, var_ratios, cev)
    - Uncentered activations for projection/regression
    - Beliefs as regression targets

    Args:
        gp: Base generative process.
        factor_idx: Index of the factor to vary (others frozen).
        num_frozen_points: Number of different frozen configurations to sample.
        batch_per_frozen: Batch size per frozen configuration.
        sequence_len: Length of sequences to generate.
        model: HookedTransformer model.
        hook_name: Name of the hook to extract activations from.
        bos_token: BOS token to prepend (if any).
        device: Device for torch tensors.
        seed_base: Base seed for random key generation.
        compute_device: Device for SVD computation ("cuda", "cpu", or "auto").

    Returns:
        FactorPCAData with PCA components, activations, and beliefs.
    """
    all_acts_centered = []
    all_acts_uncentered = []
    all_beliefs = []

    for fp_idx in range(num_frozen_points):
        frozen_key = jax.random.PRNGKey(seed_base + factor_idx * 100 + fp_idx)
        sample_key = jax.random.PRNGKey(seed_base + 1000 + factor_idx * 100 + fp_idx)

        data = generate_vary_one_beliefs(
            base_gp=gp,
            vary_factor_idx=factor_idx,
            batch_size=batch_per_frozen,
            sequence_len=sequence_len,
            frozen_key=frozen_key,
            sample_key=sample_key,
        )

        # Get activations (BOS token is prepended inside this function)
        acts = get_activations_for_inputs(
            data.observations, model, hook_name, bos_token=bos_token, device=device
        )

        # Remove BOS position (position 0)
        acts = acts[:, 1:, :]  # [batch, seq_len, d_model]

        # Store uncentered activations (flattened)
        all_acts_uncentered.append(acts.reshape(-1, acts.shape[-1]))

        # Store beliefs for the varying factor (exclude position 0)
        beliefs_i = np.array(data.belief_states[factor_idx])[:, 1:, :]
        all_beliefs.append(beliefs_i.reshape(-1, beliefs_i.shape[-1]))

        # Per-position center BEFORE flattening for PCA
        acts_centered = per_position_center(acts)
        all_acts_centered.append(acts_centered.reshape(-1, acts_centered.shape[-1]))

        # Clear JAX caches
        jax.clear_caches()

    # Combine all frozen points
    activations_centered = np.concatenate(all_acts_centered, axis=0)
    activations_uncentered = np.concatenate(all_acts_uncentered, axis=0)
    beliefs = np.concatenate(all_beliefs, axis=0)

    # Compute PCA on centered activations
    cev, Vt, var_ratios = compute_cev(activations_centered, return_components=True, compute_device=compute_device)

    # Cleanup
    del all_acts_centered, activations_centered
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FactorPCAData(
        factor_idx=factor_idx,
        Vt=Vt,
        var_ratios=var_ratios,
        cev=cev,
        activations_uncentered=activations_uncentered,
        beliefs=beliefs,
        state_dim=beliefs.shape[-1],
    )


def generate_validation_data(
    gp,
    model: torch.nn.Module,
    batch_size: int,
    sequence_len: int,
    layer_hook: str,
    bos_token: int | None,
    eos_token: int | None,
    device: torch.device | str,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate normal validation data with activations and beliefs.

    Generates data where all factors vary naturally (no frozen factors),
    extracts model activations, and returns aligned activations/beliefs.

    Args:
        gp: Generative process.
        model: HookedTransformer model.
        batch_size: Number of sequences to generate.
        sequence_len: Sequence length (excluding BOS/EOS).
        layer_hook: Layer to extract activations from.
        bos_token: BOS token (if any).
        eos_token: EOS token (if any).
        device: Torch device.
        seed: Random seed.

    Returns:
        Tuple of (activations_flat, beliefs_flat) where:
        - activations_flat: [n_samples, d_model] flattened activations
        - beliefs_flat: List of [n_samples, state_dim_i] per factor
    """
    # Expand initial state
    initial_states = expand_state_by_batch_size(gp.initial_state, batch_size)

    # Generate data with full history
    outs = generate_data_batch_with_full_history(
        initial_states,
        gp,
        batch_size,
        sequence_len,
        jax.random.PRNGKey(seed),
        bos_token=bos_token,
        eos_token=eos_token,
    )

    belief_states = outs["belief_states"]
    inputs = outs["inputs"]

    # Get activations
    model_device = next(model.parameters()).device
    # Handle both JAX arrays and potential torch tensors
    if isinstance(inputs, torch.Tensor):
        inputs_torch = inputs.to(model_device)
    else:
        # JAX array - convert via numpy (must be on CPU)
        inputs_np = np.asarray(inputs)
        inputs_torch = torch.tensor(inputs_np, device=model_device)

    with torch.no_grad():
        _, cache = model.run_with_cache(inputs_torch, return_type="logits")

    raw_activations = cache[layer_hook].cpu().numpy()  # [batch, seq_len, d_model]

    # Exclude position 0 (BOS/initial state) and flatten
    activations_flat = raw_activations[:, 1:, :].reshape(-1, raw_activations.shape[-1])

    # Prepare beliefs: exclude position 0, flatten per factor
    beliefs_flat = []
    for bs in belief_states:
        bs_np = np.array(bs)[:, 1:, :]  # [batch, seq_len-1, state_dim]
        bs_flat = bs_np.reshape(-1, bs_np.shape[-1])
        beliefs_flat.append(bs_flat)

    return activations_flat, beliefs_flat


def compute_projected_belief_regression(
    factor_pca_data: list[FactorPCAData],
    activations_flat: np.ndarray,
    beliefs_flat: list[np.ndarray],
    k_values: list[int],
    concat_belief_states: bool = True,
    rcond_values: list[float] | None = None,
) -> ProjectedBeliefRegressionData:
    """Compute projected belief regression from activations.

    For each factor:
    1. Project activations onto that factor's top-k PCA components
    2. Regress from projected activations to beliefs

    Args:
        factor_pca_data: List of FactorPCAData (one per factor).
        activations_flat: Flattened activations [n_samples, d_model].
        beliefs_flat: List of belief arrays per factor [n_samples, state_dim_i].
        k_values: Number of PCA components to use per factor.
        concat_belief_states: Whether to concatenate belief states for regression.
        rcond_values: Regularization values for SVD regression.

    Returns:
        ProjectedBeliefRegressionData with predictions and metrics.
    """
    num_factors = len(factor_pca_data)
    n_samples = activations_flat.shape[0]
    weights = jnp.ones(n_samples, dtype=jnp.float32) / n_samples

    y_pred_factors = []
    y_true_factors = []
    factor_rmse_scores = []
    factor_dims = []

    for factor_idx in range(num_factors):
        fpd = factor_pca_data[factor_idx]
        beliefs_i = beliefs_flat[factor_idx]

        # Determine actual k to use (clamped to available components)
        k = min(k_values[factor_idx], fpd.Vt.shape[0])

        # Project activations onto factor's vary-one PCs (top-k)
        Vt_k = fpd.Vt[:k]  # [k, d_model]
        X_proj = activations_flat @ Vt_k.T  # [n_samples, k]

        # Regress from projected activations to beliefs
        scalars, arrays = layer_linear_regression(
            layer_activations=jnp.array(X_proj),
            weights=weights,
            belief_states=jnp.array(beliefs_i),
            concat_belief_states=concat_belief_states,
            compute_subspace_orthogonality=False,
            use_svd=True,
            fit_intercept=True,
            rcond_values=rcond_values,
        )

        y_pred_i = np.array(arrays["projected"])
        y_true_i = beliefs_i

        y_pred_factors.append(y_pred_i)
        y_true_factors.append(y_true_i)
        factor_rmse_scores.append(float(scalars["rmse"]))
        factor_dims.append(fpd.state_dim)

    # Concatenate all factors
    y_pred = np.concatenate(y_pred_factors, axis=-1)
    y_true = np.concatenate(y_true_factors, axis=-1)

    # Compute overall RMSE
    overall_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    return ProjectedBeliefRegressionData(
        y_true=y_true,
        y_pred=y_pred,
        overall_rmse=overall_rmse,
        factor_rmse_scores=factor_rmse_scores,
        factor_dims=factor_dims,
        k_values=k_values,
        step=0,  # Will be set by caller
        layer_hook="",  # Will be set by caller
    )
