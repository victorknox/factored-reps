"""
Vary-One Analysis Utility Functions

Standalone functions for vary-one analysis of factored generative processes.
These functions support analyzing whether transformers learn factored representations
by varying one factor while freezing others.

Functions:
- setup_from_mlflow: Load config and components from an MLflow run
- create_vary_one_process: Create a process where only one factor varies
- expand_state_by_batch_size: Expand initial state to batch dimension
- generate_vary_one_beliefs: Generate beliefs with only one factor varying
- compute_cev: Compute cumulative explained variance from SVD
- compute_subspace_orthogonality_curve: Measure overlap between PCA subspaces
- compute_random_subspace_baseline: Baseline overlap for random subspaces
- per_position_center: Per-position center beliefs to isolate batch variance
- compute_cartesian_product_beliefs: Compute outer product of factor marginals
- get_activations_for_inputs: Extract activations from a model for given inputs
- collect_vary_one_activations: Collect activations across multiple frozen points
- collect_combined_activations: Collect activations where all factors vary (for combined PCA)
- build_overlap_matrix: Build factor overlap matrix from orthogonality results
"""

import gc
import tempfile
from dataclasses import dataclass
from functools import reduce
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omegaconf import OmegaConf

# Type for compute device selection
ComputeDevice = Literal["cuda", "cpu", "auto"]


def get_jax_device(requested: ComputeDevice = "auto") -> jax.Device:
    """Get a JAX device for compute, with fallback.

    Args:
        requested: "cuda" for GPU, "cpu" for CPU, or "auto" (try GPU, fall back to CPU)

    Returns:
        JAX device object
    """
    if requested == "cpu":
        return jax.devices("cpu")[0]

    # Try GPU (JAX calls it "gpu", not "cuda")
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            return gpu_devices[0]
    except RuntimeError:
        pass

    if requested == "cuda":
        print("Warning: CUDA requested but not available, falling back to CPU")

    return jax.devices("cpu")[0]

from fwh_core.generative_processes.independent_factored_generative_process import (
    IndependentFactoredGenerativeProcess,
)
from fwh_core.persistence.mlflow_persister import MLFlowPersister
from fwh_core.run_management.components import Components
from fwh_core.run_management.run_management import _setup, _setup_device
from fwh_core.structured_configs.base import resolve_base_config, validate_base_config


def setup_from_mlflow(
    run_id: str,
    experiment_id: str | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    *,
    strict: bool = False,
    verbose: bool = False,
) -> tuple:
    """Setup components from an existing MLflow run's config.yaml artifact.

    Args:
        run_id: MLflow run ID
        experiment_id: MLflow experiment ID (optional)
        tracking_uri: MLflow tracking URI (e.g., "databricks")
        registry_uri: MLflow registry URI (e.g., "databricks")
        strict: Whether to enforce strict config validation
        verbose: Whether to print verbose output

    Returns:
        Tuple of (config, components, persister)
    """
    persister = MLFlowPersister(
        experiment_id=experiment_id,
        run_id=run_id,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_config_path = persister.client.download_artifacts(
            persister.run_id,
            "config.yaml",
            dst_path=str(temp_dir),
        )
        cfg = OmegaConf.load(downloaded_config_path)

    validate_base_config(cfg)
    resolve_base_config(cfg, strict=strict)

    with _setup_device(cfg):
        components = _setup(cfg, strict=strict, verbose=verbose)

    return cfg, components, persister


def create_vary_one_process(
    base_gp,
    vary_factor_idx: int,
    frozen_key: jax.Array,
) -> IndependentFactoredGenerativeProcess:
    """Create a process where only factor vary_factor_idx varies.

    All other factors are frozen using the frozen_key, meaning they will
    produce identical emission sequences across all batch samples.

    Args:
        base_gp: Base FactoredGenerativeProcess to modify
        vary_factor_idx: Index of the factor to vary (others are frozen)
        frozen_key: JAX PRNGKey for frozen factors (same across batch)

    Returns:
        IndependentFactoredGenerativeProcess with specified factor varying
    """
    num_factors = len(base_gp.component_types)
    frozen_indices = frozenset(j for j in range(num_factors) if j != vary_factor_idx)

    return IndependentFactoredGenerativeProcess(
        component_types=base_gp.component_types,
        transition_matrices=base_gp.transition_matrices,
        normalizing_eigenvectors=base_gp.normalizing_eigenvectors,
        initial_states=base_gp.initial_states,
        structure=base_gp.structure,
        frozen_factor_indices=frozen_indices,
        frozen_key=frozen_key,
    )


def expand_state_by_batch_size(state, batch_size: int):
    """Expand initial state to batch dimension.

    Args:
        state: Initial state (single state or tuple of states)
        batch_size: Number of batch samples

    Returns:
        State expanded to [batch_size, ...] shape
    """
    if isinstance(state, tuple):
        return tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in state)
    else:
        return jnp.repeat(state[None, :], batch_size, axis=0)


@dataclass
class VaryOneBeliefsData:
    """Container for beliefs and observations from vary-one generation.

    Attributes:
        vary_factor_idx: Index of the factor that was varied
        belief_states: Per-factor belief arrays [batch, seq_len, state_dim]
        observations: Token observations [batch, seq_len]
        frozen_factor_indices: Set of factor indices that were frozen
    """

    vary_factor_idx: int
    belief_states: tuple[jax.Array, ...]
    observations: jax.Array
    frozen_factor_indices: frozenset[int]


import equinox as eqx


@eqx.filter_jit
def _jitted_generate(gp, states, keys, seq_len: int):
    """JIT-compiled wrapper for gp.generate to avoid per-call tracing overhead."""
    return gp.generate(states, keys, seq_len, True)


def generate_vary_one_beliefs(
    base_gp,
    vary_factor_idx: int,
    batch_size: int,
    sequence_len: int,
    frozen_key: jax.Array,
    sample_key: jax.Array,
) -> VaryOneBeliefsData:
    """Generate beliefs with only one factor varying.

    Args:
        base_gp: Original FactoredGenerativeProcess
        vary_factor_idx: Which factor to vary (others frozen)
        batch_size: Number of sequences to generate
        sequence_len: Length of each sequence
        frozen_key: Key for frozen factors (same across batch)
        sample_key: Key for sampling (different per batch element)

    Returns:
        VaryOneBeliefsData containing beliefs for all factors
    """
    vary_one_gp = create_vary_one_process(base_gp, vary_factor_idx, frozen_key)
    initial_states = expand_state_by_batch_size(vary_one_gp.initial_state, batch_size)
    batch_sample_keys = jax.random.split(sample_key, batch_size)

    # Use JIT-compiled generate to avoid per-call tracing overhead
    belief_states, observations = _jitted_generate(
        vary_one_gp, initial_states, batch_sample_keys, sequence_len
    )

    return VaryOneBeliefsData(
        vary_factor_idx=vary_factor_idx,
        belief_states=belief_states,
        observations=observations,
        frozen_factor_indices=vary_one_gp.frozen_factor_indices,
    )


def compute_cev(
    data: np.ndarray,
    return_components: bool = False,
    compute_device: ComputeDevice = "auto",
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative explained variance from SVD.

    Args:
        data: [n_samples, n_features] matrix (should be centered)
        return_components: If True, also return Vt and variance ratios
        compute_device: Device for SVD computation ("cuda", "cpu", or "auto")

    Returns:
        If return_components=False: CEV array [min(n_samples, n_features)]
        If return_components=True: (cev, Vt, var_ratios) where
            - Vt: [n_components, n_features] principal component directions
            - var_ratios: [n_components] individual variance ratios
    """
    jax_device = get_jax_device(compute_device)

    with jax.default_device(jax_device):
        data_jax = jnp.array(data)
        _, S, Vt = jnp.linalg.svd(data_jax, full_matrices=False)
        var_ratios = (S**2) / (S**2).sum()
        cev = jnp.cumsum(var_ratios)

    cev_np = np.array(cev)
    if return_components:
        return cev_np, np.array(Vt), np.array(var_ratios)
    return cev_np


def compute_subspace_orthogonality_curve(
    V1: np.ndarray,
    var1: np.ndarray,
    V2: np.ndarray,
    var2: np.ndarray,
    max_components: int,
    return_matrices: bool = False,
    compute_device: ComputeDevice = "auto",
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Compute orthogonality metrics as function of number of components.

    Measures overlap between two PCA subspaces using multiple metrics.

    Args:
        V1, V2: [n_components, n_features] principal components (rows are PC directions)
        var1, var2: [n_components] variance ratios for weighting
        max_components: Maximum number of components to analyze
        return_matrices: If True, also return the dot product matrices M at each k
        compute_device: Device for SVD computation ("cuda", "cpu", or "auto").
            Note: The SVD here operates on small [k, k] matrices, so GPU acceleration
            provides minimal benefit. Parameter included for API consistency.

    Returns:
        Dict with arrays indexed by k (1 to max_components):
        - normalized_overlap: sum(M²) / k, range [0, 1], 0=orthogonal, 1=aligned
        - weighted_overlap: variance-weighted using Frobenius² (sum of σᵢ²)
        - weighted_overlap_svd: variance-weighted using nuclear norm (sum of σᵢ)
        - matrices (optional): list of M matrices at each k

    Notes on normalization:
        - normalized_overlap: Normalizes by k, giving [0, 1] range where
          0 = fully orthogonal subspaces, 1 = complete overlap/containment.
        - weighted_overlap uses Frobenius² = sum(σᵢ²), normalized by var1.sum() * var2.sum()
        - weighted_overlap_svd uses nuclear norm = sum(σᵢ), normalized by sqrt(var1.sum() * var2.sum())
    """
    # Note: We keep numpy for the small [k, k] SVD operations here since GPU transfer
    # overhead would exceed any computation benefit for such small matrices.
    _ = compute_device  # Unused but kept for API consistency
    results: dict[str, list] = {
        "normalized_overlap": [],
        "weighted_overlap": [],
        "weighted_overlap_svd": [],
    }
    if return_matrices:
        results["matrices"] = []

    for k in range(1, max_components + 1):
        # Dot product matrix: M[i,j] = v1_i · v2_j
        M = V1[:k] @ V2[:k].T
        sum_sq = np.sum(M**2)  # = sum(sv²) = ||M||_F²

        # Overlap normalization: / k
        # Gives [0, 1] range with intuitive interpretation
        results["normalized_overlap"].append(sum_sq / k)

        # Weighted: scale by sqrt(variance) for each component
        # M_weighted[i,j] = sqrt(var1[i]) * M[i,j] * sqrt(var2[j])
        M_weighted = np.sqrt(var1[:k, None]) * M * np.sqrt(var2[None, :k])
        # Normalize by total variance product, then multiply by k for consistent scaling
        # This makes diagonal ≈1 when variance is roughly uniform
        weight_norm = var1[:k].sum() * var2[:k].sum()
        results["weighted_overlap"].append(
            k * np.sum(M_weighted**2) / weight_norm if weight_norm > 0 else 0.0
        )

        # SVD-based weighted overlap: use sum of singular values (nuclear norm)
        # instead of sum of squared singular values (Frobenius²)
        singular_values = np.linalg.svd(M_weighted, compute_uv=False)
        # Normalize by sqrt of weight_norm (linear vs quadratic scaling)
        # Note: No k factor here (unlike Frobenius²) because nuclear norm is linear in sv
        sqrt_weight_norm = np.sqrt(weight_norm)
        results["weighted_overlap_svd"].append(
            np.sum(singular_values) / sqrt_weight_norm if sqrt_weight_norm > 0 else 0.0
        )

        if return_matrices:
            results["matrices"].append(M.copy())

    # Convert lists to arrays (except matrices)
    return {
        k: np.array(v) if k != "matrices" else v
        for k, v in results.items()
    }


def compute_random_subspace_baseline(
    d: int,
    k_values: list[int],
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Compute expected overlap for random k-dimensional subspaces in R^d.

    This provides a baseline to interpret orthogonality metrics. For random
    subspaces, the expected overlap (using /k normalization) is approximately k/d.

    Args:
        d: Ambient dimension (e.g., d_model)
        k_values: List of subspace dimensions to test
        n_samples: Number of Monte Carlo samples per k
        seed: Random seed for reproducibility

    Returns:
        Dict with arrays (one value per k in k_values):
        - analytical: k/d for each k (theoretical expectation)
        - empirical_mean: Monte Carlo mean overlap at each k
        - empirical_p5: 5th percentile at each k
        - empirical_p95: 95th percentile at each k
    """
    rng = np.random.default_rng(seed)

    results: dict[str, list] = {
        "analytical": [],
        "empirical_mean": [],
        "empirical_p5": [],
        "empirical_p95": [],
    }

    for k in k_values:
        # Analytical baseline: E[sum(M²)/k] = k/d for random unit vectors
        results["analytical"].append(k / d)

        # Monte Carlo sampling
        overlaps = []
        for _ in range(n_samples):
            # Generate random orthonormal bases via QR decomposition
            V1 = np.linalg.qr(rng.standard_normal((d, k)))[0].T  # [k, d]
            V2 = np.linalg.qr(rng.standard_normal((d, k)))[0].T  # [k, d]
            M = V1 @ V2.T
            # Use sum/k instead of mean (which is sum/k²) to match new normalization
            overlaps.append(np.sum(M**2) / k)

        results["empirical_mean"].append(np.mean(overlaps))
        results["empirical_p5"].append(np.percentile(overlaps, 5))
        results["empirical_p95"].append(np.percentile(overlaps, 95))

    return {k: np.array(v) for k, v in results.items()}


def per_position_center(beliefs: np.ndarray) -> np.ndarray:
    """Per-position center beliefs to isolate batch variance.

    Args:
        beliefs: [batch, seq_len, dim] array

    Returns:
        Centered beliefs [batch, seq_len, dim] where mean at each position is 0
    """
    pos_mean = beliefs.mean(axis=0, keepdims=True)  # [1, seq_len, dim]
    return beliefs - pos_mean


def compute_cartesian_product_beliefs(belief_states: tuple) -> np.ndarray:
    """Compute outer product of factor marginals to get joint belief.

    For n factors with state dimensions [d1, d2, ..., dn], computes:
    b1 ⊗ b2 ⊗ ... ⊗ bn → [batch, seq_len, d1 * d2 * ... * dn]

    Args:
        belief_states: Tuple of [batch, seq_len, state_dim_i] arrays

    Returns:
        [batch, seq_len, prod(state_dim_i)] array of joint beliefs
    """
    arrays = [np.array(bs) for bs in belief_states]

    def outer_last_axis(a, b):
        # a: [batch, seq, d_a], b: [batch, seq, d_b]
        # -> [batch, seq, d_a * d_b]
        return (a[..., :, None] * b[..., None, :]).reshape(*a.shape[:-1], -1)

    return reduce(outer_last_axis, arrays)


def get_activations_for_inputs(
    inputs: jax.Array,
    model: torch.nn.Module,
    hook_name: str,
    bos_token: int | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Get activations from a specific hook for given inputs.

    Args:
        inputs: Token observations [batch, seq_len] (without BOS)
        model: HookedTransformer model
        hook_name: Name of the hook to extract activations from
        bos_token: BOS token to prepend (if any)
        device: Device for torch tensors (defaults to model's device)

    Returns:
        Activations array [batch, seq_len + has_bos, d_model]
    """
    inputs_np = np.array(inputs)

    if bos_token is not None:
        batch_size = inputs_np.shape[0]
        bos_column = np.full((batch_size, 1), bos_token, dtype=inputs_np.dtype)
        inputs_np = np.concatenate([bos_column, inputs_np], axis=1)

    if device is None:
        device = next(model.parameters()).device

    inputs_torch = torch.tensor(inputs_np, dtype=torch.long, device=device)
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs_torch, return_type="logits")
    return cache[hook_name].cpu().numpy()


def get_activations_for_multiple_layers(
    inputs: jax.Array,
    model: torch.nn.Module,
    layer_hooks: list[str],
    bos_token: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray]:
    """Get activations from multiple hooks for given inputs in one forward pass.

    Args:
        inputs: Token observations [batch, seq_len] (without BOS)
        model: HookedTransformer model
        layer_hooks: List of hook names to extract activations from
        bos_token: BOS token to prepend (if any)
        device: Device for torch tensors (defaults to model's device)

    Returns:
        Dict mapping layer_hook -> activations array [batch, seq_len + has_bos, d_model]
    """
    inputs_np = np.array(inputs)

    if bos_token is not None:
        batch_size = inputs_np.shape[0]
        bos_column = np.full((batch_size, 1), bos_token, dtype=inputs_np.dtype)
        inputs_np = np.concatenate([bos_column, inputs_np], axis=1)

    if device is None:
        device = next(model.parameters()).device

    inputs_torch = torch.tensor(inputs_np, dtype=torch.long, device=device)
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs_torch, return_type="logits")

    return {hook: cache[hook].cpu().numpy() for hook in layer_hooks}


def collect_vary_one_activations(
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
) -> np.ndarray:
    """Collect activations for vary-one analysis across multiple frozen points.

    Generates vary-one data for a single factor across multiple frozen configurations,
    extracts activations, and returns them centered and flattened.

    Args:
        gp: Base generative process
        factor_idx: Index of the factor to vary (others frozen)
        num_frozen_points: Number of different frozen configurations to sample
        batch_per_frozen: Batch size per frozen configuration
        sequence_len: Length of sequences to generate
        model: HookedTransformer model
        hook_name: Name of the hook to extract activations from
        bos_token: BOS token to prepend (if any)
        device: Device for torch tensors
        seed_base: Base seed for random key generation

    Returns:
        Centered activations [total_samples, d_model] where
        total_samples = num_frozen_points * batch_per_frozen * (sequence_len - 1)
    """
    factor_activation_data = []

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

        # Remove BOS position (position 0) and per-position center
        acts = acts[:, 1:, :]
        acts_centered = per_position_center(acts)
        acts_flat = acts_centered.reshape(-1, acts_centered.shape[-1])
        factor_activation_data.append(acts_flat)
        del acts, acts_centered  # Free intermediate arrays

    # Final cleanup (don't clear JAX caches - preserve JIT)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(factor_activation_data, axis=0)


def collect_combined_activations(
    gp,
    num_batches: int,
    batch_size: int,
    sequence_len: int,
    model: torch.nn.Module,
    hook_name: str,
    bos_token: int | None,
    device: torch.device | str,
    seed_base: int,
) -> np.ndarray:
    """Collect activations where all factors vary naturally (combined PCA).

    Generates regular data from the generative process (no frozen factors),
    extracts activations, and returns them centered and flattened.

    Args:
        gp: Generative process (all factors will vary)
        num_batches: Number of batches to generate
        batch_size: Batch size per generation
        sequence_len: Length of sequences to generate
        model: HookedTransformer model
        hook_name: Name of the hook to extract activations from
        bos_token: BOS token to prepend (if any)
        device: Device for torch tensors
        seed_base: Base seed for random key generation

    Returns:
        Centered activations [total_samples, d_model] where
        total_samples = num_batches * batch_size * (sequence_len - 1)
    """
    combined_activation_data = []

    for batch_idx in range(num_batches):
        sample_key = jax.random.PRNGKey(seed_base + batch_idx)

        # Expand initial state to batch dimension
        initial_states = expand_state_by_batch_size(gp.initial_state, batch_size)
        batch_sample_keys = jax.random.split(sample_key, batch_size)

        # Generate with all factors varying
        # Use JIT-wrapped generate to avoid per-call tracing overhead
        _, observations = _jitted_generate(
            gp, initial_states, batch_sample_keys, sequence_len
        )

        # Get activations
        acts = get_activations_for_inputs(
            observations, model, hook_name, bos_token=bos_token, device=device
        )

        # Remove BOS position (position 0) and per-position center
        acts = acts[:, 1:, :]
        acts_centered = per_position_center(acts)
        acts_flat = acts_centered.reshape(-1, acts_centered.shape[-1])
        combined_activation_data.append(acts_flat)
        del acts, acts_centered

    # Cleanup (don't clear JAX caches - preserve JIT for subsequent factor calls)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(combined_activation_data, axis=0)


def build_overlap_matrix(
    num_factors: int,
    orthogonality_results: dict[str, dict[str, np.ndarray]],
    k_func: callable,
    metric: str = "normalized_overlap",
) -> np.ndarray:
    """Build overlap matrix using k_func(i, j) to determine component count for each pair.

    Creates a symmetric matrix where entry (i, j) is the overlap between factor i
    and factor j subspaces at the number of components specified by k_func(i, j).

    Args:
        num_factors: Number of factors
        orthogonality_results: Dict mapping pair keys (e.g., "F0,F1" or "F0,F0") to
            orthogonality results from compute_subspace_orthogonality_curve.
            Must include diagonal entries (e.g., "F0,F0") for honest self-overlap.
        k_func: Function (i, j) -> int specifying the number of components to use
            for each factor pair
        metric: Which overlap metric to use. Options:
            - "normalized_overlap": /k normalization, range [0, 1]
            - "weighted_overlap": Variance-weighted (Frobenius²)
            - "weighted_overlap_svd": Variance-weighted (nuclear norm)

    Returns:
        [num_factors, num_factors] symmetric matrix of overlap values
    """
    matrix = np.zeros((num_factors, num_factors))
    for i in range(num_factors):
        for j in range(i, num_factors):  # Include diagonal (i == j)
            pair_key = f"F{i},F{j}"
            k = k_func(i, j)
            # Clamp to valid range
            k = min(k, len(orthogonality_results[pair_key][metric]))
            matrix[i, j] = orthogonality_results[pair_key][metric][k - 1]
            if i != j:
                matrix[j, i] = matrix[i, j]  # Symmetric
    return matrix
