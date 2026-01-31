"""Orthogonality analysis computation for figure generation."""

from __future__ import annotations

import gc
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
import torch

from vary_one_functions import (
    ComputeDevice,
    collect_combined_activations,
    collect_vary_one_activations,
    compute_cev,
    compute_subspace_orthogonality_curve,
)

# Import for belief regression data - import here to avoid circular imports
# The actual FactorPCAData is used optionally when store_factor_pca_for_step is set
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belief_regression import FactorPCAData


@dataclass
class OrthogonalityData:
    """Container for orthogonality analysis results across training.

    Attributes:
        steps: Training checkpoint steps analyzed.
        k_range: Array of k values [1, 2, ..., max_k].
        orthogonality_results: Per-step, per-iteration results.
            Shape: [n_steps][n_iterations] -> dict[pair_key -> dict[metric -> array]]
        avg_overlap_per_step: Average normalized overlap across iterations and pairs per step.
        std_overlap_per_step: Std of normalized overlap (None if n_seed_iterations=1).
        per_pair_overlap: Per-step list of (pair_key, normalized_overlap_array) for individual lines.
            When n_seed_iterations > 1, includes all iterations × all pairs.
        cev_per_factor: Per-step, per-factor CEV curves (averaged across iterations).
            Shape: [n_steps, n_factors, n_components]
        cev_combined: Per-step combined CEV curves (all factors varying naturally).
            Shape: [n_steps, n_components]
        cev_union: Per-step union CEV curves (concatenated vary-one data).
            Shape: [n_steps, n_components]. None for caches without this data.
        layer_hook: Layer hook name that was analyzed.
        num_factors: Number of factors in the generative process.
        max_k: Maximum number of components analyzed.
        n_seed_iterations: Number of seed iterations used.
        factor_pca_data: Optional list of FactorPCAData for belief regression (one per factor).
            Only populated when store_factor_pca_for_step is set.
        belief_regression_step: Step at which factor_pca_data was collected (None if not collected).
    """

    steps: list[int]
    k_range: np.ndarray
    orthogonality_results: list[list[dict[str, dict[str, np.ndarray]]]]
    avg_overlap_per_step: list[np.ndarray]
    std_overlap_per_step: list[np.ndarray] | None
    per_pair_overlap: list[list[tuple[str, np.ndarray]]]
    cev_per_factor: np.ndarray  # [n_steps, n_factors, n_components]
    cev_combined: np.ndarray  # [n_steps, n_components] - natural generation
    layer_hook: str
    num_factors: int
    max_k: int
    n_seed_iterations: int
    # Optional fields (with defaults for backwards compatibility)
    cev_union: np.ndarray | None = None  # [n_steps, n_components] - concatenated vary-one
    factor_pca_data: list | None = None  # list[FactorPCAData] when populated
    belief_regression_step: int | None = None


def get_cache_path(
    run_id: str,
    layer_hook: str,
    n_checkpoints: int,
    max_k: int,
    n_seed_iterations: int,
    cache_dir: str = ".",
) -> Path:
    """Generate deterministic cache filename from parameters.

    Args:
        run_id: MLflow run ID
        layer_hook: Layer hook name (e.g., "blocks.3.hook_resid_post")
        n_checkpoints: Number of checkpoints analyzed
        max_k: Maximum k for orthogonality curves
        n_seed_iterations: Number of seed iterations
        cache_dir: Directory to store cache files

    Returns:
        Path to cache file
    """
    layer_safe = layer_hook.replace(".", "_")
    filename = f"orthogonality_cache_{run_id[:8]}_{layer_safe}_n{n_checkpoints}_k{max_k}_iter{n_seed_iterations}.pkl"
    return Path(cache_dir) / filename


def save_orthogonality_cache(data: OrthogonalityData, path: Path) -> None:
    """Save orthogonality data to pickle cache.

    Args:
        data: OrthogonalityData to cache
        path: Path to save cache file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved cache to {path}")


def load_orthogonality_cache(path: Path) -> OrthogonalityData | None:
    """Load orthogonality data from pickle cache.

    Args:
        path: Path to cache file

    Returns:
        OrthogonalityData if cache exists, None otherwise
    """
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded cache from {path}")
    return data


def compute_dims_at_threshold(cev_curve: np.ndarray, threshold: float = 0.95) -> int:
    """Find number of dimensions needed to reach variance threshold.

    Args:
        cev_curve: Cumulative explained variance array [n_components]
        threshold: Variance threshold (e.g., 0.95 for 95%)

    Returns:
        Number of dimensions needed (1-indexed, minimum 1)
    """
    idx = np.searchsorted(cev_curve, threshold)
    return min(idx + 1, len(cev_curve))


def compute_factor_pca(
    gp,
    factor_idx: int,
    model: torch.nn.Module,
    layer_hook: str,
    num_frozen_points: int,
    batch_per_frozen: int,
    sequence_len: int,
    bos_token: int | None,
    device: torch.device | str,
    seed_base: int,
    compute_device: ComputeDevice = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA for a single factor using vary-one procedure.

    This function:
    1. Generates vary-one beliefs (factor varies, others frozen)
    2. Gets model activations
    3. Per-position centers activations
    4. Computes PCA via SVD

    Args:
        gp: Base generative process
        factor_idx: Which factor to analyze
        model: HookedTransformer model
        layer_hook: e.g., "blocks.3.hook_resid_post"
        num_frozen_points: Number of frozen configurations
        batch_per_frozen: Batch size per frozen configuration
        sequence_len: Sequence length for generation
        bos_token: BOS token (if any)
        device: Torch device
        seed_base: Base seed for reproducibility
        compute_device: Device for SVD computation ("cuda", "cpu", or "auto")

    Returns:
        Tuple of (cev, Vt, var_ratios) where:
        - cev: Cumulative explained variance [n_components]
        - Vt: Principal components [n_components, d_model]
        - var_ratios: Individual variance ratios [n_components]
    """
    # Collect activations across frozen points
    acts_flat = collect_vary_one_activations(
        gp=gp,
        factor_idx=factor_idx,
        num_frozen_points=num_frozen_points,
        batch_per_frozen=batch_per_frozen,
        sequence_len=sequence_len,
        model=model,
        hook_name=layer_hook,
        bos_token=bos_token,
        device=device,
        seed_base=seed_base,
    )

    # Compute PCA
    cev, Vt, var_ratios = compute_cev(acts_flat, return_components=True, compute_device=compute_device)

    return cev, Vt, var_ratios


def compute_orthogonality_at_checkpoints(
    model: torch.nn.Module,
    gp,
    persister,
    checkpoint_steps: list[int],
    layer_hook: str,
    num_factors: int,
    sequence_len: int,
    bos_token: int | None,
    device: torch.device | str,
    num_frozen_points: int = 10,
    batch_per_frozen: int = 200,
    max_k: int = 20,
    seed_base: int = 42,
    n_seed_iterations: int = 1,
    store_factor_pca_for_step: int | None = None,
    compute_device: ComputeDevice = "auto",
) -> OrthogonalityData:
    """Compute orthogonality metrics at multiple training checkpoints.

    For each checkpoint:
    1. Load model weights
    2. For each seed iteration:
       a. For each factor: compute PCA via vary-one procedure
       b. For each factor pair: compute orthogonality curve
    3. Aggregate results across iterations

    Args:
        model: HookedTransformer model
        gp: Generative process
        persister: MLFlowPersister for loading checkpoints
        checkpoint_steps: List of checkpoint steps to analyze
        layer_hook: Layer to analyze
        num_factors: Number of factors
        sequence_len: Sequence length
        bos_token: BOS token (if any)
        device: Torch device
        num_frozen_points: Frozen configurations per factor
        batch_per_frozen: Batch size per frozen point
        max_k: Maximum k for orthogonality curves
        seed_base: Base seed
        n_seed_iterations: Number of independent seed iterations (1 = current behavior)
        store_factor_pca_for_step: If provided, store full FactorPCAData (including activations
            and beliefs) for this checkpoint step to enable belief regression.
        compute_device: Device for SVD computation ("cuda", "cpu", or "auto")

    Returns:
        OrthogonalityData with all results. If store_factor_pca_for_step is provided,
        factor_pca_data will be populated with FactorPCAData for each factor.
    """
    # Per-step, per-iteration results
    all_step_results: list[list[dict[str, dict[str, np.ndarray]]]] = []
    avg_overlap_per_step: list[np.ndarray] = []
    std_overlap_per_step: list[np.ndarray] | None = [] if n_seed_iterations > 1 else None
    per_pair_overlap: list[list[tuple[str, np.ndarray]]] = []
    all_cev_per_factor: list[np.ndarray] = []  # [n_steps] list of [n_factors, n_components]
    all_cev_combined: list[np.ndarray] = []  # [n_steps] list of [n_components]
    all_cev_union: list[np.ndarray] = []  # [n_steps] list of [n_components] - concatenated vary-one

    # For belief regression data (only collected for store_factor_pca_for_step)
    factor_pca_data: list | None = None
    belief_regression_step: int | None = None

    for step_idx, step in enumerate(checkpoint_steps):
        print(f"  Checkpoint {step_idx + 1}/{len(checkpoint_steps)}: step {step}")

        # Load model weights (only once per checkpoint)
        persister.load_weights(model, step=step)
        model.eval()

        # Compute combined PCA (all factors varying)
        combined_seed = 8000 + step % 1000  # Different seed range from per-factor
        combined_acts = collect_combined_activations(
            gp=gp,
            num_batches=num_frozen_points,  # Same number as per-factor for consistency
            batch_size=batch_per_frozen,
            sequence_len=sequence_len,
            model=model,
            hook_name=layer_hook,
            bos_token=bos_token,
            device=device,
            seed_base=combined_seed,
        )

        combined_cev = compute_cev(combined_acts, compute_device=compute_device)

        all_cev_combined.append(combined_cev)
        del combined_acts
        gc.collect()

        # Compute union CEV (concatenated vary-one activations from all factors)
        print(f"    Computing union CEV (concatenated vary-one)...")
        union_acts_list = []
        union_seed = 9000 + step % 1000  # Same seed as first iteration of per-factor
        for factor_idx in range(num_factors):
            factor_acts = collect_vary_one_activations(
                gp=gp,
                factor_idx=factor_idx,
                num_frozen_points=num_frozen_points,
                batch_per_frozen=batch_per_frozen,
                sequence_len=sequence_len,
                model=model,
                hook_name=layer_hook,
                bos_token=bos_token,
                device=device,
                seed_base=union_seed,
            )
            union_acts_list.append(factor_acts)
            print(f"      Factor {factor_idx}: {factor_acts.shape[0]} samples")

        union_acts = np.concatenate(union_acts_list, axis=0)
        print(f"    Union total: {union_acts.shape[0]} samples")
        union_cev = compute_cev(union_acts, compute_device=compute_device)
        all_cev_union.append(union_cev)
        del union_acts, union_acts_list
        gc.collect()

        # Collect results across iterations for this step
        step_iteration_results: list[dict[str, dict[str, np.ndarray]]] = []
        step_all_pairs: list[tuple[str, np.ndarray]] = []
        step_cev_all_iterations: list[list[np.ndarray]] = []  # [n_iterations][n_factors]

        # Check if we should collect full PCA data for belief regression at this step
        should_store_pca_data = (
            store_factor_pca_for_step is not None
            and step == store_factor_pca_for_step
        )

        for iteration in range(n_seed_iterations):
            if n_seed_iterations > 1:
                print(f"    Iteration {iteration + 1}/{n_seed_iterations}")

            # Compute PCA for each factor
            factor_pcs: list[np.ndarray] = []
            factor_var_ratios: list[np.ndarray] = []
            factor_cevs: list[np.ndarray] = []

            # Only collect full data on first iteration to avoid redundant computation
            step_factor_pca_data: list | None = [] if (should_store_pca_data and iteration == 0) else None

            for factor_idx in range(num_factors):
                # Use same seed pattern as scratchpad: 9000 + step % 1000
                # iteration=0 produces same seeds as original code (backward compatible)
                iteration_offset = iteration * 100000  # Large offset to avoid collisions
                factor_seed = 9000 + step % 1000 + iteration_offset

                if step_factor_pca_data is not None:
                    # Collect full PCA data for belief regression
                    from belief_regression import collect_factor_pca_with_data

                    fpd = collect_factor_pca_with_data(
                        gp=gp,
                        factor_idx=factor_idx,
                        num_frozen_points=num_frozen_points,
                        batch_per_frozen=batch_per_frozen,
                        sequence_len=sequence_len,
                        model=model,
                        hook_name=layer_hook,
                        bos_token=bos_token,
                        device=device,
                        seed_base=factor_seed,
                        compute_device=compute_device,
                    )
                    step_factor_pca_data.append(fpd)
                    cev, Vt, var_ratios = fpd.cev, fpd.Vt, fpd.var_ratios
                else:
                    cev, Vt, var_ratios = compute_factor_pca(
                        gp=gp,
                        factor_idx=factor_idx,
                        model=model,
                        layer_hook=layer_hook,
                        num_frozen_points=num_frozen_points,
                        batch_per_frozen=batch_per_frozen,
                        sequence_len=sequence_len,
                        bos_token=bos_token,
                        device=device,
                        seed_base=factor_seed,
                        compute_device=compute_device,
                    )

                factor_pcs.append(Vt)
                factor_var_ratios.append(var_ratios)
                factor_cevs.append(cev)

            # Store factor PCA data if collected
            if step_factor_pca_data is not None:
                factor_pca_data = step_factor_pca_data
                belief_regression_step = step

            step_cev_all_iterations.append(factor_cevs)

            # Compute orthogonality for each factor pair
            iter_orth_results: dict[str, dict[str, np.ndarray]] = {}

            for i in range(num_factors):
                for j in range(i, num_factors):  # Include diagonal for self-overlap
                    pair_key = f"F{i},F{j}"
                    iter_orth_results[pair_key] = compute_subspace_orthogonality_curve(
                        factor_pcs[i],
                        factor_var_ratios[i],
                        factor_pcs[j],
                        factor_var_ratios[j],
                        max_components=max_k,
                        compute_device=compute_device,
                    )

            step_iteration_results.append(iter_orth_results)

            # Collect off-diagonal pairs for this iteration
            off_diag_pairs = [
                (pair_key, res)
                for pair_key, res in iter_orth_results.items()
                if pair_key.split(",")[0] != pair_key.split(",")[1]
            ]

            # Add to per-pair list (prefixed with iteration for clarity when n_seed_iterations > 1)
            for pk, res in off_diag_pairs:
                label = f"iter{iteration}_{pk}" if n_seed_iterations > 1 else pk
                step_all_pairs.append((label, res["normalized_overlap"]))

            # Memory cleanup
            del factor_pcs, factor_var_ratios, factor_cevs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            jax.clear_caches()

        all_step_results.append(step_iteration_results)
        per_pair_fwh_core.append(step_all_pairs)

        # Average CEV curves across iterations for this step
        # step_cev_all_iterations: [n_iterations][n_factors] list of CEV arrays
        # Stack to [n_iterations, n_factors, n_components] and mean over iterations
        cev_stacked = np.array([[cev for cev in iter_cevs] for iter_cevs in step_cev_all_iterations])
        step_avg_cev = cev_stacked.mean(axis=0)  # [n_factors, n_components]
        all_cev_per_factor.append(step_avg_cev)

        # Compute mean and std across all iterations and pairs
        all_overlap_values = []
        for iter_results in step_iteration_results:
            for pair_key, res in iter_results.items():
                if pair_key.split(",")[0] != pair_key.split(",")[1]:  # Off-diagonal only
                    all_overlap_values.append(res["normalized_overlap"])

        all_overlap_values = np.array(all_overlap_values)
        avg_overlap_per_step.append(all_overlap_values.mean(axis=0))

        if std_overlap_per_step is not None:
            std_overlap_per_step.append(all_overlap_values.std(axis=0))

    # Stack CEV arrays: [n_steps, n_factors, n_components]
    cev_per_factor = np.stack(all_cev_per_factor, axis=0)

    # Stack combined CEV arrays: [n_steps, n_components]
    # Note: combined CEV arrays may have different lengths, pad to max length
    max_combined_len = max(len(cev) for cev in all_cev_combined)
    cev_combined_padded = []
    for cev in all_cev_combined:
        if len(cev) < max_combined_len:
            # Pad with 1.0 (fully explained variance)
            padded = np.concatenate([cev, np.ones(max_combined_len - len(cev))])
        else:
            padded = cev
        cev_combined_padded.append(padded)
    cev_combined = np.stack(cev_combined_padded, axis=0)

    # Stack union CEV arrays: [n_steps, n_components]
    max_union_len = max(len(cev) for cev in all_cev_union)
    cev_union_padded = []
    for cev in all_cev_union:
        if len(cev) < max_union_len:
            padded = np.concatenate([cev, np.ones(max_union_len - len(cev))])
        else:
            padded = cev
        cev_union_padded.append(padded)
    cev_union = np.stack(cev_union_padded, axis=0)

    return OrthogonalityData(
        steps=checkpoint_steps,
        k_range=np.arange(1, max_k + 1),
        orthogonality_results=all_step_results,
        avg_overlap_per_step=avg_overlap_per_step,
        std_overlap_per_step=std_overlap_per_step,
        per_pair_overlap=per_pair_overlap,
        cev_per_factor=cev_per_factor,
        cev_combined=cev_combined,
        layer_hook=layer_hook,
        num_factors=num_factors,
        max_k=max_k,
        n_seed_iterations=n_seed_iterations,
        cev_union=cev_union,
        factor_pca_data=factor_pca_data,
        belief_regression_step=belief_regression_step,
    )
