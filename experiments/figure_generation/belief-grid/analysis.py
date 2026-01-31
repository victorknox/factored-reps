"""Analysis utilities for computing metrics from model checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omegaconf import DictConfig

from fwh_core.generative_processes.generator import generate_data_batch_with_full_history
from fwh_core.persistence.mlflow_persister import MLFlowPersister
from fwh_core.utils.analysis_utils import (
    make_prefix_groups,
    dedup_tuple_of_tensors_first,
    dedup_probs_sum,
)
from fwh_core.analysis.pca import layer_pca_analysis
from fwh_core.analysis.linear_regression import layer_linear_regression
from fwh_core.analysis.metric_keys import format_layer_spec


def _expand_state_by_batch_size(state: Any, batch_size: int) -> Any:
    """Expand initial state to batch size."""
    if isinstance(state, tuple):
        return tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in state)
    return jnp.repeat(state[None, :], batch_size, axis=0)


def compute_entropy_rate(
    generative_process: Any,
    cfg: DictConfig,
    n_sequences: int = 1000,
    skip_first: int = 5,
    seed: int = 0,
) -> float:
    """Estimate the entropy rate of the generative process.

    Generates sequences and computes the average negative log probability
    per token, skipping early tokens to allow the process to mix.

    Args:
        generative_process: The generative process.
        cfg: Config with predictive_model and generative_process settings.
        n_sequences: Number of sequences to generate for estimation.
        skip_first: Number of initial tokens to skip (to allow mixing).
        seed: Random seed.

    Returns:
        Estimated entropy rate (average negative log probability per token).
    """
    context_len = cfg.predictive_model.instance.cfg.n_ctx
    bos_token = getattr(cfg.generative_process, "bos_token", None)
    eos_token = getattr(cfg.generative_process, "eos_token", None)
    sequence_len = context_len - int(bos_token is not None) - int(eos_token is not None)

    initial_states = _expand_state_by_batch_size(generative_process.initial_state, n_sequences)

    # Generate sequences
    batch_keys = jax.random.split(jax.random.PRNGKey(seed), n_sequences)
    _, tokens = generative_process.generate(initial_states, batch_keys, sequence_len, False)

    # Compute per-token log probabilities
    def compute_token_log_probs(state: Any, seq: jax.Array) -> jax.Array:
        def step(carry_state: Any, token: jax.Array) -> tuple[Any, jax.Array]:
            obs_probs = generative_process.observation_probability_distribution(carry_state)
            log_prob = jnp.log(obs_probs[token])
            new_state = generative_process.transition_states(carry_state, token)
            return new_state, log_prob

        _, log_probs = jax.lax.scan(step, state, seq)
        return log_probs

    log_probs = jax.vmap(compute_token_log_probs)(initial_states, tokens)

    # Average -log(prob) over tokens after skip_first (to allow mixing)
    entropy_rate = float(-jnp.mean(log_probs[:, skip_first:]))
    return entropy_rate


@dataclass
class PreparedSequences:
    """Container for deduplicated sequences ready for model inference."""

    inputs: jax.Array  # [n_unique_prefixes, max_prefix_len] - padded unique prefixes
    belief_states: tuple[jax.Array, ...]  # tuple of [n_unique_prefixes, belief_dim]
    weights: jax.Array  # [n_unique_prefixes]
    prefix_lengths: list[int]  # length of each unique prefix
    n_samples: int
    num_factors: int
    factor_dims: list[int]


@dataclass
class PreparedActivations:
    """Container for activations ready for analysis."""

    activations: dict[str, jax.Array]  # layer_name -> [n_samples, d_model]
    belief_states: tuple[jax.Array, ...]  # tuple of [n_samples, belief_dim]
    weights: jax.Array  # [n_samples]
    n_samples: int
    num_factors: int
    factor_dims: list[int]


def prepare_sequences(
    generative_process: Any,
    cfg: DictConfig,
    batch_size: int | None = None,
    seed: int = 0,
    use_probs_as_weights: bool = True,
    max_prefix_length: int | None = None,
) -> PreparedSequences:
    """Generate sequences and deduplicate by prefix.

    This is the expensive deduplication step - run once and reuse across checkpoints.
    Does NOT run the model - just generates and deduplicates sequences.

    Args:
        generative_process: The generative process.
        cfg: Config with generative_process and predictive_model settings.
        batch_size: Number of sequences to generate (defaults to cfg.training.batch_size).
        seed: Random seed for data generation.
        use_probs_as_weights: If True, use prefix probabilities as weights.
        max_prefix_length: Maximum prefix length to include. If set, only generates
            sequences up to this length, avoiding expensive long-sequence forward passes.

    Returns:
        PreparedSequences with deduplicated sequences ready for model inference.
    """
    if batch_size is None:
        batch_size = cfg.training.batch_size

    # Get sequence length from config
    context_len = cfg.predictive_model.instance.cfg.n_ctx
    bos_token = getattr(cfg.generative_process, "bos_token", None)
    eos_token = getattr(cfg.generative_process, "eos_token", None)

    # Use max_prefix_length if specified to avoid generating unnecessarily long sequences
    if max_prefix_length is not None:
        sequence_len = max_prefix_length
    else:
        sequence_len = context_len - int(bos_token is not None) - int(eos_token is not None)

    initial_states = _expand_state_by_batch_size(generative_process.initial_state, batch_size)

    outs = generate_data_batch_with_full_history(
        initial_states,
        generative_process,
        batch_size,
        sequence_len,
        jax.random.PRNGKey(seed),
        bos_token=bos_token,
        eos_token=eos_token,
    )

    belief_states = outs["belief_states"]
    inputs = outs["inputs"]
    prefix_probs = outs["prefix_probabilities"]

    # Convert inputs to JAX array for deduplication
    if isinstance(inputs, torch.Tensor):
        inputs_jax = jnp.array(inputs.cpu().numpy())
    else:
        inputs_jax = inputs

    # Group by prefix (expensive - do once)
    prefix_to_indices = make_prefix_groups(inputs_jax)

    # Ensure belief_states is a tuple
    if not isinstance(belief_states, tuple):
        belief_states = (belief_states,)

    # Deduplicate beliefs and probs
    dedup_beliefs, prefixes = dedup_tuple_of_tensors_first(belief_states, prefix_to_indices)
    dedup_probs, _ = dedup_probs_sum(prefix_probs, prefix_to_indices)

    # Get prefix lengths and create padded input sequences
    prefix_lengths = [len(p) for p in prefixes]
    max_len = max(prefix_lengths)

    # Pad prefixes to max length for batched inference
    padded_inputs = []
    for prefix in prefixes:
        padded = list(prefix) + [0] * (max_len - len(prefix))
        padded_inputs.append(padded)
    padded_inputs = jnp.array(padded_inputs)

    # Weights
    if use_probs_as_weights:
        weights = dedup_probs
    else:
        weights = jnp.ones(len(prefixes), dtype=jnp.float32) / len(prefixes)

    # Factor info
    num_factors = len(dedup_beliefs)
    factor_dims = [b.shape[-1] for b in dedup_beliefs]

    return PreparedSequences(
        inputs=padded_inputs,
        belief_states=dedup_beliefs,
        weights=weights,
        prefix_lengths=prefix_lengths,
        n_samples=len(prefixes),
        num_factors=num_factors,
        factor_dims=factor_dims,
    )


def get_activations(
    model: Any,
    prepared_sequences: PreparedSequences,
    layers: str | list[str],
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> PreparedActivations:
    """Run model on prepared sequences and extract activations.

    This is fast - run for each checkpoint you want to analyze.

    Args:
        model: The predictive model.
        prepared_sequences: Output from prepare_sequences.
        layers: Layer name(s) to extract. Can be formatted names (e.g., "L0.resid.post")
                or raw hook names (e.g., "blocks.0.hook_resid_post").
        min_prefix_length: Minimum prefix length to include. Set to 2 to skip
            BOS-only prefixes (useful when BOS token is used).
        max_prefix_length: Maximum prefix length to include. Set to limit analysis
            to early token positions (e.g., 10 for first 10 positions).

    Returns:
        PreparedActivations ready for analysis.
    """
    model.eval()

    # Normalize to list
    if isinstance(layers, str):
        layers = [layers]

    # Convert JAX array to PyTorch tensor on the model's device
    inputs_torch = torch.tensor(
        np.array(prepared_sequences.inputs),
        device=model.cfg.device,
    )

    with torch.no_grad():
        _, cache = model.run_with_cache(inputs_torch, return_type="logits")

    # Find requested layers in the cache
    # Match by exact name or by formatted name
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

    # Filter by prefix length bounds
    valid_indices = [
        i
        for i, length in enumerate(prepared_sequences.prefix_lengths)
        if length >= min_prefix_length and (max_prefix_length is None or length <= max_prefix_length)
    ]

    if len(valid_indices) < prepared_sequences.n_samples:
        print(
            f"  Filtering: {prepared_sequences.n_samples} -> {len(valid_indices)} "
            f"prefixes (min_length={min_prefix_length})"
        )

    # Debug: show prefix length distribution
    length_counts = {}
    for i in valid_indices:
        length = prepared_sequences.prefix_lengths[i]
        length_counts[length] = length_counts.get(length, 0) + 1
    print(f"  Prefix length distribution: {dict(sorted(length_counts.items())[:5])}...")

    # Extract activation at the last real token position for each valid sequence
    # Use formatted layer names to match MLflow metric key conventions
    activations = {}
    for name, acts in raw_activations.items():
        # acts is [n_prefixes, max_seq_len, d_model]
        extracted = []
        for i in valid_indices:
            length = prepared_sequences.prefix_lengths[i]
            extracted.append(acts[i, length - 1, :])  # last real token
        stacked = torch.stack(extracted, dim=0)  # [n_valid_prefixes, d_model]
        formatted_name = format_layer_spec(name)
        activations[formatted_name] = jnp.array(stacked.cpu().numpy())

    # Filter belief states and weights to match
    valid_indices_array = jnp.array(valid_indices)
    filtered_belief_states = tuple(
        bs[valid_indices_array] for bs in prepared_sequences.belief_states
    )
    filtered_weights = prepared_sequences.weights[valid_indices_array]
    filtered_weights = filtered_weights / filtered_weights.sum()  # Renormalize

    return PreparedActivations(
        activations=activations,
        belief_states=filtered_belief_states,
        weights=filtered_weights,
        n_samples=len(valid_indices),
        num_factors=prepared_sequences.num_factors,
        factor_dims=prepared_sequences.factor_dims,
    )


# Default rcond values matching training config
DEFAULT_RCOND_VALUES = [1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]


def compute_belief_regression(
    prepared_activations: PreparedActivations,
    fit_intercept: bool = True,
    rcond_values: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute belief regression data from prepared activations.

    Args:
        prepared_activations: Output from get_activations.
        fit_intercept: Whether to fit intercept in regression.
        rcond_values: Regularization values for SVD regression. Defaults to
            [1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2] to match training.

    Returns:
        Dictionary mapping layer names to belief regression data.
    """
    if rcond_values is None:
        rcond_values = DEFAULT_RCOND_VALUES

    belief_regression_data: dict[str, dict[str, Any]] = {}

    for layer_name, acts in prepared_activations.activations.items():
        # Run linear regression with concatenated belief states
        scalars, arrays = layer_linear_regression(
            layer_activations=acts,
            weights=prepared_activations.weights,
            belief_states=prepared_activations.belief_states,
            concat_belief_states=True,
            compute_subspace_orthogonality=False,
            use_svd=True,
            fit_intercept=fit_intercept,
            rcond_values=rcond_values,
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

            belief_regression_data[layer_name] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "num_factors": prepared_activations.num_factors,
                "factor_dims": prepared_activations.factor_dims,
                "overall_rmse": overall_rmse,
                "factor_rmse_scores": factor_rmse_scores,
            }

    return belief_regression_data


def compute_cev(
    prepared_activations: PreparedActivations,
    max_components: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute CEV (cumulative explained variance) from prepared activations.

    Args:
        prepared_activations: Output from get_activations.
        max_components: Maximum number of components to include in CEV arrays.

    Returns:
        Dictionary mapping layer names to CEV arrays.
    """
    cev_data: dict[str, np.ndarray] = {}

    for layer_name, acts in prepared_activations.activations.items():
        scalars, arrays = layer_pca_analysis(
            layer_activations=acts,
            weights=prepared_activations.weights,
            n_components=None,
            variance_thresholds=(0.95,),
        )

        if "cev" in arrays:
            cev = np.array(arrays["cev"])
            if max_components is not None:
                cev = cev[:max_components]
            cev_data[layer_name] = cev

    return cev_data


def compute_belief_regression_at_checkpoint(
    model: Any,
    prepared_sequences: PreparedSequences,
    step: int,
    persister: MLFlowPersister,
    layers: str | list[str],
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute belief regression data at a specific checkpoint.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        step: The checkpoint step to load.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        min_prefix_length: Minimum prefix length to include. Set to 2 to skip
            BOS-only prefixes.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Dictionary mapping layer names to belief regression data.
    """
    persister.load_weights(model, step=step)
    prepared_activations = get_activations(
        model, prepared_sequences, layers, min_prefix_length=min_prefix_length,
        max_prefix_length=max_prefix_length
    )
    return compute_belief_regression(prepared_activations)


def compute_cev_at_checkpoint(
    model: Any,
    prepared_sequences: PreparedSequences,
    step: int,
    persister: MLFlowPersister,
    layers: str | list[str],
    max_components: int | None = None,
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute CEV at a specific checkpoint.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        step: The checkpoint step to load.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        max_components: Maximum number of components to include.
        min_prefix_length: Minimum prefix length to include. Set to 2 to skip
            BOS-only prefixes.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Dictionary mapping layer names to CEV arrays.
    """
    persister.load_weights(model, step=step)
    prepared_activations = get_activations(
        model, prepared_sequences, layers, min_prefix_length=min_prefix_length,
        max_prefix_length=max_prefix_length
    )
    return compute_cev(prepared_activations, max_components)


def compute_product_beliefs(belief_states: tuple[jax.Array, ...]) -> np.ndarray:
    """Compute outer product of all factor beliefs.

    For beliefs b1[n,d1], b2[n,d2], ..., bk[n,dk], computes the full outer
    product resulting in shape [n, d1*d2*...*dk].

    This represents the joint belief distribution over all factors.

    Args:
        belief_states: Tuple of belief arrays, each [n_samples, d_i].

    Returns:
        Joint belief array of shape [n_samples, prod(d_i)].
    """
    n_samples = belief_states[0].shape[0]
    result = np.array(belief_states[0])  # [n, d0]
    for bs in belief_states[1:]:
        bs_np = np.array(bs)
        # result: [n, d_so_far], bs_np: [n, d_i]
        # Want: [n, d_so_far * d_i]
        result = np.einsum('ni,nj->nij', result, bs_np).reshape(n_samples, -1)
    return result


def compute_belief_cev_baselines(
    prepared_sequences: PreparedSequences,
    max_components: int | None = None,
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute CEV baselines for factored and product belief representations.

    These baselines serve as ground truth references:
    - Factored: CEV of concatenated factor beliefs (compact representation)
    - Product: CEV of outer-product joint beliefs (full joint space)

    If the transformer's CEV matches the factored baseline, it suggests the
    model has learned a factored representation. If it matches the product
    baseline, it suggests a joint representation.

    Args:
        prepared_sequences: PreparedSequences containing belief_states and weights.
        max_components: Maximum number of CEV components to return.
        min_prefix_length: Minimum prefix length to include (filters out BOS-only).
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Dictionary with "factored" and "product" keys mapping to CEV arrays.
    """
    # Filter by prefix length (same filtering as activation analysis)
    valid_indices = [
        i for i, length in enumerate(prepared_sequences.prefix_lengths)
        if length >= min_prefix_length and (max_prefix_length is None or length <= max_prefix_length)
    ]
    valid_indices_array = jnp.array(valid_indices)

    filtered_beliefs = tuple(
        bs[valid_indices_array] for bs in prepared_sequences.belief_states
    )
    filtered_weights = prepared_sequences.weights[valid_indices_array]
    filtered_weights = filtered_weights / filtered_weights.sum()

    # Compute factored CEV (concatenated beliefs)
    factored_beliefs = np.concatenate([np.array(bs) for bs in filtered_beliefs], axis=-1)
    factored_scalars, factored_arrays = layer_pca_analysis(
        layer_activations=jnp.array(factored_beliefs),
        weights=filtered_weights,
        n_components=None,
        variance_thresholds=(0.95,),
    )
    factored_cev = np.array(factored_arrays["cev"])

    # Compute product CEV (outer product beliefs)
    product_beliefs = compute_product_beliefs(filtered_beliefs)
    product_scalars, product_arrays = layer_pca_analysis(
        layer_activations=jnp.array(product_beliefs),
        weights=filtered_weights,
        n_components=None,
        variance_thresholds=(0.95,),
    )
    product_cev = np.array(product_arrays["cev"])

    # Truncate to max_components if specified
    if max_components is not None:
        factored_cev = factored_cev[:max_components]
        product_cev = product_cev[:max_components]

    return {
        "factored": factored_cev,
        "product": product_cev,
    }


def compute_dims95(
    prepared_activations: PreparedActivations,
) -> dict[str, int]:
    """Compute dims@95 (number of components to reach 95% variance) from activations.

    Args:
        prepared_activations: Output from get_activations.

    Returns:
        Dictionary mapping layer names to dims@95 values.
    """
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

    return dims95_data


def compute_loss(
    model: Any,
    prepared_sequences: PreparedSequences,
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> float:
    """Compute cross-entropy loss on prepared sequences.

    Computes the average cross-entropy loss across all valid prefixes,
    weighted by prefix probability.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Weighted average cross-entropy loss.
    """
    model.eval()

    # Convert JAX array to PyTorch tensor on the model's device
    # Use long dtype for cross_entropy compatibility
    inputs_torch = torch.tensor(
        np.array(prepared_sequences.inputs),
        device=model.cfg.device,
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(inputs_torch)  # [batch, seq_len, vocab_size]

    # Filter by prefix length bounds
    valid_indices = [
        i
        for i, length in enumerate(prepared_sequences.prefix_lengths)
        if length >= min_prefix_length and (max_prefix_length is None or length <= max_prefix_length)
    ]

    # Compute per-prefix loss (cross-entropy at the last real token position)
    # For a prefix of length L, we predict token at position L-1 using logits at position L-2
    # But for single-token prediction matching our activation extraction:
    # We use logits at position L-1 to predict what would come next
    # However, we don't have the "next" token in our deduplicated prefixes.
    #
    # Alternative: compute loss over all positions in each prefix
    losses = []
    weights = []

    for i in valid_indices:
        length = prepared_sequences.prefix_lengths[i]
        if length < 2:
            continue  # Need at least 2 tokens for loss computation

        # For positions 0..L-2, predict tokens 1..L-1
        prefix_logits = logits[i, :length-1, :]  # [L-1, vocab]
        prefix_labels = inputs_torch[i, 1:length]  # [L-1]

        # Cross-entropy loss for this prefix
        loss = torch.nn.functional.cross_entropy(
            prefix_logits, prefix_labels, reduction='mean'
        )
        losses.append(loss.item())
        weights.append(float(prepared_sequences.weights[i]))

    if not losses:
        return float('nan')

    # Weighted average
    weights = np.array(weights)
    weights = weights / weights.sum()
    weighted_loss = float(np.sum(np.array(losses) * weights))

    return weighted_loss


def compute_loss_at_checkpoint(
    model: Any,
    prepared_sequences: PreparedSequences,
    step: int,
    persister: MLFlowPersister,
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> float:
    """Compute loss at a specific checkpoint.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        step: The checkpoint step to load.
        persister: MLFlowPersister for loading checkpoints.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Cross-entropy loss value.
    """
    persister.load_weights(model, step=step)
    return compute_loss(model, prepared_sequences, min_prefix_length, max_prefix_length)


def compute_dims95_at_checkpoint(
    model: Any,
    prepared_sequences: PreparedSequences,
    step: int,
    persister: MLFlowPersister,
    layers: str | list[str],
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> dict[str, int]:
    """Compute dims@95 at a specific checkpoint.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        step: The checkpoint step to load.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Dictionary mapping layer names to dims@95 values.
    """
    persister.load_weights(model, step=step)
    prepared_activations = get_activations(
        model, prepared_sequences, layers, min_prefix_length=min_prefix_length,
        max_prefix_length=max_prefix_length
    )
    return compute_dims95(prepared_activations)


def compute_dims95_at_checkpoints(
    model: Any,
    prepared_sequences: PreparedSequences,
    steps: list[int],
    persister: MLFlowPersister,
    layers: str | list[str],
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Compute dims@95 at multiple checkpoints.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        steps: List of checkpoint steps.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Dictionary mapping layer names to list of (step, dims95) tuples.
    """
    dims95_history: dict[str, list[tuple[int, int]]] = {}

    for step in steps:
        dims95_data = compute_dims95_at_checkpoint(
            model=model,
            prepared_sequences=prepared_sequences,
            step=step,
            persister=persister,
            layers=layers,
            min_prefix_length=min_prefix_length,
            max_prefix_length=max_prefix_length,
        )

        for layer_name, dims95_value in dims95_data.items():
            if layer_name not in dims95_history:
                dims95_history[layer_name] = []
            dims95_history[layer_name].append((step, dims95_value))

    return dims95_history


def compute_cev_at_checkpoints(
    model: Any,
    prepared_sequences: PreparedSequences,
    steps: list[int],
    persister: MLFlowPersister,
    layers: str | list[str],
    max_components: int | None = None,
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Compute CEV curves at multiple checkpoints.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        steps: List of checkpoint steps.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        max_components: Maximum number of components to include.
        min_prefix_length: Minimum prefix length to include. Set to 2 to skip
            BOS-only prefixes.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Dictionary mapping layer names to list of (step, cev_array) tuples.
    """
    cev_history: dict[str, list[tuple[int, np.ndarray]]] = {}

    for step in steps:
        cev_data = compute_cev_at_checkpoint(
            model=model,
            prepared_sequences=prepared_sequences,
            step=step,
            persister=persister,
            layers=layers,
            max_components=max_components,
            min_prefix_length=min_prefix_length,
            max_prefix_length=max_prefix_length,
        )

        for layer_name, cev_array in cev_data.items():
            if layer_name not in cev_history:
                cev_history[layer_name] = []
            cev_history[layer_name].append((step, cev_array))

    return cev_history


def compute_rmse_at_checkpoints(
    model: Any,
    prepared_sequences: PreparedSequences,
    steps: list[int],
    persister: MLFlowPersister,
    layers: str | list[str],
    min_prefix_length: int = 1,
    max_prefix_length: int | None = None,
) -> dict[str, list[tuple[int, float, list[float]]]]:
    """Compute RMSE at multiple checkpoints.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        steps: List of checkpoint steps.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.

    Returns:
        Dictionary mapping layer names to list of (step, overall_rmse, factor_rmse_list) tuples.
    """
    rmse_history: dict[str, list[tuple[int, float, list[float]]]] = {}

    for step in steps:
        regression_data = compute_belief_regression_at_checkpoint(
            model=model,
            prepared_sequences=prepared_sequences,
            step=step,
            persister=persister,
            layers=layers,
            min_prefix_length=min_prefix_length,
            max_prefix_length=max_prefix_length,
        )

        for layer_name, data in regression_data.items():
            if layer_name not in rmse_history:
                rmse_history[layer_name] = []
            rmse_history[layer_name].append((
                step,
                data["overall_rmse"],
                data["factor_rmse_scores"],
            ))

    return rmse_history


# Aliases for internal use to avoid name collision with boolean params in compute_all_metrics_at_checkpoints
_compute_dims95_impl = compute_dims95
_compute_loss_impl = compute_loss


@dataclass
class CheckpointMetrics:
    """Container for all metrics computed at checkpoints."""
    dims95: dict[str, list[tuple[int, int]]]  # layer -> [(step, dims95), ...]
    rmse: dict[str, list[tuple[int, float, list[float]]]]  # layer -> [(step, overall, per_factor), ...]
    loss: list[tuple[int, float]]  # [(step, loss), ...]


def compute_all_metrics_at_checkpoints(
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
) -> CheckpointMetrics:
    """Compute dims@95, RMSE, and loss at multiple checkpoints efficiently.

    This loads each checkpoint only once and computes all requested metrics,
    which is more efficient than separate calls.

    Args:
        model: The predictive model.
        prepared_sequences: Pre-computed PreparedSequences.
        steps: List of checkpoint steps.
        persister: MLFlowPersister for loading checkpoints.
        layers: Layer name(s) to extract.
        min_prefix_length: Minimum prefix length to include.
        max_prefix_length: Maximum prefix length to include.
        compute_dims95: Whether to compute dims@95.
        compute_rmse: Whether to compute RMSE.
        compute_loss: Whether to compute loss.

    Returns:
        CheckpointMetrics containing all computed metrics.
    """
    dims95_history: dict[str, list[tuple[int, int]]] = {}
    rmse_history: dict[str, list[tuple[int, float, list[float]]]] = {}
    loss_history: list[tuple[int, float]] = []

    for step in steps:
        # Load checkpoint once
        persister.load_weights(model, step=step)

        # Compute loss (doesn't need activations, just forward pass)
        if compute_loss:
            loss_val = _compute_loss_impl(
                model, prepared_sequences, min_prefix_length, max_prefix_length
            )
            loss_history.append((step, loss_val))

        # Get activations for dims95 and RMSE
        if compute_dims95 or compute_rmse:
            prepared_activations = get_activations(
                model, prepared_sequences, layers,
                min_prefix_length=min_prefix_length,
                max_prefix_length=max_prefix_length
            )

            # Compute dims@95
            if compute_dims95:
                dims95_data = _compute_dims95_impl(prepared_activations)
                for layer_name, dims95_value in dims95_data.items():
                    if layer_name not in dims95_history:
                        dims95_history[layer_name] = []
                    dims95_history[layer_name].append((step, dims95_value))

            # Compute RMSE
            if compute_rmse:
                regression_data = compute_belief_regression(prepared_activations)
                for layer_name, data in regression_data.items():
                    if layer_name not in rmse_history:
                        rmse_history[layer_name] = []
                    rmse_history[layer_name].append((
                        step,
                        data["overall_rmse"],
                        data["factor_rmse_scores"],
                    ))

    return CheckpointMetrics(
        dims95=dims95_history,
        rmse=rmse_history,
        loss=loss_history,
    )


