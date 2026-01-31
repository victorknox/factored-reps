"""
Noise Sweep Analysis Utility Functions

Standalone functions for analyzing how noise affects learned factored representations.
These functions support comparing metrics (CEV, NC@95, R², RMSE) across noise levels.

Functions:
- setup_from_mlflow: Load config and components from an MLflow run
- get_available_checkpoints: Query available checkpoint steps from MLflow
- compute_variance_ratios: Compute PCA variance ratios (JAX full SVD)
- compute_variance_ratios_torch: Compute variance ratios (PyTorch truncated SVD)
- compute_metrics_from_variance_ratios: Derive NC@90, NC@95, CEV@10
- compute_joint_belief_states: Outer product of factored beliefs
- validate_data_consistency: Ensure cache data integrity
- refresh_databricks_token: Refresh auth token
- to_numpy: Convert JAX/PyTorch arrays to numpy
"""

import json
import os
import subprocess
import tempfile

import jax.numpy as jnp
import numpy as np
import torch
from omegaconf import OmegaConf

from fwh_core.analysis.pca import compute_weighted_pca
from fwh_core.persistence.mlflow_persister import MLFlowPersister
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


def get_available_checkpoints(persister: MLFlowPersister) -> list[int]:
    """Get list of available checkpoint steps from MLflow artifacts.

    Args:
        persister: MLFlowPersister instance

    Returns:
        Sorted list of checkpoint step numbers
    """
    artifacts = persister.client.list_artifacts(persister.run_id, path="models")
    steps = []
    for artifact in artifacts:
        name = artifact.path.split("/")[-1]
        try:
            step = int(name)
            steps.append(step)
        except ValueError:
            continue
    return sorted(steps)


def compute_variance_ratios(activations: np.ndarray) -> np.ndarray:
    """Compute explained variance ratios from activations using JAX full SVD.

    Args:
        activations: Array of shape [batch, seq, hidden] or [n_samples, hidden]

    Returns:
        Array of explained variance ratios [n_components]
    """
    if activations.ndim == 3:
        batch, seq, hidden = activations.shape
        activations = activations.reshape(batch * seq, hidden)
    activations_jax = jnp.array(activations)
    pca_result = compute_weighted_pca(activations_jax, weights=None, center=True)
    return np.array(pca_result["all_explained_variance_ratio"])


def compute_variance_ratios_torch(
    activations: torch.Tensor,
    n_components: int = 100,
) -> np.ndarray:
    """Compute explained variance ratios using PyTorch truncated SVD.

    Args:
        activations: Tensor of shape [batch, seq, hidden] or [n_samples, hidden]
        n_components: Number of components to compute

    Returns:
        Array of explained variance ratios [n_components]
    """
    if activations.ndim == 3:
        batch, seq, hidden = activations.shape
        activations = activations.reshape(batch * seq, hidden)
    mean = activations.mean(dim=0)
    activations_centered = activations - mean
    n_components = min(n_components, activations_centered.shape[0], activations_centered.shape[1])
    _, S, _ = torch.pca_lowrank(activations_centered, q=n_components, center=False, niter=2)
    explained_variance = (S**2) / (activations_centered.shape[0] - 1)
    total_variance = activations_centered.var(dim=0).sum()
    explained_variance_ratio = explained_variance / total_variance
    return explained_variance_ratio.cpu().numpy()


def compute_metrics_from_variance_ratios(
    variance_ratios: np.ndarray,
    n_components_cev: int = 10,
) -> dict[str, int | float]:
    """Compute NC@90, NC@95, CEV@10 from variance ratios.

    Args:
        variance_ratios: Array of explained variance ratios
        n_components_cev: Number of components for CEV metric (default 10)

    Returns:
        Dict with keys: nc_90, nc_95, cev_10
    """
    cumulative_variance = np.cumsum(variance_ratios)
    cumulative_variance_pct = np.round(cumulative_variance * 100).astype(int)
    nc_90 = int(np.searchsorted(cumulative_variance_pct, 90) + 1)
    nc_95 = int(np.searchsorted(cumulative_variance_pct, 95) + 1)
    cev_at_10 = float(cumulative_variance[min(n_components_cev - 1, len(cumulative_variance) - 1)])
    return {"nc_90": nc_90, "nc_95": nc_95, "cev_10": cev_at_10}


def to_numpy(x) -> np.ndarray:
    """Convert tensor to numpy (handles both JAX and PyTorch).

    Args:
        x: JAX array, PyTorch tensor, or numpy array

    Returns:
        Numpy array
    """
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.array(x)


def compute_joint_belief_states(factored_states: tuple) -> np.ndarray:
    """Compute joint belief state from factored states via outer product.

    This is a self-contained implementation that doesn't require the
    compute_joint_beliefs flag from fwh_core (which is on a non-merged branch).

    Args:
        factored_states: Tuple of per-factor states, each shape [batch, seq_len, S_i]

    Returns:
        Joint belief states of shape [batch, seq_len, prod(S_i)]
    """
    arrays = [to_numpy(s) for s in factored_states]

    joint = arrays[0]
    for factor_states in arrays[1:]:
        # Outer product: [batch, seq, S_i] x [batch, seq, S_j] -> [batch, seq, S_i * S_j]
        joint = (joint[..., :, None] * factor_states[..., None, :]).reshape(
            joint.shape[0], joint.shape[1], -1
        )
    return joint


def validate_data_consistency(raw_results: dict) -> None:
    """Validate that all arrays for each layer have the same length and all layers have same steps.

    Args:
        raw_results: Dict mapping noise_level -> layer -> {steps, losses, variance_ratios, ...}

    Raises:
        AssertionError: If data is inconsistent
    """
    for noise in raw_results:
        # First, check that each layer's arrays are internally consistent
        for layer in raw_results[noise]:
            data = raw_results[noise][layer]
            n_steps = len(data["steps"])
            if n_steps == 0:
                continue
            assert len(data["losses"]) == n_steps, (
                f"ε={noise}, {layer}: losses length mismatch "
                f"({len(data['losses'])} vs {n_steps} steps)"
            )
            assert len(data["variance_ratios"]) == n_steps, (
                f"ε={noise}, {layer}: variance_ratios length mismatch "
                f"({len(data['variance_ratios'])} vs {n_steps} steps)"
            )
            if data.get("r2_overall"):
                assert len(data["r2_overall"]) == n_steps, (
                    f"ε={noise}, {layer}: r2_overall length mismatch "
                    f"({len(data['r2_overall'])} vs {n_steps} steps)"
                )
            if data.get("rmse_overall"):
                assert len(data["rmse_overall"]) == n_steps, (
                    f"ε={noise}, {layer}: rmse_overall length mismatch "
                    f"({len(data['rmse_overall'])} vs {n_steps} steps)"
                )

        # Second, check that all layers have the same steps
        layers = list(raw_results[noise].keys())
        if len(layers) > 1:
            first_layer = layers[0]
            first_steps = raw_results[noise][first_layer]["steps"]
            for layer in layers[1:]:
                layer_steps = raw_results[noise][layer]["steps"]
                assert layer_steps == first_steps, (
                    f"ε={noise}: layer {layer} has different steps than {first_layer} "
                    f"({len(layer_steps)} vs {len(first_steps)} steps)"
                )


def refresh_databricks_token() -> None:
    """Set up Databricks authentication from ~/.databrickscfg.

    Sets DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
    from the config file if not already set.

    For local MLflow, this function is a no-op - just set MLFLOW_TRACKING_URI.
    """
    import configparser
    from pathlib import Path

    # Skip if using local MLflow (not Databricks)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri and tracking_uri != "databricks":
        return  # Using local MLflow, no Databricks auth needed

    # Skip if already configured via environment
    if os.environ.get("DATABRICKS_TOKEN") and os.environ.get("DATABRICKS_HOST"):
        return

    # Try to read from ~/.databrickscfg
    config_path = Path.home() / ".databrickscfg"
    if config_path.exists():
        config = configparser.ConfigParser()
        config.read(config_path)
        if "DEFAULT" in config:
            cfg_host = config["DEFAULT"].get("host", "").strip()
            cfg_token = config["DEFAULT"].get("token", "").strip()
            if cfg_host and cfg_token:
                os.environ["DATABRICKS_HOST"] = cfg_host
                os.environ["DATABRICKS_TOKEN"] = cfg_token
                print(f"Using Databricks token from {config_path}")
                return


def compute_nc95_derivative(steps: list[int], nc_95: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute d(NC@95)/d(steps) for phase plot.

    Args:
        steps: Training step values
        nc_95: NC@95 values at each step

    Returns:
        Tuple of (nc_95_array, derivative_array) with matching indices.
    """
    steps_arr = np.array(steps, dtype=float)
    nc95_arr = np.array(nc_95, dtype=float)

    # Sort by steps
    sort_idx = np.argsort(steps_arr)
    steps_arr = steps_arr[sort_idx]
    nc95_arr = nc95_arr[sort_idx]

    # Compute derivative: d(nc95)/d(steps)
    derivative = np.gradient(nc95_arr, steps_arr)

    return nc95_arr, derivative
