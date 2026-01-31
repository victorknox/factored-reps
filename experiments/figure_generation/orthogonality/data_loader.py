"""Data loading utilities for orthogonality figure generation from MLflow runs."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

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
) -> tuple[DictConfig, Components, MLFlowPersister]:
    """Setup components from an existing MLflow run's config.yaml artifact.

    Downloads config.yaml from the MLflow run's artifacts, validates and resolves it,
    then instantiates all components.

    Args:
        run_id: The MLflow run ID to load config from.
        experiment_id: Optional experiment ID. If not provided, will be inferred from the run.
        tracking_uri: Optional MLflow tracking URI. Defaults to "databricks".
        registry_uri: Optional MLflow registry URI. Defaults to "databricks".
        strict: Whether to enable strict mode for config resolution.
        verbose: Whether to enable verbose logging.

    Returns:
        A tuple of (config, components, persister) where:
        - config: The loaded and resolved DictConfig
        - components: The instantiated Components object
        - persister: An MLFlowPersister for the run (useful for loading model checkpoints)
    """
    tracking_uri = tracking_uri or "databricks"
    registry_uri = registry_uri or "databricks"

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


def list_checkpoints(client: Any, run_id: str) -> list[int]:
    """List available checkpoint steps for a run.

    Args:
        client: MLflow tracking client.
        run_id: The MLflow run ID.

    Returns:
        Sorted list of checkpoint step numbers.
    """
    artifacts = client.list_artifacts(run_id, path="models")
    checkpoint_steps = []
    for artifact in artifacts:
        name = artifact.path.split("/")[-1]
        if name.startswith("step_"):
            try:
                step = int(name.replace("step_", ""))
                checkpoint_steps.append(step)
            except ValueError:
                pass
        elif name.isdigit():
            checkpoint_steps.append(int(name))
    return sorted(checkpoint_steps)


def select_evenly_spaced_checkpoints(
    all_steps: list[int],
    n_checkpoints: int,
) -> list[int]:
    """Select evenly spaced checkpoint steps from available checkpoints.

    Args:
        all_steps: Sorted list of all checkpoint steps.
        n_checkpoints: Number of checkpoints to select.

    Returns:
        List of selected checkpoint steps.
    """
    if len(all_steps) <= n_checkpoints:
        return all_steps

    indices = np.linspace(0, len(all_steps) - 1, n_checkpoints, dtype=int)
    return [all_steps[i] for i in indices]


def get_num_factors_from_config(cfg: DictConfig) -> int:
    """Get number of factors from the generative process config.

    Args:
        cfg: The run configuration.

    Returns:
        Number of factors in the generative process.
    """
    spec = cfg.generative_process.instance.spec
    return len(spec)


def load_random_init_baseline(baseline_path: str | Path) -> dict | None:
    """Load random initialization baseline from pickle file.

    Args:
        baseline_path: Path to the baseline pickle file

    Returns:
        Baseline data dict or None if file doesn't exist
    """
    path = Path(baseline_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_baseline_ci(
    baseline_data: dict,
    metric: str,
    ci_lower_key: str,
    ci_upper_key: str,
    layer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get mean and CI from aggregated baseline across off-diagonal pairs for a specific layer.

    Args:
        baseline_data: Loaded baseline dict from load_baseline()
        metric: "unweighted_overlap", "normalized_overlap", or "weighted_overlap"
        ci_lower_key: e.g., "percentile_5" for 90% CI
        ci_upper_key: e.g., "percentile_95" for 90% CI
        layer: Layer hook to get baseline for (e.g., "blocks.3.hook_resid_post")

    Returns:
        Tuple of (mean, lower, upper) arrays of shape [max_k]
    """
    agg = baseline_data["per_layer"][layer]["aggregated"]
    off_diag_pairs = [k for k in agg.keys() if k.split(",")[0] != k.split(",")[1]]

    means = np.array([agg[p][metric]["mean"] for p in off_diag_pairs])
    lowers = np.array([agg[p][metric][ci_lower_key] for p in off_diag_pairs])
    uppers = np.array([agg[p][metric][ci_upper_key] for p in off_diag_pairs])

    return means.mean(axis=0), lowers.mean(axis=0), uppers.mean(axis=0)


def verify_baseline_compatibility(
    baseline_data: dict,
    layer_hook: str,
    d_model: int,
    num_factors: int,
) -> tuple[bool, list[str]]:
    """Check if baseline is compatible with current config.

    Architecture match is required, but run_id mismatch is allowed
    (random init doesn't depend on trained weights).

    Args:
        baseline_data: Loaded baseline dict from load_random_init_baseline().
        layer_hook: Layer hook to check for (e.g., "blocks.3.hook_resid_post").
        d_model: Model embedding dimension.
        num_factors: Number of factors in the generative process.

    Returns:
        Tuple of (is_compatible, list_of_warnings).
        - is_compatible: True if architecture matches.
        - list_of_warnings: List of warning messages (may be non-empty even if compatible).
    """
    warnings = []
    meta = baseline_data["metadata"]

    # Check architecture (required match)
    if meta["architecture"]["d_model"] != d_model:
        return False, [f"d_model mismatch: baseline={meta['architecture']['d_model']}, current={d_model}"]
    if meta["num_factors"] != num_factors:
        return False, [f"num_factors mismatch: baseline={meta['num_factors']}, current={num_factors}"]
    if layer_hook not in meta["layers_analyzed"]:
        return False, [f"layer {layer_hook} not in baseline (available: {meta['layers_analyzed']})"]

    # Warnings (non-blocking)
    # run_id mismatch is OK - random init doesn't depend on trained weights

    return True, warnings
