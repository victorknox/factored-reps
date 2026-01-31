"""Data loading utilities for figure generation from MLflow runs."""

from __future__ import annotations

import re
import tempfile
from typing import Any

import pandas as pd
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
    device: str | None = None,
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

    # Infer experiment_id from run if not provided
    if experiment_id is None:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id

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

    if device is not None:
        # The downloaded MLflow config is often read-only; merge produces a new config.
        cfg = OmegaConf.merge(cfg, {"device": device})

    with _setup_device(cfg):
        components = _setup(cfg, strict=strict, verbose=verbose)

    return cfg, components, persister


def fetch_metric_history(
    client: Any,
    run_id: str,
    metric_key: str,
) -> pd.DataFrame:
    """Fetch full history of a metric from MLflow.

    Args:
        client: MLflow tracking client.
        run_id: The MLflow run ID.
        metric_key: The metric key to fetch.

    Returns:
        DataFrame with 'step' and 'value' columns.
    """
    history = client.get_metric_history(run_id, metric_key)
    if metric_key == "loss":
        print(f"Fetching metric history for {metric_key}: {history}")
    return pd.DataFrame([{"step": m.step, "value": m.value} for m in history])


def list_all_metrics(client: Any, run_id: str) -> list[str]:
    """List all metric keys for a run.

    Args:
        client: MLflow tracking client.
        run_id: The MLflow run ID.

    Returns:
        List of metric key names.
    """
    run = client.get_run(run_id)
    return list(run.data.metrics.keys())


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


def get_num_factors_from_config(cfg: DictConfig) -> int:
    """Get number of factors from the generative process config.

    Args:
        cfg: The run configuration.

    Returns:
        Number of factors in the generative process.
    """
    spec = cfg.generative_process.instance.spec
    return len(spec)


def get_factor_dims_from_generative_process(generative_process: Any) -> list[int]:
    """Get dimension of each factor's belief state from the generative process.

    Args:
        generative_process: The instantiated generative process.

    Returns:
        List of dimensions for each factor.
    """
    initial_state = generative_process.initial_state
    return [state.shape[-1] for state in initial_state]


def discover_layer_names(all_metrics: list[str]) -> list[str]:
    """Discover layer names from metric keys.

    Parses metric keys like 'acts/pca/nc_95/L0.resid.post' to extract layer names.
    Also handles 'Lcat' for concatenated layers.

    Args:
        all_metrics: List of all metric key names.

    Returns:
        List of unique layer names found in metrics, sorted with Lcat last if present.
    """
    # Pattern for PCA metrics: acts/pca/nc_95/{layer}
    pca_pattern = re.compile(r"^acts/pca/nc_95/(.+)$")

    layers = set()
    for key in all_metrics:
        match = pca_pattern.match(key)
        if match:
            layers.add(match.group(1))

    # Sort layers, putting Lcat last if present
    sorted_layers = sorted(layer for layer in layers if layer != "Lcat")
    if "Lcat" in layers:
        sorted_layers.append("Lcat")

    return sorted_layers


def fetch_all_metrics_for_figure(
    client: Any,
    run_id: str,
    layer_name: str,
    num_factors: int,
    loss_metric_key: str = "loss/step",
) -> dict[str, pd.DataFrame]:
    """Fetch all metrics needed for figure generation.

    Metric keys follow fwh_core naming conventions:
    - acts/pca/nc_95/{layer} - dims@95
    - acts/reg/rmse/{layer}-F{i} - per-factor RMSE
    - belief_regression/{safe_layer}/overall_rmse - overall RMSE
    - loss/step - training loss (configurable)

    Args:
        client: MLflow tracking client.
        run_id: The MLflow run ID.
        layer_name: Layer name (as discovered from metrics, e.g., "L0.resid.post").
        num_factors: Number of factors from config.
        loss_metric_key: Metric key for training loss.

    Returns:
        Dictionary mapping semantic names to DataFrames with step/value columns.
    """
    # Convert layer name to safe format for belief_regression keys
    safe_layer = layer_name.replace(".", "_")

    result: dict[str, pd.DataFrame] = {}

    # Fetch dims@95
    result["dims95"] = fetch_metric_history(client, run_id, f"acts/pca/nc_95/{layer_name}")

    # Fetch loss
    result["loss"] = fetch_metric_history(client, run_id, loss_metric_key)

    # Fetch overall RMSE
    result["overall_rmse"] = fetch_metric_history(
        client, run_id, f"belief_regression/{safe_layer}/overall_rmse"
    )

    # Fetch per-factor RMSE
    for i in range(num_factors):
        result[f"factor_{i}_rmse"] = fetch_metric_history(
            client, run_id, f"acts/reg/rmse/{layer_name}-F{i}"
        )

    return result


def get_latest_checkpoint(client: Any, run_id: str) -> int | None:
    """Get the latest checkpoint step.

    Args:
        client: MLflow tracking client.
        run_id: The MLflow run ID.

    Returns:
        Latest checkpoint step or None if no checkpoints found.
    """
    steps = list_checkpoints(client, run_id)
    return steps[-1] if steps else None


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
    import numpy as np

    if len(all_steps) <= n_checkpoints:
        return all_steps

    indices = np.linspace(0, len(all_steps) - 1, n_checkpoints, dtype=int)
    return [all_steps[i] for i in indices]
