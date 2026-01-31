# %%
"""
Noise Sweep Analysis

Goal: Analyze how observation noise affects learned factored representations
across training checkpoints. Compares CEV, NC@95, loss, R², and RMSE metrics
across noise levels (ε = 0 to 0.5) to understand when models learn factored
vs joint representations.

Generates 5 figure types per layer:
1. CEV by training step + NC@95 vs steps
2. CEV by noise level + NC@95 vs steps
3. Loss vs steps + NC@95 vs steps
4. 4-panel: Loss, NC@95, CEV, R²
5. 4-panel: Loss, NC@95, CEV, RMSE
"""
import hashlib
import os
import pickle
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =============================================================================
# Configuration
# =============================================================================

# Run names to analyze (from repro_noise_eps_* configs)
run_names = [
    "noise_eps_0",
    "noise_eps_0.001",
    "noise_eps_0.01",
    "noise_eps_0.05",
    "noise_eps_0.1",
    "noise_eps_0.2",
    "noise_eps_0.3",
    "noise_eps_0.5",
]

# Run for CEV-by-step plot (noised process)
cev_run_name = "noise_eps_0.2"
clean_run_name = "noise_eps_0"

# Compute hash of run names for cache file naming
run_names_hash = hashlib.md5("".join(sorted(run_names)).encode()).hexdigest()[:8]

# Use script's directory as base for output
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR.parent / "final_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Cache directory for large .pkl files
CACHE_DIR = SCRIPT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache file path
NOISE_CACHE_FILE = CACHE_DIR / f"{run_names_hash}_noise_checkpoints_cache.pkl"

# PCA settings
PCA_METHOD = "full"  # "full" or "truncated"
TRUNCATED_N_COMPONENTS = 100
n_components_cev = 10

# Batch size for data generation
batch_size = 5000

# Noise levels to analyze
noise_levels = ["0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.3", "0.5"]

# MLflow experiment ID for noise sweep runs
noise_mlflow_experiment_id = "2456074392123444"


# =============================================================================
# Part 1: Setup and Data Loading
# =============================================================================

# %%
# Add script directory to path for local imports
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from functions import (
    compute_joint_belief_states,
    compute_metrics_from_variance_ratios,
    compute_nc95_derivative,
    compute_variance_ratios,
    compute_variance_ratios_torch,
    get_available_checkpoints,
    refresh_databricks_token,
    setup_from_mlflow,
    to_numpy,
    validate_data_consistency,
)

# Token refresh not needed - mlflow uses ~/.databrickscfg
# refresh_databricks_token()

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from tqdm import tqdm

from simplexity.analysis.linear_regression import linear_regression_svd
from simplexity.generative_processes.torch_generator import generate_data_batch_with_full_history
from simplexity.persistence.mlflow_persister import MLFlowPersister

# Import ICML style from orthogonality module
sys.path.insert(0, str(SCRIPT_DIR.parent / "orthogonality"))
from plotting import ICML_STYLE, apply_icml_style

mlflow.set_tracking_uri("databricks")
client = mlflow.MlflowClient()

# %%
# Get runs from MLflow
runs = client.search_runs(experiment_ids=[noise_mlflow_experiment_id], max_results=1000)

noise_to_run = {}
noise_to_run_name = {}
noised_run_id = None
clean_run_id = None

for run in runs:
    run_name = run.info.run_name or ""
    if run_name in run_names and "_eps_" in run_name:
        noise_suffix = run_name.split("_eps_")[-1]
        if noise_suffix in noise_levels:
            noise_to_run[noise_suffix] = run.info.run_id
            noise_to_run_name[noise_suffix] = run_name
    if run_name == cev_run_name:
        noised_run_id = run.info.run_id
    if run_name == clean_run_name:
        clean_run_id = run.info.run_id

print("Matched runs:")
for noise, run_id in noise_to_run.items():
    print(f"  ε={noise}: {noise_to_run_name[noise]} -> {run_id}")


# =============================================================================
# Part 2: Cache Management and Checkpoint Processing
# =============================================================================

# %%
# Load existing cache if available
raw_results = {}
optimal_losses = {}
layers_to_analyze = None
layer_display_names = None
num_factors = 0
factor_dims = []

if NOISE_CACHE_FILE.exists():
    print(f"Loading cached results from {NOISE_CACHE_FILE}...")
    with open(NOISE_CACHE_FILE, "rb") as f:
        cache_data = pickle.load(f)
    raw_results = cache_data["results"]
    optimal_losses = cache_data["optimal_losses"]
    layers_to_analyze = cache_data["layers_to_analyze"]
    layer_display_names = cache_data["layer_display_names"]
    num_factors = cache_data.get("num_factors", 0)
    factor_dims = cache_data.get("factor_dims", [])
    print(f"Loaded cached results for noise levels: {list(raw_results.keys())}")
    validate_data_consistency(raw_results)
    # Only analyze resid_post layers (skip ln_final)
    layers_to_analyze = [l for l in layers_to_analyze if "resid_post" in l]
    layer_display_names = {k: v for k, v in layer_display_names.items() if "resid_post" in k}

# Setup from MLflow to get model config and check for new checkpoints
first_noise = next(n for n in noise_levels if n in noise_to_run)
first_run_id = noise_to_run[first_noise]

cfg, components, persister = setup_from_mlflow(
    run_id=first_run_id,
    experiment_id=noise_mlflow_experiment_id,
    tracking_uri="databricks",
    registry_uri="databricks",
)

# Determine layers to analyze from model config
if layers_to_analyze is None:
    n_layer = cfg.predictive_model.instance.cfg.n_layers
    last_layer_idx = n_layer - 1
    last_resid_post = f"blocks.{last_layer_idx}.hook_resid_post"
    layers_to_analyze = [last_resid_post]
    layer_display_names = {
        last_resid_post: f"L{last_layer_idx}.resid.post",
    }
print(f"Layers to analyze: {layers_to_analyze}")

# %%
# Generate data for regression and variance analysis
generative_process = components.get_generative_process()
bos_token = cfg.generative_process.bos_token
eos_token = cfg.generative_process.eos_token
context_len = cfg.predictive_model.instance.cfg.n_ctx
seq_len = context_len - int(eos_token is not None)

key = jax.random.PRNGKey(42)

initial_states = tuple(
    jnp.repeat(s[None, :], batch_size, axis=0) for s in generative_process.initial_state
)

print(f"Generating {batch_size} samples with seq_len={seq_len}...")
outs = generate_data_batch_with_full_history(
    initial_states,
    generative_process,
    batch_size,
    seq_len,
    key,
    bos_token=bos_token,
    eos_token=eos_token,
)
inputs = outs["inputs"]
labels = outs["labels"]
belief_states = outs["belief_states"]
print(f"Data generated: inputs shape = {inputs.shape}")

# Prepare belief states for regression (skip BOS token if present)
if isinstance(belief_states, tuple):
    num_factors = len(belief_states)
    if bos_token is not None:
        belief_states_for_regression = tuple(np.array(b)[:, 1:, :] for b in belief_states)
    else:
        belief_states_for_regression = tuple(np.array(b) for b in belief_states)
    factor_dims = [b.shape[-1] for b in belief_states_for_regression]
    print(f"Belief states for regression (after BOS skip): {num_factors} factors with dims {factor_dims}")
else:
    num_factors = 1
    if bos_token is not None:
        belief_states_for_regression = (np.array(belief_states)[:, 1:, :],)
    else:
        belief_states_for_regression = (np.array(belief_states),)
    factor_dims = [belief_states_for_regression[0].shape[-1]]

# %%
# Check for new checkpoints and update cache
cache_updated = False
available_noise_levels = [n for n in noise_levels if n in noise_to_run]

for noise in tqdm(available_noise_levels, desc="Checking for new checkpoints", position=0):
    run_id = noise_to_run[noise]

    # refresh_databricks_token()  # Not needed - mlflow uses ~/.databrickscfg
    client = mlflow.MlflowClient()

    persister = MLFlowPersister(
        experiment_id=noise_mlflow_experiment_id,
        run_id=run_id,
        tracking_uri="databricks",
        registry_uri="databricks",
    )

    all_checkpoints = set(get_available_checkpoints(persister))

    cached_steps = set()
    if noise in raw_results and raw_results[noise]:
        first_layer = list(raw_results[noise].keys())[0]
        cached_steps = set(raw_results[noise][first_layer]["steps"])

    new_checkpoints = sorted(all_checkpoints - cached_steps)

    if new_checkpoints:
        tqdm.write(
            f"ε={noise}: Found {len(new_checkpoints)} new checkpoints "
            f"(cached: {len(cached_steps)}, total: {len(all_checkpoints)})"
        )
        cache_updated = True

        if noise not in raw_results:
            raw_results[noise] = {
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

        if noise not in optimal_losses:
            run_data = client.get_run(run_id)
            optimal_loss = float(run_data.data.params.get("optimal_loss/average", 0))
            optimal_losses[noise] = optimal_loss

        inputs_torch = None
        labels_torch = None

        for step in tqdm(new_checkpoints, desc=f"ε={noise} new checkpoints", position=1, leave=False):
            try:
                model = persister.load_model(step=step)
                model.eval()
            except Exception as e:
                tqdm.write(f"  Step {step}: could not load model ({e}), skipping")
                continue

            device = next(model.parameters()).device

            if inputs_torch is None:
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

            with torch.no_grad():
                logits, cache = model.run_with_cache(inputs_torch, names_filter=layers_to_analyze)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels_torch.reshape(-1)
            ).item()

            # Check if all layers are available - skip checkpoint entirely if any layer is missing
            all_layers_available = all(layer in cache for layer in layers_to_analyze)
            if not all_layers_available:
                missing = [layer for layer in layers_to_analyze if layer not in cache]
                tqdm.write(f"  Step {step}: missing layers {missing}, skipping entire checkpoint")
                del model, cache
                continue

            # Collect data for ALL layers first
            layer_data = {}
            for layer in layers_to_analyze:
                activations = cache[layer]
                if bos_token is not None:
                    activations = activations[:, 1:, :]

                if PCA_METHOD == "truncated":
                    variance_ratios = compute_variance_ratios_torch(
                        activations, n_components=TRUNCATED_N_COMPONENTS
                    )
                else:
                    variance_ratios = compute_variance_ratios(activations.cpu().numpy())

                # Compute belief regression metrics
                activations_np = activations.cpu().numpy()
                batch_size_act, seq_len_act, d_model = activations_np.shape
                activations_flat = activations_np.reshape(-1, d_model)
                weights = jnp.ones(activations_flat.shape[0]) / activations_flat.shape[0]

                r2_factors = []
                rmse_factors = []

                for factor_idx, factor_beliefs in enumerate(belief_states_for_regression):
                    factor_flat = factor_beliefs.reshape(-1, factor_beliefs.shape[-1])
                    scalars, arrays = linear_regression_svd(
                        jnp.array(activations_flat),
                        jnp.array(factor_flat),
                        weights,
                        fit_intercept=True,
                    )
                    r2_factors.append(scalars["r2"])
                    rmse_factors.append(scalars["rmse"])

                r2_overall = float(np.mean(r2_factors))
                rmse_overall = float(np.mean(rmse_factors))

                layer_data[layer] = {
                    "variance_ratios": variance_ratios,
                    "r2_overall": r2_overall,
                    "rmse_overall": rmse_overall,
                    "r2_factors": r2_factors,
                    "rmse_factors": rmse_factors,
                }

            # Append data for all layers (all succeeded)
            for layer in layers_to_analyze:
                raw_results[noise][layer]["steps"].append(step)
                raw_results[noise][layer]["losses"].append(loss)
                raw_results[noise][layer]["variance_ratios"].append(layer_data[layer]["variance_ratios"])
                raw_results[noise][layer]["r2_overall"].append(layer_data[layer]["r2_overall"])
                raw_results[noise][layer]["rmse_overall"].append(layer_data[layer]["rmse_overall"])
                for factor_idx in range(num_factors):
                    raw_results[noise][layer]["r2_factors"][factor_idx].append(
                        layer_data[layer]["r2_factors"][factor_idx]
                    )
                    raw_results[noise][layer]["rmse_factors"][factor_idx].append(
                        layer_data[layer]["rmse_factors"][factor_idx]
                    )

            del model, cache

        # Sort results by step
        for layer in layers_to_analyze:
            if raw_results[noise][layer]["steps"]:
                sorted_indices = np.argsort(raw_results[noise][layer]["steps"])
                raw_results[noise][layer]["steps"] = [
                    raw_results[noise][layer]["steps"][i] for i in sorted_indices
                ]
                raw_results[noise][layer]["losses"] = [
                    raw_results[noise][layer]["losses"][i] for i in sorted_indices
                ]
                raw_results[noise][layer]["variance_ratios"] = [
                    raw_results[noise][layer]["variance_ratios"][i] for i in sorted_indices
                ]
                raw_results[noise][layer]["r2_overall"] = [
                    raw_results[noise][layer]["r2_overall"][i] for i in sorted_indices
                ]
                raw_results[noise][layer]["rmse_overall"] = [
                    raw_results[noise][layer]["rmse_overall"][i] for i in sorted_indices
                ]
                for factor_idx in range(len(raw_results[noise][layer]["r2_factors"])):
                    raw_results[noise][layer]["r2_factors"][factor_idx] = [
                        raw_results[noise][layer]["r2_factors"][factor_idx][i] for i in sorted_indices
                    ]
                    raw_results[noise][layer]["rmse_factors"][factor_idx] = [
                        raw_results[noise][layer]["rmse_factors"][factor_idx][i] for i in sorted_indices
                    ]

        first_layer = layers_to_analyze[0]
        if raw_results[noise][first_layer]["steps"]:
            final_step = raw_results[noise][first_layer]["steps"][-1]
            final_loss = raw_results[noise][first_layer]["losses"][-1]
            tqdm.write(f"ε={noise}: final step {final_step}, loss = {final_loss:.4f}")
    else:
        tqdm.write(f"ε={noise}: No new checkpoints (cached: {len(cached_steps)})")

# Validate and save updated cache
if cache_updated:
    validate_data_consistency(raw_results)
    print(f"Saving updated results to {NOISE_CACHE_FILE}...")
    with open(NOISE_CACHE_FILE, "wb") as f:
        pickle.dump(
            {
                "results": raw_results,
                "optimal_losses": optimal_losses,
                "layers_to_analyze": layers_to_analyze,
                "layer_display_names": layer_display_names,
                "num_factors": num_factors,
                "factor_dims": factor_dims,
            },
            f,
        )
    print("Cache updated.")

# %%
# Compute metrics from raw variance ratios
results = {}
for noise in raw_results:
    results[noise] = {}
    for layer in raw_results[noise]:
        results[noise][layer] = {
            "steps": raw_results[noise][layer]["steps"],
            "losses": raw_results[noise][layer]["losses"],
            "nc_90": [],
            "nc_95": [],
            "cev_10": [],
        }
        for var_ratios in raw_results[noise][layer]["variance_ratios"]:
            metrics = compute_metrics_from_variance_ratios(var_ratios, n_components_cev)
            results[noise][layer]["nc_90"].append(metrics["nc_90"])
            results[noise][layer]["nc_95"].append(metrics["nc_95"])
            results[noise][layer]["cev_10"].append(metrics["cev_10"])

# Debug: print number of data points per noise level
for noise in results:
    first_layer = list(results[noise].keys())[0]
    n_steps = len(results[noise][first_layer]["steps"])
    max_step = max(results[noise][first_layer]["steps"]) if results[noise][first_layer]["steps"] else 0
    print(f"ε={noise}: {n_steps} checkpoints, max step = {max_step}")


# =============================================================================
# Part 3: Ground Truth Computation
# =============================================================================

# %%
# Extract CEV-by-step data from noise cache (eps=0.2)
cev_noise_level = "0.2"
if cev_noise_level in raw_results:
    cev_results = {
        "steps": {layer: raw_results[cev_noise_level][layer]["steps"] for layer in layers_to_analyze},
        "variance_ratios": {
            layer: raw_results[cev_noise_level][layer]["variance_ratios"] for layer in layers_to_analyze
        },
    }
    first_layer = layers_to_analyze[0]
    print(
        f"CEV-by-step: Using {len(cev_results['steps'][first_layer])} checkpoints "
        f"from noise cache (ε={cev_noise_level})"
    )
else:
    print(f"Warning: ε={cev_noise_level} not found in noise cache, CEV-by-step plots will be empty")
    cev_results = {
        "steps": {layer: [] for layer in layers_to_analyze},
        "variance_ratios": {layer: [] for layer in layers_to_analyze},
    }

# %%
# Compute ground truth variance ratios
print("Loading generative processes for ground truth CEV...")

# Load noised generative process (eps=0.2)
cfg_noised, components_noised, _ = setup_from_mlflow(
    run_id=noised_run_id,
    experiment_id=noise_mlflow_experiment_id,
    tracking_uri="databricks",
    registry_uri="databricks",
)
noised_generative_process = components_noised.get_generative_process()

# Load clean generative process (eps=0)
cfg_clean, components_clean, _ = setup_from_mlflow(
    run_id=clean_run_id,
    experiment_id=noise_mlflow_experiment_id,
    tracking_uri="databricks",
    registry_uri="databricks",
)
clean_generative_process = components_clean.get_generative_process()

# Generate beliefs using noised process
print("Generating belief states from noised process...")
noised_initial_states = tuple(
    jnp.repeat(s[None, :], batch_size, axis=0) for s in noised_generative_process.initial_state
)
noised_outs = generate_data_batch_with_full_history(
    noised_initial_states,
    noised_generative_process,
    batch_size,
    seq_len,
    key,
    bos_token=bos_token,
    eos_token=eos_token,
)

# Generate beliefs using clean process
print("Generating belief states from clean process...")
clean_initial_states = tuple(
    jnp.repeat(s[None, :], batch_size, axis=0) for s in clean_generative_process.initial_state
)
clean_outs = generate_data_batch_with_full_history(
    clean_initial_states,
    clean_generative_process,
    batch_size,
    seq_len,
    key,
    bos_token=bos_token,
    eos_token=eos_token,
)

# Compute factored variance ratios from clean beliefs
clean_belief_states = clean_outs["belief_states"]
if isinstance(clean_belief_states, tuple):
    clean_factored_beliefs = []
    for bs in clean_belief_states:
        bs_np = to_numpy(bs)
        if bos_token is not None:
            bs_np = bs_np[:, 1:, :]
        clean_factored_beliefs.append(bs_np)
    factored_concat = np.concatenate(clean_factored_beliefs, axis=-1)
    factored_variance_ratios = compute_variance_ratios(factored_concat)
else:
    bs_np = to_numpy(clean_belief_states)
    if bos_token is not None:
        bs_np = bs_np[:, 1:, :]
    factored_variance_ratios = compute_variance_ratios(bs_np)

# Compute joint variance ratios from noised beliefs
noised_belief_states = noised_outs["belief_states"]
if isinstance(noised_belief_states, tuple):
    noised_joint_beliefs = compute_joint_belief_states(noised_belief_states)
    if bos_token is not None:
        noised_joint_beliefs = noised_joint_beliefs[:, 1:, :]
    noised_joint_variance_ratios = compute_variance_ratios(noised_joint_beliefs)
    print(f"Noised joint beliefs shape: {noised_joint_beliefs.shape}")
else:
    noised_joint_beliefs = to_numpy(noised_belief_states)
    if bos_token is not None:
        noised_joint_beliefs = noised_joint_beliefs[:, 1:, :]
    noised_joint_variance_ratios = compute_variance_ratios(noised_joint_beliefs)

print("Ground truth variance ratios computed.")


# =============================================================================
# Part 4: Figure Generation
# =============================================================================

# %%
# Apply ICML style and setup colors
apply_icml_style()

# Precompute ground truth CEV curves
factored_cev = np.cumsum(factored_variance_ratios)
factored_dims = np.arange(1, len(factored_cev) + 1)

noised_joint_cev = np.cumsum(noised_joint_variance_ratios)
noised_joint_dims = np.arange(1, len(noised_joint_cev) + 1)

gt_color = "#c44e52"  # red for ground truth

# Colormap for noise levels
noise_values = [float(n) for n in noise_levels]
noise_norm = mcolors.Normalize(vmin=0, vmax=0.5)
noise_cmap = plt.cm.viridis
noise_ticks = [0, 0.05, 0.1, 0.2, 0.3, 0.5]

# Get layer pairs for plotting
cev_layers_available = list(cev_results["variance_ratios"].keys())
layer_pairs = []
for layer in layers_to_analyze:
    if "ln_final" in layer:
        layer_pairs.append((layer, "ln_final"))
    elif "resid_post" in layer:
        layer_pairs.append((layer, "resid_post"))

# %%
# Generate figures for each layer
for nc_layer, layer_short_name in layer_pairs:
    print(f"\n=== Generating figures for {layer_short_name} ===")

    cev_layer = nc_layer
    print(f"Using layer: {nc_layer}")

    # =========================================================================
    # Figure 1: CEV by training step + NC@95
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel (a): CEV by training step
    ax_cev = axes[0]

    # Colormap for training steps
    steps = np.array(cev_results["steps"][cev_layer])
    step_norm = mcolors.LogNorm(vmin=max(steps.min(), 1), vmax=steps.max())
    step_cmap = plt.cm.viridis_r

    for i, (step, var_ratios) in enumerate(
        zip(cev_results["steps"][cev_layer], cev_results["variance_ratios"][cev_layer])
    ):
        if var_ratios is None:
            continue
        cev = np.cumsum(var_ratios)
        dims = np.arange(1, len(cev) + 1)
        color = step_cmap(step_norm(max(step, 1)))
        ax_cev.plot(dims, cev, color=color, linewidth=2.5, alpha=0.8)

    # Add ground truth lines
    ax_cev.plot(factored_dims, factored_cev, color=gt_color, linestyle="--", linewidth=3.0, zorder=100)
    ax_cev.plot(
        noised_joint_dims, noised_joint_cev, color=gt_color, linestyle="--", linewidth=3.0, zorder=100
    )

    # Text labels for ground truth
    ax_cev.text(5, 1.0, "Factored", fontsize=13, color=gt_color, ha="left", va="bottom")
    joint_label_idx = min(25, len(noised_joint_cev) - 1)
    ax_cev.text(
        noised_joint_dims[joint_label_idx],
        noised_joint_cev[joint_label_idx] - 0.07,
        "Joint (noisy)",
        fontsize=13,
        color=gt_color,
        ha="center",
        va="top",
    )

    ax_cev.text(0.65, 0.5, "ε=0.2", fontsize=16, transform=ax_cev.transAxes, ha="center", va="center")

    ax_cev.set_xlabel("Dimension", fontsize=16)
    ax_cev.set_ylabel("Cumulative Variance", fontsize=16)
    ax_cev.set_xlim(1, 64)
    ax_cev.set_ylim(0, 1.02)
    ax_cev.set_xticks([20, 40, 60])
    ax_cev.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_cev.set_yticklabels([".0", ".2", ".4", ".6", ".8", "1"])
    ax_cev.tick_params(axis="both", labelsize=14)
    ax_cev.spines["top"].set_visible(False)
    ax_cev.spines["right"].set_visible(False)
    ax_cev.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax_cev.text(-0.12, 1.05, "(a)", fontsize=18, fontweight="bold", transform=ax_cev.transAxes)

    # Colorbar for training steps
    sm_step = plt.cm.ScalarMappable(cmap=step_cmap, norm=step_norm)
    sm_step.set_array([])
    cbar_ax_cev = ax_cev.inset_axes([0.45, 0.07, 0.52, 0.08])
    cbar_cev = plt.colorbar(sm_step, cax=cbar_ax_cev, orientation="horizontal")
    cbar_cev.outline.set_visible(False)
    tick_values = [1, 10, 100, 1000, 5000]
    min_step = max(min(cev_results["steps"][cev_layer]), 1)
    max_step = max(cev_results["steps"][cev_layer])
    tick_values = [t for t in tick_values if min_step <= t <= max_step]
    cbar_cev.set_ticks(tick_values)
    cbar_cev.set_ticklabels([f"{t}" if t < 1000 else f"{t // 1000}k" for t in tick_values])
    cbar_cev.ax.tick_params(labelsize=11, top=True, bottom=False, labeltop=True, labelbottom=False)
    cbar_cev.ax.text(
        0.5,
        0.45,
        "Training Step",
        transform=cbar_cev.ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
    )

    # Right panel (b): NC@95 vs steps
    ax_nc = axes[1]

    for noise in noise_levels:
        if noise not in results or not results[noise][nc_layer]["steps"]:
            continue
        data = list(zip(results[noise][nc_layer]["steps"], results[noise][nc_layer]["nc_95"]))
        data = sorted(data, key=lambda x: x[0])
        steps_nc = [d[0] for d in data]
        dims_nc = [d[1] for d in data]
        color = noise_cmap(noise_norm(float(noise)))
        ax_nc.plot(
            steps_nc, dims_nc, color=color, linewidth=2.5, alpha=0.8, zorder=10 if noise == "0" else 1
        )

    ax_nc.axhline(y=10, color="#888888", linestyle="--", linewidth=2.0, label="Ground truth")

    ax_nc.set_xlabel("Training steps", fontsize=16)
    ax_nc.set_ylabel("Dimensions for 95% CEV", fontsize=16)
    ax_nc.set_xscale("log")
    ax_nc.tick_params(axis="both", labelsize=14)
    ax_nc.spines["top"].set_visible(False)
    ax_nc.spines["right"].set_visible(False)
    ax_nc.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax_nc.text(-0.12, 1.05, "(b)", fontsize=18, fontweight="bold", transform=ax_nc.transAxes)

    # Colorbar for noise levels
    sm_noise = plt.cm.ScalarMappable(cmap=noise_cmap, norm=noise_norm)
    sm_noise.set_array([])
    cbar_ax_nc = ax_nc.inset_axes([0.55, 0.86, 0.42, 0.08])
    cbar_nc = plt.colorbar(sm_noise, cax=cbar_ax_nc, orientation="horizontal")
    cbar_nc.outline.set_visible(False)
    cbar_nc.set_ticks(noise_ticks)
    cbar_nc.set_ticklabels(["0", ".05", ".1", ".2", ".3", ".5"])
    cbar_nc.ax.tick_params(labelsize=11, top=False, bottom=True, labeltop=False, labelbottom=True)
    cbar_nc.ax.set_xticks([0.001, 0.01], minor=True)
    cbar_nc.ax.tick_params(which="minor", top=True, bottom=False, length=2)
    cbar_nc.ax.text(
        0.5,
        0.55,
        "Noise ε",
        transform=cbar_nc.ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
    )

    fig.tight_layout()
    fig1_path = OUTPUT_DIR / f"combined_cev_nc95_figure_{layer_short_name}.png"
    plt.savefig(fig1_path, dpi=300)
    plt.show()
    print(f"Figure saved to {fig1_path}")

    # =========================================================================
    # Figure 2: CEV by noise level + NC@95
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    ax_cev2 = axes2[0]

    for noise in noise_levels:
        if noise not in raw_results or not raw_results[noise][nc_layer]["variance_ratios"]:
            continue
        final_var_ratios = raw_results[noise][nc_layer]["variance_ratios"][-1]
        cev = np.cumsum(final_var_ratios)
        dims = np.arange(1, len(cev) + 1)
        color = noise_cmap(noise_norm(float(noise)))
        ax_cev2.plot(dims, cev, color=color, linewidth=2.5, alpha=0.8, zorder=10 if noise == "0" else 1)

    ax_cev2.plot(factored_dims, factored_cev, color=gt_color, linestyle="--", linewidth=3.0, zorder=100)
    ax_cev2.plot(
        noised_joint_dims, noised_joint_cev, color=gt_color, linestyle="--", linewidth=3.0, zorder=100
    )

    ax_cev2.text(5, 1.0, "Factored", fontsize=13, color=gt_color, ha="left", va="bottom")
    joint_label_idx2 = min(25, len(noised_joint_cev) - 1)
    ax_cev2.text(
        noised_joint_dims[joint_label_idx2],
        noised_joint_cev[joint_label_idx2] - 0.07,
        "Joint (noisy)",
        fontsize=13,
        color=gt_color,
        ha="center",
        va="top",
    )

    ax_cev2.set_xlabel("Dimension", fontsize=16)
    ax_cev2.set_ylabel("Cumulative Variance", fontsize=16)
    ax_cev2.set_xlim(1, 64)
    ax_cev2.set_ylim(0, 1.02)
    ax_cev2.set_xticks([20, 40, 60])
    ax_cev2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_cev2.set_yticklabels([".0", ".2", ".4", ".6", ".8", "1"])
    ax_cev2.tick_params(axis="both", labelsize=14)
    ax_cev2.spines["top"].set_visible(False)
    ax_cev2.spines["right"].set_visible(False)
    ax_cev2.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax_cev2.text(-0.12, 1.05, "(a)", fontsize=18, fontweight="bold", transform=ax_cev2.transAxes)

    sm_noise2 = plt.cm.ScalarMappable(cmap=noise_cmap, norm=noise_norm)
    sm_noise2.set_array([])
    cbar_ax_cev2 = ax_cev2.inset_axes([0.45, 0.07, 0.52, 0.08])
    cbar_cev2 = plt.colorbar(sm_noise2, cax=cbar_ax_cev2, orientation="horizontal")
    cbar_cev2.outline.set_visible(False)
    cbar_cev2.set_ticks(noise_ticks)
    cbar_cev2.set_ticklabels(["0", ".05", ".1", ".2", ".3", ".5"])
    cbar_cev2.ax.tick_params(labelsize=11, top=True, bottom=False, labeltop=True, labelbottom=False)
    cbar_cev2.ax.set_xticks([0.001, 0.01], minor=True)
    cbar_cev2.ax.tick_params(which="minor", top=False, bottom=True, length=2)
    cbar_cev2.ax.text(
        0.5,
        0.45,
        "Noise ε",
        transform=cbar_cev2.ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
    )

    ax_nc2 = axes2[1]

    for noise in noise_levels:
        if noise not in results or not results[noise][nc_layer]["steps"]:
            continue
        data = list(zip(results[noise][nc_layer]["steps"], results[noise][nc_layer]["nc_95"]))
        data = sorted(data, key=lambda x: x[0])
        steps_nc = [d[0] for d in data]
        dims_nc = [d[1] for d in data]
        color = noise_cmap(noise_norm(float(noise)))
        ax_nc2.plot(
            steps_nc, dims_nc, color=color, linewidth=2.5, alpha=0.8, zorder=10 if noise == "0" else 1
        )

    ax_nc2.axhline(y=10, color="#888888", linestyle="--", linewidth=2.0)

    ax_nc2.set_xlabel("Training steps", fontsize=16)
    ax_nc2.set_ylabel("Dimensions for 95% CEV", fontsize=16)
    ax_nc2.set_xscale("log")
    ax_nc2.tick_params(axis="both", labelsize=14)
    ax_nc2.spines["top"].set_visible(False)
    ax_nc2.spines["right"].set_visible(False)
    ax_nc2.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax_nc2.text(-0.12, 1.05, "(b)", fontsize=18, fontweight="bold", transform=ax_nc2.transAxes)

    sm_noise3 = plt.cm.ScalarMappable(cmap=noise_cmap, norm=noise_norm)
    sm_noise3.set_array([])
    cbar_ax_nc2 = ax_nc2.inset_axes([0.55, 0.86, 0.42, 0.08])
    cbar_nc2 = plt.colorbar(sm_noise3, cax=cbar_ax_nc2, orientation="horizontal")
    cbar_nc2.outline.set_visible(False)
    cbar_nc2.set_ticks(noise_ticks)
    cbar_nc2.set_ticklabels(["0", ".05", ".1", ".2", ".3", ".5"])
    cbar_nc2.ax.tick_params(labelsize=11, top=False, bottom=True, labeltop=False, labelbottom=True)
    cbar_nc2.ax.set_xticks([0.001, 0.01], minor=True)
    cbar_nc2.ax.tick_params(which="minor", top=True, bottom=False, length=2)
    cbar_nc2.ax.text(
        0.5,
        0.55,
        "Noise ε",
        transform=cbar_nc2.ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
    )

    fig2.tight_layout()
    fig2_path = OUTPUT_DIR / f"combined_cev_nc95_by_noise_{layer_short_name}.png"
    plt.savefig(fig2_path, dpi=300)
    plt.show()
    print(f"Figure saved to {fig2_path}")

    # =========================================================================
    # Figure 3: Loss vs step + NC@95
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss = axes3[0]

    for noise in noise_levels:
        if noise not in results or not results[noise][nc_layer]["steps"]:
            continue
        data = list(zip(results[noise][nc_layer]["steps"], results[noise][nc_layer]["losses"]))
        data = sorted(data, key=lambda x: x[0])
        steps_loss = [d[0] for d in data]
        losses = [d[1] for d in data]
        color = noise_cmap(noise_norm(float(noise)))
        ax_loss.plot(
            steps_loss, losses, color=color, linewidth=2.5, alpha=0.8, zorder=10 if noise == "0" else 1
        )

    ax_loss.set_xlabel("Training steps", fontsize=16)
    ax_loss.set_ylabel("Loss", fontsize=16)
    ax_loss.set_xscale("log")
    ax_loss.tick_params(axis="both", labelsize=14)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax_loss.text(-0.12, 1.05, "(a)", fontsize=18, fontweight="bold", transform=ax_loss.transAxes)

    sm_noise_loss = plt.cm.ScalarMappable(cmap=noise_cmap, norm=noise_norm)
    sm_noise_loss.set_array([])
    cbar_ax_loss = ax_loss.inset_axes([0.55, 0.86, 0.42, 0.08])
    cbar_loss = plt.colorbar(sm_noise_loss, cax=cbar_ax_loss, orientation="horizontal")
    cbar_loss.outline.set_visible(False)
    cbar_loss.set_ticks(noise_ticks)
    cbar_loss.set_ticklabels(["0", ".05", ".1", ".2", ".3", ".5"])
    cbar_loss.ax.tick_params(labelsize=11, top=False, bottom=True, labeltop=False, labelbottom=True)
    cbar_loss.ax.set_xticks([0.001, 0.01], minor=True)
    cbar_loss.ax.tick_params(which="minor", top=True, bottom=False, length=2)
    cbar_loss.ax.text(
        0.5,
        0.55,
        "Noise ε",
        transform=cbar_loss.ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
    )

    ax_nc3 = axes3[1]

    for noise in noise_levels:
        if noise not in results or not results[noise][nc_layer]["steps"]:
            continue
        data = list(zip(results[noise][nc_layer]["steps"], results[noise][nc_layer]["nc_95"]))
        data = sorted(data, key=lambda x: x[0])
        steps_nc = [d[0] for d in data]
        dims_nc = [d[1] for d in data]
        color = noise_cmap(noise_norm(float(noise)))
        ax_nc3.plot(
            steps_nc, dims_nc, color=color, linewidth=2.5, alpha=0.8, zorder=10 if noise == "0" else 1
        )

    ax_nc3.axhline(y=10, color="#888888", linestyle="--", linewidth=2.0)

    ax_nc3.set_xlabel("Training steps", fontsize=16)
    ax_nc3.set_ylabel("Dimensions for 95% CEV", fontsize=16)
    ax_nc3.set_xscale("log")
    ax_nc3.tick_params(axis="both", labelsize=14)
    ax_nc3.spines["top"].set_visible(False)
    ax_nc3.spines["right"].set_visible(False)
    ax_nc3.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax_nc3.text(-0.12, 1.05, "(b)", fontsize=18, fontweight="bold", transform=ax_nc3.transAxes)

    sm_noise4 = plt.cm.ScalarMappable(cmap=noise_cmap, norm=noise_norm)
    sm_noise4.set_array([])
    cbar_ax_nc3 = ax_nc3.inset_axes([0.55, 0.86, 0.42, 0.08])
    cbar_nc3 = plt.colorbar(sm_noise4, cax=cbar_ax_nc3, orientation="horizontal")
    cbar_nc3.outline.set_visible(False)
    cbar_nc3.set_ticks(noise_ticks)
    cbar_nc3.set_ticklabels(["0", ".05", ".1", ".2", ".3", ".5"])
    cbar_nc3.ax.tick_params(labelsize=11, top=False, bottom=True, labeltop=False, labelbottom=True)
    cbar_nc3.ax.set_xticks([0.001, 0.01], minor=True)
    cbar_nc3.ax.tick_params(which="minor", top=True, bottom=False, length=2)
    cbar_nc3.ax.text(
        0.5,
        0.55,
        "Noise ε",
        transform=cbar_nc3.ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
    )

    fig3.tight_layout()
    fig3_path = OUTPUT_DIR / f"combined_loss_nc95_{layer_short_name}.png"
    plt.savefig(fig3_path, dpi=300)
    plt.show()
    print(f"Figure saved to {fig3_path}")

    # =========================================================================
    # Figure 4: NC@95 Derivative Phase Plot
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(8, 6))

    # Only plot ε=0.2
    noise = "0.2"
    if noise in results and results[noise][nc_layer]["steps"]:
        steps_phase = results[noise][nc_layer]["steps"]
        nc_95_phase = results[noise][nc_layer]["nc_95"]

        if len(steps_phase) >= 2:
            nc95_arr, derivative = compute_nc95_derivative(steps_phase, nc_95_phase)

            ax4.scatter(
                nc95_arr, derivative,
                color="#2d8a8a", s=30, alpha=0.8,
            )

    ax4.set_xlabel("Dimensions for 95% CEV (NC@95)", fontsize=14)
    ax4.set_ylabel("d(NC@95)/d(steps)", fontsize=14)
    ax4.set_ylim(-0.1, 0.1)
    ax4.axhline(y=0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.5)

    ax4.tick_params(axis="both", labelsize=12)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

    # Add label for noise level
    ax4.text(0.95, 0.95, "ε = 0.2", fontsize=14, transform=ax4.transAxes,
             ha="right", va="top")

    fig4.tight_layout()

    fig4_path = OUTPUT_DIR / f"nc95_derivative_phase_{layer_short_name}.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    print(f"Derivative phase plot saved to {fig4_path}")

    # =========================================================================
    # Figure 5: Steps vs NC@95 (linear scale) for ε=0.2
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(8, 6))

    noise = "0.2"
    if noise in results and results[noise][nc_layer]["steps"]:
        steps_lin = results[noise][nc_layer]["steps"]
        nc_95_lin = results[noise][nc_layer]["nc_95"]

        # Sort by steps
        data = sorted(zip(steps_lin, nc_95_lin), key=lambda x: x[0])
        steps_lin = [d[0] for d in data]
        nc_95_lin = [d[1] for d in data]

        ax5.scatter(
            steps_lin, nc_95_lin,
            color="#2d8a8a", s=30, alpha=0.8,
        )

    ax5.axhline(y=10, color="#888888", linestyle="--", linewidth=2.0)
    ax5.set_xlabel("Training steps", fontsize=14)
    ax5.set_ylabel("Dimensions for 95% CEV (NC@95)", fontsize=14)
    ax5.tick_params(axis="both", labelsize=12)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

    ax5.text(0.95, 0.95, "ε = 0.2", fontsize=14, transform=ax5.transAxes,
             ha="right", va="top")

    fig5.tight_layout()

    fig5_path = OUTPUT_DIR / f"steps_vs_nc95_linear_{layer_short_name}.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close(fig5)
    print(f"Steps vs NC@95 (linear) saved to {fig5_path}")

# =============================================================================
# Part 5: CEV by Training Step for All Noise Levels
# =============================================================================

# %%
# Generate CEV-by-step plots for each noise level (like Figure 1 left panel)
print("\n=== Generating CEV-by-step plots for all noise levels ===")

for noise in noise_levels:
    if noise not in raw_results:
        print(f"ε={noise}: No data available, skipping")
        continue

    for nc_layer, layer_short_name in layer_pairs:
        if nc_layer not in raw_results[noise]:
            continue

        layer_data = raw_results[noise][nc_layer]
        if not layer_data["steps"] or not layer_data["variance_ratios"]:
            print(f"ε={noise}, {layer_short_name}: No variance ratios, skipping")
            continue

        fig, ax_cev = plt.subplots(figsize=(6, 5))

        # Colormap for training steps
        steps = np.array(layer_data["steps"])
        if len(steps) == 0:
            plt.close(fig)
            continue

        step_norm = mcolors.LogNorm(vmin=max(steps.min(), 1), vmax=max(steps.max(), 2))
        step_cmap = plt.cm.viridis_r

        for step, var_ratios in zip(layer_data["steps"], layer_data["variance_ratios"]):
            if var_ratios is None:
                continue
            cev = np.cumsum(var_ratios)
            dims = np.arange(1, len(cev) + 1)
            color = step_cmap(step_norm(max(step, 1)))
            ax_cev.plot(dims, cev, color=color, linewidth=2.5, alpha=0.8)

        # Add ground truth lines
        ax_cev.plot(factored_dims, factored_cev, color=gt_color, linestyle="--", linewidth=3.0, zorder=100)
        ax_cev.plot(
            noised_joint_dims, noised_joint_cev, color=gt_color, linestyle="--", linewidth=3.0, zorder=100
        )

        # Text labels for ground truth
        ax_cev.text(5, 1.0, "Factored", fontsize=13, color=gt_color, ha="left", va="bottom")
        joint_label_idx = min(50, len(noised_joint_cev) - 1)
        ax_cev.text(
            noised_joint_dims[joint_label_idx],
            noised_joint_cev[joint_label_idx] - 0.05,
            f"Joint (ε={noise})",
            fontsize=13,
            color=gt_color,
            ha="center",
            va="top",
        )

        ax_cev.set_xlabel("Dimension", fontsize=16)
        ax_cev.set_ylabel("Cumulative Variance", fontsize=16)
        ax_cev.set_xlim(1, 64)
        ax_cev.set_ylim(0, 1.02)
        ax_cev.set_xticks([20, 40, 60])
        ax_cev.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_cev.set_yticklabels([".0", ".2", ".4", ".6", ".8", "1"])
        ax_cev.tick_params(axis="both", labelsize=14)
        ax_cev.spines["top"].set_visible(False)
        ax_cev.spines["right"].set_visible(False)
        ax_cev.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

        # Colorbar for training steps
        sm_step = plt.cm.ScalarMappable(cmap=step_cmap, norm=step_norm)
        sm_step.set_array([])
        cbar_ax_cev = ax_cev.inset_axes([0.45, 0.07, 0.52, 0.08])
        cbar_cev = plt.colorbar(sm_step, cax=cbar_ax_cev, orientation="horizontal")
        cbar_cev.outline.set_visible(False)
        tick_values = [1, 10, 100, 1000, 5000, 10000, 50000, 100000, 500000]
        min_step = max(min(layer_data["steps"]), 1)
        max_step = max(layer_data["steps"])
        tick_values = [t for t in tick_values if min_step <= t <= max_step]
        if len(tick_values) > 5:
            tick_values = tick_values[::2]  # Take every other tick if too many
        cbar_cev.set_ticks(tick_values)
        cbar_cev.set_ticklabels([f"{t}" if t < 1000 else f"{t // 1000}k" for t in tick_values])
        cbar_cev.ax.tick_params(labelsize=11, top=True, bottom=False, labeltop=True, labelbottom=False)
        cbar_cev.ax.text(
            0.5,
            0.45,
            "Training Step",
            transform=cbar_cev.ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="white",
            fontweight="bold",
        )

        fig.tight_layout()

        # Save with noise level in filename
        noise_str = noise.replace(".", "_")  # Replace . with _ for filename
        fig_path = OUTPUT_DIR / f"cev_by_step_eps_{noise_str}_{layer_short_name}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"ε={noise}: Saved {fig_path.name}")

print("\nAll CEV-by-step plots generated!")

print("\nAll figures generated!")
print(f"Output directory: {OUTPUT_DIR}")
