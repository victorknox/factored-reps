#!/usr/bin/env python3
"""Post-training analysis: PCA geometry, linear probes, and belief regression.

This script:
1. Loads trained model and validation data
2. Extracts residual stream activations at each layer and position
3. Runs PCA / cumulative explained variance analysis
4. Fits linear probes for component identity and belief states
5. Generates all figures for the report
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def load_model(checkpoint_dir, device="cpu"):
    with open(checkpoint_dir / "model_config.json") as f:
        mcfg = json.load(f)

    arch = mcfg.get("architecture", "attn_only")

    if arch == "full":
        from transformer_lens import HookedTransformer, HookedTransformerConfig
        cfg = HookedTransformerConfig(
            d_model=mcfg["d_model"], d_head=mcfg["d_head"],
            n_heads=mcfg["n_heads"], n_layers=mcfg["n_layers"],
            n_ctx=mcfg["n_ctx"], d_mlp=mcfg["d_mlp"],
            d_vocab=mcfg["d_vocab"], act_fn=mcfg.get("act_fn", "relu"),
            normalization_type=mcfg.get("normalization_type", "LN"),
            device=device, seed=42,
        )
        model = HookedTransformer(cfg)
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt",
                                         map_location=device, weights_only=True))
        model.eval()
    else:
        from experiments.models.attention_only import AttentionOnly, AttentionOnlyConfig
        cfg = AttentionOnlyConfig(
            d_vocab=mcfg["d_vocab"], d_model=mcfg["d_model"],
            n_heads=mcfg["n_heads"], n_layers=mcfg["n_layers"],
            n_ctx=mcfg["n_ctx"],
            normalization_type=mcfg.get("normalization_type", "LN"), seed=42,
        )
        model = AttentionOnly(cfg)
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt",
                                         map_location=device, weights_only=True))
        model.to(device)
        model.eval()

    return model, mcfg


def extract_activations(model, tokens, batch_size=256, device="cpu"):
    """Extract residual stream activations at each layer for all positions.

    Returns dict: layer_name -> (n_sequences, n_positions, d_model)
    Layer names: 'embed', 'block_0', ..., 'block_{L-1}', 'ln_final'
    """
    n_seq, seq_len = tokens.shape
    inputs = torch.tensor(tokens[:, :-1], dtype=torch.long)  # (n_seq, seq_len-1)
    n_pos = inputs.shape[1]

    activations = {}

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            batch = inputs[start:end].to(device)

            # Run with cache
            _, cache = model.run_with_cache(batch)

            for name, tensor in cache.items():
                if tensor.ndim != 3:  # only (batch, pos, d_model)
                    continue
                if name not in activations:
                    activations[name] = []
                activations[name].append(tensor.cpu().numpy())

    # Concatenate batches
    result = {}
    for name, chunks in activations.items():
        result[name] = np.concatenate(chunks, axis=0)

    return result


def get_residual_stream_keys(activations):
    """Get keys for residual stream activations in order."""
    keys = []
    # Look for the standard hook names
    for k in sorted(activations.keys()):
        if 'resid_post' in k or 'hook_embed' in k or 'hook_pos_embed' in k or 'ln_final' in k:
            keys.append(k)

    # If no standard names found, try to identify layers
    if not keys:
        keys = sorted(activations.keys())

    return keys


def run_pca_analysis(activations, layer_keys, output_dir):
    """PCA and cumulative explained variance by layer and position."""
    results = {}
    n_seq, n_pos, d_model = activations[layer_keys[0]].shape

    fig_cev, axes_cev = plt.subplots(1, len(layer_keys), figsize=(4*len(layer_keys), 4), squeeze=False)

    for li, lk in enumerate(layer_keys):
        acts = activations[lk]  # (n_seq, n_pos, d_model)
        layer_name = lk.split('.')[-1] if '.' in lk else lk

        # PCA over all positions combined
        flat = acts.reshape(-1, d_model)
        pca = PCA(n_components=min(20, d_model))
        pca.fit(flat)
        cev = np.cumsum(pca.explained_variance_ratio_)

        results[lk] = {
            "cev": cev.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components_90": int(np.searchsorted(cev, 0.9) + 1),
            "n_components_95": int(np.searchsorted(cev, 0.95) + 1),
            "n_components_99": int(np.searchsorted(cev, 0.99) + 1),
        }

        # Per-position PCA
        pos_dims = []
        for t in range(n_pos):
            pos_acts = acts[:, t, :]
            pca_pos = PCA(n_components=min(10, d_model))
            pca_pos.fit(pos_acts)
            cev_pos = np.cumsum(pca_pos.explained_variance_ratio_)
            pos_dims.append(int(np.searchsorted(cev_pos, 0.95) + 1))
        results[lk]["dims_95_by_position"] = pos_dims

        # Plot CEV
        ax = axes_cev[0, li]
        ax.plot(range(1, len(cev)+1), cev, 'o-', markersize=3)
        ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("# Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title(f"{layer_name}")
        ax.set_ylim(0, 1.05)

    fig_cev.tight_layout()
    fig_cev.savefig(output_dir / "pca_cev_by_layer.png", dpi=150)
    plt.close(fig_cev)

    # Plot dimensionality by position for each layer
    fig_dim, ax_dim = plt.subplots(figsize=(8, 4))
    for lk in layer_keys:
        layer_name = lk.split('.')[-1] if '.' in lk else lk
        dims = results[lk]["dims_95_by_position"]
        ax_dim.plot(range(len(dims)), dims, 'o-', markersize=4, label=layer_name)
    ax_dim.set_xlabel("Context Position")
    ax_dim.set_ylabel("# PCA dims for 95% var")
    ax_dim.set_title("Effective Dimensionality by Position and Layer")
    ax_dim.legend(fontsize=8)
    fig_dim.tight_layout()
    fig_dim.savefig(output_dir / "dimensionality_by_position.png", dpi=150)
    plt.close(fig_dim)

    return results


def run_pca_visualizations(activations, layer_keys, component_labels, within_beliefs,
                           comp_posteriors, output_dir):
    """2D PCA scatter plots colored by component and belief coordinates."""
    n_seq, n_pos, d_model = activations[layer_keys[0]].shape

    # Use last 3 positions for visualization
    positions_to_plot = [0, n_pos//4, n_pos//2, 3*n_pos//4, n_pos-1]
    positions_to_plot = [p for p in positions_to_plot if p < n_pos]

    colors_comp = ['#e41a1c', '#377eb8', '#4daf4a']

    for li, lk in enumerate(layer_keys):
        acts = activations[lk]
        layer_name = lk.split('.')[-1] if '.' in lk else lk

        fig, axes = plt.subplots(2, len(positions_to_plot),
                                  figsize=(4*len(positions_to_plot), 8))

        for pi, pos in enumerate(positions_to_plot):
            pos_acts = acts[:, pos, :]

            # Subsample for plotting
            n_plot = min(3000, n_seq)
            idx = np.random.RandomState(0).choice(n_seq, n_plot, replace=False)

            pca = PCA(n_components=2)
            proj = pca.fit_transform(pos_acts[idx])

            # Top row: colored by component
            ax = axes[0, pi]
            for c in range(3):
                mask = component_labels[idx] == c
                ax.scatter(proj[mask, 0], proj[mask, 1], c=colors_comp[c],
                          s=2, alpha=0.3, label=f"C{c}")
            ax.set_title(f"pos={pos}")
            if pi == 0:
                ax.set_ylabel(f"{layer_name}\nby component")
            ax.legend(fontsize=6, markerscale=3)

            # Bottom row: colored by max belief coordinate (within true component)
            ax = axes[1, pi]
            beliefs_pos = within_beliefs[idx, pos, :]  # (n_plot, 3)
            max_belief_state = np.argmax(beliefs_pos, axis=1)
            for s in range(3):
                mask = max_belief_state == s
                ax.scatter(proj[mask, 0], proj[mask, 1], c=colors_comp[s],
                          s=2, alpha=0.3, label=f"state {s}")
            if pi == 0:
                ax.set_ylabel("by hidden state")
            ax.legend(fontsize=6, markerscale=3)

        fig.suptitle(f"PCA of {layer_name} activations", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f"pca_scatter_{layer_name}.png", dpi=150)
        plt.close(fig)


def run_probe_analysis(activations, layer_keys, component_labels, within_beliefs,
                       comp_posteriors, joint_beliefs, next_token_probs, output_dir):
    """Linear probe analysis: recover component identity and belief states from activations."""
    n_seq, n_pos, d_model = activations[layer_keys[0]].shape
    n_train = int(n_seq * 0.8)

    results = {}

    for lk in layer_keys:
        acts = activations[lk]  # (n_seq, n_pos, d_model)
        layer_name = lk.split('.')[-1] if '.' in lk else lk
        layer_results = {}

        for pos in range(n_pos):
            X = acts[:, pos, :]
            X_train, X_test = X[:n_train], X[n_train:]

            pos_results = {}

            # 1. Component posterior probe
            y_comp = comp_posteriors[:, pos, :]  # (n_seq, K)
            y_train, y_test = y_comp[:n_train], y_comp[n_train:]
            reg = Ridge(alpha=1.0).fit(X_train, y_train)
            pred = reg.predict(X_test)
            r2_comp = r2_score(y_test, pred)
            pos_results["component_posterior_r2"] = r2_comp

            # 2. Component classification accuracy
            y_cls = component_labels
            y_cls_train, y_cls_test = y_cls[:n_train], y_cls[n_train:]
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(X_train, y_cls_train)
            acc = clf.score(X_test, y_cls_test)
            pos_results["component_classification_acc"] = acc

            # 3. Within-component belief probe
            y_belief = within_beliefs[:, pos, :]
            y_train, y_test = y_belief[:n_train], y_belief[n_train:]
            reg = Ridge(alpha=1.0).fit(X_train, y_train)
            pred = reg.predict(X_test)
            r2_belief = r2_score(y_test, pred)
            pos_results["within_belief_r2"] = r2_belief

            # 4. Joint belief probe
            y_joint = joint_beliefs[:, pos, :]  # (n_seq, K*3)
            y_train, y_test = y_joint[:n_train], y_joint[n_train:]
            reg = Ridge(alpha=1.0).fit(X_train, y_train)
            pred = reg.predict(X_test)
            r2_joint = r2_score(y_test, pred)
            pos_results["joint_belief_r2"] = r2_joint

            # 5. Next-token prediction probe
            y_ntp = next_token_probs[:, pos, :]
            y_train, y_test = y_ntp[:n_train], y_ntp[n_train:]
            reg = Ridge(alpha=1.0).fit(X_train, y_train)
            pred = reg.predict(X_test)
            r2_ntp = r2_score(y_test, pred)
            pos_results["next_token_r2"] = r2_ntp

            layer_results[pos] = pos_results

        results[lk] = layer_results

    # Plot probe R² by position and layer
    metrics = ["component_posterior_r2", "component_classification_acc",
               "within_belief_r2", "joint_belief_r2", "next_token_r2"]
    metric_labels = ["Component Posterior R²", "Component Clf Acc",
                     "Within-Belief R²", "Joint Belief R²", "Next-Token R²"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4), squeeze=False)

    for mi, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[0, mi]
        for lk in layer_keys:
            layer_name = lk.split('.')[-1] if '.' in lk else lk
            vals = [results[lk][pos][metric] for pos in range(n_pos)]
            ax.plot(range(n_pos), vals, 'o-', markersize=3, label=layer_name)
        ax.set_xlabel("Position")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=6)
        ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "probe_r2_by_position.png", dpi=150)
    plt.close(fig)

    # Plot probe R² by layer for fixed positions (early, mid, late)
    pos_early = 1
    pos_mid = n_pos // 2
    pos_late = n_pos - 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for pi, (pos, pos_label) in enumerate([(pos_early, "Early"), (pos_mid, "Mid"), (pos_late, "Late")]):
        ax = axes[pi]
        for mi, (metric, label) in enumerate(zip(metrics, metric_labels)):
            vals = [results[lk][pos][metric] for lk in layer_keys]
            ax.plot(range(len(layer_keys)), vals, 'o-', markersize=5, label=label)
        ax.set_xlabel("Layer")
        ax.set_xticks(range(len(layer_keys)))
        ax.set_xticklabels([lk.split('.')[-1] if '.' in lk else lk for lk in layer_keys],
                           rotation=45, fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title(f"Position {pos} ({pos_label})")
        ax.legend(fontsize=6)
        ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "probe_r2_by_layer.png", dpi=150)
    plt.close(fig)

    return results


def run_joint_geometry_visualization(activations, layer_keys, component_labels,
                                     within_beliefs, comp_posteriors, output_dir):
    """3D PCA visualization showing joint component + belief structure."""
    n_seq = activations[layer_keys[0]].shape[0]
    n_pos = activations[layer_keys[0]].shape[1]

    # Focus on last layer, multiple positions
    last_layer = layer_keys[-1]
    acts = activations[last_layer]

    positions = [0, n_pos//3, 2*n_pos//3, n_pos-1]
    positions = [p for p in positions if p < n_pos]

    fig, axes = plt.subplots(1, len(positions), figsize=(5*len(positions), 5))
    if len(positions) == 1:
        axes = [axes]

    n_plot = min(3000, n_seq)
    idx = np.random.RandomState(0).choice(n_seq, n_plot, replace=False)

    colors_comp = np.array(['#e41a1c', '#377eb8', '#4daf4a'])

    for pi, pos in enumerate(positions):
        ax = axes[pi]
        pos_acts = acts[idx, pos, :]
        pca = PCA(n_components=3)
        proj = pca.fit_transform(pos_acts)

        # Color by component, alpha by belief certainty (max belief coordinate)
        labels = component_labels[idx]
        beliefs = within_beliefs[idx, pos, :]
        certainty = np.max(beliefs, axis=1)  # how peaked the belief is

        for c in range(3):
            mask = labels == c
            ax.scatter(proj[mask, 0], proj[mask, 1],
                      c=colors_comp[c], s=3, alpha=0.2 + 0.6 * certainty[mask].mean(),
                      label=f"C{c}")

        ax.set_title(f"pos={pos}\nvar={pca.explained_variance_ratio_[:2].sum():.2f}")
        ax.legend(fontsize=7, markerscale=3)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    fig.suptitle(f"PCA of {last_layer.split('.')[-1]} by position", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "joint_geometry.png", dpi=150)
    plt.close(fig)


def plot_training_curves(output_dir, checkpoint_dir=None):
    """Plot training and validation loss curves."""
    hist_path = None
    candidates = []
    if checkpoint_dir:
        candidates.append(checkpoint_dir / "training_history.json")
    candidates += [
        output_dir.parent / "checkpoints_full" / "training_history.json",
        output_dir.parent / "checkpoints" / "training_history.json",
        output_dir / "training_history.json",
    ]
    for p in candidates:
        if p.exists():
            hist_path = p
            break
    if hist_path is None:
        print("No training history found, skipping loss curves.")
        return

    with open(hist_path) as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_losses"]) + 1)
    ax.plot(epochs, history["train_losses"], label="Train")
    ax.plot(epochs, history["val_losses"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add horizontal line for entropy of uniform over 3 tokens
    ax.axhline(np.log(3), color='gray', linestyle=':', label=f"H(uniform)={np.log(3):.3f}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "results"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else results_dir / "checkpoints"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, cfg = load_model(checkpoint_dir, args.device)

    print("Loading validation data...")
    val_data = np.load(results_dir / "val_data.npz")
    tokens = val_data["tokens"]
    component_labels = val_data["component_labels"]
    within_beliefs = val_data["within_beliefs"]
    comp_posteriors = val_data["component_posteriors"]
    joint_beliefs = val_data["joint_beliefs"]
    next_token_probs = val_data["next_token_probs"]

    print(f"Data: {tokens.shape[0]} sequences, seq_len={tokens.shape[1]}")

    print("Extracting activations...")
    activations = extract_activations(model, tokens, batch_size=512, device=args.device)
    print(f"  Extracted {len(activations)} activation tensors")
    for k, v in sorted(activations.items()):
        print(f"    {k}: {v.shape}")

    # Identify residual stream layers (must be 3D with last dim = d_model)
    d_model = cfg["d_model"]
    layer_keys = []
    for k in sorted(activations.keys()):
        if activations[k].ndim == 3 and activations[k].shape[-1] == d_model:
            if 'resid_post' in k or 'resid_mid' in k:
                layer_keys.append(k)

    # Add final ln if present
    for k in sorted(activations.keys()):
        if ('ln_final' in k or 'hook_ln_final' in k) and activations[k].ndim == 3 and activations[k].shape[-1] == d_model:
            layer_keys.append(k)
            break

    # Fallback
    if not layer_keys:
        layer_keys = [k for k in sorted(activations.keys())
                     if activations[k].ndim == 3 and activations[k].shape[-1] == d_model]

    print(f"\nResidual stream layers: {layer_keys}")

    # Adjust beliefs to match input positions (tokens[:,:-1] is input, so positions 0..seq_len-2)
    n_pos = tokens.shape[1] - 1
    within_beliefs = within_beliefs[:, :n_pos, :]
    comp_posteriors = comp_posteriors[:, :n_pos, :]
    joint_beliefs = joint_beliefs[:, :n_pos, :]
    next_token_probs = next_token_probs[:, :n_pos, :]

    # Run analyses
    print("\n=== PCA Analysis ===")
    pca_results = run_pca_analysis(activations, layer_keys, figures_dir)

    print("\n=== PCA Visualizations ===")
    run_pca_visualizations(activations, layer_keys, component_labels,
                           within_beliefs, comp_posteriors, figures_dir)

    print("\n=== Linear Probe Analysis ===")
    probe_results = run_probe_analysis(activations, layer_keys, component_labels,
                                       within_beliefs, comp_posteriors,
                                       joint_beliefs, next_token_probs, figures_dir)

    print("\n=== Joint Geometry Visualization ===")
    run_joint_geometry_visualization(activations, layer_keys, component_labels,
                                     within_beliefs, comp_posteriors, figures_dir)

    print("\n=== Training Curves ===")
    plot_training_curves(figures_dir, checkpoint_dir)

    # Save all results
    all_results = {
        "pca": pca_results,
        "probes": {lk: {str(pos): {k: float(v) for k, v in pv.items()}
                        for pos, pv in lr.items()}
                   for lk, lr in probe_results.items()},
        "layer_keys": layer_keys,
    }
    with open(results_dir / "analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_dir}/")
    print(f"Figures saved to {figures_dir}/")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Layer':<30} {'Comp Post R²':>15} {'Comp Clf Acc':>15} {'Belief R²':>12} {'Joint R²':>12} {'NTP R²':>10}")
    print("-" * 100)
    for lk in layer_keys:
        layer_name = lk.split('.')[-1] if '.' in lk else lk
        # Last position
        pos = n_pos - 1
        pr = probe_results[lk][pos]
        print(f"{layer_name:<30} {pr['component_posterior_r2']:>15.3f} "
              f"{pr['component_classification_acc']:>15.3f} {pr['within_belief_r2']:>12.3f} "
              f"{pr['joint_belief_r2']:>12.3f} {pr['next_token_r2']:>10.3f}")


if __name__ == "__main__":
    main()
