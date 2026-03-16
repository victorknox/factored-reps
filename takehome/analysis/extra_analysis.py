#!/usr/bin/env python3
"""Extra analysis: Layerwise emergence of component identity vs. within-component belief.

Hypothesis: Component identity (which Mess3 process) should emerge earlier in the network
than fine-grained within-component belief states, because component discrimination is a
coarser, more global property that can be read from token statistics.

We test this by:
1. Training linear probes at every (layer, position) for both targets
2. Comparing the R² curves to see which information appears first
3. Also: within-component PCA structure (do we see simplex-like geometry per component?)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.models.attention_only import AttentionOnly, AttentionOnlyConfig


def load_model(checkpoint_dir, device="cpu"):
    with open(checkpoint_dir / "model_config.json") as f:
        mcfg = json.load(f)
    cfg = AttentionOnlyConfig(**mcfg, seed=42)
    model = AttentionOnly(cfg)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, cfg


def extract_activations(model, tokens, batch_size=256, device="cpu"):
    n_seq = tokens.shape[0]
    inputs = torch.tensor(tokens[:, :-1], dtype=torch.long)
    activations = {}
    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            batch = inputs[start:end].to(device)
            _, cache = model.run_with_cache(batch)
            for name, tensor in cache.items():
                if tensor.ndim != 3:
                    continue
                if name not in activations:
                    activations[name] = []
                activations[name].append(tensor.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in activations.items()}


def run_emergence_analysis(activations, layer_keys, component_labels,
                           within_beliefs, comp_posteriors, output_dir):
    """Compare emergence timing of component identity vs within-component belief."""
    n_seq, n_pos, d_model = activations[layer_keys[0]].shape
    n_train = int(n_seq * 0.8)

    # Store results: (layer_idx, position) -> metrics
    comp_acc_grid = np.zeros((len(layer_keys), n_pos))
    belief_r2_grid = np.zeros((len(layer_keys), n_pos))
    comp_r2_grid = np.zeros((len(layer_keys), n_pos))

    for li, lk in enumerate(layer_keys):
        acts = activations[lk]
        layer_name = lk.split('.')[-1] if '.' in lk else lk
        print(f"  Layer {layer_name}...")

        for pos in range(n_pos):
            X = acts[:, pos, :]
            X_train, X_test = X[:n_train], X[n_train:]

            # Component classification
            y_cls_train = component_labels[:n_train]
            y_cls_test = component_labels[n_train:]
            clf = LogisticRegression(max_iter=500, C=1.0).fit(X_train, y_cls_train)
            comp_acc_grid[li, pos] = clf.score(X_test, y_cls_test)

            # Component posterior regression
            y_comp = comp_posteriors[:, pos, :]
            reg = Ridge(alpha=1.0).fit(X_train, y_comp[:n_train])
            pred = reg.predict(X_test)
            comp_r2_grid[li, pos] = r2_score(y_comp[n_train:], pred)

            # Within-component belief regression
            y_bel = within_beliefs[:, pos, :]
            reg = Ridge(alpha=1.0).fit(X_train, y_bel[:n_train])
            pred = reg.predict(X_test)
            belief_r2_grid[li, pos] = r2_score(y_bel[n_train:], pred)

    # Heatmap: component accuracy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layer_labels = [lk.split('.')[-1] if '.' in lk else lk for lk in layer_keys]

    for ax, data, title in zip(axes,
                                [comp_acc_grid, comp_r2_grid, belief_r2_grid],
                                ["Component Clf Accuracy", "Component Posterior R²",
                                 "Within-Belief R²"]):
        im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_xlabel("Context Position")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Emergence: Component Identity vs Within-Component Belief", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "emergence_heatmap.png", dpi=150)
    plt.close(fig)

    # Line plot: average over positions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(layer_keys)), comp_acc_grid.mean(axis=1), 'o-', label="Comp Clf Acc (avg)")
    ax.plot(range(len(layer_keys)), comp_r2_grid.mean(axis=1), 's-', label="Comp Post R² (avg)")
    ax.plot(range(len(layer_keys)), belief_r2_grid.mean(axis=1), '^-', label="Belief R² (avg)")
    ax.set_xlabel("Layer")
    ax.set_xticks(range(len(layer_keys)))
    ax.set_xticklabels(layer_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Layerwise Emergence (averaged over positions)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "emergence_by_layer.png", dpi=150)
    plt.close(fig)

    return {
        "comp_acc_grid": comp_acc_grid.tolist(),
        "comp_r2_grid": comp_r2_grid.tolist(),
        "belief_r2_grid": belief_r2_grid.tolist(),
        "layer_keys": layer_keys,
    }


def run_per_component_pca(activations, layer_keys, component_labels,
                          within_beliefs, output_dir):
    """PCA within each component to see if per-component belief simplices emerge."""
    n_seq, n_pos, d_model = activations[layer_keys[0]].shape
    last_layer = layer_keys[-1]
    acts = activations[last_layer]

    # Last position
    pos = n_pos - 1
    pos_acts = acts[:, pos, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for c in range(3):
        ax = axes[c]
        mask = component_labels == c
        comp_acts = pos_acts[mask]
        beliefs_c = within_beliefs[mask, pos, :]

        n_plot = min(2000, comp_acts.shape[0])
        idx = np.random.RandomState(c).choice(comp_acts.shape[0], n_plot, replace=False)

        pca = PCA(n_components=2)
        proj = pca.fit_transform(comp_acts[idx])

        # Color by barycentric coordinate (first belief dimension)
        scatter = ax.scatter(proj[:, 0], proj[:, 1],
                           c=beliefs_c[idx, 0], cmap='coolwarm',
                           s=3, alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label="Belief[0]")
        ax.set_title(f"Component {c} (n={mask.sum()})\nvar={pca.explained_variance_ratio_[:2].sum():.2f}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    fig.suptitle(f"Per-Component PCA (last layer, pos={pos})", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "per_component_pca.png", dpi=150)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "results"
    checkpoint_dir = results_dir / "checkpoints"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, cfg = load_model(checkpoint_dir, args.device)

    print("Loading data...")
    val_data = np.load(results_dir / "val_data.npz")
    tokens = val_data["tokens"]
    component_labels = val_data["component_labels"]
    within_beliefs = val_data["within_beliefs"]
    comp_posteriors = val_data["component_posteriors"]

    n_pos = tokens.shape[1] - 1
    within_beliefs = within_beliefs[:, :n_pos, :]
    comp_posteriors = comp_posteriors[:, :n_pos, :]

    print("Extracting activations...")
    activations = extract_activations(model, tokens, batch_size=512, device=args.device)

    layer_keys = [k for k in sorted(activations.keys())
                  if 'resid_post' in k or 'ln_final' in k]
    if not layer_keys:
        layer_keys = [k for k in sorted(activations.keys())
                     if activations[k].ndim == 3 and activations[k].shape[-1] == cfg.d_model]

    print(f"Layers: {layer_keys}")

    print("\n=== Emergence Analysis ===")
    emergence_results = run_emergence_analysis(
        activations, layer_keys, component_labels,
        within_beliefs, comp_posteriors, figures_dir
    )

    print("\n=== Per-Component PCA ===")
    run_per_component_pca(activations, layer_keys, component_labels,
                          within_beliefs, figures_dir)

    with open(results_dir / "extra_analysis_results.json", "w") as f:
        json.dump(emergence_results, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
