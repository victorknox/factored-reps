#!/usr/bin/env python3
"""Run all 7 experiments from the experimental specification document.

Uses full transformer (HookedTransformer with MLP blocks).
Experiments:
1. Verify hierarchical posterior (10-fold CV, RMSE over training steps)
2. Effective dimensionality vs context position (entropy overlay, KL-based ℓ*)
3. Component separability vs position & layer (heatmap)
4. Geometry visualization (PCA scatter early/mid/late, 3D)
5. Fractal geometry recovery (K×2 grid: truth vs recovered)
6. Dimensionality scaling with K (separate script)
7. Subspace orthogonality test (principal angles)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────

def load_model(checkpoint_dir, device="cpu", checkpoint_name="best_model.pt"):
    with open(checkpoint_dir / "model_config.json") as f:
        mcfg = json.load(f)

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
    model.load_state_dict(torch.load(checkpoint_dir / checkpoint_name,
                                     map_location=device, weights_only=True))
    model.eval()
    return model, mcfg


def extract_activations(model, tokens, batch_size=256, device="cpu"):
    """Extract residual stream activations: layer_name -> (n_seq, n_pos, d_model)."""
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


def get_residual_keys(activations, d_model):
    """Get ordered residual stream layer keys for HookedTransformer.

    HookedTransformer names: blocks.0.hook_resid_post, blocks.1.hook_resid_post, ...
    Also includes blocks.*.hook_resid_mid for mid-layer (post-attn, pre-MLP).
    """
    resid_post = []
    resid_mid = []
    ln_final = None

    for k in sorted(activations.keys()):
        if activations[k].ndim != 3 or activations[k].shape[-1] != d_model:
            continue
        if 'hook_resid_post' in k:
            resid_post.append(k)
        elif 'hook_resid_mid' in k:
            resid_mid.append(k)
        elif 'ln_final' in k:
            ln_final = k

    # Interleave: resid_mid_0, resid_post_0, resid_mid_1, resid_post_1, ..., ln_final
    keys = []
    resid_mid_dict = {k.split('.')[1]: k for k in resid_mid}
    resid_post_dict = {k.split('.')[1]: k for k in resid_post}
    all_blocks = sorted(set(list(resid_mid_dict.keys()) + list(resid_post_dict.keys())))

    for block in all_blocks:
        if block in resid_mid_dict:
            keys.append(resid_mid_dict[block])
        if block in resid_post_dict:
            keys.append(resid_post_dict[block])

    if ln_final:
        keys.append(ln_final)

    if not keys:
        # Fallback
        keys = [k for k in sorted(activations.keys())
                if activations[k].ndim == 3 and activations[k].shape[-1] == d_model]
    return keys


def get_resid_post_keys(activations, d_model):
    """Get only resid_post keys + ln_final (one per layer)."""
    keys = []
    for k in sorted(activations.keys()):
        if activations[k].ndim == 3 and activations[k].shape[-1] == d_model:
            if 'hook_resid_post' in k:
                keys.append(k)
    for k in sorted(activations.keys()):
        if 'ln_final' in k and activations[k].ndim == 3 and activations[k].shape[-1] == d_model:
            keys.append(k)
            break
    return keys


def short_name(lk):
    parts = lk.split('.')
    if len(parts) >= 3:
        return f"L{parts[1]}_{parts[2].replace('hook_', '')}"
    return lk.split('.')[-1] if '.' in lk else lk


# ─────────────────────────────────────────────
# Experiment 1: Verify Hierarchical Posterior
# ─────────────────────────────────────────────

def experiment_1(activations, layer_keys, component_labels, within_beliefs,
                 comp_posteriors, joint_beliefs, checkpoint_dir, results_dir,
                 tokens, device, output_dir, d_model):
    """10-fold CV regression from activations to Y, q_c, η_c.
    Also RMSE over training steps using epoch checkpoints."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Verify Hierarchical Posterior")
    print("="*60)

    n_seq, n_pos, _ = activations[layer_keys[0]].shape

    # Use last resid_post layer (before ln_final)
    # Find the last resid_post key
    last_resid_post = [k for k in layer_keys if 'resid_post' in k][-1]
    ln_final_key = [k for k in layer_keys if 'ln_final' in k]
    final_key = ln_final_key[0] if ln_final_key else last_resid_post
    acts_final = activations[final_key]

    targets = {
        "joint_posterior_Y": joint_beliefs,
        "component_posterior_q": comp_posteriors,
        "within_belief_eta": within_beliefs,
    }

    cv_results = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for target_name, target_data in targets.items():
        print(f"\n  {target_name}: 10-fold CV...")
        X_all = acts_final.reshape(-1, d_model)
        Y_all = target_data.reshape(-1, target_data.shape[-1])

        n_total = X_all.shape[0]
        if n_total > 20000:
            idx = np.random.RandomState(42).choice(n_total, 20000, replace=False)
            X_sub, Y_sub = X_all[idx], Y_all[idx]
        else:
            X_sub, Y_sub = X_all, Y_all

        rmses, r2s = [], []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_sub)):
            reg = Ridge(alpha=1.0).fit(X_sub[train_idx], Y_sub[train_idx])
            pred = reg.predict(X_sub[test_idx])
            rmses.append(np.sqrt(mean_squared_error(Y_sub[test_idx], pred)))
            r2s.append(r2_score(Y_sub[test_idx], pred))

        cv_results[target_name] = {
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
        }
        print(f"    R² = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        print(f"    RMSE = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # 1b. RMSE over training steps (epoch checkpoints)
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if not checkpoint_files:
        print("  No epoch checkpoints found, skipping RMSE-over-steps.")
        epoch_rmse = {}
    else:
        print(f"\n  Computing RMSE over {len(checkpoint_files)} checkpoints...")
        epoch_rmse = {}
        n_probe = min(2000, n_seq)
        probe_idx = np.random.RandomState(42).choice(n_seq, n_probe, replace=False)
        n_train_probe = int(n_probe * 0.8)

        for ckpt_path in checkpoint_files:
            epoch = int(ckpt_path.stem.split("epoch")[-1])
            try:
                model_ckpt, _ = load_model(checkpoint_dir, device, ckpt_path.name)
                acts_ckpt = extract_activations(model_ckpt, tokens[probe_idx],
                                                batch_size=256, device=device)
                ckpt_keys = get_resid_post_keys(acts_ckpt, d_model)
                if not ckpt_keys:
                    print(f"    Epoch {epoch}: no valid keys, skipping")
                    continue
                acts_last = acts_ckpt[ckpt_keys[-1]]

                epoch_metrics = {}
                for target_name, target_data in targets.items():
                    X_pos = acts_last[:, -1, :]
                    Y_pos = target_data[probe_idx, -1, :]
                    reg = Ridge(alpha=1.0).fit(X_pos[:n_train_probe], Y_pos[:n_train_probe])
                    pred = reg.predict(X_pos[n_train_probe:])
                    rmse = np.sqrt(mean_squared_error(Y_pos[n_train_probe:], pred))
                    r2 = r2_score(Y_pos[n_train_probe:], pred)
                    epoch_metrics[target_name] = {"rmse": float(rmse), "r2": float(r2)}
                epoch_rmse[epoch] = epoch_metrics
                print(f"    Epoch {epoch}: joint R²={epoch_metrics['joint_posterior_Y']['r2']:.3f}, "
                      f"comp R²={epoch_metrics['component_posterior_q']['r2']:.3f}, "
                      f"belief R²={epoch_metrics['within_belief_eta']['r2']:.3f}")
                del model_ckpt, acts_ckpt
            except Exception as e:
                print(f"    Epoch {epoch}: error: {e}")

    # Plot RMSE over training steps
    if epoch_rmse:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = sorted(epoch_rmse.keys())
        colors = {'joint_posterior_Y': '#1f77b4',
                  'component_posterior_q': '#ff7f0e',
                  'within_belief_eta': '#2ca02c'}
        labels = {'joint_posterior_Y': 'Joint Posterior $Y$',
                  'component_posterior_q': 'Component Posterior $q_c$',
                  'within_belief_eta': 'Within-Component Belief $\\eta_c$'}

        for target_name in targets:
            rmse_vals = [epoch_rmse[e][target_name]["rmse"] for e in epochs]
            r2_vals = [epoch_rmse[e][target_name]["r2"] for e in epochs]
            ax1.plot(epochs, rmse_vals, 'o-', color=colors[target_name],
                    label=labels[target_name], markersize=5)
            ax2.plot(epochs, r2_vals, 'o-', color=colors[target_name],
                    label=labels[target_name], markersize=5)

        ax1.set_xlabel("Training Epoch"); ax1.set_ylabel("RMSE")
        ax1.set_title("Regression RMSE over Training (last position)")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Training Epoch"); ax2.set_ylabel("R²")
        ax2.set_title("Regression R² over Training (last position)")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.05)

        fig.tight_layout()
        fig.savefig(output_dir / "exp1_rmse_over_training.png", dpi=150)
        plt.close(fig)

    results = {"cv_results": cv_results, "epoch_rmse": {str(k): v for k, v in epoch_rmse.items()}}
    print("  Experiment 1 complete.")
    return results


# ─────────────────────────────────────────────
# Experiment 2: Effective Dimensionality vs Context Position
# ─────────────────────────────────────────────

def experiment_2(activations, layer_keys, comp_posteriors, component_info, output_dir, d_model):
    """PCA dimensionality vs position with entropy overlay and KL-based ℓ*."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Effective Dimensionality vs Context Position")
    print("="*60)

    n_seq, n_pos, _ = activations[layer_keys[0]].shape

    # Use last resid_post
    resid_post_keys = [k for k in layer_keys if 'resid_post' in k]
    last_key = resid_post_keys[-1] if resid_post_keys else layer_keys[-1]
    acts = activations[last_key]

    # Compute k*_0.95 at each position
    dims_by_pos = []
    for t in range(n_pos):
        pca = PCA(n_components=min(15, d_model))
        pca.fit(acts[:, t, :])
        cev = np.cumsum(pca.explained_variance_ratio_)
        dims_by_pos.append(int(np.searchsorted(cev, 0.95) + 1))

    # Compute mean entropy of component posterior at each position
    entropy_by_pos = []
    for t in range(n_pos):
        q = comp_posteriors[:, t, :]
        q_clipped = np.clip(q, 1e-10, 1.0)
        H = -np.sum(q_clipped * np.log(q_clipped), axis=1)
        entropy_by_pos.append(float(np.mean(H)))

    # Compute KL divergences between component transition structures
    # Use the full joint transition: for a given state s, P(v, s' | c) = T_c[v][s,s']
    # KL between the full observation models (averaged over stationary distribution)
    components = component_info["components"]
    K = len(components)
    kl_divs = []
    for i in range(K):
        for j in range(K):
            if i != j:
                Ti = np.array(components[i]["transition_matrix"])  # (V, S, S)
                Tj = np.array(components[j]["transition_matrix"])  # (V, S, S)
                # For each starting state s, KL(P_i(v|s) || P_j(v|s))
                # P(v|s,c) = sum_s' T_c[v][s,s']
                pi = np.ones(3) / 3  # stationary distribution
                kl_per_state = np.zeros(3)
                for s in range(3):
                    p_i = np.array([Ti[v][s, :].sum() for v in range(3)])
                    p_j = np.array([Tj[v][s, :].sum() for v in range(3)])
                    p_i = np.clip(p_i, 1e-10, 1.0)
                    p_j = np.clip(p_j, 1e-10, 1.0)
                    p_i /= p_i.sum()
                    p_j /= p_j.sum()
                    kl_per_state[s] = np.sum(p_i * np.log(p_i / p_j))
                kl = float(np.dot(pi, kl_per_state))
                kl_divs.append(kl)

    mean_kl = np.mean(kl_divs) if kl_divs else 0
    ell_star = 1.0 / mean_kl if mean_kl > 1e-6 else n_pos * 2

    print(f"  Pairwise KL divergences: {[f'{x:.4f}' for x in kl_divs]}")
    print(f"  Mean pairwise KL = {mean_kl:.4f}")
    print(f"  Predicted ℓ* ≈ {ell_star:.1f}")
    print(f"  Dims by position: {dims_by_pos}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    positions = list(range(n_pos))

    color_dim = '#1f77b4'
    color_ent = '#ff7f0e'

    ax1.plot(positions, dims_by_pos, 'o-', color=color_dim, markersize=6, linewidth=2,
             label='$k^*_{0.95}$ (PCA dims)')
    ax1.set_xlabel("Context Position $t$", fontsize=12)
    ax1.set_ylabel("$k^*_{0.95}$ (dims for 95% var)", color=color_dim, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_dim)
    ax1.axhline(8, color=color_dim, linestyle=':', alpha=0.4, label='$3K-1=8$ (theory)')
    ax1.axhline(2, color=color_dim, linestyle=':', alpha=0.4, label='$d_c-1=2$ (single comp)')

    if ell_star < n_pos:
        ax1.axvline(ell_star, color='gray', linestyle='--', alpha=0.6,
                    label=f'$\\ell^* \\approx {ell_star:.1f}$')

    ax2 = ax1.twinx()
    ax2.plot(positions, entropy_by_pos, 's--', color=color_ent, markersize=5, alpha=0.7,
             label='Mean $H(q_c)$')
    ax2.set_ylabel("Mean Entropy $H(q_c)$", color=color_ent, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_ent)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax1.set_title("Effective Dimensionality vs Context Position (Final Layer)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "exp2_dimensionality_vs_position.png", dpi=150)
    plt.close(fig)

    results = {
        "dims_by_position": dims_by_pos,
        "entropy_by_position": entropy_by_pos,
        "mean_kl": float(mean_kl),
        "ell_star": float(ell_star),
        "kl_divs": [float(x) for x in kl_divs],
    }
    print("  Experiment 2 complete.")
    return results


# ─────────────────────────────────────────────
# Experiment 3: Component Separability vs Position & Layer
# ─────────────────────────────────────────────

def experiment_3(activations, layer_keys, component_labels, comp_posteriors, output_dir):
    """Linear probe heatmap: accuracy & R² as fn of (position, layer)."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Component Separability vs Position & Layer")
    print("="*60)

    n_seq, n_pos, d_model = activations[layer_keys[0]].shape
    n_train = int(n_seq * 0.8)

    acc_grid = np.zeros((len(layer_keys), n_pos))
    r2_grid = np.zeros((len(layer_keys), n_pos))

    for li, lk in enumerate(layer_keys):
        acts = activations[lk]
        print(f"  [{li+1}/{len(layer_keys)}] Layer {short_name(lk)}...")
        for pos in range(n_pos):
            X = acts[:, pos, :]
            X_tr, X_te = X[:n_train], X[n_train:]

            clf = LogisticRegression(max_iter=500, C=1.0).fit(X_tr, component_labels[:n_train])
            acc_grid[li, pos] = clf.score(X_te, component_labels[n_train:])

            y = comp_posteriors[:, pos, :]
            reg = Ridge(alpha=1.0).fit(X_tr, y[:n_train])
            pred = reg.predict(X_te)
            r2_grid[li, pos] = r2_score(y[n_train:], pred)

    layer_labels = [short_name(lk) for lk in layer_keys]

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title in zip(axes,
                                [acc_grid, r2_grid],
                                ["Classification Accuracy", "Comp Posterior R²"]):
        im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xlabel("Context Position $t$", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels, fontsize=7)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Experiment 3: Component Separability", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "exp3_separability_heatmap.png", dpi=150)
    plt.close(fig)

    # Line plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for li, lk in enumerate(layer_keys):
        ax.plot(range(n_pos), acc_grid[li], 'o-', markersize=3, label=short_name(lk))
    ax.set_xlabel("Context Position $t$")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Accuracy vs Position by Layer")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(0.25, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pos_early, pos_mid, pos_late = 1, n_pos // 2, n_pos - 1
    for pos, label in [(pos_early, f"Early (t={pos_early})"),
                       (pos_mid, f"Mid (t={pos_mid})"),
                       (pos_late, f"Late (t={pos_late})")]:
        ax.plot(range(len(layer_keys)), acc_grid[:, pos], 'o-', markersize=5, label=label)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(len(layer_keys)))
    ax.set_xticklabels(layer_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Accuracy vs Layer Depth")
    ax.legend(fontsize=8)
    ax.set_ylim(0.25, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "exp3_separability_lines.png", dpi=150)
    plt.close(fig)

    results = {
        "acc_grid": acc_grid.tolist(),
        "r2_grid": r2_grid.tolist(),
        "layer_keys": layer_labels,
    }
    print("  Experiment 3 complete.")
    return results


# ─────────────────────────────────────────────
# Experiment 4: Geometry Visualization
# ─────────────────────────────────────────────

def experiment_4(activations, layer_keys, component_labels, within_beliefs, output_dir, d_model):
    """PCA scatter plots: early/mid/late, colored by component and hidden state. 3D plot."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Geometry Visualization")
    print("="*60)

    n_seq, n_pos, _ = activations[layer_keys[0]].shape
    resid_post_keys = [k for k in layer_keys if 'resid_post' in k]
    last_key = resid_post_keys[-1] if resid_post_keys else layer_keys[-1]
    acts = activations[last_key]

    colors_comp = ['#e41a1c', '#377eb8', '#4daf4a']
    n_plot = min(3000, n_seq)
    idx = np.random.RandomState(42).choice(n_seq, n_plot, replace=False)

    early_pos = [0, 1, 2]
    mid_pos = [n_pos//2 - 1, n_pos//2, min(n_pos//2 + 1, n_pos-1)]
    late_pos = [max(0, n_pos - 3), n_pos - 2, n_pos - 1]

    all_groups = [("Early (t=0-2)", early_pos), ("Mid", mid_pos), ("Late", late_pos)]

    # 4a. 2D PCA colored by component — side-by-side early/mid/late
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for gi, (group_name, positions) in enumerate(all_groups):
        group_acts = np.concatenate([acts[idx, p, :] for p in positions], axis=0)
        group_labels = np.tile(component_labels[idx], len(positions))
        group_beliefs = np.concatenate([within_beliefs[idx, p, :] for p in positions], axis=0)

        pca = PCA(n_components=3)
        proj = pca.fit_transform(group_acts)

        ax = axes[0, gi]
        for c in range(3):
            mask = group_labels == c
            ax.scatter(proj[mask, 0], proj[mask, 1], c=colors_comp[c], s=2, alpha=0.3, label=f"C{c}")
        ax.set_title(f"{group_name}\nby component", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=8, markerscale=4)

        ax = axes[1, gi]
        max_state = np.argmax(group_beliefs, axis=1)
        for s in range(3):
            mask = max_state == s
            ax.scatter(proj[mask, 0], proj[mask, 1], c=colors_comp[s], s=2, alpha=0.3, label=f"state {s}")
        ax.set_title(f"{group_name}\nby hidden state", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=8, markerscale=4)

    fig.suptitle("Experiment 4: PCA Geometry — Early vs Mid vs Late (Final resid_post)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "exp4_geometry_comparison.png", dpi=150)
    plt.close(fig)

    # 4b. 3D PCA scatter at late position
    pos_late = n_pos - 1
    pos_acts = acts[idx, pos_late, :]
    pca3d = PCA(n_components=3)
    proj3d = pca3d.fit_transform(pos_acts)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for c in range(3):
        mask = component_labels[idx] == c
        ax.scatter(proj3d[mask, 0], proj3d[mask, 1], proj3d[mask, 2],
                  c=colors_comp[c], s=3, alpha=0.3, label=f"C{c}")
    ax.set_xlabel(f"PC1 ({pca3d.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca3d.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca3d.explained_variance_ratio_[2]:.1%})")
    ax.set_title(f"3D PCA at position {pos_late} (final resid_post)", fontsize=12)
    ax.legend(fontsize=9, markerscale=4)
    fig.tight_layout()
    fig.savefig(output_dir / "exp4_3d_pca.png", dpi=150)
    plt.close(fig)

    print("  Experiment 4 complete.")
    return {"positions_plotted": {"early": early_pos, "mid": mid_pos, "late": late_pos}}


# ─────────────────────────────────────────────
# Experiment 5: Fractal Geometry Recovery
# ─────────────────────────────────────────────

def experiment_5(activations, layer_keys, component_labels, within_beliefs,
                 component_info, output_dir, d_model):
    """Recover fractal belief geometry per component via linear regression."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Fractal Geometry Recovery")
    print("="*60)

    n_seq, n_pos, _ = activations[layer_keys[0]].shape
    resid_post_keys = [k for k in layer_keys if 'resid_post' in k]
    last_key = resid_post_keys[-1] if resid_post_keys else layer_keys[-1]
    acts = activations[last_key]
    K = len(component_info["components"])

    fig, axes = plt.subplots(K, 2, figsize=(12, 5*K))

    recovery_r2 = {}

    for c in range(K):
        comp_name = component_info["components"][c]["name"]
        mask = component_labels == c
        n_c = mask.sum()
        print(f"  Component {c} ({comp_name}): {n_c} sequences")

        comp_acts = acts[mask]
        comp_beliefs = within_beliefs[mask]

        X = comp_acts.reshape(-1, d_model)
        Y = comp_beliefs.reshape(-1, 3)

        n_total = X.shape[0]
        n_tr = int(n_total * 0.8)
        reg = Ridge(alpha=1.0).fit(X[:n_tr], Y[:n_tr])
        pred = reg.predict(X[n_tr:])
        r2 = r2_score(Y[n_tr:], pred)
        recovery_r2[comp_name] = float(r2)
        print(f"    Recovery R² = {r2:.4f}")

        recovered = reg.predict(X)

        n_plot = min(10000, n_total)
        plot_idx = np.random.RandomState(c).choice(n_total, n_plot, replace=False)

        def simplex_to_2d(beliefs):
            x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
            y = (np.sqrt(3) / 2) * beliefs[:, 2]
            return x, y

        # Left: ground truth
        ax = axes[c, 0]
        gt_x, gt_y = simplex_to_2d(Y[plot_idx])
        ax.scatter(gt_x, gt_y, c=Y[plot_idx, 0], cmap='coolwarm', s=0.5, alpha=0.3, vmin=0, vmax=1)
        ax.set_title(f"{comp_name}: Ground Truth $\\eta_c$", fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.95)

        # Right: recovered
        ax = axes[c, 1]
        rec_beliefs = recovered[plot_idx]
        rec_clipped = np.clip(rec_beliefs, 0, 1)
        sums = rec_clipped.sum(axis=1, keepdims=True)
        sums = np.maximum(sums, 1e-10)
        rec_clipped = rec_clipped / sums
        rec_x, rec_y = simplex_to_2d(rec_clipped)
        ax.scatter(rec_x, rec_y, c=rec_clipped[:, 0], cmap='coolwarm', s=0.5, alpha=0.3, vmin=0, vmax=1)
        ax.set_title(f"{comp_name}: Recovered (R²={r2:.3f})", fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.95)

    fig.suptitle("Experiment 5: Fractal Geometry Recovery\n(Left: Ground Truth, Right: Linear Recovery from Activations)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "exp5_fractal_recovery.png", dpi=150)
    plt.close(fig)

    print("  Experiment 5 complete.")
    return {"recovery_r2": recovery_r2}


# ─────────────────────────────────────────────
# Experiment 7: Subspace Orthogonality Test
# ─────────────────────────────────────────────

def experiment_7(activations, layer_keys, component_labels, within_beliefs,
                 comp_posteriors, output_dir, d_model):
    """Test factored structure: measure overlaps between component-ID and belief subspaces."""
    print("\n" + "="*60)
    print("EXPERIMENT 7: Subspace Orthogonality Test")
    print("="*60)

    n_seq, n_pos, _ = activations[layer_keys[0]].shape
    resid_post_keys = [k for k in layer_keys if 'resid_post' in k]
    last_key = resid_post_keys[-1] if resid_post_keys else layer_keys[-1]
    acts = activations[last_key]
    K = 3

    # Use last position
    pos = n_pos - 1
    X = acts[:, pos, :]
    n_train = int(n_seq * 0.8)
    X_train = X[:n_train]

    # 1. Component-identity subspace: regression from X to q_c
    y_comp = comp_posteriors[:n_train, pos, :]
    reg_comp = Ridge(alpha=1.0).fit(X_train, y_comp)
    W_comp = reg_comp.coef_  # (K, d_model)
    U_comp, S_comp, Vt_comp = np.linalg.svd(W_comp, full_matrices=False)
    Q_comp = Vt_comp[:K-1].T  # (d_model, K-1)

    # 2. Within-component belief subspaces
    Q_beliefs = []
    for c in range(K):
        mask_train = component_labels[:n_train] == c
        X_c = X_train[mask_train]
        y_c = within_beliefs[:n_train][mask_train, pos, :]
        reg_c = Ridge(alpha=1.0).fit(X_c, y_c)
        W_c = reg_c.coef_  # (3, d_model)
        U_c, S_c, Vt_c = np.linalg.svd(W_c, full_matrices=False)
        Q_c = Vt_c[:2].T  # (d_model, 2)
        Q_beliefs.append(Q_c)

    # 3. Compute pairwise overlaps
    subspaces = {"comp_ID": Q_comp}
    for c in range(K):
        subspaces[f"belief_C{c}"] = Q_beliefs[c]

    names = list(subspaces.keys())
    n_sub = len(names)
    overlap_matrix = np.zeros((n_sub, n_sub))

    for i in range(n_sub):
        for j in range(n_sub):
            Qa = subspaces[names[i]]
            Qb = subspaces[names[j]]
            d_min = min(Qa.shape[1], Qb.shape[1])
            overlap = np.linalg.norm(Qa.T @ Qb, 'fro')**2 / d_min
            overlap_matrix[i, j] = overlap

    print("  Subspace overlap matrix (after training):")
    header = f"{'':>12}"
    for n in names:
        header += f"  {n:>12}"
    print(f"  {header}")
    for i, ni in enumerate(names):
        row = f"  {ni:>12}"
        for j in range(n_sub):
            row += f"  {overlap_matrix[i,j]:>12.4f}"
        print(row)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(n_sub))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_sub))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Experiment 7: Subspace Overlap Matrix\n(Principal Angles Metric)", fontsize=12)

    for i in range(n_sub):
        for j in range(n_sub):
            ax.text(j, i, f"{overlap_matrix[i,j]:.3f}",
                   ha="center", va="center", fontsize=9,
                   color="white" if overlap_matrix[i,j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Overlap")
    fig.tight_layout()
    fig.savefig(output_dir / "exp7_subspace_orthogonality.png", dpi=150)
    plt.close(fig)

    comp_belief_overlaps = {f"comp_ID_vs_belief_C{c}": float(overlap_matrix[0, 1+c]) for c in range(K)}
    belief_belief_overlaps = {}
    for i in range(K):
        for j in range(i+1, K):
            belief_belief_overlaps[f"belief_C{i}_vs_belief_C{j}"] = float(overlap_matrix[1+i, 1+j])

    off_diag = [float(overlap_matrix[i, j]) for i in range(n_sub) for j in range(i+1, n_sub)]

    results = {
        "overlap_matrix": overlap_matrix.tolist(),
        "subspace_names": names,
        "comp_belief_overlaps": comp_belief_overlaps,
        "belief_belief_overlaps": belief_belief_overlaps,
        "mean_off_diagonal": float(np.mean(off_diag)),
    }
    print(f"\n  Mean off-diagonal overlap: {np.mean(off_diag):.4f}")
    print(f"  Comp-ID vs belief overlaps: {comp_belief_overlaps}")
    print("  Experiment 7 complete.")
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--experiments", type=str, default="1,2,3,4,5,7",
                        help="Comma-separated experiment numbers to run")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "results"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else results_dir / "checkpoints_full"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_to_run = [int(x) for x in args.experiments.split(",")]
    print(f"Running experiments: {experiments_to_run}")
    print(f"Results dir: {results_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")

    # Load model
    print("\nLoading model...")
    model, mcfg = load_model(checkpoint_dir, args.device)
    d_model = mcfg["d_model"]
    print(f"  d_model={d_model}, n_layers={mcfg['n_layers']}, n_heads={mcfg['n_heads']}, "
          f"d_mlp={mcfg.get('d_mlp', 'N/A')}")

    # Load data
    print("Loading validation data...")
    val_data = np.load(results_dir / "val_data.npz")
    tokens = val_data["tokens"]
    component_labels = val_data["component_labels"]
    within_beliefs = val_data["within_beliefs"]
    comp_posteriors = val_data["component_posteriors"]
    joint_beliefs = val_data["joint_beliefs"]
    next_token_probs = val_data["next_token_probs"]

    n_pos = tokens.shape[1] - 1
    within_beliefs = within_beliefs[:, :n_pos, :]
    comp_posteriors = comp_posteriors[:, :n_pos, :]
    joint_beliefs = joint_beliefs[:, :n_pos, :]
    next_token_probs = next_token_probs[:, :n_pos, :]

    print(f"Data: {tokens.shape[0]} sequences, seq_len={tokens.shape[1]}, n_pos={n_pos}")

    # Extract activations
    print("Extracting activations...")
    activations = extract_activations(model, tokens, batch_size=512, device=args.device)
    layer_keys = get_residual_keys(activations, d_model)
    print(f"Layers ({len(layer_keys)}): {[short_name(k) for k in layer_keys]}")

    # Load component info
    with open(results_dir / "component_info.json") as f:
        component_info = json.load(f)

    all_results = {}

    if 1 in experiments_to_run:
        all_results["experiment_1"] = experiment_1(
            activations, layer_keys, component_labels, within_beliefs,
            comp_posteriors, joint_beliefs, checkpoint_dir, results_dir,
            tokens, args.device, output_dir, d_model)

    if 2 in experiments_to_run:
        all_results["experiment_2"] = experiment_2(
            activations, layer_keys, comp_posteriors, component_info, output_dir, d_model)

    if 3 in experiments_to_run:
        all_results["experiment_3"] = experiment_3(
            activations, layer_keys, component_labels, comp_posteriors, output_dir)

    if 4 in experiments_to_run:
        all_results["experiment_4"] = experiment_4(
            activations, layer_keys, component_labels, within_beliefs, output_dir, d_model)

    if 5 in experiments_to_run:
        all_results["experiment_5"] = experiment_5(
            activations, layer_keys, component_labels, within_beliefs,
            component_info, output_dir, d_model)

    if 7 in experiments_to_run:
        all_results["experiment_7"] = experiment_7(
            activations, layer_keys, component_labels, within_beliefs,
            comp_posteriors, output_dir, d_model)

    # Save results
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Results: {output_dir / 'experiment_results.json'}")
    print(f"Figures: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
