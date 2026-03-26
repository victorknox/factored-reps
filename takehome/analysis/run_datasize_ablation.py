#!/usr/bin/env python3
"""Dataset size ablation: measure key metrics as a function of training set size.

Fixes K=3 and sweeps total dataset size to check stability of:
  - k*_0.95 (PCA effective dimensionality, late positions)
  - Joint posterior R², component posterior R², within-belief R² (linear probes)
  - Subspace overlap: comp-ID vs belief (orthogonality)

All models use identical architecture and hyperparameters matching main_full.json.
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
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from fwh_core.generative_processes.transition_matrices import mess3
from fwh_core.generative_processes.hidden_markov_model import HiddenMarkovModel
import jax
import jax.numpy as jnp

K = 3
VOCAB_SIZE = 3
COMPONENTS = [
    {"name": "C0_slow",  "x": 0.08, "a": 0.75},
    {"name": "C1_mid",   "x": 0.15, "a": 0.55},
    {"name": "C2_fast",  "x": 0.25, "a": 0.40},
]
MIXTURE_WEIGHTS = np.ones(K) / K

TRAIN_HP = {
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "batch_size": 512,
}


def build_hmm(comp):
    return HiddenMarkovModel(mess3(comp["x"], comp["a"]))


def generate_data(n_train, n_val, seq_len=16, seed=42):
    """Generate K=3 mixture data with exact belief computation."""
    rng = np.random.RandomState(seed)
    jax_key = jax.random.PRNGKey(seed)
    hmms = [build_hmm(c) for c in COMPONENTS]
    trans_mats_np = [np.array(h.transition_matrices) for h in hmms]
    init_states_np = [np.array(h.initial_state) for h in hmms]

    # Precompute observation probability sums
    obs_sums = []
    for c in range(K):
        obs_c = np.zeros((VOCAB_SIZE, 3))
        for v in range(VOCAB_SIZE):
            obs_c[v] = trans_mats_np[c][v].sum(axis=1)
        obs_sums.append(obs_c)

    datasets = {}
    for split, n_total in [("train", n_train), ("val", n_val)]:
        labels = rng.choice(K, size=n_total, p=MIXTURE_WEIGHTS)
        all_tokens = np.zeros((n_total, seq_len), dtype=np.int32)

        for c in range(K):
            mask = labels == c
            n_c = mask.sum()
            if n_c == 0:
                continue
            jax_key, subkey = jax.random.split(jax_key)
            initial = hmms[c].initial_state
            batch_init = jnp.broadcast_to(initial, (n_c, initial.shape[0]))
            keys = jax.random.split(subkey, n_c)
            _, tokens_c = hmms[c].generate(batch_init, keys, seq_len, True)
            all_tokens[mask] = np.array(tokens_c)

        # Compute beliefs
        within_beliefs = np.zeros((n_total, seq_len, 3), dtype=np.float32)
        comp_posteriors = np.zeros((n_total, seq_len, K), dtype=np.float32)

        batch_size = min(5000, n_total)
        for batch_start in range(0, n_total, batch_size):
            batch_end = min(batch_start + batch_size, n_total)
            bs = batch_end - batch_start
            batch_tokens = all_tokens[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end]

            states_all = np.stack([np.tile(init_states_np[c], (bs, 1)) for c in range(K)], axis=1)
            log_evidence = np.tile(np.log(MIXTURE_WEIGHTS), (bs, 1))
            true_states = np.zeros((bs, 3), dtype=np.float64)
            for i in range(bs):
                true_states[i] = init_states_np[batch_labels[i]]

            for t in range(seq_len):
                log_post = log_evidence - log_evidence.max(axis=1, keepdims=True)
                post_c = np.exp(log_post)
                post_c /= post_c.sum(axis=1, keepdims=True)
                comp_posteriors[batch_start:batch_end, t, :] = post_c
                within_beliefs[batch_start:batch_end, t, :] = true_states

                tok = batch_tokens[:, t]
                for c in range(K):
                    obs_prob = np.zeros(bs, dtype=np.float64)
                    for v in range(VOCAB_SIZE):
                        mask_v = tok == v
                        if mask_v.any():
                            obs_prob[mask_v] = (states_all[mask_v, c, :] * obs_sums[c][v]).sum(axis=1)
                    log_evidence[:, c] += np.log(np.maximum(obs_prob, 1e-30))

                    new_states = np.zeros((bs, 3), dtype=np.float64)
                    for v in range(VOCAB_SIZE):
                        mask_v = tok == v
                        if mask_v.any():
                            new_states[mask_v] = states_all[mask_v, c, :] @ trans_mats_np[c][v]
                    ns = np.maximum(new_states.sum(axis=1, keepdims=True), 1e-30)
                    states_all[:, c, :] = new_states / ns

                new_true = np.zeros((bs, 3), dtype=np.float64)
                for i in range(bs):
                    tc = batch_labels[i]
                    v = tok[i]
                    new_true[i] = true_states[i] @ trans_mats_np[tc][v]
                ns = np.maximum(new_true.sum(axis=1, keepdims=True), 1e-30)
                true_states = new_true / ns

        # Compute joint beliefs
        joint_beliefs = np.zeros((n_total, seq_len, K * 3), dtype=np.float32)
        for c in range(K):
            joint_beliefs[:, :, c*3:(c+1)*3] = comp_posteriors[:, :, c:c+1] * within_beliefs

        datasets[split] = {
            "tokens": all_tokens,
            "component_labels": labels,
            "within_beliefs": within_beliefs,
            "component_posteriors": comp_posteriors,
            "joint_beliefs": joint_beliefs,
        }

    return datasets


def build_model(d_model=128, n_heads=4, n_layers=4, n_ctx=15, device="cpu"):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    d_head = d_model // n_heads
    d_mlp = d_model * 4
    cfg = HookedTransformerConfig(
        d_model=d_model, d_head=d_head,
        n_heads=n_heads, n_layers=n_layers,
        n_ctx=n_ctx, d_mlp=d_mlp,
        d_vocab=VOCAB_SIZE, act_fn="relu",
        normalization_type="LN",
        device=device, seed=42,
    )
    return HookedTransformer(cfg)


def train_model(model, train_tokens, val_tokens, num_epochs=200,
                batch_size=512, device="cpu"):
    lr = TRAIN_HP["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=TRAIN_HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr / 10)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.RandomState(42)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        idx = np.arange(len(train_tokens))
        rng.shuffle(idx)
        epoch_loss, epoch_tokens = 0.0, 0

        for start in range(0, len(train_tokens), batch_size):
            end = min(start + batch_size, len(train_tokens))
            batch = train_tokens[idx[start:end]]
            inputs = torch.tensor(batch[:, :-1], dtype=torch.long).to(device)
            targets = torch.tensor(batch[:, 1:], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_HP["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item() * targets.numel()
            epoch_tokens += targets.numel()

        scheduler.step()
        train_loss = epoch_loss / epoch_tokens

        model.eval()
        val_loss_sum, val_tokens_n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_tokens), batch_size):
                end = min(start + batch_size, len(val_tokens))
                batch = val_tokens[start:end]
                inputs = torch.tensor(batch[:, :-1], dtype=torch.long).to(device)
                targets = torch.tensor(batch[:, 1:], dtype=torch.long).to(device)
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss_sum += loss.item() * targets.numel()
                val_tokens_n += targets.numel()
        val_loss = val_loss_sum / val_tokens_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:4d} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


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


def get_last_layer_key(activations, d_model):
    keys = []
    for k in sorted(activations.keys()):
        if activations[k].ndim == 3 and activations[k].shape[-1] == d_model:
            if 'resid_post' in k or 'ln_final' in k:
                keys.append(k)
    return keys[-1] if keys else None


def measure_all_metrics(model, val_data, d_model, min_pos, device):
    """Measure k*_0.95, probe R² values, and subspace overlap for one trained model."""
    tokens = val_data["tokens"]
    labels = val_data["component_labels"]
    within_beliefs = val_data["within_beliefs"]
    comp_posteriors = val_data["component_posteriors"]
    joint_beliefs = val_data["joint_beliefs"]

    # Extract activations
    activations = extract_activations(model, tokens, batch_size=512, device=device)
    last_key = get_last_layer_key(activations, d_model)
    acts = activations[last_key]  # (n_seq, n_pos, d_model)
    n_seq, n_pos, _ = acts.shape

    # --- k*_0.95 on late positions ---
    acts_late = acts[:, min_pos:, :]
    acts_flat = acts_late.reshape(-1, d_model)
    n_comp = min(20, d_model, acts_flat.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(acts_flat)
    cev = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.searchsorted(cev, 0.95) + 1)

    # --- Linear probes (10-fold CV, last position) ---
    pos = n_pos - 1
    X = acts[:, pos, :]

    targets = {
        "joint_R2": joint_beliefs[:, pos, :],
        "comp_R2": comp_posteriors[:, pos, :],
        "belief_R2": within_beliefs[:, pos, :],
    }

    probe_results = {}
    n_sub = min(5000, n_seq)
    sub_idx = np.random.RandomState(42).choice(n_seq, n_sub, replace=False)

    for name, Y in targets.items():
        X_sub, Y_sub = X[sub_idx], Y[sub_idx]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2s = []
        for train_idx, test_idx in kf.split(X_sub):
            reg = Ridge(alpha=1.0).fit(X_sub[train_idx], Y_sub[train_idx])
            pred = reg.predict(X_sub[test_idx])
            r2s.append(r2_score(Y_sub[test_idx], pred))
        probe_results[name] = float(np.mean(r2s))

    # --- Subspace overlap (comp-ID vs belief) ---
    n_train = int(n_seq * 0.8)
    X_train = X[:n_train]

    # Component-identity subspace
    y_comp = comp_posteriors[:n_train, pos, :]
    reg_comp = Ridge(alpha=1.0).fit(X_train, y_comp)
    W_comp = reg_comp.coef_
    _, _, Vt_comp = np.linalg.svd(W_comp, full_matrices=False)
    Q_comp = Vt_comp[:K-1].T  # (d_model, K-1)

    # Within-component belief subspaces
    comp_id_overlaps = []
    for c in range(K):
        mask_train = labels[:n_train] == c
        X_c = X_train[mask_train]
        y_c = within_beliefs[:n_train][mask_train, pos, :]
        reg_c = Ridge(alpha=1.0).fit(X_c, y_c)
        W_c = reg_c.coef_
        _, _, Vt_c = np.linalg.svd(W_c, full_matrices=False)
        Q_c = Vt_c[:2].T  # (d_model, 2)
        d_min = min(Q_comp.shape[1], Q_c.shape[1])
        overlap = np.linalg.norm(Q_comp.T @ Q_c, 'fro')**2 / d_min
        comp_id_overlaps.append(float(overlap))

    mean_overlap = float(np.mean(comp_id_overlaps))

    del activations
    torch.cuda.empty_cache()

    return {
        "k95": k95,
        "cev": cev.tolist(),
        **probe_results,
        "comp_belief_overlap": mean_overlap,
        "comp_belief_overlaps_per_c": comp_id_overlaps,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--min_position", type=int, default=10)
    parser.add_argument("--dataset_sizes", type=str, default="20000,40000,60000,100000,150000,200000",
                        help="Comma-separated list of total training set sizes")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "results" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    d_model = args.d_model
    min_pos = args.min_position
    dataset_sizes = [int(x) for x in args.dataset_sizes.split(",")]

    print(f"Dataset size ablation for K={K}")
    print(f"Sizes: {dataset_sizes}")
    print(f"Hyperparameters: lr={TRAIN_HP['lr']}, wd={TRAIN_HP['weight_decay']}, clip={TRAIN_HP['grad_clip']}")
    print(f"Architecture: d_model={d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print(f"PCA on positions >= {min_pos}")

    all_results = {}

    for n_train in dataset_sizes:
        n_val = max(n_train // 10, 5000)
        print(f"\n{'='*60}")
        print(f"Dataset size: {n_train} train, {n_val} val")
        print(f"{'='*60}")

        print(f"  Generating data...")
        datasets = generate_data(n_train=n_train, n_val=n_val, seq_len=16, seed=42)
        train_tokens = datasets["train"]["tokens"]
        val_tokens = datasets["val"]["tokens"]

        n_ctx = train_tokens.shape[1] - 1
        print(f"  Training model...")
        model = build_model(d_model=d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, n_ctx=n_ctx, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    {n_params:,} parameters")

        model, best_val = train_model(model, train_tokens, val_tokens,
                                       num_epochs=args.num_epochs, device=device)
        print(f"  Best val loss: {best_val:.4f}")

        print(f"  Measuring metrics...")
        metrics = measure_all_metrics(model, datasets["val"], d_model, min_pos, device)
        metrics["best_val_loss"] = float(best_val)
        metrics["n_train"] = n_train
        metrics["n_val"] = n_val

        all_results[n_train] = metrics
        print(f"  k*_0.95={metrics['k95']}, joint_R2={metrics['joint_R2']:.3f}, "
              f"comp_R2={metrics['comp_R2']:.3f}, belief_R2={metrics['belief_R2']:.3f}, "
              f"overlap={metrics['comp_belief_overlap']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ─── Plots ───
    print("\nGenerating plots...")
    sizes = sorted(all_results.keys())
    sizes_k = [s / 1000 for s in sizes]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. k*_0.95 vs dataset size
    ax = axes[0, 0]
    k95s = [all_results[s]["k95"] for s in sizes]
    ax.plot(sizes_k, k95s, 'o-', color='steelblue', markersize=8, linewidth=2)
    ax.axhline(8, color='coral', linestyle='--', alpha=0.6, label='Theory $3K-1=8$')
    ax.set_xlabel("Training Set Size (k)")
    ax.set_ylabel("$k^*_{0.95}$")
    ax.set_title("Effective Dimensionality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Joint R² vs dataset size
    ax = axes[0, 1]
    joint_r2 = [all_results[s]["joint_R2"] for s in sizes]
    ax.plot(sizes_k, joint_r2, 'o-', color='#1f77b4', markersize=8, linewidth=2)
    ax.set_xlabel("Training Set Size (k)")
    ax.set_ylabel("R²")
    ax.set_title("Joint Posterior $Y$ (R²)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 3. Component R² vs dataset size
    ax = axes[0, 2]
    comp_r2 = [all_results[s]["comp_R2"] for s in sizes]
    ax.plot(sizes_k, comp_r2, 'o-', color='#ff7f0e', markersize=8, linewidth=2)
    ax.set_xlabel("Training Set Size (k)")
    ax.set_ylabel("R²")
    ax.set_title("Component Posterior $q_c$ (R²)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 4. Within-belief R² vs dataset size
    ax = axes[1, 0]
    belief_r2 = [all_results[s]["belief_R2"] for s in sizes]
    ax.plot(sizes_k, belief_r2, 'o-', color='#2ca02c', markersize=8, linewidth=2)
    ax.set_xlabel("Training Set Size (k)")
    ax.set_ylabel("R²")
    ax.set_title("Within-Component Belief $\\eta_c$ (R²)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 5. Subspace overlap vs dataset size
    ax = axes[1, 1]
    overlaps = [all_results[s]["comp_belief_overlap"] for s in sizes]
    ax.plot(sizes_k, overlaps, 'o-', color='#9467bd', markersize=8, linewidth=2)
    ax.set_xlabel("Training Set Size (k)")
    ax.set_ylabel("Mean Overlap")
    ax.set_title("Comp-ID vs Belief Subspace Overlap")
    ax.set_ylim(0, max(0.15, max(overlaps) * 1.5))
    ax.grid(True, alpha=0.3)

    # 6. Val loss vs dataset size
    ax = axes[1, 2]
    val_losses = [all_results[s]["best_val_loss"] for s in sizes]
    ax.plot(sizes_k, val_losses, 'o-', color='#d62728', markersize=8, linewidth=2)
    ax.axhline(np.log(3), color='gray', linestyle=':', alpha=0.5, label='Uniform $\\log 3$')
    ax.set_xlabel("Training Set Size (k)")
    ax.set_ylabel("Val Loss (nats)")
    ax.set_title("Best Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Dataset Size Ablation (K=3, positions $\\geq$ 10)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "datasize_ablation.png", dpi=150)
    plt.close(fig)

    # Save results
    results_out = {
        "K": K,
        "min_position": min_pos,
        "hyperparameters": TRAIN_HP,
        "results_by_size": {str(s): v for s, v in all_results.items()},
    }
    with open(output_dir / "datasize_ablation_results.json", "w") as f:
        json.dump(results_out, f, indent=2, default=str)

    print(f"\nAblation complete. Results saved to {output_dir}")
    print(f"\nSummary:")
    print(f"  {'Size':>8} | k*_0.95 | Joint R² | Comp R² | Belief R² | Overlap | Val Loss")
    print(f"  {'-'*80}")
    for s in sizes:
        r = all_results[s]
        print(f"  {s:>8} | {r['k95']:>7} | {r['joint_R2']:>8.3f} | {r['comp_R2']:>7.3f} | "
              f"{r['belief_R2']:>9.3f} | {r['comp_belief_overlap']:>7.4f} | {r['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
