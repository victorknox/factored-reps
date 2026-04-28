#!/usr/bin/env python3
"""Convergence study: how do metrics behave as dataset size and epoch count vary?

Two sweeps:
1. Size sweep (fixed epochs=200): 20k, 40k, 60k, 100k, 150k, 200k, 300k, 500k
2. Epoch sweep (fixed size=200k): 50, 100, 200, 400

Reports k*_0.95, k*_0.99, joint R², comp R², belief R², comp-belief overlap.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from fwh_core.generative_processes.transition_matrices import mess3
from fwh_core.generative_processes.hidden_markov_model import HiddenMarkovModel
import jax
import jax.numpy as jnp

ALL_COMPONENTS = [
    {"name": "C0_slow",   "x": 0.08, "a": 0.75},
    {"name": "C1_mid",    "x": 0.15, "a": 0.55},
    {"name": "C2_fast",   "x": 0.25, "a": 0.40},
]
VOCAB_SIZE = 3
K = 3

TRAIN_HP = {
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "batch_size": 512,
}


def build_hmm(comp):
    return HiddenMarkovModel(mess3(comp["x"], comp["a"]))


def generate_data(n_train, n_val, seq_len=16, seed=42):
    components = ALL_COMPONENTS[:K]
    mixture_weights = np.ones(K) / K
    rng = np.random.RandomState(seed)
    jax_key = jax.random.PRNGKey(seed)
    hmms = [build_hmm(c) for c in components]

    datasets = {}
    for split, n_total in [("train", n_train), ("val", n_val)]:
        labels = rng.choice(K, size=n_total, p=mixture_weights)
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

        datasets[split] = {"tokens": all_tokens, "component_labels": labels}

    return datasets, components


def compute_ground_truth(tokens, components):
    hmms = [build_hmm(c) for c in components]
    n_seq, seq_len = tokens.shape
    n_pos = seq_len - 1

    comp_posterior = np.zeros((n_seq, n_pos, K))
    within_belief = np.zeros((n_seq, n_pos, K * 3))
    joint_posterior = np.zeros((n_seq, n_pos, K * 3))

    for i in range(n_seq):
        log_likelihoods = np.zeros(K)
        beliefs = [np.array(hmms[c].initial_state) for c in range(K)]
        for t in range(n_pos):
            obs = int(tokens[i, t])
            for c in range(K):
                tm = np.array(hmms[c].transition_matrices[obs])
                new_belief = tm @ beliefs[c]
                ll = np.log(new_belief.sum() + 1e-30)
                log_likelihoods[c] += ll
                beliefs[c] = new_belief / (new_belief.sum() + 1e-30)

            log_q = log_likelihoods - np.max(log_likelihoods)
            q = np.exp(log_q)
            q = q / (q.sum() + 1e-30)
            comp_posterior[i, t] = q
            for c in range(K):
                within_belief[i, t, c*3:(c+1)*3] = beliefs[c]
                joint_posterior[i, t, c*3:(c+1)*3] = q[c] * beliefs[c]

    return comp_posterior, within_belief, joint_posterior


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


def train_model(model, train_tokens, val_tokens, num_epochs=200, device="cpu"):
    lr = TRAIN_HP["lr"]
    batch_size = TRAIN_HP["batch_size"]
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


def extract_activations(model, tokens, batch_size=512, device="cpu"):
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
    if not keys:
        keys = [k for k in sorted(activations.keys())
                if activations[k].ndim == 3 and activations[k].shape[-1] == d_model]
    return keys[-1] if keys else None


def measure_all_metrics(model, val_tokens, components, d_model=128, min_pos=10, device="cpu"):
    """Measure k*_0.95, k*_0.99, probe R², and subspace overlap."""
    # Extract activations
    activations = extract_activations(model, val_tokens, batch_size=512, device=device)
    last_key = get_last_layer_key(activations, d_model)
    acts = activations[last_key]  # (n_seq, n_pos, d_model)
    n_seq, n_pos, _ = acts.shape

    # Late-position activations for PCA
    acts_late = acts[:, min_pos:, :]
    acts_flat = acts_late.reshape(-1, d_model)

    # PCA
    n_comp = min(30, d_model, acts_flat.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(acts_flat)
    cev = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.searchsorted(cev, 0.95) + 1)
    k99 = int(np.searchsorted(cev, 0.99) + 1)
    evr = pca.explained_variance_ratio_

    # Ground truth (subsample for speed if large)
    max_probe_seqs = 10000
    if n_seq > max_probe_seqs:
        probe_idx = np.random.RandomState(0).choice(n_seq, max_probe_seqs, replace=False)
    else:
        probe_idx = np.arange(n_seq)

    probe_tokens = val_tokens[probe_idx]
    probe_acts = acts[probe_idx]

    comp_post, within_bel, joint_post = compute_ground_truth(probe_tokens, components)

    # Flatten for probes (late positions only)
    X = probe_acts[:, min_pos:, :].reshape(-1, d_model)
    y_joint = joint_post[:, min_pos:, :].reshape(-1, K * 3)
    y_comp = comp_post[:, min_pos:, :].reshape(-1, K)
    y_belief = within_bel[:, min_pos:, :].reshape(-1, K * 3)

    # Linear probes
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y_joint)
    joint_r2 = r2_score(y_joint, ridge.predict(X))

    ridge_c = Ridge(alpha=1.0)
    ridge_c.fit(X, y_comp)
    comp_r2 = r2_score(y_comp, ridge_c.predict(X))

    ridge_b = Ridge(alpha=1.0)
    ridge_b.fit(X, y_belief)
    belief_r2 = r2_score(y_belief, ridge_b.predict(X))

    # Classification: use component labels from ground truth
    # For each (seq, pos) pair, the true component is argmax of comp_posterior
    # But we also have the dataset labels; use argmax(comp_post) as ground truth
    y_comp_label = np.argmax(y_comp, axis=1)

    # Linear classification (Logistic Regression)
    logreg = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    logreg.fit(X, y_comp_label)
    linear_clf_acc = float(accuracy_score(y_comp_label, logreg.predict(X)))

    # MLP classification (nonlinear probe)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                             early_stopping=True, validation_fraction=0.1,
                             random_state=42)
    mlp_clf.fit(X, y_comp_label)
    mlp_clf_acc = float(accuracy_score(y_comp_label, mlp_clf.predict(X)))

    # MLP regression on joint posterior (nonlinear probe)
    mlp_reg = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                            early_stopping=True, validation_fraction=0.1,
                            random_state=42)
    mlp_reg.fit(X, y_joint)
    mlp_joint_r2 = float(r2_score(y_joint, mlp_reg.predict(X)))

    # MLP regression on comp posterior
    mlp_comp_reg = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                 early_stopping=True, validation_fraction=0.1,
                                 random_state=42)
    mlp_comp_reg.fit(X, y_comp)
    mlp_comp_r2 = float(r2_score(y_comp, mlp_comp_reg.predict(X)))

    # Subspace overlap
    W_comp = ridge_c.coef_  # (K, d_model)
    U_comp, _, _ = np.linalg.svd(W_comp, full_matrices=False)
    Q_comp = U_comp[:min(K-1, U_comp.shape[0]), :].T  # (d_model, K-1)

    overlaps_per_c = []
    for c in range(K):
        W_c = ridge_b.coef_[c*3:(c+1)*3, :]  # (3, d_model)
        U_c, _, _ = np.linalg.svd(W_c, full_matrices=False)
        Q_c = U_c[:min(2, U_c.shape[0]), :].T  # (d_model, 2)
        overlap = np.linalg.norm(Q_comp.T @ Q_c, 'fro')**2 / min(Q_comp.shape[1], Q_c.shape[1])
        overlaps_per_c.append(float(overlap))
    mean_overlap = float(np.mean(overlaps_per_c))

    del activations
    return {
        "k95": k95,
        "k99": k99,
        "cev": cev.tolist(),
        "evr": evr.tolist(),
        "joint_R2": float(joint_r2),
        "comp_R2": float(comp_r2),
        "belief_R2": float(belief_r2),
        "linear_clf_acc": linear_clf_acc,
        "mlp_clf_acc": mlp_clf_acc,
        "mlp_joint_R2": mlp_joint_r2,
        "mlp_comp_R2": mlp_comp_r2,
        "comp_belief_overlap": mean_overlap,
        "comp_belief_overlaps_per_c": overlaps_per_c,
    }


def run_single(n_train, num_epochs, device, d_model=128, n_layers=4, n_heads=4):
    """Train one model and measure all metrics."""
    n_val = max(n_train // 10, 5000)
    datasets, components = generate_data(n_train=n_train, n_val=n_val, seq_len=16, seed=42)
    train_tokens = datasets["train"]["tokens"]
    val_tokens = datasets["val"]["tokens"]

    n_ctx = train_tokens.shape[1] - 1
    model = build_model(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        n_ctx=n_ctx, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {n_params:,} parameters")

    model, best_val = train_model(model, train_tokens, val_tokens,
                                   num_epochs=num_epochs, device=device)
    print(f"  Best val loss: {best_val:.4f}")

    print(f"  Measuring metrics...")
    metrics = measure_all_metrics(model, val_tokens, components, d_model=d_model,
                                   min_pos=10, device=device)
    metrics["best_val_loss"] = float(best_val)
    metrics["n_train"] = n_train
    metrics["num_epochs"] = num_epochs

    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sweep", type=str, required=True,
                        choices=["size", "epochs", "both"],
                        help="Which sweep to run")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "results" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    all_results = {}

    if args.sweep in ("size", "both"):
        # Size sweep: fixed 200 epochs, vary dataset size
        sizes = [20000, 40000, 60000, 100000, 150000, 200000, 300000, 500000]
        print("=" * 70)
        print("SIZE SWEEP (200 epochs, varying dataset size)")
        print("=" * 70)
        size_results = {}
        for n_train in sizes:
            print(f"\n--- Size={n_train} ---")
            metrics = run_single(n_train, num_epochs=200, device=device)
            size_results[str(n_train)] = metrics
            print(f"  k95={metrics['k95']}, k99={metrics['k99']}, "
                  f"joint_R2={metrics['joint_R2']:.3f}, comp_R2={metrics['comp_R2']:.3f}, "
                  f"lin_clf={metrics['linear_clf_acc']:.3f}, mlp_clf={metrics['mlp_clf_acc']:.3f}, "
                  f"mlp_joint={metrics['mlp_joint_R2']:.3f}, overlap={metrics['comp_belief_overlap']:.4f}")
        all_results["size_sweep"] = size_results

    if args.sweep in ("epochs", "both"):
        # Epoch sweep: fixed 200k, vary epochs
        epoch_counts = [50, 100, 200, 400]
        print("\n" + "=" * 70)
        print("EPOCH SWEEP (200k dataset, varying epochs)")
        print("=" * 70)
        epoch_results = {}
        for n_epochs in epoch_counts:
            print(f"\n--- Epochs={n_epochs} ---")
            metrics = run_single(200000, num_epochs=n_epochs, device=device)
            epoch_results[str(n_epochs)] = metrics
            print(f"  k95={metrics['k95']}, k99={metrics['k99']}, "
                  f"joint_R2={metrics['joint_R2']:.3f}, comp_R2={metrics['comp_R2']:.3f}, "
                  f"lin_clf={metrics['linear_clf_acc']:.3f}, mlp_clf={metrics['mlp_clf_acc']:.3f}, "
                  f"mlp_joint={metrics['mlp_joint_R2']:.3f}, overlap={metrics['comp_belief_overlap']:.4f}")
        all_results["epoch_sweep"] = epoch_results

    # Save raw results
    with open(output_dir / "convergence_study_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate plots
    print("\nGenerating plots...")

    if "size_sweep" in all_results:
        sr = all_results["size_sweep"]
        sizes = sorted(sr.keys(), key=int)
        xs = [int(s) / 1000 for s in sizes]  # in thousands

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))

        # k*_0.95 and k*_0.99
        ax = axes[0, 0]
        ax.plot(xs, [sr[s]["k95"] for s in sizes], 'o-', color='steelblue', label='$k^*_{0.95}$')
        ax.plot(xs, [sr[s]["k99"] for s in sizes], 's--', color='coral', label='$k^*_{0.99}$')
        ax.set_xlabel("Training set size (×1000)")
        ax.set_ylabel("Effective dimensionality")
        ax.set_title("PCA Dimensionality")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Linear Probe R²
        ax = axes[0, 1]
        ax.plot(xs, [sr[s]["joint_R2"] for s in sizes], 'o-', label='Joint $Y$')
        ax.plot(xs, [sr[s]["comp_R2"] for s in sizes], 's-', label='Component $q_c$')
        ax.plot(xs, [sr[s]["belief_R2"] for s in sizes], '^-', label='Belief $\\eta_c$')
        ax.set_xlabel("Training set size (×1000)")
        ax.set_ylabel("R²")
        ax.set_title("Linear Probe R²")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # MLP Probe R²
        ax = axes[0, 2]
        ax.plot(xs, [sr[s]["mlp_joint_R2"] for s in sizes], 'o-', label='MLP Joint $Y$')
        ax.plot(xs, [sr[s]["mlp_comp_R2"] for s in sizes], 's-', label='MLP Component $q_c$')
        ax.plot(xs, [sr[s]["joint_R2"] for s in sizes], 'o--', alpha=0.4, label='Linear Joint (ref)')
        ax.plot(xs, [sr[s]["comp_R2"] for s in sizes], 's--', alpha=0.4, label='Linear Comp (ref)')
        ax.set_xlabel("Training set size (×1000)")
        ax.set_ylabel("R²")
        ax.set_title("MLP Probe R² vs Linear")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Classification accuracy
        ax = axes[0, 3]
        ax.plot(xs, [sr[s]["linear_clf_acc"] for s in sizes], 'o-', color='steelblue', label='Linear (LogReg)')
        ax.plot(xs, [sr[s]["mlp_clf_acc"] for s in sizes], 's-', color='coral', label='MLP')
        ax.axhline(1/3, color='gray', linestyle=':', alpha=0.5, label='Chance (1/3)')
        ax.set_xlabel("Training set size (×1000)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Overlap
        ax = axes[1, 0]
        ax.plot(xs, [sr[s]["comp_belief_overlap"] for s in sizes], 'o-', color='purple')
        ax.set_xlabel("Training set size (×1000)")
        ax.set_ylabel("Comp-Belief Overlap")
        ax.set_title("Subspace Overlap")
        ax.grid(True, alpha=0.3)

        # Val loss
        ax = axes[1, 1]
        ax.plot(xs, [sr[s]["best_val_loss"] for s in sizes], 'o-', color='green')
        ax.set_xlabel("Training set size (×1000)")
        ax.set_ylabel("Best Val Loss")
        ax.set_title("Validation Loss")
        ax.grid(True, alpha=0.3)

        # Eigenvalue spectra overlay
        ax = axes[1, 2]
        for s in sizes:
            evr = sr[s]["evr"]
            ax.plot(range(1, len(evr)+1), evr, 'o-', markersize=3,
                    label=f'{int(s)//1000}k', alpha=0.7)
        ax.set_xlabel("PCA Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Eigenvalue Spectra")
        ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # CEV curves overlay
        ax = axes[1, 3]
        for s in sizes:
            cev = sr[s]["cev"]
            ax.plot(range(1, len(cev)+1), cev, 'o-', markersize=3,
                    label=f'{int(s)//1000}k', alpha=0.7)
        ax.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='0.95')
        ax.axhline(0.99, color='gray', linestyle='--', alpha=0.5, label='0.99')
        ax.set_xlabel("# PCA Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("CEV Curves")
        ax.set_ylim(0.2, 1.02)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Size Sweep: K=3, 200 epochs", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "convergence_size_sweep.png", dpi=150)
        plt.close(fig)

    if "epoch_sweep" in all_results:
        er = all_results["epoch_sweep"]
        epochs = sorted(er.keys(), key=int)
        xe = [int(e) for e in epochs]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        ax = axes[0]
        ax.plot(xe, [er[e]["k95"] for e in epochs], 'o-', color='steelblue', label='$k^*_{0.95}$')
        ax.plot(xe, [er[e]["k99"] for e in epochs], 's--', color='coral', label='$k^*_{0.99}$')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Effective dimensionality")
        ax.set_title("Dimensionality (200k)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(xe, [er[e]["joint_R2"] for e in epochs], 'o-', label='Linear Joint')
        ax.plot(xe, [er[e]["comp_R2"] for e in epochs], 's-', label='Linear Comp')
        ax.plot(xe, [er[e]["mlp_joint_R2"] for e in epochs], 'o--', label='MLP Joint')
        ax.plot(xe, [er[e]["mlp_comp_R2"] for e in epochs], 's--', label='MLP Comp')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("R²")
        ax.set_title("Probe R² (200k)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(xe, [er[e]["linear_clf_acc"] for e in epochs], 'o-', color='steelblue', label='Linear')
        ax.plot(xe, [er[e]["mlp_clf_acc"] for e in epochs], 's-', color='coral', label='MLP')
        ax.axhline(1/3, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification (200k)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax.plot(xe, [er[e]["comp_belief_overlap"] for e in epochs], 'o-', color='purple')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Comp-Belief Overlap")
        ax.set_title("Overlap (200k)")
        ax.grid(True, alpha=0.3)

        ax = axes[4]
        ax.plot(xe, [er[e]["best_val_loss"] for e in epochs], 'o-', color='green')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Best Val Loss")
        ax.set_title("Val Loss (200k)")
        ax.grid(True, alpha=0.3)

        fig.suptitle("Epoch Sweep: K=3, 200k sequences", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "convergence_epoch_sweep.png", dpi=150)
        plt.close(fig)

    print(f"\nConvergence study complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
