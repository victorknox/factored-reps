#!/usr/bin/env python3
"""Experiment 6: Dimensionality Scaling with K.

Train models on K=2,3,4,5 component mixtures and measure
effective dimensionality scaling against the 3K-1 prediction.

Two modes:
  --n_per_component N  : each K gets N sequences per component (default: 20000)
  --n_total N          : fixed total dataset size across all K (e.g. 100000)

All models use identical hyperparameters (lr=3e-4, weight_decay=1e-4,
grad_clip=1.0, cosine schedule) matching the main K=3 training config.
PCA is measured on late positions only (t >= min_position) where
the posterior has had time to converge.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from fwh_core.generative_processes.transition_matrices import mess3
from fwh_core.generative_processes.hidden_markov_model import HiddenMarkovModel
import jax
import jax.numpy as jnp

# Component bank — pick first K from these
ALL_COMPONENTS = [
    {"name": "C0_slow",   "x": 0.08, "a": 0.75},
    {"name": "C1_mid",    "x": 0.15, "a": 0.55},
    {"name": "C2_fast",   "x": 0.25, "a": 0.40},
    {"name": "C3_xfast",  "x": 0.30, "a": 0.30},
    {"name": "C4_xslow",  "x": 0.05, "a": 0.85},
]
VOCAB_SIZE = 3

# Hyperparameters matching main K=3 training config (main_full.json)
TRAIN_HP = {
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "batch_size": 512,
}


def build_hmm(comp):
    return HiddenMarkovModel(mess3(comp["x"], comp["a"]))


def generate_data(K, n_train, n_val, seq_len=16, seed=42):
    """Generate data for K components with specified total counts."""
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


def build_model(d_model=128, n_heads=4, n_layers=4, n_ctx=15, device="cpu"):
    """Build a full HookedTransformer with MLP blocks."""
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
    model = HookedTransformer(cfg)
    return model


def train_model(model, train_tokens, val_tokens, num_epochs=200,
                batch_size=512, device="cpu"):
    """Train model on next-token prediction with consistent hyperparameters."""
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

        # Validation
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

    # Load best state
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
    if not keys:
        keys = [k for k in sorted(activations.keys())
                if activations[k].ndim == 3 and activations[k].shape[-1] == d_model]
    return keys[-1] if keys else None


def measure_dimensionality(acts_flat, max_components=20):
    """Compute CEV curve and k*_0.95."""
    n_comp = min(max_components, acts_flat.shape[1], acts_flat.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(acts_flat)
    cev = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.searchsorted(cev, 0.95) + 1)
    return cev, k95, pca.explained_variance_ratio_


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--n_per_component", type=int, default=20000,
                        help="Sequences per component (ignored if --n_total is set)")
    parser.add_argument("--n_total", type=int, default=None,
                        help="Fixed total dataset size across all K (overrides --n_per_component)")
    parser.add_argument("--min_position", type=int, default=10,
                        help="Minimum context position for PCA (default: 10, use late positions only)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output filenames (e.g. '_fixed100k')")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "results" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    K_values = [2, 3, 4, 5]
    d_model = args.d_model
    min_pos = args.min_position

    # Determine data mode
    if args.n_total is not None:
        data_mode = "fixed_total"
        print(f"Data mode: fixed total = {args.n_total} train sequences for all K")
    else:
        data_mode = "per_component"
        print(f"Data mode: {args.n_per_component} sequences per component")

    print(f"Hyperparameters: lr={TRAIN_HP['lr']}, weight_decay={TRAIN_HP['weight_decay']}, "
          f"grad_clip={TRAIN_HP['grad_clip']}")
    print(f"Architecture: d_model={d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print(f"PCA on positions >= {min_pos}")

    results = {}
    cev_curves = {}

    for K in K_values:
        print(f"\n{'='*60}")
        print(f"K={K}: Generate data, train, and measure dimensionality")
        print(f"{'='*60}")

        # Compute train/val sizes
        if args.n_total is not None:
            n_train = args.n_total
            n_val = max(args.n_total // 10, 5000)
        else:
            n_train = args.n_per_component * K
            n_val = max(args.n_per_component // 10, 1000) * K

        print(f"  Generating data for K={K} ({n_train} train, {n_val} val sequences)...")
        datasets, components = generate_data(K, n_train=n_train, n_val=n_val,
                                              seq_len=16, seed=42)
        train_tokens = datasets["train"]["tokens"]
        val_tokens = datasets["val"]["tokens"]

        # Build and train model
        n_ctx = train_tokens.shape[1] - 1
        print(f"  Training model (d_model={d_model}, n_layers={args.n_layers})...")
        model = build_model(d_model=d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, n_ctx=n_ctx, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    {n_params:,} parameters")

        model, best_val = train_model(model, train_tokens, val_tokens,
                                       num_epochs=args.num_epochs, device=device)
        print(f"  Best val loss: {best_val:.4f}")

        # Extract activations
        print(f"  Extracting activations...")
        activations = extract_activations(model, val_tokens, batch_size=512, device=device)
        last_key = get_last_layer_key(activations, d_model)
        acts = activations[last_key]  # (n_seq, n_positions, d_model)

        # Use only late positions (t >= min_position) for PCA
        n_positions = acts.shape[1]
        if min_pos < n_positions:
            acts_late = acts[:, min_pos:, :]
        else:
            acts_late = acts
        acts_flat = acts_late.reshape(-1, d_model)
        print(f"  PCA on positions [{min_pos}, {n_positions-1}]: {acts_flat.shape[0]} vectors")

        cev, k95, evr = measure_dimensionality(acts_flat)
        results[K] = {"k95": k95, "best_val_loss": float(best_val),
                       "n_train": n_train, "n_val": n_val}
        cev_curves[K] = cev.tolist()
        print(f"  K={K}: k*_0.95 = {k95} (theory: {3*K-1})")

        del model, activations
        torch.cuda.empty_cache()

    # ─── Plots ───
    print("\nGenerating plots...")
    suffix = args.output_suffix

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    K_vals = sorted(results.keys())
    measured = [results[K]["k95"] for K in K_vals]
    predicted = [3*K - 1 for K in K_vals]

    x = np.arange(len(K_vals))
    width = 0.35
    ax1.bar(x - width/2, measured, width, label='Measured $k^*_{0.95}$', color='steelblue')
    ax1.bar(x + width/2, predicted, width, label='Theory $3K-1$', color='coral', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={K}" for K in K_vals])
    ax1.set_ylabel("Effective Dimensionality")
    title_mode = f"(fixed {args.n_total} total)" if args.n_total else f"({args.n_per_component}/comp)"
    ax1.set_title(f"Dimensionality Scaling {title_mode}")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # CEV curves
    for K in K_vals:
        cev = cev_curves[K]
        ax2.plot(range(1, len(cev)+1), cev, 'o-', markersize=4, label=f'K={K} (measured)')
    for K in K_vals:
        theory = 3*K - 1
        if theory <= 20:
            ax2.axvline(theory, color='gray', linestyle='--', alpha=0.5)
            ax2.text(theory + 0.2, 0.5, f"3x{K}-1={theory}", fontsize=8, alpha=0.6)

    ax2.axhline(0.95, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel("# PCA Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title(f"CEV Curves by K (positions >= {min_pos})")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.5, 1.02)

    fig.suptitle("Experiment 6: Dimensionality Scaling with K", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"exp6_dimensionality_scaling{suffix}.png", dpi=150)
    plt.close(fig)

    # Save results
    exp6_results = {
        "data_mode": data_mode,
        "n_total": args.n_total,
        "n_per_component": args.n_per_component,
        "min_position": min_pos,
        "hyperparameters": TRAIN_HP,
        "results_by_K": {str(K): v for K, v in results.items()},
        "cev_curves": {str(K): v for K, v in cev_curves.items()},
        "theoretical_prediction": {str(K): 3*K-1 for K in K_vals},
    }
    with open(output_dir / f"exp6_results{suffix}.json", "w") as f:
        json.dump(exp6_results, f, indent=2, default=str)

    print(f"\nExperiment 6 complete. Results saved to {output_dir}")
    for K in K_vals:
        print(f"  K={K}: measured k*_0.95={results[K]['k95']}, theory 3K-1={3*K-1}")


if __name__ == "__main__":
    main()
