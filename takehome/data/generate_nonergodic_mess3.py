#!/usr/bin/env python3
"""Generate non-ergodic Mess3 training data.

Dataset: mixture over K=3 distinct Mess3 ergodic components.
Each sequence is generated entirely by one component (no switching).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import jax
import jax.numpy as jnp
import json
from pathlib import Path

from fwh_core.generative_processes.transition_matrices import mess3
from fwh_core.generative_processes.hidden_markov_model import HiddenMarkovModel

COMPONENTS = [
    {"name": "C0_slow",  "x": 0.08, "a": 0.75},
    {"name": "C1_mid",   "x": 0.15, "a": 0.55},
    {"name": "C2_fast",  "x": 0.25, "a": 0.40},
]
K = len(COMPONENTS)
VOCAB_SIZE = 3
MIXTURE_WEIGHTS = np.ones(K) / K


def build_hmm(comp):
    return HiddenMarkovModel(mess3(comp["x"], comp["a"]))


def verify_ergodicity(comp):
    T = mess3(comp["x"], comp["a"])
    S = jnp.sum(T, axis=0)
    all_positive = bool(jnp.all(S > 0))
    eigenvalues, eigenvectors = jnp.linalg.eig(S.T)
    idx = jnp.argmin(jnp.abs(eigenvalues - 1.0))
    stationary = jnp.abs(eigenvectors[:, idx].real)
    stationary = stationary / jnp.sum(stationary)
    sorted_eigs = jnp.sort(jnp.abs(eigenvalues))[::-1]
    spectral_gap = float(1.0 - sorted_eigs[1])
    return {
        "name": comp["name"], "x": comp["x"], "a": comp["a"],
        "all_positive": all_positive, "ergodic": all_positive,
        "stationary": [float(s) for s in stationary],
        "spectral_gap": spectral_gap,
    }


def generate_sequences_jax(hmm, n_sequences, seq_len, rng_key):
    initial_state = hmm.initial_state
    batch_initial = jnp.broadcast_to(initial_state, (n_sequences, initial_state.shape[0]))
    keys = jax.random.split(rng_key, n_sequences)
    belief_trajectory, tokens = hmm.generate(batch_initial, keys, seq_len, True)
    return np.array(tokens), np.array(belief_trajectory)


def compute_all_beliefs(tokens, trans_mats_np, init_states_np, component_labels):
    """Vectorized belief computation using numpy broadcasting.

    Computes within-component beliefs, component posteriors, joint beliefs,
    and next-token predictive distributions.
    """
    n_seq, seq_len = tokens.shape

    within_beliefs = np.zeros((n_seq, seq_len, 3), dtype=np.float32)
    comp_posteriors = np.zeros((n_seq, seq_len, K), dtype=np.float32)
    joint_beliefs = np.zeros((n_seq, seq_len, K * 3), dtype=np.float32)
    next_token_probs = np.zeros((n_seq, seq_len, VOCAB_SIZE), dtype=np.float32)

    # Precompute observation probability sums for each component and token
    # obs_sum[c][v] = T[v].sum(axis=1) shape (3,) — used for P(emit v | state s) = sum_{s'} T[v][s,s']
    obs_sums = []
    for c in range(K):
        obs_c = np.zeros((VOCAB_SIZE, 3))
        for v in range(VOCAB_SIZE):
            obs_c[v] = trans_mats_np[c][v].sum(axis=1)
        obs_sums.append(obs_c)  # (vocab, states)

    # Process in batches for memory efficiency
    batch_size = min(5000, n_seq)
    for batch_start in range(0, n_seq, batch_size):
        batch_end = min(batch_start + batch_size, n_seq)
        bs = batch_end - batch_start
        if batch_start > 0 and batch_start % 10000 == 0:
            print(f"    Processing {batch_start}/{n_seq}")

        batch_tokens = tokens[batch_start:batch_end]  # (bs, seq_len)
        batch_labels = component_labels[batch_start:batch_end]

        # Initialize states for all components: (bs, K, 3)
        states_all = np.stack([np.tile(init_states_np[c], (bs, 1)) for c in range(K)], axis=1)
        log_evidence = np.tile(np.log(MIXTURE_WEIGHTS), (bs, 1))  # (bs, K)

        # True component states
        true_states = np.zeros((bs, 3), dtype=np.float64)
        for i in range(bs):
            true_states[i] = init_states_np[batch_labels[i]]

        for t in range(seq_len):
            # Component posterior
            log_post = log_evidence - log_evidence.max(axis=1, keepdims=True)
            post_c = np.exp(log_post)
            post_c /= post_c.sum(axis=1, keepdims=True)
            comp_posteriors[batch_start:batch_end, t, :] = post_c

            # Within-component belief (true component)
            within_beliefs[batch_start:batch_end, t, :] = true_states

            # Joint beliefs and next-token probs
            for c in range(K):
                joint_beliefs[batch_start:batch_end, t, c*3:(c+1)*3] = \
                    post_c[:, c:c+1] * states_all[:, c, :]
                # P(x_t = v | component c, belief) = belief @ obs_sums[c][v]
                # For all vocab: (bs, 3) @ (3, vocab) -> (bs, vocab)
                obs_probs_c = states_all[:, c, :] @ obs_sums[c].T  # (bs, vocab)
                next_token_probs[batch_start:batch_end, t, :] += \
                    post_c[:, c:c+1] * obs_probs_c

            # Update: observe token at position t
            tok = batch_tokens[:, t]  # (bs,)

            for c in range(K):
                # Observation probability for observed token
                # P(tok | c, belief) = belief @ obs_sums[c][tok] for each sequence
                obs_prob = np.zeros(bs, dtype=np.float64)
                for v in range(VOCAB_SIZE):
                    mask_v = tok == v
                    if mask_v.any():
                        obs_prob[mask_v] = (states_all[mask_v, c, :] * obs_sums[c][v]).sum(axis=1)
                log_evidence[:, c] += np.log(np.maximum(obs_prob, 1e-30))

                # State transition: new_state = belief @ T[tok] / normalizer
                new_states = np.zeros((bs, 3), dtype=np.float64)
                for v in range(VOCAB_SIZE):
                    mask_v = tok == v
                    if mask_v.any():
                        new_states[mask_v] = states_all[mask_v, c, :] @ trans_mats_np[c][v]
                ns = new_states.sum(axis=1, keepdims=True)
                ns = np.maximum(ns, 1e-30)
                states_all[:, c, :] = new_states / ns

            # True component state transition
            new_true = np.zeros((bs, 3), dtype=np.float64)
            for i in range(bs):
                tc = batch_labels[i]
                v = tok[i]
                new_true[i] = true_states[i] @ trans_mats_np[tc][v]
            ns = new_true.sum(axis=1, keepdims=True)
            ns = np.maximum(ns, 1e-30)
            true_states = new_true / ns

    return within_beliefs, comp_posteriors, joint_beliefs, next_token_probs


def generate_dataset(n_train, n_val, seq_len, seed=42):
    rng = np.random.RandomState(seed)
    jax_key = jax.random.PRNGKey(seed)
    hmms = [build_hmm(comp) for comp in COMPONENTS]

    print("Verifying ergodicity:")
    ergodicity_info = [verify_ergodicity(comp) for comp in COMPONENTS]
    for info in ergodicity_info:
        print(f"  {info['name']}: ergodic={info['ergodic']}, gap={info['spectral_gap']:.4f}")

    trans_mats_np = [np.array(hmm.transition_matrices) for hmm in hmms]
    init_states_np = [np.array(hmm.initial_state) for hmm in hmms]

    datasets = {}
    for split, n_total in [("train", n_train), ("val", n_val)]:
        print(f"\nGenerating {split} ({n_total} sequences, seq_len={seq_len})...")
        component_labels = rng.choice(K, size=n_total, p=MIXTURE_WEIGHTS)
        all_tokens = np.zeros((n_total, seq_len), dtype=np.int32)

        for c in range(K):
            mask = component_labels == c
            n_c = mask.sum()
            if n_c == 0:
                continue
            jax_key, subkey = jax.random.split(jax_key)
            tokens_c, _ = generate_sequences_jax(hmms[c], n_c, seq_len, subkey)
            all_tokens[mask] = tokens_c

        print(f"  Computing beliefs/posteriors...")
        within_beliefs, comp_post, joint_beliefs, ntp = \
            compute_all_beliefs(all_tokens, trans_mats_np, init_states_np, component_labels)

        datasets[split] = {
            "tokens": all_tokens,
            "component_labels": component_labels,
            "within_beliefs": within_beliefs,
            "component_posteriors": comp_post,
            "joint_beliefs": joint_beliefs,
            "next_token_probs": ntp,
        }

    return datasets, ergodicity_info, hmms


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent.parent / "results")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets, ergodicity_info, _ = generate_dataset(
        args.n_train, args.n_val, args.seq_len, args.seed
    )

    for split in ["train", "val"]:
        data = datasets[split]
        np.savez_compressed(
            output_dir / f"{split}_data.npz",
            tokens=data["tokens"],
            component_labels=data["component_labels"],
            within_beliefs=data["within_beliefs"],
            component_posteriors=data["component_posteriors"],
            joint_beliefs=data["joint_beliefs"],
            next_token_probs=data["next_token_probs"],
        )
        print(f"Saved {split}: tokens {data['tokens'].shape}")

    comp_info = {
        "components": [{**c, "transition_matrix": np.array(mess3(c["x"], c["a"])).tolist()}
                       for c in COMPONENTS],
        "mixture_weights": MIXTURE_WEIGHTS.tolist(),
        "ergodicity": ergodicity_info,
        "vocab_size": VOCAB_SIZE, "K": K,
    }
    with open(output_dir / "component_info.json", "w") as f:
        json.dump(comp_info, f, indent=2)

    for split in ["train", "val"]:
        data = datasets[split]
        labels = data["component_labels"]
        print(f"\n{split}:")
        for c in range(K):
            n = (labels == c).sum()
            print(f"  {COMPONENTS[c]['name']}: {n} ({100*n/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
