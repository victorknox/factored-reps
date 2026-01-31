#!/usr/bin/env python3
"""Pre-generate mess3 belief data for Blender rendering."""

import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp

from fwh_core.generative_processes.builder import build_factored_process_from_spec
from fwh_core.generative_processes.generator import generate_data_batch_with_full_history

SEED = 7
BATCH_SIZE = 3000
SEQ_LEN = 64

def build_process():
    return build_factored_process_from_spec(
        structure_type="independent",
        spec=[
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
        ],
    )


def generate_beliefs(process, batch_size: int, seq_len: int, seed: int):
    key = jax.random.PRNGKey(seed)

    initial_states = process.initial_states
    gen_states = tuple(jnp.broadcast_to(s, (batch_size, s.shape[0])) for s in initial_states)

    result = generate_data_batch_with_full_history(
        gen_states=gen_states,
        data_generator=process,
        batch_size=batch_size,
        sequence_len=seq_len,
        key=key,
    )

    belief_states = result["belief_states"]
    belief_states = tuple(np.array(bs) for bs in belief_states)

    b0 = belief_states[0].reshape(-1, belief_states[0].shape[-1])
    b1 = belief_states[1].reshape(-1, belief_states[1].shape[-1])
    return b0, b1


def main():
    output_dir = Path(__file__).parent

    print("Building process...")
    process = build_process()

    print(f"Generating beliefs (batch_size={BATCH_SIZE}, seq_len={SEQ_LEN})...")
    b0, b1 = generate_beliefs(process, BATCH_SIZE, SEQ_LEN, SEED)

    print(f"Generated {b0.shape[0]} belief states")
    print(f"b0 shape: {b0.shape}, b1 shape: {b1.shape}")

    # Save to npz file
    output_path = output_dir / "mess3_beliefs.npz"
    np.savez(output_path, b0=b0, b1=b1)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
