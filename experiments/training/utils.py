"""Utility functions for training experiments."""

import jax
import jax.numpy as jnp


def expand_state_by_batch_size(
    state: jax.Array | tuple[jax.Array, ...], batch_size: int
) -> jax.Array | tuple[jax.Array, ...]:
    """Expand a single state to a batch of identical states.

    Args:
        state: A single state array or tuple of state arrays
        batch_size: Number of copies to create

    Returns:
        Batched state with shape [batch_size, ...] or tuple of such arrays
    """
    if isinstance(state, tuple):
        return tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in state)
    else:
        return jnp.repeat(state[None, :], batch_size, axis=0)
