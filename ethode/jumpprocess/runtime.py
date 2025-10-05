"""Runtime structures for jump processes.

This module provides Penzai structs for JAX-compatible jump process simulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
from penzai import pz

from ..runtime import QuantityNode


@pz.pytree_dataclass
class JumpProcessRuntime(pz.Struct):
    """JAX-compatible runtime structure for jump processes.

    All fields are JAX arrays for use in jax.lax.scan, jax.jit, etc.

    Uses QuantityNode for rate to match ethode conventions.

    Attributes:
        process_type: 0=Poisson, 1=deterministic
        rate: Event rate as QuantityNode (events per time unit)
        seed: Random seed (auxiliary data, not differentiated)
    """
    process_type: jax.Array  # 0=Poisson, 1=deterministic
    rate: QuantityNode       # Event rate as QuantityNode (not raw float)
    seed: int                # Random seed (auxiliary data, not differentiated)


@pz.pytree_dataclass
class JumpProcessState(pz.Struct):
    """State for jump process simulation.

    Tracks the current state of the jump process for sequential simulation.

    Attributes:
        last_jump_time: Time of last event
        next_jump_time: Time of next scheduled event
        rng_key: JAX random number generator key
        event_count: Total events generated (for diagnostics)
    """
    last_jump_time: jax.Array     # Time of last event
    next_jump_time: jax.Array     # Time of next scheduled event
    rng_key: jax.Array            # JAX random number generator key
    event_count: jax.Array        # Total events generated (for diagnostics)

    @classmethod
    def zero(cls, seed: int = 0, start_time: float = 0.0) -> 'JumpProcessState':
        """Initialize state at given start time.

        Args:
            seed: Random seed
            start_time: Starting time for the process

        Returns:
            Initialized JumpProcessState
        """
        return cls(
            last_jump_time=jnp.array(float(start_time)),
            next_jump_time=jnp.array(float(start_time)),  # Will be updated immediately
            rng_key=jrandom.PRNGKey(seed),
            event_count=jnp.array(0, dtype=jnp.int32),
        )
