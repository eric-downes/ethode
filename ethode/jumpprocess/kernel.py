"""Jump process kernel functions for JAX.

This module provides JIT-compiled functions for Poisson and deterministic jump processes.
"""

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Tuple

from .runtime import JumpProcessRuntime, JumpProcessState


def generate_next_jump_time(
    runtime: JumpProcessRuntime,
    state: JumpProcessState,
    current_time: jax.Array,
) -> Tuple[JumpProcessState, jax.Array]:
    """
    Generate the next jump time given current time.

    Args:
        runtime: Jump process configuration
        state: Current state
        current_time: Current time (used as base for next jump)

    Returns:
        (new_state, next_jump_time)
    """
    # Split RNG key
    key, subkey = jrandom.split(state.rng_key)

    # Extract rate value from QuantityNode
    rate_value = runtime.rate.value

    # Generate next jump time based on process type
    dt = jax.lax.cond(
        runtime.process_type == 0,  # Poisson
        lambda: jrandom.exponential(subkey) / rate_value,
        lambda: 1.0 / rate_value,  # Deterministic
    )

    next_time = current_time + dt

    new_state = dataclasses.replace(
        state,
        last_jump_time=current_time,
        next_jump_time=next_time,
        rng_key=key,
    )

    return new_state, next_time


def check_jump_occurred(
    state: JumpProcessState,
    current_time: jax.Array,
) -> jax.Array:
    """
    Check if a jump occurred by current_time.

    Args:
        state: Current state
        current_time: Time to check

    Returns:
        Boolean array (True if jump occurred)
    """
    return current_time >= state.next_jump_time


def step(
    runtime: JumpProcessRuntime,
    state: JumpProcessState,
    current_time: jax.Array,
    dt: jax.Array,
) -> Tuple[JumpProcessState, bool]:
    """
    Step the jump process forward in time.

    Args:
        runtime: Jump process configuration
        state: Current state
        current_time: Current simulation time
        dt: Time step size

    Returns:
        (new_state, jump_occurred)
    """
    new_time = current_time + dt

    # Check if jump occurs in this interval
    jump_occurred = jnp.logical_and(
        state.next_jump_time >= current_time,
        state.next_jump_time < new_time
    )

    # Generate next jump if this one occurred, and increment event count
    def handle_jump(s):
        new_s, _ = generate_next_jump_time(runtime, s, state.next_jump_time)
        return dataclasses.replace(new_s, event_count=s.event_count + 1)

    new_state = jax.lax.cond(
        jump_occurred,
        handle_jump,
        lambda s: s,
        state
    )

    return new_state, jump_occurred


def generate_jumps_in_interval(
    runtime: JumpProcessRuntime,
    t_start: float,
    t_end: float,
    seed: int = 0,
) -> jax.Array:
    """
    Generate all jump times in interval [t_start, t_end).

    Useful for batch generation of events.

    Properly seeds the recurrence from t_start instead of
    using the initial state's next_jump_time=inf.

    Args:
        runtime: Jump process configuration
        t_start: Start time
        t_end: End time
        seed: Random seed

    Returns:
        Array of jump times (variable length, padded with inf)
    """
    # Initialize state with proper start time
    state = JumpProcessState.zero(seed=seed, start_time=t_start)

    # Generate first jump starting from t_start
    state, _ = generate_next_jump_time(runtime, state, jnp.array(t_start))

    # Generate jumps until we exceed t_end
    def scan_fn(carry, _):
        state, jumps, idx = carry

        # Current jump time
        jump_time = state.next_jump_time

        # Only process if within interval
        def process_jump(_):
            # Add to list
            new_jumps = jumps.at[idx].set(jump_time)

            # Generate next jump from current jump time
            new_state, _ = generate_next_jump_time(runtime, state, jump_time)

            # Increment index
            new_idx = idx + 1

            return new_state, new_jumps, new_idx

        def skip_jump(_):
            # Don't modify anything if jump is outside interval
            return state, jumps, idx

        # Only advance state if current jump is within interval
        new_state, new_jumps, new_idx = jax.lax.cond(
            jump_time < t_end,
            process_jump,
            skip_jump,
            None
        )

        return (new_state, new_jumps, new_idx), None

    # Estimate max events (rate * interval * 10 for safety, capped at 100k)
    rate_value = float(runtime.rate.value)
    max_events = min(int((t_end - t_start) * rate_value * 10) + 100, 100000)
    jumps = jnp.full(max_events, jnp.inf)

    (final_state, final_jumps, n_jumps), _ = jax.lax.scan(
        scan_fn,
        (state, jumps, 0),
        None,
        length=max_events
    )

    # Return only non-inf values (actual jumps within interval)
    return final_jumps[final_jumps < t_end]
