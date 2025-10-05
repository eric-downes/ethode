"""Hawkes event schedule pre-computation for JumpDiffusion integration."""

from typing import Tuple
import jax
import jax.numpy as jnp

from .runtime import HawkesRuntime, HawkesState
from .kernel import update_intensity, generate_event


def generate_schedule(
    runtime: HawkesRuntime,
    t_span: Tuple[float, float],
    dt: jax.Array,
    max_events: int,
    seed: int,
    dtype: jnp.dtype = jnp.float32
) -> Tuple[jax.Array, HawkesState]:
    """Pre-compute Hawkes event times using thinning algorithm.

    Uses jax.lax.scan for JIT compilation. Advances intensity and checks
    for events at each dt step.

    Args:
        runtime: HawkesRuntime configuration
        t_span: (t_start, t_end) time span
        dt: Time step for thinning (should be << 1/λ_max)
        max_events: Safety cap on number of events
        seed: Random seed
        dtype: Data type for event_times (should match initial_state.dtype)

    Returns:
        Tuple of:
        - event_times: Array of event times, shape (max_events,), padded with inf
        - final_state: Final HawkesState after simulation

    Note:
        For accurate Hawkes simulation, dt should satisfy:
        dt << min(1/λ_max, τ_decay)
        where λ_max is the maximum expected intensity.
    """
    t_start, t_end = t_span
    base_rate = float(runtime.jump_rate.value)

    # Clamp dt to avoid pathological cases (0, inf, or extremely large values)
    # Minimum dt floor: 1e-6 * (t_end - t_start) to avoid division issues
    dt_floor = 1e-6 * (t_end - t_start)
    dt_clamped = jnp.maximum(dt, dt_floor)

    # Initialize state and PRNG
    state = HawkesState.initialize(base_rate)
    key = jax.random.PRNGKey(seed)

    # Pre-allocate event buffer with matching dtype
    event_times = jnp.full(max_events, jnp.inf, dtype=dtype)
    event_count = jnp.array(0, dtype=jnp.int32)

    # Calculate number of time steps using clamped dt
    n_steps = int(jnp.ceil((t_end - t_start) / dt_clamped))

    def scan_fn(carry, _):
        """Single time step: update intensity, check for event."""
        state, key, event_times, event_count, current_time = carry

        # Advance intensity and check for event (use clamped dt)
        key, subkey = jax.random.split(key)
        new_state, event_occurred, _ = generate_event(
            runtime, state, subkey, dt_clamped
        )

        # If event occurred and buffer not full, record time
        can_record = jnp.logical_and(
            event_occurred,
            event_count < max_events
        )

        event_times = jax.lax.cond(
            can_record,
            lambda: event_times.at[event_count].set(current_time + dt_clamped),
            lambda: event_times,
        )

        event_count = jax.lax.cond(
            can_record,
            lambda: event_count + 1,
            lambda: event_count,
        )

        return (new_state, key, event_times, event_count, current_time + dt_clamped), None

    # Run scan over time steps
    init_carry = (state, key, event_times, event_count, jnp.array(t_start))
    (final_state, _, final_times, final_count, _), _ = jax.lax.scan(
        scan_fn,
        init_carry,
        None,
        length=n_steps
    )

    return final_times, final_state
