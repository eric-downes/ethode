"""Runtime structures for jump diffusion processes.

This module provides Penzai structs for JAX-compatible ODE+Jump simulation.
"""

from __future__ import annotations

from typing import Callable, Any, Optional
import dataclasses
import jax
import jax.numpy as jnp
from penzai.core import struct

from ..runtime import QuantityNode
from ..jumpprocess import JumpProcessRuntime, JumpProcessState
from ..hawkes import HawkesRuntime


@struct.pytree_dataclass
class JumpSchedulerRuntime(struct.Struct):
    """Unified scheduler runtime for lazy and pre-generated event sources.

    Attributes:
        mode: Scheduler type (0=poisson, 1=pregen_hawkes, 2=online_hawkes)
        scheduled: JumpProcessRuntime if mode==0
        hawkes: HawkesRuntime if mode==1 or mode==2
        hawkes_dt: Time step for Hawkes thinning (only used if mode==1)
        hawkes_max_events: Safety cap (mode 1: buffer size, mode 2: thinning rejection limit)
        seed: Random seed for reproducibility

        # Mode 2 only: Static (non-pytree) callables for custom excitation kernels
        # These MUST be JAX-compatible pure functions (jax.Array inputs/outputs only)
        lambda_0_fn: State-dependent base rate (ode_state) -> lambda_0
        excitation_decay_fn: Excitation decay (E, dt, hawkes_runtime) -> E_new
        excitation_jump_fn: Excitation jump (E, hawkes_runtime) -> E_new
        intensity_fn: Intensity computation (lambda_0, E, hawkes_runtime) -> lambda
    """
    mode: int  # 0=poisson, 1=pregen_hawkes, 2=online_hawkes
    scheduled: Optional[JumpProcessRuntime] = None
    hawkes: Optional[HawkesRuntime] = None
    hawkes_dt: jax.Array = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    hawkes_max_events: jax.Array = dataclasses.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    seed: jax.Array = dataclasses.field(default_factory=lambda: jnp.array(0, dtype=jnp.uint32))

    # Mode 2 static callables (use dataclasses.field with metadata to exclude from pytree)
    # IMPORTANT: Storing callables in pytree breaks JAX flattening/jit
    lambda_0_fn: Optional[Callable] = dataclasses.field(
        default=None, metadata={'pytree_node': False}
    )
    excitation_decay_fn: Optional[Callable] = dataclasses.field(
        default=None, metadata={'pytree_node': False}
    )
    excitation_jump_fn: Optional[Callable] = dataclasses.field(
        default=None, metadata={'pytree_node': False}
    )
    intensity_fn: Optional[Callable] = dataclasses.field(
        default=None, metadata={'pytree_node': False}
    )


@struct.pytree_dataclass
class ScheduledJumpBuffer(struct.Struct):
    """Unified buffer for jump event times (lazy or pre-generated).

    Filling strategy and memory usage by mode:
    - Mode 0 (Poisson): Lazy fill, buffer size = max_steps (reuse from simulate())
    - Mode 1 (Pregen Hawkes): Pre-filled upfront, buffer size = hawkes_max_events
    - Mode 2 (Online Hawkes): Lazy fill, buffer size = 100 (small fixed constant)
      ↳ Only stores 1-2 events at a time, hawkes_max_events bounds thinning rejections

    Attributes:
        event_times: Array of jump times, padded with inf
            - Shape: (max_steps,) for mode 0, (hawkes_max_events,) for mode 1, (100,) for mode 2
        count: Total valid events in buffer (mode 1) or capacity (mode 0, 2)
        next_index: Pointer to next unused event
        rng_key: PRNG key for lazy generation (mode 0, 2)
        cumulative_excitation: E(t) for Hawkes intensity (mode 2 only, 0.0 otherwise)
            - Can be Any pytree (not just scalar) to track additional state
            - Example: {"value": E, "elapsed_time": t} for power-law kernels
        last_update_time: Time when cumulative_excitation was last updated (mode 2 only)
    """
    event_times: jax.Array  # shape varies by mode (see docstring)
    count: jax.Array        # total valid events (mode 1) or buffer capacity (mode 0, 2)
    next_index: jax.Array   # pointer to next unused event
    rng_key: jax.Array      # PRNG key for lazy modes
    cumulative_excitation: Any  # E(t) for online Hawkes (mode 2) - can be pytree!
    last_update_time: jax.Array  # Last time E(t) was updated (mode 2)


@struct.pytree_dataclass
class JumpDiffusionRuntime(struct.Struct):
    """JAX-compatible runtime structure for ODE+Jump simulation.

    Attributes:
        dynamics_fn: ODE right-hand side function
        jump_effect_fn: Jump effect function
        scheduler: Unified scheduler runtime (replaces jump_runtime)
        solver_type: Solver identifier (0=euler, 1=rk4, 2=dopri5, 3=dopri8)
        dt_max: Maximum integration step as QuantityNode
        rtol: Relative tolerance
        atol: Absolute tolerance
        params: User parameters (pytree)
    """

    dynamics_fn: Callable
    jump_effect_fn: Callable
    scheduler: JumpSchedulerRuntime  # ← Changed from jump_runtime
    solver_type: int  # 0=euler, 1=rk4, 2=dopri5, 3=dopri8
    dt_max: QuantityNode
    rtol: float
    atol: float
    params: Any


@struct.pytree_dataclass
class JumpDiffusionState(struct.Struct):
    """Current simulation state.

    Attributes:
        t: Current time
        state: Current state vector
        jump_buffer: Scheduled jump buffer (replaces jump_state)
        step_count: Number of ODE steps taken
        jump_count: Number of jumps processed
    """

    t: jax.Array  # Current time (scalar)
    state: jax.Array  # State vector
    jump_buffer: ScheduledJumpBuffer  # ← Changed from jump_state
    step_count: jax.Array  # Int array for JAX compatibility
    jump_count: jax.Array  # Int array for JAX compatibility

    @classmethod
    def zero(
        cls,
        initial_state: jax.Array,
        jump_buffer: ScheduledJumpBuffer,
        t0: float = 0.0
    ) -> 'JumpDiffusionState':
        """Create initial state."""
        return cls(
            t=jnp.array(t0),
            state=initial_state,
            jump_buffer=jump_buffer,
            step_count=jnp.array(0, dtype=jnp.int32),
            jump_count=jnp.array(0, dtype=jnp.int32),
        )
