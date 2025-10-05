"""Runtime structures for jump diffusion processes.

This module provides Penzai structs for JAX-compatible ODE+Jump simulation.
"""

from __future__ import annotations

from typing import Callable, Any
import jax
import jax.numpy as jnp
from penzai.core import struct

from ..runtime import QuantityNode
from ..jumpprocess import JumpProcessRuntime, JumpProcessState


@struct.pytree_dataclass
class JumpDiffusionRuntime(struct.Struct):
    """JAX-compatible runtime structure for ODE+Jump simulation.

    Attributes:
        dynamics_fn: ODE right-hand side function
        jump_effect_fn: Jump effect function
        jump_runtime: JumpProcessRuntime for timing
        solver_type: Solver identifier (0=euler, 1=rk4, 2=dopri5, 3=dopri8)
        dt_max: Maximum integration step as QuantityNode
        rtol: Relative tolerance
        atol: Absolute tolerance
        params: User parameters (pytree)
    """

    dynamics_fn: Callable
    jump_effect_fn: Callable
    jump_runtime: JumpProcessRuntime
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
        jump_state: JumpProcessState for next jump
        step_count: Number of ODE steps taken
        jump_count: Number of jumps processed
    """

    t: jax.Array  # Current time (scalar)
    state: jax.Array  # State vector
    jump_state: JumpProcessState  # JumpProcessState (pytree)
    step_count: jax.Array  # Int array for JAX compatibility
    jump_count: jax.Array  # Int array for JAX compatibility

    @classmethod
    def zero(
        cls,
        initial_state: jax.Array,
        jump_state: JumpProcessState,
        t0: float = 0.0
    ) -> 'JumpDiffusionState':
        """Create initial state."""
        return cls(
            t=jnp.array(t0),
            state=initial_state,
            jump_state=jump_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            jump_count=jnp.array(0, dtype=jnp.int32),
        )
