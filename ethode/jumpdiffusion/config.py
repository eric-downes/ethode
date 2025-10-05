"""Configuration for jump diffusion processes.

This module provides Pydantic configuration classes for ODE+Jump simulation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Callable, Literal, Optional, Tuple, Any
import jax.numpy as jnp
import pint

from ..units import UnitManager, UnitSpec, QuantityInput
from ..jumpprocess import JumpProcessConfig


class JumpDiffusionConfig(BaseModel):
    """Configuration for ODE+Jump hybrid simulation.

    Attributes:
        initial_state: Initial state vector
        dynamics_fn: Function computing dstate/dt for ODE
        jump_effect_fn: Function computing state after jump
        jump_process: Configuration for jump timing (Poisson, Hawkes, etc.)
        solver: ODE solver method ('euler', 'rk4', 'dopri5')
        dt_max: Maximum ODE integration step size
        rtol: Relative tolerance for adaptive solvers
        atol: Absolute tolerance for adaptive solvers
        params: User parameters passed to dynamics/jump functions

    Note:
        Current implementation saves state only at jump times and final time.
        Dense output (save at every ODE step) and custom save_at times are
        planned for future versions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Initial conditions
    initial_state: jnp.ndarray = Field(
        description="Initial state vector (JAX array)"
    )

    # User-defined dynamics
    dynamics_fn: Callable[[float, jnp.ndarray, Any], jnp.ndarray] = Field(
        description="Function: (t, state, params) -> dstate_dt"
    )

    jump_effect_fn: Callable[[float, jnp.ndarray, Any], jnp.ndarray] = Field(
        description="Function: (t, state_before, params) -> state_after"
    )

    # Jump process configuration
    jump_process: JumpProcessConfig = Field(
        description="Configuration for jump timing (uses JumpProcessAdapter)"
    )

    # ODE solver configuration
    solver: Literal['euler', 'rk4', 'dopri5', 'dopri8'] = Field(
        default='dopri5',
        description="ODE integration method"
    )

    dt_max: Tuple[float, UnitSpec] = Field(
        description="Maximum integration step size (with units)"
    )

    rtol: float = Field(
        default=1e-6,
        description="Relative tolerance for adaptive solvers"
    )

    atol: float = Field(
        default=1e-9,
        description="Absolute tolerance for adaptive solvers"
    )

    # Optional parameters passed to dynamics/jump functions
    params: Optional[Any] = Field(
        default=None,
        description="User parameters passed to dynamics_fn and jump_effect_fn"
    )

    @field_validator("dt_max", mode="before")
    @classmethod
    def validate_dt_max(cls, v):
        """Validate dt_max has time dimension."""
        manager = UnitManager.instance()

        if isinstance(v, str):
            q = manager.ensure_quantity(v)
        elif isinstance(v, tuple):
            return v  # Already validated
        else:
            # Bare number - assume days
            q = manager.ensure_quantity(f"{v} day", "day")

        try:
            dt_value, dt_spec = manager.to_canonical(q, "time")
        except Exception as e:
            raise ValueError(f"dt_max must have time dimension, got {q}: {e}")

        if dt_value <= 0:
            raise ValueError(f"dt_max must be positive, got {dt_value}")

        return (dt_value, dt_spec)

    def to_runtime(self, check_units: bool = True) -> 'JumpDiffusionRuntime':
        """Convert config to JAX-ready runtime structure."""
        from ..runtime import QuantityNode
        from .runtime import JumpDiffusionRuntime

        dt_max_value, dt_max_spec = self.dt_max

        # Map solver string to int for JAX compatibility
        solver_map = {'euler': 0, 'rk4': 1, 'dopri5': 2, 'dopri8': 3}
        solver_type_int = solver_map[self.solver]

        return JumpDiffusionRuntime(
            dynamics_fn=self.dynamics_fn,
            jump_effect_fn=self.jump_effect_fn,
            jump_runtime=self.jump_process.to_runtime(check_units=check_units),
            solver_type=solver_type_int,
            dt_max=QuantityNode.from_float(dt_max_value, dt_max_spec),
            rtol=self.rtol,
            atol=self.atol,
            params=self.params,
        )


class JumpDiffusionConfigOutput(BaseModel):
    """Output wrapper for introspection.

    Attributes:
        config: The JumpDiffusionConfig used
        runtime: JAX-ready runtime structure
        state_dimension: Dimension of state vector
        solver_name: Human-readable solver name
    """

    config: JumpDiffusionConfig
    runtime: 'JumpDiffusionRuntime'
    state_dimension: int
    solver_name: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
