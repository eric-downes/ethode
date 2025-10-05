"""Configuration for jump diffusion processes.

This module provides Pydantic configuration classes for ODE+Jump simulation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Callable, Literal, Optional, Tuple, Any, Union
import jax
import jax.numpy as jnp
import pint
import warnings

from ..units import UnitManager, UnitSpec, QuantityInput
from ..jumpprocess import JumpProcessConfig
from ..hawkes import HawkesConfig


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
    jump_process: Union[JumpProcessConfig, HawkesConfig] = Field(
        description="Configuration for jump timing (Poisson, deterministic, or Hawkes)"
    )

    # Explicit Hawkes mode selection
    hawkes_mode: Optional[Literal["pregen", "online"]] = Field(
        default=None,
        description="Hawkes generation mode: 'pregen' (Mode 1) or 'online' (Mode 2). "
                    "Only relevant for HawkesConfig. Defaults to 'pregen' if not specified."
    )

    # Hawkes-specific discretization parameters (Mode 1 only)
    hawkes_dt: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Time step for Hawkes thinning (Mode 1 only, ignored for Mode 2). "
                    "Should be << 1/λ_max for accuracy. Auto-computed if not provided for Mode 1."
    )

    hawkes_max_events: Optional[int] = Field(
        default=None,
        description="Safety cap on Hawkes events. "
                    "Mode 1: Buffer size for pre-generation. Mode 2: Thinning iteration limit. "
                    "Default: 10000"
    )

    # Mode 2 only: Custom excitation kernel callables
    # These will be stored as static fields in JumpSchedulerRuntime (excluded from pytree)
    lambda_0_fn: Optional[Callable[[jax.Array], jax.Array]] = Field(
        default=None,
        description="State-dependent base rate function (ode_state) -> lambda_0. "
                    "Mode 2 only. Default: constant base rate from HawkesConfig."
    )

    excitation_decay_fn: Optional[Callable[[Any, jax.Array, Any], Any]] = Field(
        default=None,
        description="Excitation decay function (E, dt, hawkes_runtime) -> E_new. "
                    "Mode 2 only. Default: exponential decay."
    )

    excitation_jump_fn: Optional[Callable[[Any, Any], Any]] = Field(
        default=None,
        description="Excitation jump function (E, hawkes_runtime) -> E_new. "
                    "Mode 2 only. Default: unit jump (E + 1)."
    )

    intensity_fn: Optional[Callable[[jax.Array, Any, Any], jax.Array]] = Field(
        default=None,
        description="Intensity function (lambda_0, E, hawkes_runtime) -> lambda. "
                    "Mode 2 only. Default: linear (lambda_0 + alpha * E)."
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

    @field_validator("hawkes_mode", mode="before")
    @classmethod
    def validate_hawkes_mode(cls, v, info):
        """Validate hawkes_mode and set default."""
        jump_process = info.data.get('jump_process')

        # Only relevant for HawkesConfig
        if jump_process is not None and isinstance(jump_process, HawkesConfig):
            return v or "pregen"  # Default to Mode 1

        return v

    @field_validator("hawkes_dt", mode="before")
    @classmethod
    def validate_hawkes_dt(cls, v, info):
        """Validate hawkes_dt has time dimension if provided."""
        if v is None:
            # Auto-computation handled by model_validator
            return v

        # Validate time dimension
        manager = UnitManager.instance()
        if isinstance(v, str):
            q = manager.ensure_quantity(v)
        elif isinstance(v, tuple):
            return v
        else:
            q = manager.ensure_quantity(f"{v} second")

        dt_value, dt_spec = manager.to_canonical(q, "time")
        if dt_value <= 0:
            raise ValueError(f"hawkes_dt must be positive, got {dt_value}")

        return (dt_value, dt_spec)

    @field_validator("hawkes_max_events", mode="before")
    @classmethod
    def validate_hawkes_max_events(cls, v, info):
        """Validate hawkes_max_events is required for Hawkes."""
        jump_process = info.data.get('jump_process')

        if jump_process is not None and isinstance(jump_process, HawkesConfig):
            if v is None:
                # Auto-compute default: 10x expected events
                # This is conservative; adjust based on excitation strength
                return 10000  # Reasonable default

            if v <= 0:
                raise ValueError(f"hawkes_max_events must be positive, got {v}")

        return v

    @field_validator("lambda_0_fn", mode="before")
    @classmethod
    def validate_lambda_0_fn(cls, v, info):
        """Warn if using default lambda_0 for Mode 2."""
        hawkes_mode = info.data.get('hawkes_mode')
        jump_process = info.data.get('jump_process')

        if hawkes_mode == 'online' and isinstance(jump_process, HawkesConfig):
            if v is None:
                warnings.warn(
                    "Mode 2 (Online Hawkes): No lambda_0_fn provided. "
                    "Using default state-independent base rate (constant λ₀). "
                    "To enable state-dependent intensity, provide lambda_0_fn=your_function. "
                    "See docs for signature: (ode_state: jax.Array) -> jax.Array",
                    UserWarning
                )
        return v

    @field_validator("excitation_decay_fn", mode="before")
    @classmethod
    def validate_excitation_decay_fn(cls, v, info):
        """Warn if using default excitation decay for Mode 2."""
        hawkes_mode = info.data.get('hawkes_mode')
        jump_process = info.data.get('jump_process')

        if hawkes_mode == 'online' and isinstance(jump_process, HawkesConfig):
            if v is None:
                warnings.warn(
                    "Mode 2 (Online Hawkes): No excitation_decay_fn provided. "
                    "Using default exponential decay: E(t+dt) = E(t) * exp(-β*dt). "
                    "To customize, provide excitation_decay_fn=your_function. "
                    "See docs for signature: (E: Any, dt: jax.Array, hawkes_runtime: HawkesRuntime) -> Any",
                    UserWarning
                )
        return v

    @field_validator("excitation_jump_fn", mode="before")
    @classmethod
    def validate_excitation_jump_fn(cls, v, info):
        """Warn if using default excitation jump for Mode 2."""
        hawkes_mode = info.data.get('hawkes_mode')
        jump_process = info.data.get('jump_process')

        if hawkes_mode == 'online' and isinstance(jump_process, HawkesConfig):
            if v is None:
                warnings.warn(
                    "Mode 2 (Online Hawkes): No excitation_jump_fn provided. "
                    "Using default unit jump: E(t+) = E(t-) + 1. "
                    "To customize, provide excitation_jump_fn=your_function. "
                    "See docs for signature: (E: Any, hawkes_runtime: HawkesRuntime) -> Any",
                    UserWarning
                )
        return v

    @field_validator("intensity_fn", mode="before")
    @classmethod
    def validate_intensity_fn(cls, v, info):
        """Warn if using default intensity function for Mode 2."""
        hawkes_mode = info.data.get('hawkes_mode')
        jump_process = info.data.get('jump_process')

        if hawkes_mode == 'online' and isinstance(jump_process, HawkesConfig):
            if v is None:
                warnings.warn(
                    "Mode 2 (Online Hawkes): No intensity_fn provided. "
                    "Using default linear intensity: λ(t) = λ₀ + α*E(t). "
                    "To customize, provide intensity_fn=your_function. "
                    "See docs for signature: (lambda_0: jax.Array, E: Any, hawkes_runtime: HawkesRuntime) -> jax.Array",
                    UserWarning
                )
        return v

    @model_validator(mode='after')
    def _auto_compute_hawkes_dt(self):
        """Auto-compute hawkes_dt if not provided for Mode 1."""
        # Only auto-compute for Mode 1 Hawkes
        if (isinstance(self.jump_process, HawkesConfig) and
            self.hawkes_mode == 'pregen' and
            self.hawkes_dt is None):

            # Extract parameters from validated HawkesConfig
            base_rate = self.jump_process.jump_rate[0]
            excitation = self.jump_process.excitation_strength[0]
            decay = self.jump_process.excitation_decay[0]

            # Compute maximum intensity accounting for self-excitation
            lambda_max = base_rate / (1.0 - excitation)
            default_dt = min(0.1 / lambda_max, 0.25 / decay)

            warnings.warn(
                f"Mode 1 (Pre-gen Hawkes): Auto-computed hawkes_dt={default_dt:.6f} seconds. "
                f"Based on λ_max={lambda_max:.3f}/s (base={base_rate:.3f}/s, α={excitation:.2f}) "
                f"and τ_decay={decay:.1f}s. To override, provide hawkes_dt explicitly.",
                UserWarning
            )

            self.hawkes_dt = (default_dt, "second")

        return self

    def to_runtime(self, check_units: bool = True) -> 'JumpDiffusionRuntime':
        """Convert config to JAX-ready runtime structure."""
        from ..runtime import QuantityNode
        from .runtime import JumpDiffusionRuntime, JumpSchedulerRuntime

        dt_max_value, dt_max_spec = self.dt_max

        # Map solver string to int for JAX compatibility
        solver_map = {'euler': 0, 'rk4': 1, 'dopri5': 2, 'dopri8': 3}
        solver_type_int = solver_map[self.solver]

        # Build scheduler runtime based on jump_process type
        if isinstance(self.jump_process, HawkesConfig):
            # Hawkes mode
            hawkes_runtime = self.jump_process.to_runtime()

            # Mode selection based on explicit hawkes_mode field
            if self.hawkes_mode == 'pregen':
                # Mode 1: Pre-generated Hawkes
                mode = 1
                hawkes_dt_value, _ = self.hawkes_dt  # Required for Mode 1
            else:  # self.hawkes_mode == 'online'
                # Mode 2: Online Hawkes (lazy generation with cumulative excitation)
                mode = 2
                hawkes_dt_value = float('nan')  # Set to NaN for Mode 2 (unused, explicit marker)

            scheduler = JumpSchedulerRuntime(
                mode=mode,
                scheduled=None,
                hawkes=hawkes_runtime,
                hawkes_dt=jnp.array(hawkes_dt_value),
                hawkes_max_events=jnp.array(self.hawkes_max_events, dtype=jnp.int32),
                seed=jnp.array(self.jump_process.seed or 0, dtype=jnp.uint32),
                # Mode 2: Store callables as static fields (excluded from pytree flattening)
                lambda_0_fn=self.lambda_0_fn,
                excitation_decay_fn=self.excitation_decay_fn,
                excitation_jump_fn=self.excitation_jump_fn,
                intensity_fn=self.intensity_fn,
            )
        else:
            # Mode 0: Poisson/Deterministic
            jump_runtime = self.jump_process.to_runtime(check_units=check_units)
            scheduler = JumpSchedulerRuntime(
                mode=0,
                scheduled=jump_runtime,
                hawkes=None,
                hawkes_dt=jnp.array(0.0),
                hawkes_max_events=jnp.array(0, dtype=jnp.int32),
                seed=jnp.array(self.jump_process.seed or 0, dtype=jnp.uint32),
                # Mode 0: No callables needed
                lambda_0_fn=None,
                excitation_decay_fn=None,
                excitation_jump_fn=None,
                intensity_fn=None,
            )

        return JumpDiffusionRuntime(
            dynamics_fn=self.dynamics_fn,
            jump_effect_fn=self.jump_effect_fn,
            scheduler=scheduler,
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
