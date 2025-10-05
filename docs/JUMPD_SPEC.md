# Ethode ODE+Jump Simulation API Specification

**Version**: 1.1 (all blockers resolved)
**Date**: 2025-10-05
**Status**: Ready for implementation
**Dependencies**: Requires JumpProcess APIs (already implemented)

---

## 1. Overview

### 1.1 Purpose

This spec defines APIs for **hybrid ODE+Jump simulations** where:
- **Continuous dynamics** evolve via ODEs between events
- **Discrete jumps** occur at stochastic times (Poisson, Hawkes, etc.)
- **Jump effects** instantaneously modify the state

This is the classic **jump-diffusion** or **piecewise-deterministic process** framework, essential for modeling systems with both continuous and discrete dynamics.

### 1.2 Use Case: RAI Dollar Interest Rate Model

```python
class RDInterestSim(JumpDiffusionSimulation):
    """Debt grows continuously (ODE), jumps occur at drip times."""

    def dynamics(self, t: float, state: Array, params: Any) -> Array:
        """Continuous dynamics between jumps.

        dD/dt = α(t) * D    # Debt grows at interest rate
        dR/dt = 0           # Supply unchanged between drips
        dα/dt = 0           # Rate unchanged between controller updates
        """
        D, R, alpha = state
        return jnp.array([alpha * D, 0.0, 0.0])

    def jump_effect(self, t: float, state: Array, params: Any) -> Array:
        """Discrete jump: mint new supply to match debt."""
        D, R, alpha = state
        accrued_interest = D - R
        R_new = R + accrued_interest  # Mint and distribute
        return jnp.array([D, R_new, alpha])

# Usage
config = JumpDiffusionConfig(
    initial_state=jnp.array([1e6, 1e6, 0.005]),  # D, R, α
    dynamics_fn=sim.dynamics,
    jump_effect_fn=sim.jump_effect,
    jump_process=JumpProcessConfig(
        process_type='poisson',
        rate="100 / day",  # 100 drips/day
    ),
    solver='dopri5',
    dt_max=0.1,  # Max ODE step: 0.1 days
)

adapter = JumpDiffusionAdapter(config)
times, states = adapter.simulate(t_span=(0.0, 365.0))
```

### 1.3 Architecture

Following ethode's layered design:

1. **Config Layer** (`JumpDiffusionConfig`) - Pydantic validation, user-facing
2. **Runtime Layer** (`JumpDiffusionRuntime`, `JumpDiffusionState`) - JAX pytrees
3. **Kernel Layer** (`integrate_step`, `apply_jump`, `simulate`) - Pure JAX functions
4. **Adapter Layer** (`JumpDiffusionAdapter`) - High-level stateful API

---

## 2. API Design

### 2.1 Config Layer

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Callable, Literal, Optional, Tuple, Any
import jax.numpy as jnp
from ethode import JumpProcessConfig, UnitManager, UnitSpec

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
        from ethode.runtime import QuantityNode
        from ethode.jumpdiffusion.runtime import JumpDiffusionRuntime

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
```

### 2.2 Runtime Layer

```python
from typing import Callable, Any
import jax
import jax.numpy as jnp
from penzai.core import struct
from ethode.runtime import QuantityNode
from ethode.jumpprocess import JumpProcessRuntime, JumpProcessState


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
```

### 2.3 Kernel Layer

```python
import dataclasses
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Any
from ethode.jumpprocess.kernel import generate_next_jump_time
from ethode.jumpdiffusion.runtime import JumpDiffusionRuntime, JumpDiffusionState


def integrate_step(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
    t_end: jax.Array,
) -> Tuple[JumpDiffusionState, jax.Array]:
    """Integrate ODE from current time until next jump or t_end.

    Args:
        runtime: Runtime configuration
        state: Current simulation state
        t_end: Maximum time to integrate to

    Returns:
        (updated_state, time_reached)
        - updated_state: State after integration
        - time_reached: Actual time reached (min of next_jump_time, t_end)
    """
    # Determine integration end time
    next_jump_time = state.jump_state.next_jump_time
    t_target = jnp.minimum(next_jump_time, t_end)

    # Integrate ODE from state.t to t_target
    # Use runtime.solver_type to select solver
    state_new = _ode_integrate(
        runtime.dynamics_fn,
        state.state,
        state.t,
        t_target,
        runtime.dt_max.value,
        runtime.rtol,
        runtime.atol,
        runtime.params,
        runtime.solver_type,
    )

    # Update state using dataclasses.replace for Penzai structs
    updated_state = dataclasses.replace(
        state,
        t=t_target,
        state=state_new,
        step_count=state.step_count + 1,
    )

    return updated_state, t_target


def apply_jump(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
) -> JumpDiffusionState:
    """Apply jump effect and generate next jump time.

    Args:
        runtime: Runtime configuration
        state: State at jump time (before jump)

    Returns:
        State after jump with new next_jump_time
    """
    # Apply jump effect to state
    state_after_jump = runtime.jump_effect_fn(
        state.t,
        state.state,
        runtime.params
    )

    # Generate next jump time
    jump_state_new, _ = generate_next_jump_time(
        runtime.jump_runtime,
        state.jump_state,
        state.t
    )

    # Update state using dataclasses.replace for Penzai structs
    return dataclasses.replace(
        state,
        state=state_after_jump,
        jump_state=jump_state_new,
        jump_count=state.jump_count + 1,
    )


def simulate(
    runtime: JumpDiffusionRuntime,
    initial_state: jax.Array,
    t_span: Tuple[float, float],
    max_steps: int = 100000,
) -> Tuple[jax.Array, jax.Array]:
    """Run full ODE+Jump simulation using jax.lax.scan for JIT compatibility.

    Saves state at: initial time, each jump time, and final time.
    Does NOT save at every ODE integration step (see Future Extensions for dense_output).

    Args:
        runtime: Runtime configuration
        initial_state: Initial state vector
        t_span: (t_start, t_end) simulation time span
        max_steps: Maximum number of steps (safety limit for total saves)

    Returns:
        (times, states)
        - times: Array of time points [shape: (n_saves,)] padded with inf
        - states: Array of states [shape: (n_saves,) + state_shape] padded with final state

    Note:
        - Output is padded to max_steps length. Filter by `times < t_end` to get actual trajectory.
        - Current implementation: saves only at jump times and t_end
        - Future: dense_output (save every ODE step) and custom save_at times
    """
    from ethode.jumpprocess.runtime import JumpProcessState

    t_start, t_end = t_span

    # Tolerance for time comparisons to avoid spurious iterations from diffrax roundoff
    TIME_ATOL = 1e-9

    # Initialize jump process state
    jump_state_init = JumpProcessState.zero(
        seed=runtime.jump_runtime.seed,
        start_time=t_start
    )
    jump_state_init, _ = generate_next_jump_time(
        runtime.jump_runtime,
        jump_state_init,
        jnp.array(t_start)
    )

    # Initialize simulation state
    sim_state = JumpDiffusionState.zero(
        initial_state,
        jump_state_init,
        t0=t_start
    )

    # Pre-allocate arrays for results
    # Note: Use full shape + dtype to handle arbitrary state dimensions
    times = jnp.full(max_steps, jnp.inf)
    states = jnp.zeros((max_steps,) + initial_state.shape, dtype=initial_state.dtype)

    # Set initial values
    times = times.at[0].set(t_start)
    states = states.at[0].set(initial_state)

    def scan_fn(carry, _):
        """Single scan step: integrate + possibly jump."""
        sim_state, times, states, idx = carry

        # Stop if we've reached t_end
        def continue_sim(_):
            # Integrate to next jump or t_end
            new_state, t_reached = integrate_step(runtime, sim_state, jnp.array(t_end))

            # Check if we hit a jump
            # Use tolerance for both jump detection and end-time check to avoid roundoff issues
            jump_occurred = jnp.logical_and(
                jnp.isclose(new_state.t, new_state.jump_state.next_jump_time, rtol=0, atol=TIME_ATOL),
                new_state.t < t_end - TIME_ATOL
            )

            # Apply jump if occurred
            new_state = jax.lax.cond(
                jump_occurred,
                lambda s: apply_jump(runtime, s),
                lambda s: s,
                new_state
            )

            # Save state if we jumped or reached end (with tolerance)
            should_save = jnp.logical_or(jump_occurred, new_state.t >= t_end - TIME_ATOL)

            # Guard against overflow: only save if idx+1 < max_steps
            can_save = idx + 1 < max_steps
            should_save_safe = jnp.logical_and(should_save, can_save)

            new_idx = jax.lax.cond(
                should_save_safe,
                lambda: idx + 1,
                lambda: idx,
                ()
            )

            new_times = jax.lax.cond(
                should_save_safe,
                lambda: times.at[new_idx].set(new_state.t),
                lambda: times,
                ()
            )

            new_states = jax.lax.cond(
                should_save_safe,
                lambda: states.at[new_idx].set(new_state.state),
                lambda: states,
                ()
            )

            return new_state, new_times, new_states, new_idx

        def stop_sim(_):
            # Already at t_end, don't modify anything
            return sim_state, times, states, idx

        # Only continue if current time < t_end (with tolerance to avoid spurious iterations)
        new_state, new_times, new_states, new_idx = jax.lax.cond(
            sim_state.t < t_end - TIME_ATOL,
            continue_sim,
            stop_sim,
            None
        )

        return (new_state, new_times, new_states, new_idx), None

    # Run scan
    (final_state, final_times, final_states, final_idx), _ = jax.lax.scan(
        scan_fn,
        (sim_state, times, states, jnp.array(0, dtype=jnp.int32)),
        None,
        length=max_steps
    )

    return final_times, final_states


def _ode_integrate(
    dynamics_fn: Callable,
    y0: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    dt_max: float,
    rtol: float,
    atol: float,
    params: Any,
    solver_type: int,
) -> jax.Array:
    """Internal ODE integration using diffrax.

    Args:
        dynamics_fn: Right-hand side function
        y0: Initial state
        t0: Start time
        t1: End time
        dt_max: Maximum step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        params: User parameters
        solver_type: Solver identifier (0=euler, 1=rk4, 2=dopri5, 3=dopri8)

    Returns:
        Final state at t1
    """
    import diffrax

    # Map solver type int to diffrax solver
    # 0=euler, 1=rk4, 2=dopri5, 3=dopri8
    solvers = [
        diffrax.Euler(),
        diffrax.RungeKutta4(),  # Fixed: was Kvaerno4
        diffrax.Dopri5(),
        diffrax.Dopri8(),
    ]
    solver = solvers[solver_type]

    # Define ODE term
    def vector_field(t, y, args):
        return dynamics_fn(t, y, params)

    term = diffrax.ODETerm(vector_field)

    # Solve
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt_max,
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(t1=True),
    )

    return solution.ys[-1]
```

### 2.4 Adapter Layer (High-level API)

```python
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any
from ethode.jumpdiffusion.config import JumpDiffusionConfig
from ethode.jumpdiffusion.runtime import JumpDiffusionRuntime, JumpDiffusionState
from ethode.jumpdiffusion.kernel import simulate, integrate_step, apply_jump
from ethode.jumpprocess.runtime import JumpProcessState
from ethode.jumpprocess.kernel import generate_next_jump_time


class JumpDiffusionAdapter:
    """High-level adapter for ODE+Jump simulations with stateful API.

    This is the primary API for hybrid continuous/discrete simulations.
    Follows the ethode adapter pattern with stateful and functional interfaces.

    Example:
        >>> # Define dynamics
        >>> def my_dynamics(t, state, params):
        ...     x, v = state
        ...     return jnp.array([v, -params['k'] * x])  # Harmonic oscillator
        >>>
        >>> def my_jump(t, state, params):
        ...     x, v = state
        ...     return jnp.array([x, -v * 0.9])  # Damped collision
        >>>
        >>> # Configure
        >>> config = JumpDiffusionConfig(
        ...     initial_state=jnp.array([1.0, 0.0]),
        ...     dynamics_fn=my_dynamics,
        ...     jump_effect_fn=my_jump,
        ...     jump_process=JumpProcessConfig(
        ...         process_type='poisson',
        ...         rate="10 / second",
        ...     ),
        ...     solver='dopri5',
        ...     dt_max="0.01 second",
        ...     params={'k': 1.0},
        ... )
        >>>
        >>> # Stateful API
        >>> adapter = JumpDiffusionAdapter(config)
        >>> times, states = adapter.simulate(t_span=(0.0, 10.0))
        >>>
        >>> # Or step-by-step
        >>> adapter.reset()
        >>> for i in range(100):
        ...     jump_occurred = adapter.step(t_end=adapter.state.t + 0.1)
        ...     if jump_occurred:
        ...         print(f"Jump at t={adapter.state.t}")

    Args:
        config: JumpDiffusionConfig instance
        check_units: Whether to validate dimensional consistency

    Attributes:
        config: The configuration used
        runtime: JAX-ready runtime structure
        state: Current simulation state (JumpDiffusionState)
    """

    def __init__(
        self,
        config: JumpDiffusionConfig,
        *,
        check_units: bool = True
    ):
        self.config = config
        self.runtime = config.to_runtime(check_units=check_units)

        # Initialize state
        jump_state_init = JumpProcessState.zero(
            seed=self.runtime.jump_runtime.seed,
            start_time=0.0
        )
        jump_state_init, _ = generate_next_jump_time(
            self.runtime.jump_runtime,
            jump_state_init,
            jnp.array(0.0)
        )

        self.state = JumpDiffusionState.zero(
            self.config.initial_state,
            jump_state_init,
            t0=0.0
        )

    def simulate(
        self,
        t_span: Tuple[float, float],
        max_steps: int = 100000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full ODE+Jump simulation over time span (functional).

        Saves state at: initial time, each jump time, and final time.
        Does not modify internal state. For stateful stepping, use step().

        Args:
            t_span: (t_start, t_end) simulation interval
            max_steps: Maximum number of steps (safety limit for total saves)

        Returns:
            (times, states) where:
            - times: 1D array of time points at jump times + t_end
            - states: 2D array (n_times, state_dim) with corresponding states

        Note:
            Internal padding is automatically filtered out. You receive only actual trajectory.
        """
        times_jax, states_jax = simulate(
            self.runtime,
            self.config.initial_state,
            t_span,
            max_steps=max_steps,
        )

        # Filter out padding
        mask = times_jax < t_span[1]
        return np.array(times_jax[mask]), np.array(states_jax[mask])

    def step(
        self,
        t_end: float,
    ) -> bool:
        """
        Take single step: integrate to next jump or t_end (stateful).

        Updates internal self.state.

        Args:
            t_end: Maximum time to integrate to

        Returns:
            jump_occurred: True if a jump occurred during this step
        """
        new_state, t_reached = integrate_step(
            self.runtime,
            self.state,
            jnp.array(t_end)
        )

        # Check if jump occurred
        jump_occurred = jnp.logical_and(
            jnp.isclose(new_state.t, new_state.jump_state.next_jump_time, rtol=0, atol=1e-9),
            new_state.t < t_end
        )

        # Apply jump if occurred
        if jump_occurred:
            new_state = apply_jump(self.runtime, new_state)

        self.state = new_state
        return bool(jump_occurred)

    def reset(self, t0: float = 0.0, seed: Optional[int] = None):
        """
        Reset simulation to initial conditions.

        Args:
            t0: Initial time
            seed: New random seed (uses config seed if None)
        """
        seed = seed if seed is not None else self.runtime.jump_runtime.seed

        # Re-initialize jump process state
        jump_state_init = JumpProcessState.zero(seed=seed, start_time=t0)
        jump_state_init, _ = generate_next_jump_time(
            self.runtime.jump_runtime,
            jump_state_init,
            jnp.array(t0)
        )

        self.state = JumpDiffusionState.zero(
            self.config.initial_state,
            jump_state_init,
            t0=t0
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get current simulation state as dictionary.

        Returns:
            Dictionary with:
            - 't': current time
            - 'state': current state vector
            - 'next_jump_time': time of next scheduled jump
            - 'step_count': number of ODE steps taken
            - 'jump_count': number of jumps processed
        """
        return {
            't': float(self.state.t),
            'state': np.array(self.state.state),
            'next_jump_time': float(self.state.jump_state.next_jump_time),
            'step_count': int(self.state.step_count),
            'jump_count': int(self.state.jump_count),
        }

    def set_state(self, state: JumpDiffusionState):
        """
        Set simulation state directly.

        Args:
            state: JumpDiffusionState to use
        """
        self.state = state
```

---

## 3. Integration with Existing Ethode

### 3.1 Dependencies

**Required:**
- `JumpProcessConfig`, `JumpProcessAdapter` (already implemented)
- `HawkesConfig`, `HawkesAdapter` (already implemented)
- `UnitManager`, `QuantityNode` (already implemented)
- `diffrax` for ODE integration (already a dependency)

**New files to add:**
```
ethode/
  jumpdiffusion/
    __init__.py
    config.py      # JumpDiffusionConfig, JumpDiffusionConfigOutput
    runtime.py     # JumpDiffusionRuntime, JumpDiffusionState
    kernel.py      # integrate_step, apply_jump, simulate, _ode_integrate
```

**Update:**
```
ethode/adapters.py  # Add JumpDiffusionAdapter
ethode/__init__.py  # Export new classes
```

### 3.2 Example Usage

Migration from legacy `JumpDiffusionSim` pattern:

```python
# BEFORE (legacy)
class MyJumpDiffusionSim(JumpDiffusionSim):  # Legacy!
    ...

# AFTER (new ethode)
from ethode import JumpDiffusionConfig, JumpDiffusionAdapter, JumpProcessConfig

def create_simulation(params):
    """Create hybrid ODE+Jump simulation."""

    def dynamics(t, state, params):
        """Continuous dynamics between jumps."""
        x, y, z = state
        dx_dt = params['alpha'] * x
        dy_dt = 0.0
        dz_dt = params['beta'] * z
        return jnp.array([dx_dt, dy_dt, dz_dt])

    def jump_effect(t, state, params):
        """Discrete jump effect on state."""
        x, y, z = state
        # Example: redistribute accumulated values
        accumulated = x - y
        y_new = y + accumulated

        # Apply jump logic...
        return jnp.array([x, y_new, z])

    config = JumpDiffusionConfig(
        initial_state=jnp.array([
            params['init_x'],
            params['init_y'],
            params['init_z'],
        ]),
        dynamics_fn=dynamics,
        jump_effect_fn=jump_effect,
        jump_process=JumpProcessConfig(
            process_type='poisson',
            rate=f"{params['jump_rate']} / day",
            seed=params.get('seed', 42),
        ),
        solver='dopri5',
        dt_max="0.1 day",
        params=params,
    )

    return JumpDiffusionAdapter(config)

# Usage
sim = create_simulation(params)
times, states = sim.simulate(t_span=(0.0, 365.0))
```

---

## 4. Testing Requirements

### 4.1 Unit Tests

**Config validation:**
```python
def test_jump_diffusion_config_validation():
    """Test that config validates inputs correctly."""
    # Valid config
    config = JumpDiffusionConfig(
        initial_state=jnp.array([1.0, 0.0]),
        dynamics_fn=lambda t, s, p: s,
        jump_effect_fn=lambda t, s, p: s,
        jump_process=JumpProcessConfig(process_type='poisson', rate="10/day"),
        dt_max="0.01 day",
    )
    assert config is not None

    # Invalid dt_max (wrong dimension)
    with pytest.raises(ValueError, match="time dimension"):
        JumpDiffusionConfig(
            ...,
            dt_max="1.0 kg",  # Wrong dimension!
        )
```

**Integration accuracy:**
```python
def test_ode_integration_accuracy():
    """Test ODE integration matches analytical solution."""
    # Exponential growth: dy/dt = k*y, y(0) = 1
    # Solution: y(t) = exp(k*t)
    k = 0.5

    config = JumpDiffusionConfig(
        initial_state=jnp.array([1.0]),
        dynamics_fn=lambda t, y, p: jnp.array([p['k'] * y[0]]),
        jump_effect_fn=lambda t, y, p: y,  # No jumps
        jump_process=JumpProcessConfig(process_type='poisson', rate="0/day"),
        dt_max="0.01 day",
        params={'k': k},
    )

    adapter = JumpDiffusionAdapter(config)
    times, states = adapter.simulate(t_span=(0.0, 1.0))

    # Check final value
    expected = jnp.exp(k * 1.0)
    assert jnp.abs(states[-1, 0] - expected) < 1e-6
```

**Jump handling:**
```python
def test_jump_application():
    """Test that jumps are applied correctly."""
    # Constant dynamics, periodic jumps
    config = JumpDiffusionConfig(
        initial_state=jnp.array([1.0]),
        dynamics_fn=lambda t, y, p: jnp.array([0.0]),  # Constant
        jump_effect_fn=lambda t, y, p: jnp.array([y[0] * 2]),  # Double on jump
        jump_process=JumpProcessConfig(
            process_type='deterministic',
            rate="1/day",  # One jump per day
        ),
        dt_max="0.1 day",
    )

    adapter = JumpDiffusionAdapter(config)
    times, states = adapter.simulate(t_span=(0.0, 3.0))

    # Should have jumps at t=1, 2, 3
    # State after 3 jumps: 1 * 2 * 2 * 2 = 8
    assert jnp.abs(states[-1, 0] - 8.0) < 1e-9
```

### 4.2 Integration Test (RAI Dollar Example)

```python
def test_rai_dollar_interest_simulation():
    """Test RAI Dollar interest rate model."""
    # Simple case: constant rate, one drip
    params = {
        'init_D': 1e6,
        'init_R': 1e6,
        'alpha': 0.05,  # 5% annual
    }

    def dynamics(t, state, p):
        D, R = state
        return jnp.array([p['alpha'] * D, 0.0])

    def jump_effect(t, state, p):
        D, R = state
        return jnp.array([D, D])  # Set R = D

    config = JumpDiffusionConfig(
        initial_state=jnp.array([params['init_D'], params['init_R']]),
        dynamics_fn=dynamics,
        jump_effect_fn=jump_effect,
        jump_process=JumpProcessConfig(
            process_type='deterministic',
            rate="365/year",  # Daily drips
        ),
        dt_max="0.1 day",
        params=params,
    )

    adapter = JumpDiffusionAdapter(config)
    times, states = adapter.simulate(t_span=(0.0, 10.0))  # 10 days

    # Check debt grew
    assert states[-1, 0] > states[0, 0]

    # Check supply matches debt at end (last drip)
    assert jnp.abs(states[-1, 0] - states[-1, 1]) < 1e-3
```

---

## 5. Implementation Notes

### 5.1 Solver Selection

Use `diffrax` for ODE integration (already ethode dependency):
- **Explicit methods**: Euler, RK4 (RungeKutta4), Dopri5, Dopri8
- **Implicit methods**: (future) for stiff systems
- **Adaptive stepping**: PID controller for step size

**Important**: Solver type is stored as int (0-3) for JAX compatibility, not string.

### 5.2 Performance Considerations

1. **JIT compilation**: All kernel functions use `jax.lax.scan` for JIT compatibility
2. **Pre-allocated arrays**: `simulate()` pre-allocates result arrays (padded with `inf`)
3. **Vectorization**: `jax.vmap` over multiple simulations works out of the box
4. **Memory**: Default max_steps=100000 may need tuning for long simulations
5. **Padding**: Output arrays are padded; filter by `times < t_end` to get actual trajectory

### 5.3 JAX Compatibility

**Key design decisions for JAX**:
- Penzai structs (`@pz.pytree_dataclass`) instead of NamedTuple
- `dataclasses.replace()` for struct updates
- `jnp.logical_and()` instead of Python `and`
- `jax.lax.cond()` for conditional execution
- `jax.lax.scan()` for main simulation loop
- Integer arrays for counters (`step_count`, `jump_count`)

### 5.4 Edge Cases

1. **No jumps**: `rate="0/day"` → pure ODE (but still initializes jump state)
2. **Many jumps**: High-frequency jumps may require smaller `dt_max` and larger `max_steps`
3. **Stiff systems**: May need implicit solvers (future extension)
4. **State-dependent jumps**: Not supported in this version (jumps are time-only)
5. **Exceeding max_steps**: Simulation stops, returns padded arrays (check for `inf` in times)

---

## 6. Timeline Estimate

**Implementation effort**: ~3-4 days

1. **Day 1**: Config + Runtime layers, basic validation
2. **Day 2**: Kernel layer (integrate_step, apply_jump, simulate)
3. **Day 3**: Adapter layer + unit tests
4. **Day 4**: Integration tests + documentation

---

## 7. Future Extensions

### High Priority (v2.0)

1. **Dense output**: Save state at every ODE integration step (not just jumps)
   - Add `dense_output: bool` to config
   - Modify scan_fn to save at every iteration when enabled
   - Implementation: ~0.5 days

2. **Custom save times**: Save state at user-specified times via `save_at` parameter
   - Add `save_at: Optional[jax.Array]` to simulate signature
   - Interpolate state at requested times
   - Implementation: ~1 day

### Medium Priority

3. **State-dependent jump rates**: Jump rate depends on current state
4. **Multiple jump processes**: Different types of events (compound process)
5. **Implicit solvers**: For stiff ODEs (Kvaerno solvers)
6. **Event detection**: Trigger jumps when state crosses threshold (diffrax event detection)

### Low Priority

7. **Parallel simulations**: `vmap` over ensemble (already supported, needs documentation)
8. **Checkpointing**: Save/load simulation state for long runs

---

## Appendix: Comparison with Legacy

| Feature | Legacy `JumpDiffusionSim` | New `JumpDiffusionAdapter` |
|---------|---------------------------|----------------------------|
| Config | Class attributes | Pydantic config |
| Units | Manual | Automatic validation (pint) |
| Jump process | Coupled | Separate JumpProcessConfig |
| ODE solver | Fixed (scipy) | Configurable (diffrax) |
| JAX compatibility | None | Full (Penzai pytrees) |
| Batching | Manual loops | `jax.vmap` ready |
| State management | Mutable (Python lists) | Immutable (functional) |
| Simulation loop | Python `while` | `jax.lax.scan` |
| JIT compilation | Not supported | Full support |
| Gradient computation | Not supported | JAX autodiff ready |

---

**End of Specification**

---

## Status Summary

**Specification status**: ✅ Ready for implementation (all blockers resolved)
**Dependencies**: All required (JumpProcess, Hawkes, UnitManager, diffrax, Penzai)
**Architecture**: Fully aligned with ethode patterns (Config/Runtime/Kernel/Adapter)
**JAX compatibility**: Complete (scan-based, JIT-ready, vmap-ready)
**Estimated implementation time**: 3-4 days

**Key improvements over v1.0**:
- Penzai structs (`from penzai.core import struct`) instead of NamedTuple
- scan-based simulation loop (no Python lists/while)
- Complete stateful adapter API (reset, get_state, set_state)
- Fixed solver mapping (RK4 = RungeKutta4, not Kvaerno4)
- Integer solver_type for JAX compatibility
- Proper use of dataclasses.replace()
- jnp.logical_and() instead of Python `and`

**Critical fixes (v1.1)**:
- Correct Penzai import: `from penzai.core import struct` (not penzai.toolshed)
- State buffer handles arbitrary shapes/dtypes: `jnp.zeros((max_steps,) + initial_state.shape, dtype=initial_state.dtype)`
- Overflow guard: Check `idx + 1 < max_steps` before scatter operations
- Time tolerance: Use `t < t_end - TIME_ATOL` consistently to avoid spurious iterations from diffrax roundoff

**Minor clarifications (v1.1)**:
- Removed `dense_output` config field (moved to Future Extensions)
- Clarified that `simulate()` saves only at jump times and t_end (not every ODE step)
- Added roadmap for dense output and custom `save_at` times in v2.0

Questions or feedback? Contact ethode development team.
