# Ethode ODE+Jump Simulation API Specification

**Version**: 1.0
**Date**: 2025-10-05
**Status**: Draft for ethode team
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
        dense_output: Whether to save state at every integration step
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

    # Output configuration
    dense_output: bool = Field(
        default=False,
        description="Save state at every integration step (not just jumps)"
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

        return JumpDiffusionRuntime(
            dynamics_fn=self.dynamics_fn,
            jump_effect_fn=self.jump_effect_fn,
            jump_runtime=self.jump_process.to_runtime(check_units=check_units),
            solver_type=self.solver,
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
from typing import NamedTuple, Callable, Any
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from ethode.runtime import QuantityNode
from ethode.jumpprocess import JumpProcessRuntime


@register_pytree_node_class
class JumpDiffusionRuntime(NamedTuple):
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
    solver_type: str
    dt_max: QuantityNode
    rtol: float
    atol: float
    params: Any

    def tree_flatten(self):
        """JAX pytree flatten.

        IMPORTANT: jump_runtime is a pytree (contains RNG state), so it must
        be in children, not aux. Only truly static data goes in aux.
        """
        children = (self.dt_max, self.jump_runtime)
        aux = (
            self.dynamics_fn,
            self.jump_effect_fn,
            self.solver_type,
            self.rtol,
            self.atol,
            self.params,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """JAX pytree unflatten."""
        dt_max, jump_runtime = children
        dynamics_fn, jump_effect_fn, solver_type, rtol, atol, params = aux
        return cls(
            dynamics_fn, jump_effect_fn, jump_runtime,
            solver_type, dt_max, rtol, atol, params
        )


@register_pytree_node_class
class JumpDiffusionState(NamedTuple):
    """Current simulation state.

    Attributes:
        t: Current time
        state: Current state vector
        jump_state: JumpProcessState for next jump
        step_count: Number of ODE steps taken
        jump_count: Number of jumps processed
    """

    t: jnp.ndarray  # Current time (scalar)
    state: jnp.ndarray  # State vector
    jump_state: Any  # JumpProcessState (pytree)
    step_count: int
    jump_count: int

    def tree_flatten(self):
        """JAX pytree flatten."""
        children = (self.t, self.state, self.jump_state)
        aux = (self.step_count, self.jump_count)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """JAX pytree unflatten."""
        t, state, jump_state = children
        step_count, jump_count = aux
        return cls(t, state, jump_state, step_count, jump_count)

    @classmethod
    def from_initial(
        cls,
        initial_state: jnp.ndarray,
        jump_state: Any,
        t0: float = 0.0
    ):
        """Create initial state."""
        return cls(
            t=jnp.array(t0),
            state=initial_state,
            jump_state=jump_state,
            step_count=0,
            jump_count=0,
        )
```

### 2.3 Kernel Layer

```python
import jax
import jax.numpy as jnp
from typing import Tuple
from ethode.jumpprocess.kernel import generate_next_jump_time


def integrate_step(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
    t_end: jnp.ndarray,
) -> Tuple[JumpDiffusionState, jnp.ndarray]:
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
    from ethode.jumpprocess.kernel import check_jump_occurred

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

    updated_state = JumpDiffusionState(
        t=t_target,
        state=state_new,
        jump_state=state.jump_state,
        step_count=state.step_count + 1,
        jump_count=state.jump_count,
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
    from ethode.jumpprocess.kernel import generate_next_jump_time

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

    return JumpDiffusionState(
        t=state.t,
        state=state_after_jump,
        jump_state=jump_state_new,
        step_count=state.step_count,
        jump_count=state.jump_count + 1,
    )


def simulate(
    runtime: JumpDiffusionRuntime,
    initial_state: jnp.ndarray,
    t_span: Tuple[float, float],
    save_at: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run full ODE+Jump simulation.

    Args:
        runtime: Runtime configuration
        initial_state: Initial state vector
        t_span: (t_start, t_end) simulation time span
        save_at: Optional array of times to save state (default: jump times only)

    Returns:
        (times, states)
        - times: Array of time points [shape: (n_saves,)]
        - states: Array of states [shape: (n_saves, state_dim)]
    """
    from ethode.jumpprocess.runtime import JumpProcessState
    from ethode.jumpprocess.kernel import generate_next_jump_time

    t_start, t_end = t_span

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
    sim_state = JumpDiffusionState.from_initial(
        initial_state,
        jump_state_init,
        t0=t_start
    )

    # Storage for results
    times_list = [t_start]
    states_list = [initial_state]

    # Main simulation loop
    while sim_state.t < t_end:
        # Integrate to next jump or t_end
        sim_state, t_reached = integrate_step(runtime, sim_state, jnp.array(t_end))

        # Check if we hit a jump (use tolerance to handle floating-point errors from ODE integration)
        jump_occurred = jnp.isclose(sim_state.t, sim_state.jump_state.next_jump_time, rtol=0, atol=1e-9) and (sim_state.t < t_end)

        if jump_occurred:
            # Apply jump
            sim_state = apply_jump(runtime, sim_state)

            # Save post-jump state
            times_list.append(float(sim_state.t))
            states_list.append(sim_state.state)
        elif sim_state.t >= t_end:
            # Reached end time
            times_list.append(float(sim_state.t))
            states_list.append(sim_state.state)
            break

    times = jnp.array(times_list)
    states = jnp.stack(states_list)

    return times, states


def _ode_integrate(
    dynamics_fn: Callable,
    y0: jnp.ndarray,
    t0: jnp.ndarray,
    t1: jnp.ndarray,
    dt_max: float,
    rtol: float,
    atol: float,
    params: Any,
    solver_type: str,
) -> jnp.ndarray:
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
        solver_type: Solver name

    Returns:
        Final state at t1
    """
    import diffrax

    # Map solver type to diffrax solver
    solver_map = {
        'euler': diffrax.Euler(),
        'rk4': diffrax.Kvaerno4(),
        'dopri5': diffrax.Dopri5(),
        'dopri8': diffrax.Dopri8(),
    }
    solver = solver_map.get(solver_type, diffrax.Dopri5())

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
from typing import Tuple, Optional


class JumpDiffusionAdapter:
    """High-level adapter for ODE+Jump simulations.

    This is the primary API for hybrid continuous/discrete simulations.

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
        >>> # Simulate
        >>> adapter = JumpDiffusionAdapter(config)
        >>> times, states = adapter.simulate(t_span=(0.0, 10.0))

    Args:
        config: JumpDiffusionConfig instance
        check_units: Whether to validate dimensional consistency

    Attributes:
        config: The configuration used
        runtime: JAX-ready runtime structure
    """

    def __init__(
        self,
        config: JumpDiffusionConfig,
        *,
        check_units: bool = True
    ):
        self.config = config
        self.runtime = config.to_runtime(check_units=check_units)

    def simulate(
        self,
        t_span: Tuple[float, float],
        save_at: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run ODE+Jump simulation over time span.

        Args:
            t_span: (t_start, t_end) simulation interval
            save_at: Optional times to save state (default: jump times only)

        Returns:
            (times, states) where:
            - times: 1D array of time points
            - states: 2D array (n_times, state_dim)
        """
        times_jax, states_jax = simulate(
            self.runtime,
            self.config.initial_state,
            t_span,
            save_at=jnp.array(save_at) if save_at is not None else None,
        )

        return np.array(times_jax), np.array(states_jax)

    def step(
        self,
        state: JumpDiffusionState,
        t_end: float,
    ) -> Tuple[JumpDiffusionState, bool]:
        """
        Take single step: integrate to next jump or t_end.

        Args:
            state: Current simulation state
            t_end: Maximum time to integrate to

        Returns:
            (new_state, jump_occurred)
        """
        new_state, t_reached = integrate_step(
            self.runtime,
            state,
            jnp.array(t_end)
        )

        # Check if jump occurred (use tolerance to handle floating-point errors from ODE integration)
        jump_occurred = (
            jnp.isclose(new_state.t, new_state.jump_state.next_jump_time, rtol=0, atol=1e-9) and
            new_state.t < t_end
        )

        if jump_occurred:
            new_state = apply_jump(self.runtime, new_state)

        return new_state, bool(jump_occurred)
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
  jumpd diffusion/
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

### 3.2 Usage in rd-sim

Once implemented, `rd_interest_model.py` can migrate from legacy:

```python
# BEFORE (legacy)
class RDInterestSim(JumpDiffusionSim):  # Legacy!
    ...

# AFTER (new ethode)
from ethode import JumpDiffusionConfig, JumpDiffusionAdapter, JumpProcessConfig

def create_rd_interest_simulation(params: RDInterestParams):
    """Create RAI Dollar interest simulation."""

    def dynamics(t, state, params):
        """Continuous dynamics: debt grows at interest rate."""
        D, R, alpha, integral, p_RD = state
        dD_dt = alpha * D
        dR_dt = 0.0
        dalpha_dt = 0.0
        dintegral_dt = 0.0
        dp_RD_dt = 0.0
        return jnp.array([dD_dt, dR_dt, dalpha_dt, dintegral_dt, dp_RD_dt])

    def jump_effect(t, state, params):
        """Drip effect: mint supply to match debt."""
        D, R, alpha, integral, p_RD = state
        accrued = D - R
        R_new = R + accrued

        # Controller update logic...
        # (same as current implementation)

        return jnp.array([D, R_new, alpha_new, integral_new, p_RD])

    config = JumpDiffusionConfig(
        initial_state=jnp.array([
            params.init_D,
            params.init_R,
            params.alpha_bias,
            0.0,
            1.0
        ]),
        dynamics_fn=dynamics,
        jump_effect_fn=jump_effect,
        jump_process=JumpProcessConfig(
            process_type='poisson',
            rate=f"{params.jump_rate / 365} / day",
            seed=params.seed,
        ),
        solver='dopri5',
        dt_max="0.1 day",
        params=params,
    )

    return JumpDiffusionAdapter(config)

# Usage
sim = create_rd_interest_simulation(params)
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
- **Explicit methods**: Euler, RK4, Dopri5, Dopri8
- **Implicit methods**: (future) for stiff systems
- **Adaptive stepping**: PID controller for step size

### 5.2 Performance Considerations

1. **JIT compilation**: All kernel functions should be `@jax.jit`-able
2. **Vectorization**: `jax.vmap` over multiple simulations
3. **Checkpointing**: For long simulations, save state periodically
4. **Memory**: Dense output can be expensive - default to jump times only

### 5.3 Edge Cases

1. **No jumps**: `rate="0/day"` → pure ODE
2. **Many jumps**: High-frequency jumps may require smaller `dt_max`
3. **Stiff systems**: May need implicit solvers (future extension)
4. **State-dependent jumps**: Not supported in this version (jumps are time-only)

---

## 6. Timeline Estimate

**Implementation effort**: ~3-4 days

1. **Day 1**: Config + Runtime layers, basic validation
2. **Day 2**: Kernel layer (integrate_step, apply_jump, simulate)
3. **Day 3**: Adapter layer + unit tests
4. **Day 4**: Integration tests + documentation

---

## 7. Future Extensions

1. **State-dependent jump rates**: Jump rate depends on current state
2. **Multiple jump processes**: Different types of events
3. **Implicit solvers**: For stiff ODEs
4. **Event detection**: Trigger jumps when state crosses threshold
5. **Parallel simulations**: `vmap` over ensemble

---

## Appendix: Comparison with Legacy

| Feature | Legacy `JumpDiffusionSim` | New `JumpDiffusionAdapter` |
|---------|---------------------------|----------------------------|
| Config | Class attributes | Pydantic config |
| Units | Manual | Automatic validation |
| Jump process | Coupled | Separate JumpProcessConfig |
| ODE solver | Fixed | Configurable (diffrax) |
| JAX compatibility | Partial | Full (pytrees) |
| Batching | Manual | `jax.vmap` ready |
| State management | Mutable | Immutable (functional) |

---

**End of Specification**

Questions or feedback? Contact ethode development team.
