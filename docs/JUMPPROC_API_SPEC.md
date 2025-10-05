# JumpProcess API Specification for Ethode

**Request from**: rd-sim migration effort
**Purpose**: Enable discrete event simulation with jump processes in new ethode architecture
**Priority**: Critical blocker for full rd-sim migration
**Version**: 3.0 (JAX-native with architectural improvements)

---

## Revision History

- **v3.0**: JAX-native architecture
  - Updated to use diffrax for ODE integration (primary)
  - Added jax.lax.scan fallback for simple ODEs
  - Removed scipy.integrate.solve_ivp option (full JAX)
  - Changed module structure to `ethode/jumpprocess/` (no underscore)
  - Updated terminology: `jumpprocess` in variable names (consistent with other subsystems)
  - Added integration with Simulation class
  - Added comprehensive examples and test coverage
- **v2.0**: Integrated feedback from ethode team
  - Fixed `validate_rate` to use `manager.to_canonical()` instead of string comparison
  - Fixed `generate_jumps_in_interval` recurrence seeding bug
  - Updated runtime to use `QuantityNode` for consistency
  - Added `start_time` parameter to state initialization
- **v1.0**: Initial specification

---

## 1. Overview

The rd-sim codebase currently uses `JumpProcess`, `JumpProcessParams`, and `JumpDiffusionSim` from the legacy `stochastic_extensions.py` module. We need equivalent functionality in the new ethode API to support:

1. **Poisson jump processes** - Random events at constant rate
2. **Deterministic jump processes** - Regular periodic events
3. **Integration with continuous dynamics** - Combining ODEs with discrete jumps (JAX-native)
4. **Event-driven simulation** - Triggering state changes at jump times
5. **Full JAX compatibility** - JIT compilation, automatic differentiation, vmap

**Key Design Principle**: Full JAX integration using diffrax for ODE solving, enabling:
- Gradient-based optimization through entire simulation
- JIT compilation for performance
- Parallel simulations via vmap
- No NumPy/SciPy dependencies in critical path

---

## 2. Current Usage in rd-sim

### 2.1 Legacy Code Pattern

**Jump process generation:**
```python
from stochastic_extensions import JumpProcess, JumpProcessParams

# Configuration
params = JumpProcessParams(
    jump_rate=100.0,              # Events per time unit
    jump_process_type='poisson',  # or 'deterministic'
    seed=12345
)

# Usage
jp = JumpProcess(params)
jump_times = jp.generate_jumps(t_start=0.0, t_end=1.0)
# Returns: [0.0234, 0.1567, 0.3421, ...] - event times
```

**Jump-diffusion simulation:**
```python
from stochastic_extensions import JumpDiffusionSim, JumpProcessParams

class MyModel(JumpDiffusionSim):
    params: JumpProcessParams

    @staticmethod
    def func(t, v, p):
        """Continuous dynamics between jumps."""
        x, y = v
        dx_dt = -0.1 * x
        dy_dt = 0.2 * y
        return (dx_dt, dy_dt)

    def jump_effect(self, t: float, state: np.ndarray) -> np.ndarray:
        """Effect of a jump on state."""
        state[0] += np.random.normal(0, 0.1)  # Add noise to x
        return state

# Run simulation
sim = MyModel(params)
sim.sim()  # Integrates ODEs between jumps, applies jump_effect at each jump
```

### 2.2 Specific Use Cases in rd-sim

**Use Case 1: Interest rate drip events** (`rd_interest_model.py`)
- Continuous: Debt grows continuously via `dD/dt = α * D`
- Jumps: Periodic "drip" events that update interest rate controller
- Need: Generate jump times, integrate ODEs between jumps, execute controller at each jump

**Use Case 2: AMM swap events** (`rd_par_discrete_model.py`)
- Continuous: TWAP dynamics evolve continuously
- Jumps: Random swap events (Poisson or Hawkes) that affect spot price
- Need: Generate stochastic jump times, apply discrete state changes

**Use Case 3: Multiple event types** (`rd_par_discrete_model.py`)
- Different jump processes for different event types:
  - Redemptions (user-triggered, Hawkes process)
  - AMM swaps (market activity, Poisson or Hawkes)
  - Borrowing (new CDPs, Poisson)
  - Liquidations (price-triggered)
  - PAR updates (deterministic periodic)
- Need: Combine multiple jump processes, merge event queues

---

## 3. Module Structure

Following ethode conventions, jumpprocess will be organized as:

```
ethode/jumpprocess/
├── __init__.py          # Exports all public API
├── config.py            # JumpProcessConfig, JumpProcessConfigOutput
├── runtime.py           # JumpProcessRuntime, JumpProcessState
└── kernel.py            # Pure JAX kernel functions
```

Then add `JumpProcessAdapter` to `ethode/adapters.py` (consistent with other adapters).

---

## 4. Proposed API Design

### 4.1 Config Layer (Pydantic)

**File**: `ethode/jumpprocess/config.py`

Following the existing ethode pattern from `ControllerConfig`, `HawkesConfig`, etc.

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional, Tuple
import pint
import jax.numpy as jnp

from ..units import UnitManager, UnitSpec, QuantityInput

class JumpProcessConfig(BaseModel):
    """Configuration for jump (point) processes.

    Supports Poisson (constant rate) and deterministic (periodic) processes.
    For self-exciting (Hawkes) processes, use HawkesConfig.

    Example:
        >>> config = JumpProcessConfig(
        ...     process_type='poisson',
        ...     rate="100 / day",
        ...     seed=42
        ... )
        >>> adapter = JumpProcessAdapter(config)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Process type
    process_type: Literal['poisson', 'deterministic'] = Field(
        default='poisson',
        description="Type of jump process"
    )

    # Rate parameter (unit-aware)
    rate: Tuple[float, UnitSpec] = Field(
        description="Event rate (events per time unit)"
    )

    # Random seed for reproducibility
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for stochastic processes"
    )

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, v: QuantityInput, info) -> Tuple[float, UnitSpec]:
        """Validate rate has dimension 1/time (frequency).

        Uses manager.to_canonical() to properly handle pint's dimensionality
        representation (e.g., "1 / [time]").
        """
        manager = UnitManager.instance()

        # Parse input to quantity
        if isinstance(v, str):
            q = manager.ensure_quantity(v)
        elif isinstance(v, pint.Quantity):
            q = v
        else:
            # Bare number - assume 1/year
            q = manager.ensure_quantity(f"{v} 1/year", "1/year")

        # Validate and convert to canonical form using to_canonical
        # This properly handles pint's "1 / [time]" dimensionality format
        try:
            rate_value, rate_spec = manager.to_canonical(q, "1/time")
        except (ValueError, pint.DimensionalityError) as e:
            raise ValueError(
                f"Rate must have frequency dimension (1/time), got {q.dimensionality}. "
                f"Error: {e}"
            )

        return (rate_value, rate_spec)

    def to_runtime(self, check_units: bool = True) -> 'JumpProcessRuntime':
        """Convert config to JAX-ready runtime structure."""
        from ..runtime import QuantityNode
        from .runtime import JumpProcessRuntime

        rate_value, rate_spec = self.rate

        # Store rate as QuantityNode for consistency with rest of ethode
        rate_node = QuantityNode.from_float(rate_value, rate_spec)

        return JumpProcessRuntime(
            process_type=0 if self.process_type == 'poisson' else 1,  # Enum for JAX
            rate=rate_node,
            seed=self.seed if self.seed is not None else 0,
        )


class JumpProcessConfigOutput(BaseModel):
    """Output wrapper for JumpProcessConfig (for introspection)."""

    config: JumpProcessConfig
    runtime: 'JumpProcessRuntime'

    # Diagnostic info
    expected_events_per_unit_time: float
    process_type_name: str
```

### 4.2 Runtime Layer (JAX Structures)

**File**: `ethode/jumpprocess/runtime.py`

```python
from typing import NamedTuple
import jax
import jax.numpy as jnp
from penzai import pz

from ..runtime import QuantityNode

@pz.pytree_dataclass
class JumpProcessRuntime(pz.Struct):
    """JAX-compatible runtime structure for jump processes.

    All fields are JAX arrays for use in jax.lax.scan, jax.jit, etc.

    Uses QuantityNode for rate to match ethode conventions.
    """
    process_type: jax.Array  # 0=Poisson, 1=deterministic
    rate: QuantityNode       # Event rate as QuantityNode (not raw float)
    seed: int                # Random seed (auxiliary data, not differentiated)


@pz.pytree_dataclass
class JumpProcessState(pz.Struct):
    """State for jump process simulation.

    Tracks the current state of the jump process for sequential simulation.
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
        import jax.random as jrandom
        return cls(
            last_jump_time=jnp.array(float(start_time)),
            next_jump_time=jnp.array(float(start_time)),  # Will be updated immediately
            rng_key=jrandom.PRNGKey(seed),
            event_count=jnp.array(0, dtype=jnp.int32),
        )
```

### 4.3 Kernel Functions (Pure JAX)

**File**: `ethode/jumpprocess/kernel.py`

```python
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
        event_count=state.event_count + 1,
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

    # Generate next jump if this one occurred
    new_state = jax.lax.cond(
        jump_occurred,
        lambda s: generate_next_jump_time(runtime, s, state.next_jump_time)[0],
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
```

### 4.4 Adapter Layer (High-level API)

**File**: `ethode/adapters.py` (add to existing file)

```python
import numpy as np
import jax.numpy as jnp
from typing import Optional

class JumpProcessAdapter:
    """High-level adapter for jump processes with stateful API.

    This is the primary high-level API for jump process usage.

    Example:
        >>> config = JumpProcessConfig(
        ...     process_type='poisson',
        ...     rate="100 / day",
        ...     seed=42
        ... )
        >>> adapter = JumpProcessAdapter(config)
        >>>
        >>> # Sequential usage
        >>> for t in range(100):
        ...     jump_occurred = adapter.step(t * 0.1, dt=0.1)
        ...     if jump_occurred:
        ...         # Handle event
        ...         pass
        >>>
        >>> # Batch generation
        >>> jump_times = adapter.generate_jumps(0.0, 10.0)

    Args:
        config: JumpProcessConfig instance
        check_units: Whether to validate dimensional consistency (default: True)

    Attributes:
        config: The JumpProcessConfig used
        runtime: JAX-ready runtime structure
        state: Current JumpProcessState
    """

    def __init__(
        self,
        config: 'JumpProcessConfig',
        *,
        check_units: bool = True
    ):
        from .jumpprocess.config import JumpProcessConfig
        from .jumpprocess.runtime import JumpProcessState
        from .jumpprocess.kernel import generate_next_jump_time

        self.config = config
        self.runtime = config.to_runtime(check_units=check_units)
        self.state = JumpProcessState.zero(
            seed=config.seed if config.seed is not None else 0,
            start_time=0.0
        )

        # Initialize first jump time from t=0
        self.state, _ = generate_next_jump_time(
            self.runtime, self.state, jnp.array(0.0)
        )

    def step(self, current_time: float, dt: float) -> bool:
        """
        Step forward in time, check if jump occurred.

        Updates internal state.

        Args:
            current_time: Current time
            dt: Time step

        Returns:
            True if jump occurred in [current_time, current_time + dt)
        """
        from .jumpprocess.kernel import step

        self.state, occurred = step(
            self.runtime,
            self.state,
            jnp.array(float(current_time)),
            jnp.array(float(dt))
        )
        return bool(occurred)

    def generate_jumps(self, t_start: float, t_end: float) -> np.ndarray:
        """
        Generate all jump times in interval [t_start, t_end).

        Args:
            t_start: Start time
            t_end: End time

        Returns:
            Numpy array of jump times
        """
        from .jumpprocess.kernel import generate_jumps_in_interval

        jumps = generate_jumps_in_interval(
            self.runtime,
            t_start,
            t_end,
            seed=self.config.seed if self.config.seed is not None else 0
        )
        return np.array(jumps)

    def reset(self, seed: Optional[int] = None, start_time: float = 0.0):
        """Reset state to initial conditions.

        Args:
            seed: Random seed (uses config seed if None)
            start_time: Time to start from (default 0.0)
        """
        from .jumpprocess.runtime import JumpProcessState
        from .jumpprocess.kernel import generate_next_jump_time

        seed = seed if seed is not None else self.config.seed
        seed = seed if seed is not None else 0
        self.state = JumpProcessState.zero(seed, start_time)

        # Generate first jump
        self.state, _ = generate_next_jump_time(
            self.runtime, self.state, jnp.array(start_time)
        )

    def get_expected_rate(self) -> float:
        """Get expected event rate (events per unit time)."""
        return float(self.runtime.rate.value)

    def get_state(self) -> dict:
        """Get current state as dictionary."""
        return {
            'last_jump_time': float(self.state.last_jump_time),
            'next_jump_time': float(self.state.next_jump_time),
            'event_count': int(self.state.event_count),
        }
```

### 4.5 Integration with ODE Simulation (JAX-Native)

The primary integration pattern uses **diffrax** for ODE solving with JAX. This enables:
- Full JIT compilation
- Automatic differentiation through entire simulation
- Vmap for parallel simulations
- No NumPy/SciPy dependencies

#### Primary: Using diffrax (Recommended)

**Installation**: `pip install diffrax`

**Helper function** (document in examples):

```python
import diffrax
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

def simulate_jump_diffusion_diffrax(
    jump_adapter: JumpProcessAdapter,
    ode_func: Callable[[jax.Array, jax.Array], jax.Array],  # (t, y) -> dy/dt
    jump_effect: Callable[[jax.Array, jax.Array], jax.Array],  # (t, y) -> y_new
    t_span: Tuple[float, float],
    y0: jax.Array,
    dt0: float = 0.01,
    solver=None,
) -> Tuple[jax.Array, jax.Array]:
    """
    Simulate ODE with jump process - fully JAX-native using diffrax.

    This function integrates continuous ODE dynamics between discrete jump events.
    Supports adaptive stepping, stiff solvers, and is fully differentiable.

    Args:
        jump_adapter: JumpProcessAdapter for event generation
        ode_func: ODE right-hand side: dy/dt = f(t, y)
        jump_effect: Jump effect function: y_new = g(t, y)
        t_span: (t_start, t_end)
        y0: Initial conditions (JAX array)
        dt0: Initial time step for adaptive solver
        solver: Diffrax solver (default: Dopri5 adaptive RK45)

    Returns:
        (times, states) both as JAX arrays

    Example:
        >>> def ode_func(t, y):
        ...     # dy/dt = -0.1 * y
        ...     return -0.1 * y
        >>>
        >>> def jump_effect(t, y):
        ...     # Add random noise at jumps
        ...     return y + 0.1
        >>>
        >>> config = JumpProcessConfig(rate="10 / day", seed=42)
        >>> adapter = JumpProcessAdapter(config)
        >>> times, states = simulate_jump_diffusion_diffrax(
        ...     adapter, ode_func, jump_effect,
        ...     t_span=(0.0, 10.0),
        ...     y0=jnp.array([1.0, 0.0])
        ... )
    """
    if solver is None:
        solver = diffrax.Dopri5()  # Adaptive RK45

    # Generate all jump times
    jump_times = jump_adapter.generate_jumps(t_span[0], t_span[1])
    n_jumps = len(jump_times)

    # Create ODE term
    term = diffrax.ODETerm(lambda t, y, args: ode_func(t, y))

    # Storage for results
    all_times = []
    all_states = []

    # Current state
    y = y0

    # Integrate between each jump
    for i in range(n_jumps + 1):
        t_start = t_span[0] if i == 0 else float(jump_times[i-1])
        t_end = t_span[1] if i == n_jumps else float(jump_times[i])

        if t_end > t_start:
            # Continuous integration using diffrax
            sol = diffrax.diffeqsolve(
                terms=term,
                solver=solver,
                t0=t_start,
                t1=t_end,
                dt0=dt0,
                y0=y,
                saveat=diffrax.SaveAt(ts=jnp.linspace(t_start, t_end, 50)),
            )

            # Store results
            all_times.append(sol.ts)
            all_states.append(sol.ys)

            # Update state to end of interval
            y = sol.ys[-1]

        # Apply jump effect if this is a jump time
        if i < n_jumps:
            y = jump_effect(jump_times[i], y)
            # Record post-jump state
            all_times.append(jnp.array([float(jump_times[i])]))
            all_states.append(jnp.expand_dims(y, axis=0))

    # Concatenate all results
    times = jnp.concatenate(all_times)
    states = jnp.concatenate(all_states, axis=0)

    return times, states


# JIT-compilable version (for performance)
@jax.jit
def simulate_jump_diffusion_diffrax_jit(
    runtime: JumpProcessRuntime,
    ode_func: Callable,
    jump_effect: Callable,
    t_span: Tuple[float, float],
    y0: jax.Array,
    seed: int,
):
    """JIT-compiled version for production use."""
    # Implementation similar to above
    # Can be used in gradient-based optimization!
    pass
```

#### Fallback: Using jax.lax.scan (Simple ODEs)

For simple, non-stiff ODEs where adaptive stepping is not needed:

```python
def simulate_jump_diffusion_scan(
    jump_adapter: JumpProcessAdapter,
    ode_func: Callable[[jax.Array, jax.Array], jax.Array],
    jump_effect: Callable[[jax.Array, jax.Array], jax.Array],
    t_span: Tuple[float, float],
    y0: jax.Array,
    dt: float = 0.01,
) -> Tuple[jax.Array, jax.Array]:
    """
    Simulate ODE with jumps using simple Euler integration via jax.lax.scan.

    Faster compilation than diffrax, but less accurate. Use for simple ODEs only.

    Args:
        jump_adapter: JumpProcessAdapter
        ode_func: dy/dt = f(t, y)
        jump_effect: y_new = g(t, y)
        t_span: (t_start, t_end)
        y0: Initial conditions
        dt: Fixed time step (Euler method)

    Returns:
        (times, states) as JAX arrays
    """
    # Generate jump times
    jump_times = jump_adapter.generate_jumps(t_span[0], t_span[1])

    def euler_step(y, t):
        """Single Euler step."""
        dydt = ode_func(t, y)
        return y + dt * dydt

    def integrate_between_jumps(y_initial, t_start, t_end):
        """Integrate ODE from t_start to t_end using Euler."""
        n_steps = int((t_end - t_start) / dt)

        def scan_fn(y, i):
            t = t_start + i * dt
            y_new = euler_step(y, t)
            return y_new, (t, y_new)

        final_y, (ts, ys) = jax.lax.scan(scan_fn, y_initial, jnp.arange(n_steps))
        return final_y, ts, ys

    # Process each interval
    y = y0
    all_times = []
    all_states = []

    for i in range(len(jump_times) + 1):
        t_start = t_span[0] if i == 0 else float(jump_times[i-1])
        t_end = t_span[1] if i == len(jump_times) else float(jump_times[i])

        # Integrate continuous dynamics
        y_end, ts, ys = integrate_between_jumps(y, t_start, t_end)
        all_times.append(ts)
        all_states.append(ys)

        # Apply jump
        if i < len(jump_times):
            y = jump_effect(jump_times[i], y_end)
        else:
            y = y_end

    # Concatenate
    times = jnp.concatenate(all_times)
    states = jnp.concatenate(all_states, axis=0)

    return times, states
```

### 4.6 Integration with Simulation Class

Extend the `Simulation` class to support jump processes:

**File**: `ethode/simulation.py` (update)

```python
class Simulation:
    """High-level simulation facade supporting multiple subsystems."""

    def __init__(
        self,
        *,
        controller: Optional[ControllerAdapter] = None,
        fee: Optional[FeeAdapter] = None,
        liquidity: Optional[LiquidityAdapter] = None,
        hawkes: Optional[HawkesAdapter] = None,
        jumpprocess: Optional[JumpProcessAdapter] = None,  # NEW
    ):
        """Initialize simulation with subsystems.

        Args:
            controller: ControllerAdapter instance (optional)
            fee: FeeAdapter instance (optional)
            liquidity: LiquidityAdapter instance (optional)
            hawkes: HawkesAdapter instance (optional)
            jumpprocess: JumpProcessAdapter instance (optional)
        """
        self.controller = controller
        self.fee = fee
        self.liquidity = liquidity
        self.hawkes = hawkes
        self.jumpprocess = jumpprocess

        # Validate subsystems
        if controller is not None and not isinstance(controller, ControllerAdapter):
            raise TypeError(f"controller must be a ControllerAdapter")
        if fee is not None and not isinstance(fee, FeeAdapter):
            raise TypeError(f"fee must be a FeeAdapter")
        if liquidity is not None and not isinstance(liquidity, LiquidityAdapter):
            raise TypeError(f"liquidity must be a LiquidityAdapter")
        if hawkes is not None and not isinstance(hawkes, HawkesAdapter):
            raise TypeError(f"hawkes must be a HawkesAdapter")
        if jumpprocess is not None and not isinstance(jumpprocess, JumpProcessAdapter):
            raise TypeError(f"jumpprocess must be a JumpProcessAdapter")

    def step(self, inputs: dict, dt: float) -> dict:
        """Execute one simulation step (stateful).

        Args:
            inputs: Dictionary with subsystem inputs:
                - 'error': float - For controller
                - 'time': float - Current time (for jumpprocess)
                - 'market_volatility': float - For fee
                - 'volume_ratio': float - For fee
            dt: Time step size

        Returns:
            Dictionary with subsystem outputs:
                - 'control': float - Controller output
                - 'fee': float - Fee amount
                - 'liquidity': float - Liquidity level
                - 'event_occurred': bool - Hawkes event flag
                - 'jump_occurred': bool - JumpProcess event flag
        """
        outputs = {}

        # 1. Controller subsystem
        if self.controller is not None:
            if 'error' not in inputs:
                raise ValueError("'error' required when controller is active")
            control = self.controller.step(inputs['error'], dt)
            outputs['control'] = control

        # 2. Fee subsystem
        if self.fee is not None:
            if 'market_volatility' in inputs:
                self.fee.update_stress(
                    volatility=inputs['market_volatility'],
                    volume_ratio=inputs.get('volume_ratio', 1.0)
                )

            transaction_amount = inputs.get('transaction_amount', 0.0)
            if transaction_amount == 0.0 and 'control' in outputs:
                transaction_amount = abs(outputs['control'])

            fee = self.fee.step(transaction_amount, dt)
            outputs['fee'] = fee

        # 3. Liquidity subsystem
        if self.liquidity is not None:
            liquidity = self.liquidity.step(dt)
            outputs['liquidity'] = liquidity

        # 4. Hawkes subsystem
        if self.hawkes is not None:
            event_occurred = self.hawkes.step(dt)
            outputs['event_occurred'] = event_occurred

        # 5. JumpProcess subsystem (NEW)
        if self.jumpprocess is not None:
            current_time = inputs.get('time', 0.0)
            jump_occurred = self.jumpprocess.step(current_time, dt)
            outputs['jump_occurred'] = jump_occurred

        return outputs

    def reset(self):
        """Reset all subsystem states."""
        if self.controller is not None:
            self.controller.reset()
        if self.fee is not None:
            self.fee.reset()
        if self.liquidity is not None:
            self.liquidity.reset()
        if self.hawkes is not None:
            self.hawkes.reset()
        if self.jumpprocess is not None:  # NEW
            self.jumpprocess.reset()

    def get_state(self) -> dict:
        """Get current state of all subsystems."""
        state = {}

        if self.controller is not None:
            state['controller'] = self.controller.get_state()
        if self.fee is not None:
            state['fee'] = self.fee.get_state()
        if self.liquidity is not None:
            state['liquidity'] = self.liquidity.get_state()
        if self.hawkes is not None:
            state['hawkes'] = self.hawkes.get_state()
        if self.jumpprocess is not None:  # NEW
            state['jumpprocess'] = self.jumpprocess.get_state()

        return state
```

---

## 5. Testing Requirements

The implementation should include tests for:

### 5.1 Poisson Process Statistics

```python
def test_poisson_rate():
    """Verify Poisson process has correct average rate."""
    config = JumpProcessConfig(process_type='poisson', rate="100 / year", seed=42)
    adapter = JumpProcessAdapter(config)

    jumps = adapter.generate_jumps(0.0, 10.0)  # 10 years
    expected_events = 100 * 10  # rate * duration

    # Should be close to expected (allow 3 sigma)
    assert 900 < len(jumps) < 1100  # ~95% confidence
```

### 5.2 Deterministic Process

```python
def test_deterministic_spacing():
    """Verify deterministic process has uniform spacing."""
    config = JumpProcessConfig(process_type='deterministic', rate="10 / day")
    adapter = JumpProcessAdapter(config)

    jumps = adapter.generate_jumps(0.0, 1.0)  # 1 day
    intervals = np.diff(jumps)

    assert np.allclose(intervals, 1/10)  # 0.1 day spacing
```

### 5.3 JAX Compatibility

```python
def test_jax_jit():
    """Verify kernel functions are JIT-compilable."""
    config = JumpProcessConfig(process_type='poisson', rate="100 / year")
    runtime = config.to_runtime()
    state = JumpProcessState.zero(42, start_time=0.0)

    from ethode.jumpprocess.kernel import step
    jitted_step = jax.jit(step)
    new_state, occurred = jitted_step(runtime, state, jnp.array(0.0), jnp.array(0.01))

    assert new_state is not None
```

### 5.4 Unit Validation

```python
def test_unit_validation():
    """Verify rate must have dimension 1/time."""
    with pytest.raises(ValueError, match="frequency dimension"):
        JumpProcessConfig(rate="100 USD")  # Wrong dimension

    # These should work
    JumpProcessConfig(rate="100 / year")
    JumpProcessConfig(rate="10 / day")
    JumpProcessConfig(rate=100)  # Dimensionless -> assume 1/year
```

### 5.5 Jump Generation Correctness

```python
def test_jumps_in_interval():
    """Verify all jumps are within requested interval."""
    config = JumpProcessConfig(process_type='poisson', rate="100 / day", seed=42)
    adapter = JumpProcessAdapter(config)

    t_start, t_end = 5.0, 10.0
    jumps = adapter.generate_jumps(t_start, t_end)

    # All jumps should be in [t_start, t_end)
    assert all(t_start <= t < t_end for t in jumps)
    assert len(jumps) > 0  # Should generate some jumps
```

### 5.6 State Continuity

```python
def test_step_vs_batch_generation():
    """Verify sequential step() matches batch generate_jumps()."""
    config = JumpProcessConfig(process_type='deterministic', rate="10 / day", seed=42)

    # Batch generation
    adapter1 = JumpProcessAdapter(config)
    jumps_batch = adapter1.generate_jumps(0.0, 1.0)

    # Sequential stepping
    adapter2 = JumpProcessAdapter(config)
    jumps_seq = []
    for t in np.arange(0.0, 1.0, 0.01):
        if adapter2.step(t, 0.01):
            jumps_seq.append(adapter2.state.last_jump_time)

    # Should produce same jump times (deterministic)
    assert len(jumps_batch) == len(jumps_seq)
    assert np.allclose(jumps_batch, jumps_seq, atol=0.01)
```

### 5.7 Reset Behavior

```python
def test_reset_with_start_time():
    """Verify reset() can start from non-zero time."""
    config = JumpProcessConfig(process_type='poisson', rate="100 / day", seed=42)
    adapter = JumpProcessAdapter(config)

    # Reset to t=5.0
    adapter.reset(start_time=5.0)

    # First jump should be > 5.0
    state = adapter.get_state()
    assert state['next_jump_time'] > 5.0
```

### 5.8 Integration with Other Subsystems

```python
def test_jumpprocess_in_simulation():
    """Verify JumpProcess works in Simulation with other subsystems."""
    from ethode import Simulation, ControllerAdapter, ControllerConfig

    controller_config = ControllerConfig(
        kp="0.2 / day", ki="0.02 / day**2", kd=0.0,
        tau="7 day", noise_band=("0.001 USD", "0.003 USD")
    )
    jump_config = JumpProcessConfig(
        process_type='deterministic',
        rate="1 / day"
    )

    sim = Simulation(
        controller=ControllerAdapter(controller_config),
        jumpprocess=JumpProcessAdapter(jump_config)
    )

    outputs = sim.step({'error': 1.0, 'time': 0.5}, dt=0.1)

    assert 'control' in outputs
    assert 'jump_occurred' in outputs
    assert isinstance(outputs['jump_occurred'], bool)
```

### 5.9 ODE Integration Tests

```python
def test_jump_diffusion_with_diffrax():
    """Test ODE+jump integration using diffrax."""
    import diffrax

    # Simple exponential decay
    def ode_func(t, y):
        return -0.1 * y

    # Jump adds noise
    def jump_effect(t, y):
        return y + 0.1

    config = JumpProcessConfig(rate="5 / day", seed=42)
    adapter = JumpProcessAdapter(config)

    times, states = simulate_jump_diffusion_diffrax(
        adapter,
        ode_func,
        jump_effect,
        t_span=(0.0, 10.0),
        y0=jnp.array([1.0])
    )

    # Should have trajectory
    assert len(times) > 0
    assert len(states) > 0
    # States should be affected by both decay and jumps
    assert not np.allclose(states, states[0])
```

---

## 6. Priority & Dependencies

**Priority**: **P0 - Critical blocker** for rd-sim migration

**Dependencies**:
- ✅ Existing ethode infrastructure (units, validation, adapters)
- ✅ JAX integration patterns
- ✅ Pydantic config layer
- ✅ QuantityNode runtime structure
- ⚠️ **NEW**: diffrax (optional dependency, ~1MB)

**Estimated effort**: 2-3 days
- Day 1: Config + Runtime structures + Kernel functions
- Day 2: Adapter + Simulation integration
- Day 3: Testing + diffrax integration + documentation

---

## 7. Questions & Decisions

### 7.1 Multiple Jump Processes

**Decision**: Start simple, extend later
- Phase 1: Single JumpProcessAdapter (current spec)
- Phase 2 (if needed): Create helper to merge multiple jump processes
- Users can manually combine multiple adapters for now

### 7.2 Time-Varying Rates

**Decision**: Not in v1, add callback pattern later
- Current constant-rate is sufficient for rd-sim migration
- Future: Could add `rate_fn: Callable[[float], float]` parameter
- Maintains backward compatibility

### 7.3 Marked Point Processes

**Decision**: Not in v1
- Jump magnitudes handled in user's `jump_effect` function
- Keeps JumpProcess focused on timing
- Document pattern in examples

### 7.4 Diffrax Dependency

**Decision**: Optional dependency
- Primary integration uses diffrax (better accuracy, adaptive stepping)
- Fallback to jax.lax.scan for simple ODEs (no dependency)
- Document both patterns

---

## 8. Migration Impact

Once this API is available, rd-sim can migrate:

- ✅ `rd_interest_model.py` - Interest rate drip events
- ✅ `rd_par_discrete_model.py` - AMM swaps, redemptions, borrowing
- ✅ All test files using jump processes
- ✅ Scripts using stochastic simulations

**Unblocks**: ~8 files in rd-sim, ~50% of migration effort

**Key Improvements Over Legacy**:
- Full JAX integration (JIT, grad, vmap)
- Better unit handling
- Differentiable simulations
- Consistent with ethode architecture
- Better performance

---

## 9. Documentation & Examples

After implementation, add to `docs/adapter_examples.md`:

### Example 38: Basic Poisson Jump Process

```python
from ethode import JumpProcessAdapter, JumpProcessConfig

config = JumpProcessConfig(
    process_type='poisson',
    rate="100 / day",
    seed=42
)
adapter = JumpProcessAdapter(config)

# Sequential simulation
for t in range(100):
    current_time = t * 0.1
    jump_occurred = adapter.step(current_time, dt=0.1)
    if jump_occurred:
        print(f"Event at t={current_time:.2f}")

# Batch generation
jump_times = adapter.generate_jumps(0.0, 10.0)
print(f"Generated {len(jump_times)} events")
```

### Example 39: Deterministic Periodic Events

```python
config = JumpProcessConfig(
    process_type='deterministic',
    rate="4 / hour"  # Every 15 minutes
)
adapter = JumpProcessAdapter(config)

jumps = adapter.generate_jumps(0.0, 2.0)  # 2 hours
intervals = np.diff(jumps)

print(f"Events every {intervals[0]:.4f} hours")
assert np.allclose(intervals, 0.25)  # 15 min = 0.25 hours
```

### Example 40: Jump-Diffusion Simulation with Diffrax

```python
import diffrax
from ethode import JumpProcessAdapter, JumpProcessConfig

# Exponential decay ODE
def ode_func(t, y):
    return -0.5 * y  # dy/dt = -0.5 * y

# Jump effect: add 10% noise
def jump_effect(t, y):
    return y * 1.1

# Configure jump process
config = JumpProcessConfig(
    process_type='poisson',
    rate="10 / day",
    seed=42
)
adapter = JumpProcessAdapter(config)

# Simulate
times, states = simulate_jump_diffusion_diffrax(
    adapter,
    ode_func,
    jump_effect,
    t_span=(0.0, 10.0),
    y0=jnp.array([100.0])
)

# Plot
import matplotlib.pyplot as plt
plt.plot(times, states)
plt.xlabel('Time (days)')
plt.ylabel('State')
plt.title('Jump-Diffusion Simulation')
plt.show()
```

### Example 41: Multi-Subsystem with JumpProcess

```python
from ethode import (
    Simulation,
    ControllerAdapter, ControllerConfig,
    JumpProcessAdapter, JumpProcessConfig,
)

# Configure subsystems
controller_config = ControllerConfig(
    kp="0.3 / day",
    ki="0.03 / day**2",
    kd=0.0,
    tau="7 day",
    noise_band=("0.001 USD", "0.005 USD")
)

jump_config = JumpProcessConfig(
    process_type='poisson',
    rate="20 / day",  # ~1 event per hour
    seed=42
)

# Create simulation
sim = Simulation(
    controller=ControllerAdapter(controller_config),
    jumpprocess=JumpProcessAdapter(jump_config)
)

# Run simulation tracking jumps
error = 1.0
jump_count = 0

for t in range(1000):
    current_time = t * 0.01
    outputs = sim.step({'error': error, 'time': current_time}, dt=0.01)

    if outputs.get('jump_occurred', False):
        jump_count += 1
        # Apply jump effect to error (example)
        error = max(0.0, error - 0.1)

    # Update error based on control
    error -= outputs['control'] * 0.01

print(f"Total jumps: {jump_count}")
print(f"Final error: {error:.4f}")
```

---

## 10. Contact

Questions or clarifications needed? Contact rd-sim team or create issue in rd-sim repo.

**Timeline requested**: Ideally within 1 week to unblock rd-sim migration.
