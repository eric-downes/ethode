# Hawkes Process Integration for JumpDiffusion

**Version**: 3.2
**Date**: 2025-10-05
**Status**: Requirement for migration
**Priority**: High 

---

## 1. Problem Statement

### 1.1 Current Limitation

`JumpDiffusionConfig` currently only accepts `JumpProcessConfig` for jump timing:

```python
class JumpDiffusionConfig(BaseModel):
    # ...
    jump_process: JumpProcessConfig = Field(
        description="Configuration for jump timing (uses JumpProcessAdapter)"
    )
```

This restricts jump processes to:
- ✅ Poisson (constant rate)
- ✅ Deterministic (periodic)
- ❌ Hawkes (self-exciting or reactive) - **NOT SUPPORTED**

### 1.2 Impact

Smart contract interest rate models need **Hawkes processes** to model:
- **Reward/drip clustering**: High activity periods trigger more disbursement
- **Attack resistance analysis**: Compare Poisson vs self-exciting behavior
- **Realistic timing**: On-chain events often cluster (gas price spikes, MEV opportunities)

**Current workaround**: Using Poisson process for all simulations, losing self-exciting dynamics.

### 1.3 Technical Incompatibility

The current JumpDiffusion kernel assumes **pre-scheduled jump times** via `JumpProcessState`:

```python
# In jumpdiffusion/kernel.py:35-36
next_jump_time = state.jump_state.next_jump_time
t_target = jnp.minimum(next_jump_time, t_end)
```

**JumpProcessState** (what kernel expects):
```python
last_jump_time: jax.Array
next_jump_time: jax.Array    # Pre-scheduled next event
rng_key: jax.Array
event_count: jax.Array
```

**HawkesState** (what Hawkes provides):
```python
current_intensity: jax.Array  # Intensity function λ(t)
time: jax.Array
event_count: jax.Array
last_event_time: jax.Array
cumulative_impact: jax.Array
```

**Why they're incompatible:**
- JumpProcess: Pre-schedules events, integrates until `next_jump_time`
- Hawkes: Uses intensity function, checks at each dt if event occurs with probability λ(t)*dt
- Cannot simply swap `HawkesState` for `JumpProcessState` without kernel redesign

---

## 2. Proposed Solution: Unified Event Buffer with Lazy/Pregen Modes

### 2.1 High-Level Approach

**Keep single `JumpDiffusionConfig`** but use a unified buffer that supports three event generation strategies:

- **Mode 0 (Poisson/Deterministic)**: Lazy generation - fill buffer one event at a time
- **Mode 1 (Pre-generated Hawkes)**: Pre-compute all events upfront using JAX thinning
- **Mode 2 (Online Hawkes)**: Lazy generation with state-dependent intensity and **cumulative excitation accumulator**

**Key insights:**
1. `apply_jump()` has access to current ODE state, past events in buffer, and runtime params
2. **Cumulative excitation** eliminates need for history: Instead of O(n)-summing `Σ exp(-β(t-tᵢ))` over all past events, maintain an accumulator `E(t)` that updates in O(1)
3. User-provided JIT-compatible functions define excitation kernel (exponential, power-law, critical, etc.)

**Benefits:**
- ✅ Single config API (no user-facing breaking changes)
- ✅ Uniform kernel logic (lazy filling for all modes)
- ✅ Full JAX JIT compilation (no Python loops)
- ✅ O(1) memory for Mode 2 (no history storage needed)
- ✅ Extensible: Custom excitation kernels via user functions
- ✅ Type-safe: Use Union types to distinguish lazy vs pregen event sources

### 2.2 Type System: Lazy vs Pre-generated Event Sources

**Use type hints to distinguish event generation strategies:**

```python
from typing import Protocol, Union, Callable, Tuple
import jax
import jax.numpy as jnp

# Type alias for lazy event generators (Mode 0, 2)
LazyEventGenerator = Callable[
    [JumpDiffusionState, JumpDiffusionRuntime, jax.Array],  # (state, runtime, rng_key)
    Tuple[float, jax.Array]  # (next_event_time, new_rng_key)
]

# Type alias for pre-generated schedules (Mode 1)
PregenEventSchedule = jax.Array  # shape (max_events,), pre-computed times padded with inf

# Union type for runtime configuration
EventSource = Union[LazyEventGenerator, PregenEventSchedule]
```

**Mode Selection Guide:**

| Mode | Name | Event Source Type | Use When |
|------|------|------------------|----------|
| 0 | Poisson/Deterministic | `LazyEventGenerator` | Constant rate or periodic timing |
| 1 | Pre-generated Hawkes | `PregenEventSchedule` | Exogenous clustering (gas, MEV) |
| 2 | Online Hawkes | `LazyEventGenerator` | State-dependent base rate λ₀(D,R,α) |

**Mode 0 vs Mode 1 vs Mode 2:**

**Mode 1 (Pre-generated) Preserves:**
- ✅ Temporal clustering (bursts and quiet periods)
- ✅ Self-exciting dynamics (past events trigger future events)
- ✅ Memory effects (excitation decay)
- ✅ All statistical properties of Hawkes process
- ✅ Fixed base rate λ₀ (constant)

**Mode 2 (Online) Adds:**
- ✅ State-dependent base rate: λ₀ = λ₀(D, R, α)
- ✅ Intensity responds to ODE state evolution
- ✅ Feedback loops: system state → event rate → system state

**For Interest distribution:**

- **Use Mode 1** if drip clustering is **exogenous** (gas prices, MEV, social coordination)
  - State-dependent incentives captured in `jump_effect()` (sees current state)
  - Analysis goal: "Compare clustering patterns" not "model causal intensity feedback"

- **Use Mode 2** if drip rate **causally depends** on system state:
  - Example: λ₀ = λ_base + k·(D - R) ("more accrued interest → higher base drip rate")
  - Intensity must respond to ODE state within simulation timestep
  - Requires feedback mechanism as core model feature

### 2.3 Config Layer Changes

**Add Hawkes mode selection and discretization parameters:**

```python
from typing import Union, Optional, Literal
from ..jumpprocess import JumpProcessConfig
from ..hawkes import HawkesConfig

class JumpDiffusionConfig(BaseModel):
    """Configuration for ODE+Jump hybrid simulation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ... existing fields ...

    # Accept either JumpProcessConfig or HawkesConfig
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
                    "Should be << 1/λ_max for accuracy. Default: min(0.1/λ_max, 0.25*τ_decay)"
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

    excitation_decay_fn: Optional[Callable[[jax.Array, jax.Array, Any], jax.Array]] = Field(
        default=None,
        description="Excitation decay function (E, dt, hawkes_runtime) -> E_new. "
                    "Mode 2 only. Default: exponential decay."
    )

    excitation_jump_fn: Optional[Callable[[jax.Array, Any], jax.Array]] = Field(
        default=None,
        description="Excitation jump function (E, hawkes_runtime) -> E_new. "
                    "Mode 2 only. Default: unit jump (E + 1)."
    )

    intensity_fn: Optional[Callable[[jax.Array, jax.Array, Any], jax.Array]] = Field(
        default=None,
        description="Intensity function (lambda_0, E, hawkes_runtime) -> lambda. "
                    "Mode 2 only. Default: linear (lambda_0 + alpha * E)."
    )

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
        """Validate hawkes_dt has time dimension (Mode 1 only)."""
        jump_process = info.data.get('jump_process')
        hawkes_mode = info.data.get('hawkes_mode', 'pregen')

        # Only relevant for Mode 1 (pregen)
        if jump_process is not None and isinstance(jump_process, HawkesConfig):
            if hawkes_mode == 'pregen':
                if v is None:
                    # Auto-compute default: dt << min(1/λ_max, τ_decay)
                    base_rate = jump_process.jump_rate[0]  # Extract value from tuple
                    excitation = jump_process.excitation_strength  # float, not tuple
                    decay = jump_process.excitation_decay[0]

                    # Stability check
                    if excitation >= 1.0:
                        raise ValueError(
                            f"excitation_strength must be < 1.0 for stability, got {excitation}"
                        )

                    # Compute maximum intensity accounting for self-excitation
                    lambda_max = base_rate / (1.0 - excitation)
                    default_dt = min(0.1 / lambda_max, 0.25 * decay)
                    return (default_dt, "second")

                # Validate time dimension if provided
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
            else:
                # Mode 2 (online): hawkes_dt is ignored, warn if provided
                if v is not None:
                    import warnings
                    warnings.warn(
                        "hawkes_dt is ignored for hawkes_mode='online' (Mode 2). "
                        "Event generation uses cumulative excitation, not pre-generation.",
                        UserWarning
                    )
                return None

        return v

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
```

### 2.4 Runtime Layer Changes

**New scheduler runtime struct:**

```python
# In ethode/jumpdiffusion/runtime.py

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
        excitation_decay_fn: Excitation decay (E, dt, params) -> E_new
        excitation_jump_fn: Excitation jump (E, params) -> E_new
        intensity_fn: Intensity computation (lambda_0, E, params) -> lambda
    """
    mode: int  # 0=poisson, 1=pregen_hawkes, 2=online_hawkes
    scheduled: Optional[JumpProcessRuntime] = None
    hawkes: Optional[HawkesRuntime] = None
    hawkes_dt: jax.Array = jnp.array(0.0)
    hawkes_max_events: jax.Array = jnp.array(0, dtype=jnp.int32)
    seed: jax.Array = jnp.array(0, dtype=jnp.uint32)

    # Mode 2 static callables (use pz.static_field to exclude from pytree)
    # IMPORTANT: Storing callables in pytree breaks JAX flattening/jit
    lambda_0_fn: Callable = struct.static_field(default=None)
    excitation_decay_fn: Callable = struct.static_field(default=None)
    excitation_jump_fn: Callable = struct.static_field(default=None)
    intensity_fn: Callable = struct.static_field(default=None)


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
    solver_type: int
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
    t: jax.Array
    state: jax.Array
    jump_buffer: ScheduledJumpBuffer  # ← Changed from jump_state
    step_count: jax.Array
    jump_count: jax.Array

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
```

**Updated config.to_runtime():**

```python
def to_runtime(self, check_units: bool = True) -> 'JumpDiffusionRuntime':
    """Convert config to JAX-ready runtime structure."""
    from ..runtime import QuantityNode
    from .runtime import JumpDiffusionRuntime, JumpSchedulerRuntime

    dt_max_value, dt_max_spec = self.dt_max
    solver_map = {'euler': 0, 'rk4': 1, 'dopri5': 2, 'dopri8': 3}
    solver_type_int = solver_map[self.solver]

    # Build scheduler runtime based on jump_process type
    if isinstance(self.jump_process, HawkesConfig):
        # Hawkes mode
        hawkes_runtime = self.jump_process.to_runtime(check_units=check_units)
        hawkes_dt_value, _ = self.hawkes_dt

        scheduler = JumpSchedulerRuntime(
            mode=1,
            scheduled=None,
            hawkes=hawkes_runtime,
            hawkes_dt=jnp.array(hawkes_dt_value),
            hawkes_max_events=jnp.array(self.hawkes_max_events, dtype=jnp.int32),
            seed=jnp.array(
                self.jump_process.seed if self.jump_process.seed is not None else 0,
                dtype=jnp.uint32
            ),
        )
    else:
        # Scheduled mode (Poisson/deterministic)
        jump_runtime = self.jump_process.to_runtime(check_units=check_units)

        scheduler = JumpSchedulerRuntime(
            mode=0,
            scheduled=jump_runtime,
            hawkes=None,
            hawkes_dt=jnp.array(0.0),
            hawkes_max_events=jnp.array(0, dtype=jnp.int32),
            seed=jnp.array(
                self.jump_process.seed if self.jump_process.seed is not None else 0,
                dtype=jnp.uint32
            ),
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
```

### 2.5 Cumulative Excitation Accumulator (Mode 2)

**Problem:** Online Hawkes needs intensity `λ(t) = λ₀(state) + Σᵢ α·exp(-β·(t-tᵢ))` but storing/summing all past events is O(n).

**Solution:** Maintain a decaying accumulator `E(t)` that updates in O(1):

```
E(t) = Σᵢ: tᵢ<t exp(-β·(t - tᵢ))
```

**Update rules:**
- **Between events:** `E(t + Δt) = E(t) · exp(-β·Δt)` (pure decay)
- **After event at t:** `E(t⁺) = E(t⁻) + 1` (jump by 1)
- **Intensity:** `λ(t) = λ₀(state) + α·E(t)`

This requires **NO event history** - just `E(t)`, `t_last_update`, and current time.

#### 2.5.1 Customizable Excitation Kernels

**Users can customize via `params` dict with JIT-compatible functions:**

```python
# Default: Exponential Hawkes
def exponential_decay(E: jax.Array, dt: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """E(t + dt) = E(t) * exp(-beta * dt)"""
    beta = hawkes.excitation_decay.value
    return E * jnp.exp(-beta * dt)

def exponential_jump(E: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """E(t+) = E(t-) + 1"""
    return E + 1.0

def linear_intensity(lambda_0: jax.Array, E: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """λ(t) = λ₀ + α·E(t)"""
    alpha = hawkes.excitation_strength
    return lambda_0 + alpha * E

# Power-law Hawkes (critical phenomena)
# For this, make cumulative_excitation a pytree to track elapsed time
def powerlaw_decay(E: jax.Array, dt: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """E(t + dt) = E(t) / (1 + (t - t_start))^γ

    Note: For power-law depending on absolute time, make cumulative_excitation a pytree:
        E = {"value": float, "t_start": float}

    This example assumes E is a dict with "value" and "elapsed_time" fields.
    """
    # Extract fields from pytree (cumulative_excitation can be Any pytree, not just scalar)
    if isinstance(E, dict):
        value = E["value"]
        elapsed = E["elapsed_time"] + dt
        gamma = hawkes.excitation_decay.value  # Store gamma here or in custom params
        new_value = value / jnp.power(1.0 + elapsed, gamma)
        return {"value": new_value, "elapsed_time": elapsed}
    else:
        # Fallback for scalar E (simplified power-law that only depends on dt)
        gamma = 0.5
        return E / jnp.power(1.0 + dt, gamma)

def powerlaw_jump(E: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """E(t+) = E(t-) + constant"""
    # Could use hawkes.excitation_strength as jump size
    jump_size = hawkes.excitation_strength
    return E + jump_size

# No decay (pure branching)
def no_decay(E: jax.Array, dt: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """E(t + dt) = E(t)"""
    return E

# State-dependent base rate
def state_dependent_lambda_0(ode_state: jax.Array) -> jax.Array:
    """Compute base rate from ODE state.

    Example: λ₀ = base + k·(D - R) for RAI Dollar
    """
    D, R, alpha, integral, p_RD, last_update = ode_state
    base_rate = 50.0 / 86400  # 50 per day
    state_term = 0.1 * (D - R) / 1e6 / 86400
    return base_rate + state_term

# User passes callables to config (NOT in params dict!)
# They get stored as static fields in JumpSchedulerRuntime
config = JumpDiffusionConfig(
    jump_process=HawkesConfig(...),
    hawkes_mode="online",
    # Callables are passed separately and stored as static fields
    # (Implementation detail: config.to_runtime() stores them in scheduler static fields)
)

# In to_runtime(), these would be set as:
# scheduler.lambda_0_fn = state_dependent_lambda_0  # static field
# scheduler.excitation_decay_fn = exponential_decay  # static field
# scheduler.excitation_jump_fn = exponential_jump  # static field
# scheduler.intensity_fn = linear_intensity  # static field
```

**Function signatures (must be JIT-compatible):**

```python
# Type aliases for Mode 2 callables
# IMPORTANT: These are stored as static fields in JumpSchedulerRuntime (NOT in params dict)

ExcitationDecayFn = Callable[[Any, jax.Array, HawkesRuntime], Any]
# (E, dt, hawkes_runtime) -> E_new
# E can be jax.Array (scalar) or Any pytree (e.g., dict for power-law with elapsed time)

ExcitationJumpFn = Callable[[Any, HawkesRuntime], Any]
# (E, hawkes_runtime) -> E_new
# E can be jax.Array (scalar) or Any pytree

IntensityFn = Callable[[jax.Array, Any, HawkesRuntime], jax.Array]
# (lambda_0, E, hawkes_runtime) -> lambda
# E can be jax.Array (scalar) or Any pytree
# lambda output is always jax.Array (scalar intensity)

StateBasedRateFn = Callable[[jax.Array], jax.Array]
# (ode_state,) -> lambda_0

# All functions MUST:
# - Take only jax.Array or pytree inputs (HawkesRuntime is a pytree)
# - Return jax.Array or pytree outputs (E can be pytree, lambda must be Array)
# - Be pure (no Python side effects, no mutable state)
# - Be compatible with jax.jit, jax.vmap, jax.grad
#
# IMPORTANT: cumulative_excitation (E) can be Any pytree to track additional state:
# - Scalar: E = jnp.array(0.0)  # Simple case (exponential kernel)
# - Dict: E = {"value": 0.0, "elapsed_time": 0.0}  # Power-law with time tracking
# - Custom struct: E = MyExcitationState(value=0.0, history=[...])  # Complex tracking
```

**Storage location:**
- Callables: JumpSchedulerRuntime static fields (using `struct.static_field`)
- Numerical parameters: HawkesRuntime pytree fields (e.g., excitation_decay, excitation_strength)
- User data: params dict (NOT for callables!)

**Default behavior (if not provided):**
- Uses exponential decay: `E·exp(-β·dt)` where β from hawkes_runtime.excitation_decay
- Uses unit jump: `E + 1`
- Uses linear intensity: `λ₀ + α·E` where α from hawkes_runtime.excitation_strength
- Uses constant base rate: `λ₀ = hawkes_runtime.jump_rate.value` (state-independent)

**IMPORTANT: No silent fallbacks policy**

Following the "no silent failures" philosophy, the config validator MUST warn if Mode 2 callables are not provided:

```python
@field_validator("lambda_0_fn", mode="before")
@classmethod
def validate_lambda_0_fn(cls, v, info):
    """Warn if using default lambda_0 for Mode 2."""
    hawkes_mode = info.data.get('hawkes_mode')
    jump_process = info.data.get('jump_process')

    if hawkes_mode == 'online' and isinstance(jump_process, HawkesConfig):
        if v is None:
            import warnings
            warnings.warn(
                "Mode 2 (Online Hawkes): No lambda_0_fn provided. "
                "Using default state-independent base rate (constant λ₀). "
                "To enable state-dependent intensity, provide lambda_0_fn=your_function. "
                "See docs for signature: (ode_state: jax.Array) -> jax.Array",
                UserWarning
            )
    return v

# Similar validators for excitation_decay_fn, excitation_jump_fn, intensity_fn
# Each warns with specific guidance about default behavior
```

This ensures users are explicitly aware when defaults are used, making the behavior transparent.

### 2.6 Hawkes Scheduler Module (Mode 1)

**New file: `ethode/hawkes/scheduler.py`**

```python
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
```

### 2.6 Kernel Layer Changes

**Update key functions in `jumpdiffusion/kernel.py`:**

```python
def integrate_step(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
    t_end: jax.Array,
) -> Tuple[JumpDiffusionState, jax.Array]:
    """Integrate ODE from current time until next jump or t_end."""
    # Get next jump time from buffer (same for all modes)
    idx = state.jump_buffer.next_index
    next_jump_time = state.jump_buffer.event_times[idx]

    t_target = jnp.minimum(next_jump_time, t_end)

    # Integrate ODE (unchanged)
    state_new = _ode_integrate(...)

    return updated_state, t_target


def apply_jump(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
) -> JumpDiffusionState:
    """Apply jump effect and generate next event (mode-dependent).

    This is the ONLY function that differs by mode:
    - Mode 0: Generate next Poisson event
    - Mode 1: Advance buffer pointer (events pre-generated)
    - Mode 2: Generate next Hawkes event with state-dependent λ₀
    """
    # Apply jump effect to state (same for all modes)
    state_after_jump = runtime.jump_effect_fn(
        state.t,
        state.state,
        runtime.params
    )

    # Generate/retrieve next event time based on mode
    mode = runtime.scheduler.mode

    if mode == 0:
        # Mode 0: Poisson - lazy generation
        next_time, new_key = _generate_poisson_event(
            runtime.scheduler.scheduled,
            state.t,
            state.jump_buffer.rng_key
        )
    elif mode == 1:
        # Mode 1: Pre-generated Hawkes - just advance pointer
        next_idx = state.jump_buffer.next_index + 1
        next_time = state.jump_buffer.event_times[next_idx]
        new_key = state.jump_buffer.rng_key  # Unchanged
    else:  # mode == 2
        # Mode 2: Online Hawkes - lazy generation with cumulative excitation

        # Decay excitation from last update to now
        dt_since_last = state.t - state.jump_buffer.last_update_time
        decay_fn = runtime.scheduler.excitation_decay_fn or _default_exponential_decay
        decayed_excitation = decay_fn(
            state.jump_buffer.cumulative_excitation,
            dt_since_last,
            runtime.scheduler.hawkes  # HawkesRuntime pytree (JAX-compatible)
        )

        # Add excitation from this event
        jump_fn = runtime.scheduler.excitation_jump_fn or _default_unit_jump
        new_excitation = jump_fn(decayed_excitation, runtime.scheduler.hawkes)

        # Generate next event using cumulative excitation
        next_time, new_key, final_excitation = _generate_hawkes_event_online(
            runtime.scheduler,  # Pass scheduler (has static callable fields)
            cumulative_excitation=new_excitation,
            current_t=state.t,
            ode_state=state.state,  # ← Access current ODE state!
            rng_key=state.jump_buffer.rng_key,
            max_rejections=int(runtime.scheduler.hawkes_max_events)  # Use as rejection limit
        )
        new_buffer_excitation = final_excitation
        new_buffer_update_time = state.t

    # Update buffer with next event
    next_idx = state.jump_buffer.next_index

    # IMPORTANT: Guard against buffer overflow (no silent failures!)
    # Mode 2 uses fixed 100-element buffer - check capacity before writing
    # See Critical Implementation Requirements section 3 for JAX-compatible approach
    buffer_capacity = state.jump_buffer.count

    if mode == 2:
        # Apply buffer overflow guard: if full, set next_time to inf (stop generating)
        next_time = jax.lax.cond(
            next_idx + 1 < buffer_capacity,
            lambda: next_time,
            lambda: jnp.inf  # Buffer full - no more events
        )

        # Update buffer with cumulative excitation fields
        new_buffer = dataclasses.replace(
            state.jump_buffer,
            event_times=state.jump_buffer.event_times.at[next_idx + 1].set(next_time),
            next_index=next_idx + 1,
            rng_key=new_key,
            cumulative_excitation=new_buffer_excitation,
            last_update_time=new_buffer_update_time
        )
    else:
        # Mode 0, 1: Standard buffer update
        new_buffer = dataclasses.replace(
            state.jump_buffer,
            event_times=state.jump_buffer.event_times.at[next_idx + 1].set(next_time),
            next_index=next_idx + 1,
            rng_key=new_key
        )

    return dataclasses.replace(
        state,
        state=state_after_jump,
        jump_buffer=new_buffer,
        jump_count=state.jump_count + 1,
    )


def _generate_poisson_event(
    runtime: JumpProcessRuntime,
    current_t: jax.Array,
    rng_key: jax.Array
) -> Tuple[float, jax.Array]:
    """Generate next Poisson event time."""
    from ..jumpprocess.kernel import generate_next_jump_time

    # Create temporary state (only for interface compatibility)
    temp_state = JumpProcessState(
        last_jump_time=current_t,
        next_jump_time=current_t,  # Will be overwritten
        rng_key=rng_key,
        event_count=jnp.array(0, dtype=jnp.int32)
    )

    new_state, next_time = generate_next_jump_time(runtime, temp_state, current_t)
    return next_time, new_state.rng_key


def _default_exponential_decay(E: jax.Array, dt: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """Default excitation decay: E(t+dt) = E(t) * exp(-beta * dt).

    Args:
        E: Current cumulative excitation
        dt: Time since last event
        hawkes: HawkesRuntime containing excitation_decay parameter

    Returns:
        Decayed excitation value

    Note:
        This function signature MUST be JAX-compatible (jax.Array inputs/outputs only).
        HawkesRuntime is a pytree, so it can be passed through JAX transformations.
    """
    beta = hawkes.excitation_decay.value
    return E * jnp.exp(-beta * dt)


def _default_unit_jump(E: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """Default excitation jump: E(t+) = E(t-) + 1.

    Args:
        E: Cumulative excitation before event
        hawkes: HawkesRuntime (unused in default, but allows custom implementations)

    Returns:
        Excitation after event
    """
    return E + 1.0


def _default_linear_intensity(lambda_0: jax.Array, E: jax.Array, hawkes: HawkesRuntime) -> jax.Array:
    """Default intensity: lambda(t) = lambda_0 + alpha * E(t).

    Args:
        lambda_0: State-dependent base rate
        E: Current cumulative excitation
        hawkes: HawkesRuntime containing excitation_strength parameter

    Returns:
        Current intensity
    """
    alpha = hawkes.excitation_strength
    return lambda_0 + alpha * E


def _generate_hawkes_event_online(
    scheduler: JumpSchedulerRuntime,
    cumulative_excitation: jax.Array,
    current_t: jax.Array,
    ode_state: jax.Array,
    rng_key: jax.Array,
    max_rejections: int = 1000
) -> Tuple[float, jax.Array, jax.Array]:
    """Generate next Hawkes event using JAX-compatible thinning with bounded loop.

    Uses cumulative excitation accumulator E(t) instead of reconstructing from history.
    Implements Ogata's thinning algorithm with jax.lax.while_loop.

    Args:
        scheduler: JumpSchedulerRuntime with static callable fields
        cumulative_excitation: Current value of E(t) accumulator
        current_t: Current simulation time
        ode_state: Current ODE state vector (for state-dependent λ₀)
        rng_key: JAX PRNG key
        max_rejections: Safety limit on thinning iterations

    Returns:
        Tuple of:
        - next_event_time: Time of next event (jnp.inf if max_rejections hit)
        - new_rng_key: Updated PRNG key
        - updated_excitation: E(t) value at next_event_time

    Note:
        All callable functions (lambda_0_fn, excitation_decay_fn, intensity_fn) are
        retrieved from scheduler static fields. These MUST be JAX-compatible pure
        functions (jax.Array inputs/outputs only, no Python side effects).
    """
    # Get user-provided functions or defaults from static fields
    lambda_0_fn = scheduler.lambda_0_fn or (lambda s: scheduler.hawkes.jump_rate.value)
    decay_fn = scheduler.excitation_decay_fn or _default_exponential_decay
    intensity_fn = scheduler.intensity_fn or _default_linear_intensity

    # Compute state-dependent base rate
    lambda_0 = lambda_0_fn(ode_state)

    def cond_fn(carry):
        """Continue while not accepted and iterations < max."""
        _, _, _, accepted, iter_count = carry
        return jnp.logical_and(
            jnp.logical_not(accepted),
            iter_count < max_rejections
        )

    def body_fn(carry):
        """Single thinning iteration: propose candidate, accept/reject."""
        rng, t_candidate, E_current, _, iter_count = carry

        # Current intensity at candidate time
        lambda_current = intensity_fn(lambda_0, E_current, scheduler.hawkes)

        # Sample inter-event time using current intensity as upper bound
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        dt = jax.random.exponential(subkey1) / lambda_current
        t_next = t_candidate + dt

        # Decay excitation to t_next
        E_next = decay_fn(E_current, dt, scheduler.hawkes)

        # Intensity at proposed time
        lambda_next = intensity_fn(lambda_0, E_next, scheduler.hawkes)

        # Accept/reject using thinning (accept if λ(t_next) / λ_current >= U)
        u = jax.random.uniform(subkey2)
        accepted = u * lambda_current <= lambda_next

        return (rng, t_next, E_next, accepted, iter_count + 1)

    # Run bounded thinning loop
    init = (rng_key, current_t, cumulative_excitation, False, 0)
    final_rng, final_t, final_E, final_accepted, final_iter = jax.lax.while_loop(
        cond_fn, body_fn, init
    )

    # Fail-safe: if max_rejections hit, return inf (no more events in this simulation)
    t_event = jnp.where(final_accepted, final_t, jnp.inf)

    return t_event, final_rng, final_E
```

**Critical: simulate() must initialize buffer based on scheduler mode:**

```python
def simulate(
    runtime: JumpDiffusionRuntime,
    initial_state: jax.Array,
    t_span: Tuple[float, float],
    max_steps: int = 100000,
) -> Tuple[jax.Array, jax.Array]:
    """Run full ODE+Jump simulation with mode-dependent buffer initialization."""
    t_start, t_end = t_span

    # Initialize RNG key
    rng_key = jax.random.PRNGKey(int(runtime.scheduler.seed))

    # Initialize jump buffer based on scheduler mode
    mode = runtime.scheduler.mode

    if mode == 0:
        # Mode 0 (Poisson): Lazy - initialize empty buffer with first event
        key1, key2 = jax.random.split(rng_key)
        first_event, new_key = _generate_poisson_event(
            runtime.scheduler.scheduled,
            jnp.array(t_start),
            key1
        )

        event_times = jnp.full(max_steps, jnp.inf, dtype=initial_state.dtype)
        event_times = event_times.at[0].set(first_event)

        jump_buffer = ScheduledJumpBuffer(
            event_times=event_times,
            count=jnp.array(max_steps, dtype=jnp.int32),  # Capacity
            next_index=jnp.array(0, dtype=jnp.int32),
            rng_key=new_key,
            cumulative_excitation=jnp.array(0.0, dtype=initial_state.dtype),  # Not used for mode 0
            last_update_time=jnp.array(t_start, dtype=initial_state.dtype)
        )

    elif mode == 1:
        # Mode 1 (Pre-gen Hawkes): Pre-generate all events upfront
        from ..hawkes.scheduler import generate_schedule

        events, _ = generate_schedule(
            runtime.scheduler.hawkes,
            t_span,
            runtime.scheduler.hawkes_dt,
            int(runtime.scheduler.hawkes_max_events),
            int(runtime.scheduler.seed),
            dtype=initial_state.dtype  # Match dtype to avoid upcasting
        )

        # Count valid events
        valid_mask = events < t_end
        count = jnp.sum(valid_mask.astype(jnp.int32))

        jump_buffer = ScheduledJumpBuffer(
            event_times=events,
            count=count,
            next_index=jnp.array(0, dtype=jnp.int32),
            rng_key=rng_key,  # Not used for mode 1
            cumulative_excitation=jnp.array(0.0, dtype=initial_state.dtype),  # Not used for mode 1
            last_update_time=jnp.array(t_start, dtype=initial_state.dtype)
        )

    else:  # mode == 2
        # Mode 2 (Online Hawkes): Lazy generation with cumulative excitation accumulator
        key1, key2 = jax.random.split(rng_key)

        # Initialize with zero excitation (no past events)
        initial_excitation = jnp.array(0.0, dtype=initial_state.dtype)

        # Generate first event
        first_event, new_key, first_excitation = _generate_hawkes_event_online(
            runtime.scheduler.hawkes,
            cumulative_excitation=initial_excitation,
            current_t=jnp.array(t_start),
            ode_state=initial_state,
            rng_key=key1,
            params=runtime.params,
            max_rejections=int(runtime.scheduler.hawkes_max_events)
        )

        # Mode 2: Fixed buffer size of 100 (small constant for lazy filling)
        # IMPORTANT: This is usually sufficient (stores 1-2 events at a time)
        # If buffer fills, apply_jump() will raise clear error (no silent failure)
        MODE2_BUFFER_SIZE = 100
        event_times = jnp.full(MODE2_BUFFER_SIZE, jnp.inf, dtype=initial_state.dtype)
        event_times = event_times.at[0].set(first_event)

        jump_buffer = ScheduledJumpBuffer(
            event_times=event_times,
            count=jnp.array(MODE2_BUFFER_SIZE, dtype=jnp.int32),  # Capacity (not max_events!)
            next_index=jnp.array(0, dtype=jnp.int32),
            rng_key=new_key,
            cumulative_excitation=first_excitation,
            last_update_time=jnp.array(t_start, dtype=initial_state.dtype)
        )

    # Initialize simulation state with buffer
    sim_state = JumpDiffusionState.zero(
        initial_state,
        jump_buffer,
        t0=t_start
    )

    # ... rest of simulate() logic (scan loop unchanged)
```

### 2.7 Adapter Layer Changes

**Update `JumpDiffusionAdapter` in `ethode/adapters.py`:**

**IMPORTANT: Buffer initialization is done in simulate(), not __init__()**

The adapter's `simulate()` method handles all buffer initialization (see Section 2.6). The adapter state is minimal - just holds runtime config.

```python
class JumpDiffusionAdapter:
    def __init__(self, config: 'JumpDiffusionConfig', *, check_units: bool = True):
        """Initialize adapter.

        Note: Event buffer initialization is deferred to simulate() to support:
        1. Mode-dependent initialization (lazy vs pre-gen)
        2. Multiple simulations with different t_span values
        3. Clean separation: config/runtime vs simulation state
        """
        from .jumpdiffusion.config import JumpDiffusionConfig

        self.config = config
        self.runtime = config.to_runtime(check_units=check_units)
        # No state stored here - simulate() creates fresh state each time

    def simulate(self, t_span: Tuple[float, float]) -> Tuple[jax.Array, jax.Array]:
        """Run simulation over t_span.

        Buffer initialization strategy by mode:
        - Mode 0: Lazy Poisson - first event generated, rest filled in apply_jump()
        - Mode 1: Pre-gen Hawkes - all events pre-computed using thinning
        - Mode 2: Online Hawkes - first event generated, rest filled in apply_jump()

        Each call creates a fresh buffer, allowing the adapter to be reused.
        """
        from .jumpdiffusion.kernel import simulate

        return simulate(
            self.runtime,
            self.config.initial_state,
            t_span,
            max_steps=100000
        )
```

---

## 3. Usage Examples

### 3.1 Interest Model with Hawkes

```python
from ethode import JumpDiffusionConfig, JumpDiffusionAdapter, HawkesConfig, JumpProcessConfig
import jax.numpy as jnp

def create_rd_interest_simulation(params: RDInterestParams) -> JumpDiffusionAdapter:
    """Create interest simulation."""

    # ... define dynamics() and jump_effect() functions ...

    # Create jump process config
    if params.use_hawkes:
        jump_config = HawkesConfig(
            jump_rate=f"{params.drips_per_day} / day",
            excitation_strength=params.excitation_strength,
            excitation_decay=f"{params.excitation_decay_days} day",
            seed=params.seed,
        )

        # Hawkes discretization: dt << 1/λ_max
        base_rate = params.drips_per_day / 86400  # Convert to per-second
        max_rate = base_rate / (1 - params.excitation_strength)  # Account for self-excitation
        hawkes_dt = 0.1 / max_rate  # 10x safety factor

        config = JumpDiffusionConfig(
            initial_state=jnp.array([
                params.init_D,
                params.init_R,
                params.init_alpha,
                params.init_integral,
                params.init_p,
                0.0,  # last_controller_update
            ]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=jump_config,
            hawkes_dt=f"{hawkes_dt} second",  # Explicit discretization
            hawkes_max_events=10000,  # Safety cap
            solver='dopri5',
            dt_max="0.1 day",
            params=params_dict,
        )
    else:
        # Poisson case (no hawkes_dt needed)
        jump_config = JumpProcessConfig(
            process_type='poisson',
            rate=f"{params.drips_per_day} / day",
            seed=params.seed,
        )

        config = JumpDiffusionConfig(
            initial_state=jnp.array([...]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=jump_config,
            solver='dopri5',
            dt_max="0.1 day",
            params=params_dict,
        )

    # Single constructor for both!
    return JumpDiffusionAdapter(config, check_units=False)
```

### 3.2 Comparing Poisson vs Hawkes

```python
# Define shared dynamics and jump functions
def dynamics(t, state, params):
    D, R, alpha, integral, p, last_update = state
    alpha_per_day = alpha / 365
    return jnp.array([alpha_per_day * D, 0.0, 0.0, 0.0, 0.0, 0.0])

def jump_effect(t, state, params):
    D, R, alpha, integral, p, last_update = state
    return jnp.array([D, D, alpha, integral, p, t])  # R = D (drip)

# Poisson simulation
poisson_config = JumpDiffusionConfig(
    initial_state=jnp.array([1e6, 1e6, 0.005, 0.0, 1.0, 0.0]),
    dynamics_fn=dynamics,
    jump_effect_fn=jump_effect,
    jump_process=JumpProcessConfig(
        process_type='poisson',
        rate="100 / day",
        seed=42,
    ),
    solver='dopri5',
    dt_max="0.1 day",
)

# Hawkes simulation
hawkes_config = JumpDiffusionConfig(
    initial_state=jnp.array([1e6, 1e6, 0.005, 0.0, 1.0, 0.0]),
    dynamics_fn=dynamics,
    jump_effect_fn=jump_effect,
    jump_process=HawkesConfig(
        jump_rate="100 / day",
        excitation_strength=0.3,  # 30% self-excitation
        excitation_decay="1 day",  # 1-day memory
        seed=42,
    ),
    hawkes_dt="100 second",  # ~1/10 of mean inter-event time
    hawkes_max_events=5000,
    solver='dopri5',
    dt_max="0.1 day",
)

# Run simulations (same API!)
adapter_poisson = JumpDiffusionAdapter(poisson_config)
adapter_hawkes = JumpDiffusionAdapter(hawkes_config)

times_p, states_p = adapter_poisson.simulate(t_span=(0.0, 5.0))
times_h, states_h = adapter_hawkes.simulate(t_span=(0.0, 5.0))

# Compare clustering behavior
intervals_p = np.diff(times_p)
intervals_h = np.diff(times_h)

print(f"Poisson: mean={np.mean(intervals_p):.3f}, var={np.var(intervals_p):.3f}")
print(f"Hawkes: mean={np.mean(intervals_h):.3f}, var={np.var(intervals_h):.3f}")
print(f"Variance ratio (Hawkes/Poisson): {np.var(intervals_h)/np.var(intervals_p):.2f}")
# Expected: ratio > 1 due to clustering
```

### 3.3 Mode 2: Online Hawkes with State-Dependent Base Rate

**Use case:** Drip rate causally depends on system state (e.g., more accrued interest → higher drip rate)

```python
from ethode import JumpDiffusionConfig, JumpDiffusionAdapter, HawkesConfig
import jax.numpy as jnp

# Define state-dependent base rate function
def lambda_0_fn(ode_state):
    """Base drip rate depends on accrued interest (D - R)."""
    D, R, alpha, integral, p, last_update = ode_state
    accrued_interest = D - R

    # Example: base rate = 50/day + 0.1*(D-R)/1e6
    # Higher accrued interest → higher base drip rate
    base_rate = 50.0 / 86400  # 50 per day in per-second
    state_term = 0.1 * accrued_interest / 1e6 / 86400

    return base_rate + state_term

# Define dynamics (interest accrues continuously)
def dynamics(t, state, params):
    D, R, alpha, integral, p, last_update = state
    alpha_per_second = alpha / (365 * 86400)
    dD_dt = alpha_per_second * D  # Interest accrues on debt
    return jnp.array([dD_dt, 0.0, 0.0, 0.0, 0.0, 0.0])

# Define jump effect (drip: R increases to match D)
def jump_effect(t, state, params):
    D, R, alpha, integral, p, last_update = state
    # Drip: redemption matches debt
    return jnp.array([D, D, alpha, integral, p, t])

# Create Online Hawkes config (Mode 2)
online_hawkes_config = JumpDiffusionConfig(
    initial_state=jnp.array([1e6, 1e6, 0.005, 0.0, 1.0, 0.0]),
    dynamics_fn=dynamics,
    jump_effect_fn=jump_effect,
    jump_process=HawkesConfig(
        jump_rate="50 / day",  # Base rate (will be modulated by lambda_0_fn)
        excitation_strength=0.3,
        excitation_decay="1 day",
        seed=42,
    ),
    hawkes_mode="online",  # ← Explicit Mode 2 selection
    hawkes_max_events=1000,  # Used as max_rejections limit for thinning
    solver='dopri5',
    dt_max="0.1 day",
    params={
        'lambda_0_fn': lambda_0_fn,  # State-dependent base rate
        # Optional: custom excitation kernel (defaults to exponential if not provided)
        # 'excitation_decay_fn': my_custom_decay,
        # 'excitation_jump_fn': my_custom_jump,
        # 'intensity_fn': my_custom_intensity,
        'excitation_decay': 1.0 / 86400,  # 1 day in seconds (for default exponential)
        'excitation_strength': 0.3,
    },
)

# Run simulation
adapter_online = JumpDiffusionAdapter(online_hawkes_config)
times, states = adapter_online.simulate(t_span=(0.0, 5.0))

# Analyze rate evolution
D_vals = states[:, 0]
R_vals = states[:, 1]
accrued = D_vals - R_vals

print(f"Initial accrued interest: {accrued[0]:.0f}")
print(f"Final accrued interest: {accrued[-1]:.0f}")
print(f"Number of drips: {len(times) - 1}")

# Expected: More drips when accrued interest is higher (feedback loop)
```

**Key difference from Mode 1:**
- **Mode 1 (Pre-gen)**: Base rate λ₀ is constant, clustering is exogenous
- **Mode 2 (Online)**: Base rate λ₀ = λ₀(state) responds to ODE state, creating feedback

**When to use Mode 2:**
- Need causal feedback: system state → event rate → system state
- Base rate must respond to ODE state within simulation timestep
- Example: "Higher accrued interest causally increases drip rate"

**Performance note:** Mode 2 is lazy (like Mode 0), so O(1) memory but if you need to track the whole history (don't: use an accumulator as we recommend) this requires O(n_events) intensity reconstructions from past events.

---

## 4. Performance and Accuracy Considerations

### 4.1 Discretization Requirement

**Hawkes simulations require small dt for accuracy (Mode 1 only):**

Mode 1 (pre-generation) uses thinning with discrete time steps. The validator auto-estimates `dt` if not provided:

```python
# Rule of thumb: dt << min(1/λ_max, τ_decay)

base_rate = 100 / 86400  # 100 per day in per-second
excitation = 0.3
decay = 86400  # 1 day in seconds

# Maximum intensity (worst case: infinite history of events)
lambda_max = base_rate / (1 - excitation)  # ≈ 1.43 × base_rate

# Recommended dt (auto-computed if not provided)
dt_recommended = min(0.1 / lambda_max, 0.25 * decay)
# ≈ min(6000s, 21600s) = 6000s ≈ 1.7 hours

# For higher accuracy, use dt ≈ 0.01 / lambda_max
dt_high_accuracy = 0.01 / lambda_max
# ≈ 600s ≈ 10 minutes
```

**Auto-estimation (Mode 1):**
The config validator automatically computes `hawkes_dt` if not provided, using the formula above with standard exponential kernel parameters. For custom excitation kernels, you must provide `hawkes_dt` manually.

**Mode 2 does not use `hawkes_dt`:**
Online Hawkes (Mode 2) uses continuous-time thinning with `jax.lax.while_loop`, so no discretization parameter is needed. If you provide `hawkes_dt` for Mode 2, a warning is issued and the value is set to `jnp.nan` (ignored).

**Why `jnp.nan` instead of 0.0?**
Using `NaN` is more principled than 0.0 because:
- Makes it explicit that the value is unused (not just zero)
- Causes errors if accidentally used in computations (fail-fast behavior)
- Self-documenting: seeing `NaN` in runtime immediately signals "this field is not applicable"

### 4.2 Performance Impact

**Comparison of step counts and memory:**

| Mode | Process Type | ODE Steps (5 day sim) | Memory | Notes |
|------|--------------|----------------------|--------|-------|
| 0 | Poisson (100/day) | ~50-100 | O(max_steps) buffer | Lazy generation, integrates between jumps |
| 0 | Deterministic (100/day) | 500 | O(max_steps) buffer | One step per jump |
| 1 | Pre-gen Hawkes (dt=600s) | ~720 | O(max_events) | Fixed dt sampling, all events upfront |
| 1 | Pre-gen Hawkes (dt=60s) | ~7200 | O(max_events) | 10x finer sampling |
| 2 | Online Hawkes | ~50-100 | O(100) buffer | Lazy like Mode 0, **O(1) with accumulator** |

**Memory scaling:**

- **Mode 0 (Poisson)**: O(max_steps) buffer - lazy generation, fills buffer one event at a time
- **Mode 1 (Pre-gen Hawkes)**: O(hawkes_max_events) - full buffer pre-allocated
- **Mode 2 (Online Hawkes)**: O(100) - fixed small buffer + cumulative excitation accumulator

**Computational cost:**

- **Mode 1**: Upfront O(n_steps) during pre-generation, then O(1) per jump
- **Mode 2**: O(1) per ODE step, **O(1) per jump using cumulative excitation accumulator**
  - **Important:** If you choose NOT to use the accumulator pattern and instead reconstruct intensity from full event history, hot loops will be O(n_events) per jump (slow). We strongly recommend using the accumulator pattern (default behavior).

**Recommendation**:
- Start with Mode 1 for Hawkes if clustering is exogenous
- Use Mode 2 only if state-dependent base rate is required
- Default dt (auto-computed) is usually sufficient

---

## 5. Implementation Checklist

### 5.1 Code Changes

**File: `ethode/jumpdiffusion/config.py`**
- [ ] Add `Union[JumpProcessConfig, HawkesConfig]` type
- [ ] Add `hawkes_mode: Optional[Literal["pregen", "online"]]` field with validator
- [ ] Add `hawkes_dt` and `hawkes_max_events` optional fields with validators
- [ ] Add callable fields: `lambda_0_fn`, `excitation_decay_fn`, `excitation_jump_fn`, `intensity_fn`
- [ ] **Fix validator**: Use `lambda_max = base_rate / (1 - excitation)` for auto-default (Mode 1 only)
- [ ] **Add stability check**: Raise error if `excitation >= 1.0`
- [ ] **Warn if hawkes_dt provided for Mode 2** (it's ignored, set to NaN)
- [ ] **Warn if callables are None for Mode 2** (using defaults - no silent fallbacks!)
  - Separate validator for each callable with specific guidance on default behavior
- [ ] Update `to_runtime()` to build `JumpSchedulerRuntime` with mode 0/1/2 based on `hawkes_mode`
- [ ] Store callables in scheduler static fields (not in params dict!)

**File: `ethode/jumpdiffusion/runtime.py`**
- [ ] Add `JumpSchedulerRuntime` struct with `mode: int` (0=poisson, 1=pregen_hawkes, 2=online_hawkes)
- [ ] Add `ScheduledJumpBuffer` struct with fields:
  - `rng_key` for lazy modes
  - `cumulative_excitation` for Mode 2
  - `last_update_time` for Mode 2
- [ ] Update `JumpDiffusionRuntime` to use `scheduler` (replacing `jump_runtime`)
- [ ] Update `JumpDiffusionState` to use `jump_buffer` (replacing `jump_state`)

**File: `ethode/hawkes/scheduler.py`** (new, Mode 1 only)
- [ ] Implement `generate_schedule()` function for pre-generation
- [ ] Add docstring explaining dt requirement
- [ ] **Clamp dt with floor** (1e-6 * t_span) to avoid 0/inf cases
- [ ] **Add dtype parameter** and use for event_times to match initial_state.dtype

**File: `ethode/jumpdiffusion/kernel.py`**
- [ ] Update `integrate_step()` to read from `state.jump_buffer.event_times[next_index]` (uniform for all modes)
- [ ] **Rewrite `apply_jump()`** with mode branching:
  - Mode 0: Call `_generate_poisson_event()` (lazy)
  - Mode 1: Advance buffer pointer (pre-generated)
  - Mode 2: Decay excitation, add jump, call `_generate_hawkes_event_online()` with cumulative excitation
  - **Mode 2 buffer overflow guard**: Use `jax.lax.cond` to set `next_time=inf` if buffer full
- [ ] **Add `_generate_poisson_event()`** helper function
- [ ] **Add default excitation kernel functions** (retrieve from scheduler static fields or use these):
  - `_default_exponential_decay(E, dt, hawkes_runtime) -> E_new`
  - `_default_unit_jump(E, hawkes_runtime) -> E_new`
  - `_default_linear_intensity(lambda_0, E, hawkes_runtime) -> lambda`
  - All functions accept pytree `E` (not just scalar) for extensibility
- [ ] **Add `_generate_hawkes_event_online()`** using **jax.lax.while_loop**:
  - Input: cumulative_excitation (not history)
  - Retrieve callables from scheduler static fields (or use defaults)
  - Implement Ogata's thinning with bounded max_rejections
  - Return (next_event_time, new_rng_key, updated_excitation)
  - Fail-safe: return jnp.inf if max_rejections hit
- [ ] **Update `simulate()`** to initialize buffer with cumulative_excitation fields:
  - Mode 0: Initialize with cumulative_excitation=0.0 (unused)
  - Mode 1: Initialize with cumulative_excitation=0.0 (unused)
  - Mode 2: Initialize with cumulative_excitation=0.0, MODE2_BUFFER_SIZE=100, generate first event
  - **Post-simulation check**: Raise error if Mode 2 buffer was exhausted (next_index >= capacity-1)
- [ ] **Update buffer update logic** in apply_jump to handle cumulative_excitation for Mode 2

**File: `ethode/adapters.py`**
- [ ] **Simplify `__init__()`**: Remove state initialization (deferred to simulate())
- [ ] **Document in `simulate()`**: Buffer initialized fresh for each t_span, mode-dependent strategy

**Testing**
- [ ] **Add JIT compilation test**: Verify full simulate() compiles with jax.jit for Mode 2
- [ ] **Test Mode 2 thinning loop**: Verify no Python loops, all JAX primitives
- [ ] **Test cumulative excitation**: Verify E(t) updates correctly between events
- [ ] **Test custom excitation kernels**: Power-law, no-decay, etc.
- [ ] **Test max_rejections fail-safe**: Verify jnp.inf returned when limit hit
- [ ] **Test default callable warnings**: Verify warnings issued when Mode 2 callables not provided
- [ ] **Test buffer overflow protection**:
  - Verify simulation stops gracefully (next_time=inf) when Mode 2 buffer fills
  - Verify clear error message raised post-simulation if buffer exhausted
  - Test pathological case: extremely high intensity → buffer overflow

---

## 6. Timeline Estimate

**Implementation effort**: ~12-15 hours (1.5-2 days + testing/docs)

**Breakdown:**
1. **Config changes** (2 hours):
   - Union type, hawkes_mode field with validator
   - Validators with lambda_max fix (Mode 1 only)
   - to_runtime() mode selection (0/1/2)
2. **Runtime structs** (1.5 hours):
   - JumpSchedulerRuntime with mode field
   - ScheduledJumpBuffer with rng_key, cumulative_excitation, last_update_time
3. **Hawkes scheduler** (2 hours):
   - generate_schedule() with jax.lax.scan (Mode 1 only)
4. **Cumulative excitation pattern** (2 hours):
   - Default excitation kernel functions (_default_exponential_decay, etc.)
   - Documentation of accumulator pattern
5. **Kernel updates** (3.5 hours):
   - `apply_jump()` mode branching with excitation decay/jump for Mode 2 (1.5 hours)
   - `_generate_poisson_event()` helper (30 min)
   - `_generate_hawkes_event_online()` with **jax.lax.while_loop** (1.5 hours)
6. **simulate() buffer init** (1 hour): Mode 0/1/2 initialization with cumulative_excitation fields
7. **Adapter updates** (30 min): Simplify __init__, document simulate()
8. **Testing** (4 hours):
   - Mode 0: Existing tests still pass (30 min)
   - Mode 1: Pre-gen Hawkes clustering validation (1 hour)
   - Mode 2: Online Hawkes with cumulative excitation (1.5 hours)
   - **JAX JIT compilation test**: Verify no Python loops (1 hour)
9. **Documentation** (2 hours): Docstrings, examples, accumulator pattern docs, spec updates

**Risk**: Medium
- Well-defined interfaces, uniform buffer approach
- **JAX compatibility critical**: jax.lax.while_loop for thinning (not Python loops)
- **Cumulative excitation correctness**: E(t) decay/jump logic must be exact
- **Custom kernel extensibility**: User-provided functions must be JIT-compatible

---

## 7. Summary

**What we're adding:**

1. **Config**: `Union[JumpProcessConfig, HawkesConfig]` + explicit `hawkes_mode` field + hawkes_dt/max_events
2. **Runtime**: `JumpSchedulerRuntime` (mode 0/1/2) + `ScheduledJumpBuffer` with rng_key and cumulative excitation fields
3. **Type system**: `LazyEventGenerator` vs `PregenEventSchedule` type aliases
4. **Scheduler**: `hawkes.scheduler.generate_schedule()` for Mode 1 (pre-generation using jax.lax.scan)
5. **Kernel**: Unified buffer approach with JAX-compatible event generation in `apply_jump()`
6. **Cumulative excitation**: O(1) accumulator pattern for Mode 2 (no event history needed)
7. **Custom excitation kernels**: User-provided JIT-compatible functions for power-law, critical phenomena, etc.

**Three Event Generation Modes:**

| Mode | Name | Strategy | Memory | JAX Primitive | Use Case |
|------|------|----------|--------|---------------|----------|
| 0 | Poisson | Lazy | O(max_steps) buffer | - | Constant rate or periodic timing |
| 1 | Pre-gen Hawkes | Pre-compute | O(hawkes_max_events) | jax.lax.scan | Exogenous clustering (gas, MEV) |
| 2 | Online Hawkes | Lazy + accumulator | O(100) buffer | jax.lax.while_loop | Feedback: state → rate → state |

**Memory details:**
- Mode 0: Buffer size = max_steps (from simulate()), fills lazily
- Mode 1: Buffer size = hawkes_max_events, pre-filled upfront
- Mode 2: Buffer size = 100 (fixed small constant), fills lazily, `hawkes_max_events` only bounds rejection loop

**Key Design Decisions:**
- **Unified buffer**: All modes use `ScheduledJumpBuffer`, filled lazily (Mode 0, 2) or upfront (Mode 1)
- **Single extension point**: `apply_jump()` branches on mode for event generation
- **Explicit mode selection**: `hawkes_mode="pregen"` or `"online"` field (not implicit)
- **JAX compatibility**: All event generation uses JAX primitives (no Python loops)
- **Cumulative excitation (Mode 2)**: Maintains `E(t)` accumulator instead of reconstructing from history
  - **Between events**: `E(t+dt) = decay_fn(E(t), dt, params)`
  - **After event**: `E(t+) = jump_fn(E(t-), params)`
  - **Intensity**: `λ(t) = intensity_fn(λ₀(state), E(t), params)`
  - **O(1) per event** (no history sum)
- **Custom kernels**: Users provide JIT-compatible functions via params dict
  - `excitation_decay_fn(E, dt, params) -> E_new`
  - `excitation_jump_fn(E, params) -> E_new`
  - `intensity_fn(lambda_0, E, params) -> lambda`
  - Warn and Default to exponential Hawkes if not provided
- **Bounded thinning**: jax.lax.while_loop with max_rejections limit, fail-safe returns jnp.inf

**Mode 2 Extensibility (Key Insight):**

`apply_jump()` has access to:
- `state.state` (current ODE state)
- `state.jump_buffer.cumulative_excitation` (E(t) accumulator, **not** event history)
- `state.jump_buffer.last_update_time` (for decaying E(t) since last event)
- `runtime.params` (user functions: `lambda_0_fn`, `excitation_decay_fn`, etc.)

This is **sufficient** to generate next event with state-dependent λ₀(D, R, α) and custom excitation kernels in **O(1)** without history storage.

**What stays the same:**

- Single `JumpDiffusionConfig` API
- Single `JumpDiffusionAdapter` constructor
- `integrate_step()` logic (reads from buffer uniformly)
- **Full JAX JIT compilation support** (all modes)

**Trade-offs:**

- ✅ Clean single-config API
- ✅ Uniform kernel logic (lazy buffer filling)
- ✅ **Full JAX JIT compilation** (jax.lax.scan for Mode 1, jax.lax.while_loop for Mode 2)
- ✅ **Callable storage**: Static fields (struct.static_field) prevent pytree flattening issues
- ✅ **O(100) memory for Mode 2** (small fixed buffer + cumulative excitation, no history)
- ✅ Extensible custom excitation kernels (power-law, critical phenomena)
- ✅ Type-safe: Union types distinguish lazy vs pregen
- ⚠️ Mode 1 requires fine dt (more steps during pre-generation)
- ⚠️ Mode 1 memory scales with hawkes_max_events
- ⚠️ Mode 2 has bounded thinning loop (max_rejections limit)

**Unblocks:**
- Full migration of `rd_interest_model.py` with Poisson vs Hawkes comparisons (Mode 1)
- Future work: State-dependent drip rates with causal feedback (Mode 2)

---

**End of Specification v3.2**

---

## Changes from v3.0 → v3.1

**Critical JAX Compatibility Updates:**

1. **Cumulative Excitation Accumulator**: Replaced O(n) history reconstruction with O(1) accumulator pattern
   - Mode 2 no longer sums over past events
   - Maintains `E(t)` with decay/jump update rules
   - Fully generalizable via user-provided JIT-compatible functions

2. **JAX Primitives Only**: All event generation uses JAX control flow
   - Mode 1: `jax.lax.scan` for pre-generation
   - Mode 2: `jax.lax.while_loop` for bounded thinning
   - No Python `while True:` loops

3. **Explicit Mode Selection**: Added `hawkes_mode` field instead of implicit detection
   - `hawkes_mode="pregen"` → Mode 1
   - `hawkes_mode="online"` → Mode 2
   - Clear, not fragile

4. **Custom Excitation Kernels**: Extensible beyond exponential decay
   - Power-law decay for critical phenomena
   - No-decay for pure branching
   - Any user-provided JIT-compatible function

5. **Random Stream Management**: Explicit `jax.random.split` operations throughout

6. **Bounded Thinning**: Added `max_rejections` parameter with `jnp.inf` fail-safe

**Testing Requirements:**
- Must verify `jax.jit(simulate)` compiles successfully for Mode 2
- Must verify no Python loops in execution path
- Must test E(t) accumulator correctness

---

## Changes from v3.1 → v3.2

**No Silent Failures Policy - Implementation Details:**

1. **Pytree-based Generic Tracker (Q1)**
   - `cumulative_excitation` field type changed from `jax.Array` to `Any` (can be pytree)
   - Enables power-law kernels with elapsed time tracking: `E = {"value": 0.0, "elapsed_time": 0.0}`
   - All excitation function signatures updated to accept `Any` for E (not just scalar)
   - Documentation: Custom structs supported for complex tracking needs

2. **Auto-Estimation of hawkes_dt (Q2)**
   - Validator already computes default: `dt = min(0.1/λ_max, 0.25*τ_decay)` for Mode 1
   - Clarified in docs: Works for exponential kernel, manual override needed for custom kernels
   - Mode 2 uses `jnp.nan` (not 0.0) to signal "unused" - fail-fast if accidentally referenced

3. **Performance Impact Wording Fix (Q3)**
   - Corrected: Mode 2 is **O(1) per jump** with accumulator (was incorrectly stated as O(n))
   - Added warning: "If you choose NOT to use accumulator, hot loops will be O(n) (slow)"
   - Made clear that accumulator pattern is default and recommended

4. **Built-in Defaults with Warnings (Follow-up 1)**
   - Added config validators to warn if Mode 2 callables are None
   - Each warning provides specific guidance: "Using default [behavior]. To customize, provide [callable]=your_function"
   - No silent fallbacks - user is explicitly notified of default usage
   - Defaults stored in scheduler static fields during runtime creation

5. **Buffer Overflow Protection (Follow-up 2)**
   - Mode 2: 100-element fixed buffer with overflow guards
   - In-JIT: `jax.lax.cond` to set `next_time=inf` if buffer full (graceful degradation)
   - Post-simulation: Explicit check raises clear error if buffer exhausted
   - Error message includes actionable guidance: "Check intensity, reduce excitation_strength, or use Mode 1"
   - No silent failures - simulation stops and reports issue

**Key Additions:**
- Section 2.5: Pytree support for cumulative_excitation documented
- Section 4.1: Auto-estimation of dt with NaN rationale
- Section 4.2: Corrected O(1) performance claim
- Critical Requirements Section 3: Buffer overflow protection (JAX-compatible)
- Config validators: Warnings for default callable usage
- Testing checklist: Buffer overflow and warning tests

**Philosophy reinforced:**
- ✅ Sensible defaults (convenience)
- ✅ Warn when using defaults (transparency)
- ✅ Fail-fast with clear messages (debuggability)
- ❌ No silent degradation
- ❌ No mysterious failures

---

## Critical Implementation Requirements

**Two key issues resolved in v3.1:**

### 1. **Callable Storage (JAX Pytree Compatibility)**

**Problem:** Storing Python callables in `params` dict (which becomes part of pytree) breaks JAX flattening/jit.

**Solution:**
- Callables stored as **static fields** in `JumpSchedulerRuntime` using `struct.static_field()`
- Config fields: `lambda_0_fn`, `excitation_decay_fn`, `excitation_jump_fn`, `intensity_fn`
- These are excluded from pytree flattening, preserving JAX compatibility

**Requirements:**
- All callables MUST be JAX-compatible pure functions
- Inputs: Only `jax.Array` or pytrees (e.g., `HawkesRuntime`)
- Outputs: Only `jax.Array`
- No Python side effects, no mutable state
- Must work with `jax.jit`, `jax.vmap`, `jax.grad`

### 2. **Memory Usage (Clarified O(1) Claim)**

**Problem:** Claimed "O(1) memory for Mode 2" but allocated `event_times` array of size `hawkes_max_events`.

**Solution:**
- **Mode 0**: Buffer size = `max_steps` (from `simulate()`), fills lazily
- **Mode 1**: Buffer size = `hawkes_max_events`, pre-filled upfront
- **Mode 2**: Buffer size = **100 (fixed small constant)**, fills lazily
  - `hawkes_max_events` ONLY bounds thinning rejection loop (not buffer size)
  - True O(100) = O(1) memory usage

**Implementation:**
```python
# In simulate() for Mode 2:
MODE2_BUFFER_SIZE = 100  # Small fixed constant
event_times = jnp.full(MODE2_BUFFER_SIZE, jnp.inf, dtype=initial_state.dtype)
```

**Why this works:** Mode 2 only ever stores 1-2 events in buffer at a time (current + next). Small fixed buffer is sufficient.

### 3. **Buffer Overflow Protection (Mode 2)**

**Problem:** Mode 2 uses a fixed 100-element buffer. What if a simulation needs more concurrent events?

**Solution (JAX-compatible):**

```python
# In apply_jump(), guard against overflow using jax.lax.cond:

def safe_buffer_update(next_idx, buffer_capacity, next_time):
    """Return next_time if space available, else jnp.inf (stop generating events)."""
    has_space = next_idx + 1 < buffer_capacity
    return jax.lax.cond(
        has_space,
        lambda: next_time,
        lambda: jnp.inf  # Buffer full - stop generating events
    )

# After simulation completes, check if buffer was exhausted:
final_idx = final_state.jump_buffer.next_index
buffer_capacity = final_state.jump_buffer.count

if final_idx >= buffer_capacity - 1:
    raise RuntimeError(
        f"Mode 2 buffer overflow: Simulation exhausted {buffer_capacity}-element buffer. "
        f"This indicates extremely rapid jumps or pathological simulation. "
        f"Possible causes: (1) Intensity too high, (2) Stuck in loop, (3) Bug in kernel. "
        f"Consider: Reducing excitation_strength, checking lambda_0_fn, or using Mode 1."
    )
```

**Why this approach:**
- JAX-compatible: Uses `jax.lax.cond`, not Python exceptions
- Fail-safe: Returns `inf` to gracefully stop event generation (no crashes)
- Clear diagnostics: Post-simulation check provides actionable error message
- No silent failures: User is explicitly notified if buffer limit is hit

**When would this trigger?**
- Extremely high intensity (λ >> expected)
- Bug in custom `lambda_0_fn` or `intensity_fn` (returns negative decay, causing runaway)
- Pathological simulation (stuck in tight loop)

**Normal operation:** Buffer should rarely exceed 5-10 elements (most simulations use 1-2).

### Summary: Implementation Requirements (No Silent Failures)

Both requirements follow the "no silent fallbacks" philosophy:

| Requirement | Implementation | Error Detection | User Guidance |
|-------------|----------------|-----------------|---------------|
| **1. Default callables** | Provide built-in implementations | Warn at config creation | "Using default exponential decay. To customize, provide excitation_decay_fn=your_function" |
| **2. Buffer overflow** | Guard with `jax.lax.cond` (in JIT) + post-check | Stop event generation at inf, raise error after simulation | "Buffer exhausted. Check intensity, reduce excitation_strength, or use Mode 1" |

**Key principles:**
- ✅ Provide sensible defaults (user convenience)
- ✅ Warn when using defaults (transparency)
- ✅ Fail-fast with clear messages (debuggability)
- ❌ No silent degradation
- ❌ No mysterious failures

---

## 8. Mode Selection Logic

**How to determine which mode to use in `to_runtime()`:**

```python
def to_runtime(self, check_units: bool = True) -> 'JumpDiffusionRuntime':
    # ...existing code...

    if isinstance(self.jump_process, HawkesConfig):
        # Hawkes process - mode determined by hawkes_mode field
        hawkes_runtime = self.jump_process.to_runtime(check_units=check_units)

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

    # ... rest of to_runtime() ...
```

**User-facing API:**

```python
# Mode 0 (Poisson)
config = JumpDiffusionConfig(
    jump_process=JumpProcessConfig(process_type='poisson', rate="100/day"),
)

# Mode 1 (Pre-gen Hawkes) - explicit mode selection
config = JumpDiffusionConfig(
    jump_process=HawkesConfig(jump_rate="100/day", excitation_strength=0.3, ...),
    hawkes_mode="pregen",  # ← Explicit Mode 1
    hawkes_dt="100 second",  # Required for Mode 1
    hawkes_max_events=5000,
)

# Mode 2 (Online Hawkes) - explicit mode selection
config = JumpDiffusionConfig(
    jump_process=HawkesConfig(jump_rate="100/day", excitation_strength=0.3, ...),
    hawkes_mode="online",  # ← Explicit Mode 2
    hawkes_max_events=1000,  # Used as max_rejections limit
    params={
        'lambda_0_fn': my_state_dependent_rate_fn,
        'excitation_decay_fn': exponential_decay,  # Optional, defaults provided
        'excitation_jump_fn': unit_jump,  # Optional
        'intensity_fn': linear_intensity,  # Optional
        'excitation_decay': decay_rate,  # Parameters for the functions
        'excitation_strength': alpha,
    },
)
```

---

Questions or feedback? Contact ethode team.
