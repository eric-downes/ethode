# Adapter Pattern for Ethode Subsystems

## Overview

This document describes the **generic adapter pattern** used in ethode to bridge user-friendly configuration with JAX-optimized runtime structures. This pattern has been successfully implemented for the controller subsystem and is designed to be reusable for all subsystems (fee, liquidity, hawkes, etc.).

## Pattern Philosophy

The adapter pattern solves a fundamental tension:
- **Users want**: Friendly APIs with units, validation, stateful objects, and familiar patterns
- **JAX wants**: Pure functions, immutable structures, and arrays without metadata

The solution: **Two representations connected by an adapter**

```
HighLevel Config (Pydantic)  →  Adapter (Stateful)  →  Low-Level Runtime (JAX/Penzai)
     ↓                              ↓                           ↓
  validation                   state mgmt                  pure functions
  unit handling                convenience                 transformations
  user-friendly                reset/step                  jit/vmap/scan
```

## Architecture: Five Components

Every subsystem following this pattern needs five components:

### 1. Config (Pydantic Model)
**Purpose**: User-facing configuration with unit validation

**Location**: `ethode/{subsystem}/config.py`

**Responsibilities**:
- Accept user-friendly inputs (strings with units, floats, etc.)
- Validate units and dimensional consistency
- Convert to canonical units
- Provide `.to_runtime()` method

**Example**: `ControllerConfig`, `FeeConfig`, `LiquidityConfig`

### 2. Runtime (Penzai Struct)
**Purpose**: JAX-compatible parameter structure

**Location**: `ethode/{subsystem}/runtime.py`

**Responsibilities**:
- Store canonical parameters as `QuantityNode` (value + unit metadata)
- Be a valid JAX pytree (immutable, frozen)
- Support JAX transformations (jit, vmap, scan)

**Example**: `ControllerRuntime`, `FeeRuntime`, `LiquidityRuntime`

### 3. State (Penzai Struct)
**Purpose**: JAX-compatible state structure

**Location**: `ethode/{subsystem}/runtime.py`

**Responsibilities**:
- Store evolving state as JAX arrays
- Be a valid JAX pytree (immutable, frozen)
- Provide `.zero()` class method for initialization
- Support JAX transformations

**Example**: `ControllerState`, `FeeState`, `LiquidityState`

### 4. Kernel (Pure Function)
**Purpose**: Core computation as pure JAX function

**Location**: `ethode/{subsystem}/kernel.py`

**Responsibilities**:
- Pure function: `(runtime, state, inputs...) → (new_state, outputs)`
- No side effects or mutations
- JAX-compatible (jit, vmap, grad)
- Optional diagnostics version

**Example**: `controller_step()`, `calculate_fee()`, `update_liquidity()`

### 5. Adapter (Stateful Wrapper)
**Purpose**: High-level convenience API

**Location**: `ethode/adapters.py`

**Responsibilities**:
- Wrap kernel in stateful interface
- Manage internal state automatically
- Provide multiple step variants
- Validate units on construction
- Expose runtime/state for JAX power users

**Example**: `ControllerAdapter`, (future: `FeeAdapter`, `LiquidityAdapter`)

---

## Implementation Template

Here's a step-by-step template for creating a new subsystem adapter:

### Step 1: Define Config (Pydantic Model)

```python
# ethode/{subsystem}/config.py

from pydantic import BaseModel, Field, field_validator
from typing import Tuple, Optional
from ..units import UnitManager
from ..runtime import UnitSpec
from .runtime import {Subsystem}Runtime

class {Subsystem}Config(BaseModel):
    """Configuration for {subsystem}.

    All parameters support unit-aware inputs.
    """

    # Parameters (stored as Tuple[float, UnitSpec])
    param1: Tuple[float, UnitSpec] = Field(
        description="Description with expected units"
    )

    param2: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Optional parameter"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Validators
    @field_validator('param1', mode='before')
    @classmethod
    def validate_param1(cls, v):
        """Validate and convert param1 to canonical units."""
        from ..fields import quantity_field
        return quantity_field(
            v,
            expected_dimension="dimension_name",  # e.g., "time", "1/time"
            field_name="param1"
        )

    def to_runtime(self, *, check_units: bool = False) -> {Subsystem}Runtime:
        """Convert config to JAX-ready runtime structure."""
        from ..runtime import QuantityNode

        runtime = {Subsystem}Runtime(
            param1=QuantityNode.from_float(self.param1[0], self.param1[1]),
            param2=QuantityNode.from_float(self.param2[0], self.param2[1]) if self.param2 else None,
        )

        # Optional: unit validation
        if check_units:
            from ..validation import validate_{subsystem}_dimensions
            validate_{subsystem}_dimensions(runtime)

        return runtime
```

### Step 2: Define Runtime & State (Penzai Structs)

```python
# ethode/{subsystem}/runtime.py

from penzai import struct
import jax.numpy as jnp
from ..runtime import QuantityNode

@struct.pytree_dataclass
class {Subsystem}Runtime(struct.Struct):
    """Runtime parameters for {subsystem} (JAX-compatible).

    All fields are QuantityNodes containing JAX arrays with unit metadata.
    """

    param1: QuantityNode
    param2: Optional[QuantityNode] = None

    # Helper methods for accessing values
    def get_param1(self) -> float:
        """Get param1 value as Python float."""
        return float(self.param1.value)


@struct.pytree_dataclass
class {Subsystem}State(struct.Struct):
    """Runtime state for {subsystem} (JAX-compatible).

    Tracks evolving state during simulation.
    """

    state_var1: jax.Array  # Description
    state_var2: jax.Array  # Description
    time: jax.Array        # Simulation time

    @classmethod
    def zero(cls, dtype=jnp.float32) -> '{Subsystem}State':
        """Create zero-initialized state."""
        return cls(
            state_var1=jnp.array(0.0, dtype=dtype),
            state_var2=jnp.array(0.0, dtype=dtype),
            time=jnp.array(0.0, dtype=dtype)
        )
```

### Step 3: Define Kernel (Pure Function)

```python
# ethode/{subsystem}/kernel.py

import jax
import jax.numpy as jnp
from typing import Tuple
from .runtime import {Subsystem}Runtime, {Subsystem}State

def {subsystem}_step(
    runtime: {Subsystem}Runtime,
    state: {Subsystem}State,
    input1: jax.Array,
    dt: jax.Array
) -> Tuple[{Subsystem}State, jax.Array]:
    """Pure functional step for {subsystem}.

    Args:
        runtime: Immutable parameters
        state: Current state
        input1: Input signal
        dt: Time step

    Returns:
        Tuple of (new_state, output)
    """
    # Extract parameters
    param1 = runtime.param1.value

    # Compute new state (functional, no mutation!)
    new_state_var1 = state.state_var1 + input1 * dt
    new_state_var2 = ...  # Your logic here

    # Compute output
    output = param1 * new_state_var1

    # Create new state (immutable)
    new_state = {Subsystem}State(
        state_var1=new_state_var1,
        state_var2=new_state_var2,
        time=state.time + dt
    )

    return new_state, output


# Optional: diagnostics version
def {subsystem}_step_with_diagnostics(
    runtime: {Subsystem}Runtime,
    state: {Subsystem}State,
    input1: jax.Array,
    dt: jax.Array
) -> Tuple[{Subsystem}State, jax.Array, dict]:
    """Step with diagnostic information."""
    new_state, output = {subsystem}_step(runtime, state, input1, dt)

    diagnostics = {
        'state_var1': float(new_state.state_var1),
        'state_var2': float(new_state.state_var2),
        # ... other useful info
    }

    return new_state, output, diagnostics
```

### Step 4: Define Adapter (Stateful Wrapper)

```python
# ethode/adapters.py

from .{subsystem} import {Subsystem}Config
from .{subsystem}.runtime import {Subsystem}Runtime, {Subsystem}State
from .{subsystem}.kernel import {subsystem}_step, {subsystem}_step_with_diagnostics
import jax.numpy as jnp

class {Subsystem}Adapter:
    """High-level adapter for {subsystem}.

    Provides stateful API with automatic unit validation and
    convenient access to JAX runtime structures.

    Example:
        >>> config = {Subsystem}Config(param1="1.0 / second")
        >>> adapter = {Subsystem}Adapter(config)
        >>> output = adapter.step(input1=1.0, dt=0.1)
    """

    def __init__(
        self,
        config: {Subsystem}Config,
        *,
        check_units: bool = True
    ):
        """Initialize adapter.

        Args:
            config: Configuration instance
            check_units: Whether to validate units (default: True)
        """
        self.config = config

        # Build runtime
        self.runtime = config.to_runtime(check_units=False)

        # Validate units if requested
        if check_units:
            from .validation import validate_{subsystem}_dimensions
            try:
                validate_{subsystem}_dimensions(self.runtime)
            except (ValueError, Exception) as e:
                raise ValueError(f"Unit validation failed: {e}")

        # Initialize state
        self.state = {Subsystem}State.zero()

    def step(self, input1: float, dt: float) -> float:
        """Execute one step (stateful).

        Args:
            input1: Input signal
            dt: Time step

        Returns:
            Output value (float)
        """
        # Convert inputs to JAX arrays
        input1_jax = jnp.array(input1, dtype=jnp.float32)
        dt_jax = jnp.array(dt, dtype=jnp.float32)

        # Call pure kernel
        self.state, output = {subsystem}_step(
            self.runtime, self.state, input1_jax, dt_jax
        )

        # Return as Python float
        return float(output)

    def step_with_diagnostics(self, input1: float, dt: float) -> Tuple[float, dict]:
        """Step with diagnostic information."""
        input1_jax = jnp.array(input1, dtype=jnp.float32)
        dt_jax = jnp.array(dt, dtype=jnp.float32)

        self.state, output, diagnostics = {subsystem}_step_with_diagnostics(
            self.runtime, self.state, input1_jax, dt_jax
        )

        return float(output), diagnostics

    def reset(self):
        """Reset state to zero."""
        self.state = {Subsystem}State.zero()

    def get_state(self) -> dict:
        """Get current state as dictionary."""
        return {
            'state_var1': float(self.state.state_var1),
            'state_var2': float(self.state.state_var2),
            'time': float(self.state.time)
        }
```

### Step 5: Update Simulation Facade

```python
# ethode/simulation.py

class Simulation:
    def __init__(
        self,
        *,
        controller: Optional[ControllerAdapter] = None,
        fee: Optional[FeeAdapter] = None,
        liquidity: Optional[LiquidityAdapter] = None,
        {subsystem}: Optional[{Subsystem}Adapter] = None
    ):
        """Initialize simulation with subsystems."""
        self.controller = controller
        self.fee = fee
        self.liquidity = liquidity
        self.{subsystem} = {subsystem}

    def step(self, error: float, dt: float) -> dict:
        """Execute one simulation step, orchestrating all subsystems."""
        results = {}

        # Orchestrate in dependency order
        if self.controller:
            control = self.controller.step(error, dt)
            results['control'] = control

        if self.{subsystem}:
            {subsystem}_output = self.{subsystem}.step(..., dt)
            results['{subsystem}'] = {subsystem}_output

        # ... other subsystems

        return results
```

---

## Design Considerations

### Unit Validation

**Where to validate:**
- Config validators: Validate on construction (Pydantic)
- Adapter init: Optional cross-field validation (`check_units=True`)
- Never in kernel: Kernel assumes valid inputs

**How to validate:**
- Use `quantity_field()` for single parameters
- Use `tuple_quantity_field()` for tuples/ranges
- Custom validators for complex constraints
- Leverage existing `validate_*_dimensions()` functions

### State Management

**Stateful vs Functional:**
- Adapter: Stateful (maintains `self.state`)
- Kernel: Functional (returns new state)
- Simulation: Stateful (orchestrates adapters)

**State Access:**
- Expose `.state` attribute for JAX access
- Expose `.get_state()` for debugging
- Expose `.reset()` for re-initialization

### Error Handling

**Config Validation:**
```python
try:
    config = SubsystemConfig(param="invalid value")
except ValidationError as e:
    # Pydantic validation error
    print(f"Invalid config: {e}")
```

**Unit Validation:**
```python
try:
    adapter = SubsystemAdapter(config, check_units=True)
except ValueError as e:
    # Unit dimension error
    print(f"Unit mismatch: {e}")
```

### JAX Compatibility

**Requirements:**
- Runtime: Must be a valid pytree (frozen dataclass)
- State: Must be a valid pytree (frozen dataclass)
- Kernel: Pure function (no side effects)
- Arrays: Use `jax.numpy` not `numpy`

**Testing JAX Compatibility:**
```python
# Test pytree flattening
leaves, treedef = jax.tree_util.tree_flatten(runtime)
reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

# Test JIT compilation
jitted_step = jax.jit(subsystem_step)
output = jitted_step(runtime, state, input, dt)

# Test vmap
vmapped_step = jax.vmap(subsystem_step, in_axes=(None, None, 0, None))
outputs = vmapped_step(runtime, state, inputs_array, dt)
```

---

## Multi-Subsystem Integration

### Dependency Order

Define clear execution order for subsystems:

```
1. Controller → computes control signal
2. Fee → applies fees based on transaction
3. Liquidity → updates liquidity based on trading
4. Market → processes market dynamics
```

### State Synchronization

**Option 1: Sequential Updates (Recommended)**
```python
def step(self, error, dt):
    # Each subsystem sees updated state from previous
    control = self.controller.step(error, dt)
    fee = self.fee.step(transaction_amount=control, dt=dt)
    liquidity = self.liquidity.step(trade_size=control, dt=dt)
    return {'control': control, 'fee': fee, 'liquidity': liquidity}
```

**Option 2: Parallel with Shared State**
```python
def step(self, market_state, dt):
    # All subsystems read from shared state, update independently
    control = self.controller.step(market_state.error, dt)
    fee = self.fee.step(market_state.volume, dt)
    # Combine results
    return combined_output
```

### Cross-Subsystem Validation

Validate compatibility between subsystems:

```python
def __init__(self, *, controller, fee, liquidity):
    self.controller = controller
    self.fee = fee
    self.liquidity = liquidity

    # Validate dimensional consistency
    if controller and fee:
        controller_output_dim = controller.runtime.kp.units.dimension
        fee_input_dim = fee.runtime.base_fee_rate.units.dimension
        # Check compatibility...
```

---

## Testing Checklist

For each new adapter, ensure:

- [ ] **Config validation** - Invalid inputs raise clear errors
- [ ] **Unit conversion** - Units converted to canonical form
- [ ] **Runtime creation** - `to_runtime()` produces valid pytree
- [ ] **State initialization** - `.zero()` creates valid initial state
- [ ] **Kernel purity** - Function has no side effects
- [ ] **Adapter step** - Stateful step works correctly
- [ ] **Adapter reset** - Reset returns to zero state
- [ ] **JAX JIT** - Kernel compiles with `jax.jit`
- [ ] **JAX vmap** - Kernel vectorizes with `jax.vmap`
- [ ] **JAX scan** - Works with `jax.lax.scan`
- [ ] **Simulation integration** - Works in `Simulation.step()`

---

## Examples in Codebase

### Fully Implemented: Controller

- **Config**: `ethode/controller/config.py` - `ControllerConfig`
- **Runtime**: `ethode/runtime.py` - `ControllerRuntime`, `ControllerState`
- **Kernel**: `ethode/controller/kernel.py` - `controller_step()`
- **Adapter**: `ethode/adapters.py` - `ControllerAdapter`
- **Tests**: `test_adapters.py`, `test_simulation.py`

### Partially Implemented: Fee, Liquidity, Hawkes

- **Configs**: Available in `ethode/{subsystem}/config.py`
- **Runtimes**: Available in `ethode/{subsystem}/runtime.py`
- **Kernels**: Available in `ethode/{subsystem}/kernel.py`
- **Adapters**: Not yet implemented (follow this pattern!)

---

## Summary

The adapter pattern provides:

✅ **User-Friendly Interface** - Pydantic configs with unit validation
✅ **JAX Performance** - Pure functions with runtime structs
✅ **Stateful Convenience** - Adapter wraps complexity
✅ **Composability** - Easy integration in Simulation
✅ **Testability** - Clear separation of concerns
✅ **Extensibility** - Generic pattern for all subsystems

**Next Steps**:
1. Apply this pattern to create `FeeAdapter`
2. Apply this pattern to create `LiquidityAdapter`
3. Apply this pattern to create `HawkesAdapter`
4. Enhance `Simulation` for multi-subsystem orchestration
