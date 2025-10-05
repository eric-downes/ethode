# PID Controller Migration Guide

## Overview
The old `PIDController` implementations have been replaced with a new unit-aware, JAX-based implementation that provides:
- Full unit support with pint
- JAX compilation and vectorization
- Improved numerical stability
- Backward compatibility with existing code

## Migration Status ✅
- **Old implementations removed**: The duplicate implementations in `ethode_extended.py` and `stochastic_extensions.py` have been replaced with imports from the new module
- **Backward compatibility**: A legacy wrapper provides the exact same API as before
- **All tests passing**: Existing code continues to work without modification

## Using the Legacy API (No Changes Required)

Your existing code will continue to work:

```python
from ethode.controller import PIDController, PIDParams

# Old style with PIDParams
params = PIDParams(
    kp=1.0,
    ki=0.1,
    kd=0.01,
    integral_leak=0.1,
    output_min=-10.0,
    output_max=10.0,
    noise_threshold=0.001
)
pid = PIDController(params)
output = pid.update(error, dt)
pid.reset()

# Old style with kwargs
pid = PIDController(kp=0.5, ki=0.05, kd=0.005)
output = pid.update(error, dt)
```

## Using the New Unit-Aware API (Recommended)

The new API provides better unit handling and JAX optimization:

```python
from ethode import ControllerConfig, controller_step, ControllerState
import jax.numpy as jnp

# Create config with unit-aware inputs
config = ControllerConfig(
    kp="0.2 / day",                      # Units are validated
    ki="0.2 / day / 7 day",             # Complex units supported
    kd="0.0 hour",                       # Different time units OK
    tau="1 week",                        # Integral leak time constant
    noise_band=("1 milliUSD", "3 milliUSD"),  # Range with units
    output_min="-100 USD",               # Optional limits
    output_max="100 USD",
)

# Convert to runtime structure for JAX
runtime = config.to_runtime()
state = ControllerState.zero()

# Run control step (JAX-compiled)
new_state, output = controller_step(runtime, state, error, dt)
```

## Key Differences

### Old API
- Stateful object (`pid.update()` modifies internal state)
- Python floats only
- No unit validation
- Single-threaded execution
- `integral_leak` as decay rate

### New API
- Pure functions (state passed explicitly)
- JAX arrays with JIT compilation
- Full unit support and validation
- Vectorizable with `vmap`
- `tau` as time constant (inverse of leak rate)

## Migration Strategies

### Option 1: Keep Using Legacy API (No Changes)
Your existing code continues to work. The legacy wrapper uses the new implementation internally.

### Option 2: Gradual Migration
Replace PIDController usage with ControllerConfig where convenient:

```python
# Old code
pid = PIDController(kp=0.2/86400, ki=0.02/86400, kd=0)  # Manual unit conversion

# New code
config = ControllerConfig(
    kp="0.2 / day",   # Units handled automatically
    ki="0.02 / day",
    kd=0.0,
    tau="1 week",
    noise_band=(0.001, 0.003)
)
```

### Option 3: Full Migration for Performance
Use the JAX-based API directly for maximum performance:

```python
# Vectorized control over multiple errors
errors = jnp.array([0.001, 0.002, 0.003, 0.004])

@jax.vmap
def run_control(error):
    return controller_step(runtime, state, error, dt)

outputs = run_control(errors)  # Parallel execution!
```

## Benefits of Migration

1. **Type Safety**: Pydantic validation catches configuration errors
2. **Unit Safety**: Automatic unit conversion and validation
3. **Performance**: JAX compilation provides 10-100x speedup for loops
4. **Reproducibility**: Pure functions make testing easier
5. **Scalability**: Vectorization for parallel control loops

## Testing

Run the migration test suite to verify compatibility:

```bash
python test_pid_migration.py  # Test legacy compatibility
python test_controller.py     # Test new features
```

## Support

The old PIDController API will be maintained for backward compatibility, but new features will only be added to the ControllerConfig API.

---

# New High-Level API: ControllerAdapter & Simulation

## Overview

The ethode library now provides a **two-tier API architecture**:

### Core Tier (Low-Level, JAX-Focused)
For users building custom solvers or needing direct JAX access:
- `ControllerConfig` — Configuration with unit validation
- `ControllerRuntime` — JAX-ready parameter struct (Penzai)
- `ControllerState` — JAX-ready state struct (Penzai)
- `controller_step()` — Pure JAX kernel function

### Adapter Tier (High-Level, Day-to-Day Use)
For users who want convenience and stateful APIs:
- **`ControllerAdapter`** — Stateful wrapper with automatic unit validation
- **`Simulation`** — Multi-subsystem orchestration (controller now, fee/liquidity later)

### Legacy (Deprecated, Removed in v3.0)
- `PIDParams` + `PIDController` — Backward compatibility only

## The New High-Level API: ControllerAdapter

`ControllerAdapter` is now **THE recommended high-level API** for day-to-day controller usage. It provides:

- ✓ Stateful API (like old `PIDController`)
- ✓ Automatic unit validation by default
- ✓ Direct access to JAX runtime when needed
- ✓ Multiple step variants (basic, diagnostics, with units)
- ✓ Works with both `ControllerConfig` (new) and `PIDParams` (legacy)

### Basic Usage

```python
from ethode import ControllerAdapter, ControllerConfig

# Create config with units
config = ControllerConfig(
    kp="0.2 / day",
    ki="0.02 / day**2",
    kd=0.0,
    tau="7 day",
    noise_band=("0.001 USD", "0.003 USD")
)

# Create adapter (validates units automatically)
adapter = ControllerAdapter(config)

# Use like the old PIDController
output = adapter.step(error=1.0, dt=0.1)
output = adapter.step(error=0.5, dt=0.1)

# Reset when needed
adapter.reset()
```

### Migration from PIDController to ControllerAdapter

**Old code (deprecated):**
```python
from ethode.controller import PIDController, PIDParams

params = PIDParams(kp=1.0, ki=0.1, kd=0.01)
pid = PIDController(params)
output = pid.update(error, dt)
```

**New code (recommended):**
```python
from ethode import ControllerAdapter, ControllerConfig

config = ControllerConfig(kp=1.0, ki=0.1, kd=0.01)
adapter = ControllerAdapter(config)
output = adapter.step(error, dt)
```

Key differences:
- `PIDController.update()` → `ControllerAdapter.step()`
- Unit validation runs automatically (pass `check_units=False` to disable)
- Access to JAX runtime via `adapter.runtime` and `adapter.state`

### Advanced: Direct JAX Access

When you need JAX transformations, access the runtime directly:

```python
from ethode import ControllerAdapter, ControllerConfig
from ethode.controller.kernel import controller_step
import jax

# Create adapter as usual
config = ControllerConfig(kp="0.2/day", ki="0.02/day**2", kd=0.0, tau="7 day")
adapter = ControllerAdapter(config)

# Access JAX runtime structures
runtime = adapter.runtime  # ControllerRuntime (Penzai struct)
state = adapter.state       # ControllerState (Penzai struct)

# Use with JAX transformations
@jax.jit
def fast_control_step(state, error, dt):
    return controller_step(runtime, state, error, dt)

# JIT-compiled execution
new_state, output = fast_control_step(state, jnp.array(1.0), jnp.array(0.1))
```

### Step Variants

`ControllerAdapter` provides multiple step methods:

#### 1. Basic Step
```python
output = adapter.step(error=1.0, dt=0.1)
```

#### 2. Step with Diagnostics
```python
output, diagnostics = adapter.step_with_diagnostics(error=1.0, dt=0.1)
# diagnostics = {'proportional': ..., 'integral': ..., 'derivative': ...}
```

#### 3. Step with Units (Debugging)
```python
from pint import Quantity

error_qty = Quantity(1.0, "USD")
dt_qty = Quantity(0.1, "second")
output_qty = adapter.step_with_units(error_qty, dt_qty)
# Returns pint Quantity with units preserved
```

## Simulation Facade for Multi-Subsystem Orchestration

The `Simulation` class provides high-level orchestration of multiple subsystems (controller, fee, liquidity, etc.). Currently supports controller, with fee/liquidity integration planned.

### Stateful API (Notebook/Interactive Use)

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig

# Setup controller
config = ControllerConfig(
    kp="0.2/day", ki="0.02/day**2", kd=0.0,
    tau="7 day", noise_band=("0.001 USD", "0.003 USD")
)
adapter = ControllerAdapter(config)

# Create simulation
sim = Simulation(controller=adapter)

# Run simulation steps
for t in range(100):
    error = target - measurement
    control = sim.step(error, dt=0.1)
    # Apply control...

# Reset for new simulation
sim.reset()
```

### Functional API (JAX Transformations)

For users who want functional programming with JAX:

```python
from ethode import simulate_controller_step
import jax
import jax.numpy as jnp

# Access runtime structures
runtime = sim.controller.runtime
initial_state = sim.controller.state

# Define scan function
def step_fn(state, inputs):
    error, dt = inputs
    return simulate_controller_step(runtime, state, error, dt)

# Batch process with jax.lax.scan
errors = jnp.array([1.0, 0.5, 0.2, 0.0, -0.1])
dts = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
final_state, outputs = jax.lax.scan(step_fn, initial_state, (errors, dts))
```

### Convenience: Simulation.scan()

The `.scan()` method provides a convenient wrapper for batch processing:

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig
import jax.numpy as jnp

# Setup
config = ControllerConfig(kp="0.2/day", ki="0.02/day**2", kd=0.0, tau="7 day")
adapter = ControllerAdapter(config)
sim = Simulation(controller=adapter)

# Batch process multiple steps efficiently
errors = jnp.array([1.0, 0.5, 0.2, 0.0, -0.1])
dts = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])

# Single call processes all steps with JAX scan
outputs, final_state = sim.scan(errors, dts)

# Internal state is automatically updated
print(sim.controller.state.integral)  # Updated to final_state.integral
```

Benefits of `.scan()`:
- ✓ No need to write `jax.lax.scan` loops manually
- ✓ Efficient JAX compilation under the hood
- ✓ Automatically updates internal state
- ✓ Works with varying time steps
- ✓ Can chain multiple scan calls

### State Management

Access and inspect state at any time:

```python
# Get current state
state_dict = sim.get_state()
print(state_dict['controller']['integral'])

# Reset all subsystems
sim.reset()

# Direct access to controller state
integral_value = float(sim.controller.state.integral)
```

## Complete Migration Example

Here's a complete example showing migration from old to new API:

### Old Code (Deprecated)
```python
from ethode.controller import PIDController

# Manual unit conversion, no validation
pid = PIDController(
    kp=0.2/86400,      # 0.2 per day → per second
    ki=0.02/86400,     # 0.02 per day → per second
    kd=0.0,
    integral_leak=1.0/(7*86400),  # 7 days → leak rate
    noise_threshold=0.001
)

# Stateful updates
for error in errors:
    output = pid.update(error, dt=0.1)
    results.append(output)

pid.reset()
```

### New Code (Recommended)
```python
from ethode import Simulation, ControllerAdapter, ControllerConfig
import jax.numpy as jnp

# Units handled automatically, validation built-in
config = ControllerConfig(
    kp="0.2 / day",           # Units explicit!
    ki="0.02 / day**2",       # Dimensional analysis automatic
    kd=0.0,
    tau="7 day",              # Time constant (not leak rate)
    noise_band=("0.001 USD", "0.003 USD")  # Range instead of threshold
)

adapter = ControllerAdapter(config)
sim = Simulation(controller=adapter)

# Option 1: Stateful (similar to old API)
results = []
for error in errors:
    output = sim.step(error, dt=0.1)
    results.append(output)

# Option 2: Batch with scan (much faster!)
errors_jax = jnp.array(errors)
dts_jax = jnp.array([0.1] * len(errors))
outputs, final_state = sim.scan(errors_jax, dts_jax)

sim.reset()
```

## API Reference Summary

### ControllerAdapter
- `__init__(config, *, check_units=True)` — Create adapter with validation
- `.step(error, dt)` → `float` — Execute one control step
- `.step_with_diagnostics(error, dt)` → `(float, dict)` — Step + term values
- `.step_with_units(error_qty, dt_qty)` → `Quantity` — Step with pint units
- `.reset()` — Reset internal state to zero
- `.get_state()` → `dict` — Get current state as dictionary
- `.runtime` — Access JAX runtime structure (ControllerRuntime)
- `.state` — Access JAX state structure (ControllerState)

### Simulation
- `__init__(*, controller, fee=None, liquidity=None)` — Create simulation
- `.step(error, dt)` → `float` — Execute one simulation step (stateful)
- `.scan(errors, dts)` → `(Array, State)` — Batch process with JAX scan
- `.reset()` — Reset all subsystem states
- `.get_state()` → `dict` — Get all subsystem states
- `.controller` — Access controller adapter
- `.fee` — Reserved for FeeAdapter (future)
- `.liquidity` — Reserved for LiquidityAdapter (future)

### Pure Functions (Functional API)
- `simulate_controller_step(runtime, state, error, dt)` — Pure wrapper for JAX
- `controller_step(runtime, state, error, dt)` — Core JAX kernel

## Recommended Migration Path

1. **Phase 1**: Replace `PIDController` with `ControllerAdapter`
   - Minimal code changes
   - Keep stateful `.step()` API
   - Get automatic unit validation

2. **Phase 2**: Replace `PIDParams` with `ControllerConfig`
   - Add explicit units to configuration
   - Benefit from dimensional analysis
   - Catch unit errors early

3. **Phase 3**: Use `Simulation` for orchestration
   - Prepare for multi-subsystem integration
   - Use `.scan()` for batch processing
   - Access JAX runtime when needed

4. **Phase 4**: Optimize with JAX (optional)
   - Use direct `controller_step()` for custom solvers
   - Apply JAX transformations (jit, vmap, grad)
   - Maximum performance

## Questions & Troubleshooting

### Q: Should I use ControllerAdapter or the core API?
**A:** Use `ControllerAdapter` for day-to-day work. Only use the core API (`ControllerConfig` + `controller_step`) if you're building custom solvers or need maximum control over JAX transformations.

### Q: Can I disable unit validation?
**A:** Yes: `adapter = ControllerAdapter(config, check_units=False)`. But we recommend keeping validation enabled—it catches bugs early!

### Q: How do I migrate integral_leak to tau?
**A:** They're inverses:
- Old: `integral_leak=0.1` → New: `tau=10.0` (or `tau="10 seconds"`)
- Old: `integral_leak=1/(7*86400)` → New: `tau="7 day"`

### Q: What about error filters?
**A:** The legacy `error_filter` parameter is only supported in `PIDController`. For `ControllerAdapter`, apply filtering before calling `.step()`:
```python
filtered_error = my_filter(error)
output = adapter.step(filtered_error, dt)
```

### Q: When will fee/liquidity adapters be available?
**A:** The adapter pattern is designed to be generic. Fee and liquidity subsystems will follow the same pattern: `FeeConfig` → `FeeAdapter` → `Simulation(fee=adapter)`.

### Q: Do I need to learn JAX?
**A:** No! The `ControllerAdapter` and `Simulation` APIs work without any JAX knowledge. JAX is only needed if you want to write custom transformations or solvers.