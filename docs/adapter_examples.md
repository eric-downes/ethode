# ControllerAdapter & Simulation: Code Examples

This document provides practical examples for using the new high-level API.

## Table of Contents
- [Basic Controller Usage](#basic-controller-usage)
- [Using Simulation for Orchestration](#using-simulation-for-orchestration)
- [Direct JAX Access](#direct-jax-access)
- [Batch Processing with scan()](#batch-processing-with-scan)
- [Unit Validation and Debugging](#unit-validation-and-debugging)
- [Advanced Patterns](#advanced-patterns)

---

## Basic Controller Usage

### Example 1: Simple Controller with Units

```python
from ethode import ControllerAdapter, ControllerConfig

# Create configuration with explicit units
config = ControllerConfig(
    kp="0.2 / day",          # Proportional gain
    ki="0.02 / day**2",      # Integral gain
    kd=0.0,                   # No derivative
    tau="7 day",              # Integral leak time constant
    noise_band=("0.001 USD", "0.003 USD")  # Dead zone
)

# Create adapter (automatically validates units)
adapter = ControllerAdapter(config)

# Simulate control loop
target = 1.0
measurement = 0.0

for t in range(100):
    error = target - measurement
    control = adapter.step(error, dt=0.1)

    # Apply control (simplified dynamics)
    measurement += control * 0.1

    if t % 10 == 0:
        print(f"t={t}: error={error:.4f}, control={control:.4f}")

# Reset when starting a new simulation
adapter.reset()
```

### Example 2: Controller with Output Limits

```python
from ethode import ControllerAdapter, ControllerConfig

config = ControllerConfig(
    kp="0.5 / day",
    ki="0.05 / day**2",
    kd="0.01 day",
    tau="10 day",
    noise_band=("0 USD", "0.01 USD"),
    output_min="-5 USD",      # Minimum control output
    output_max="5 USD"        # Maximum control output
)

adapter = ControllerAdapter(config)

# Large error gets clamped
error = 100.0
output = adapter.step(error, dt=1.0)
print(f"Output (clamped): {output}")  # Will be at most 5.0
```

### Example 3: Dimensionless Controller

```python
from ethode import ControllerAdapter, ControllerConfig

# Simple dimensionless controller (like old API)
config = ControllerConfig(
    kp=1.0,
    ki=0.1,
    kd=0.01
)

adapter = ControllerAdapter(config)

# Use like old PIDController
output = adapter.step(error=0.5, dt=0.01)
```

---

## Using Simulation for Orchestration

### Example 4: Basic Simulation Setup

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig

# Setup controller
config = ControllerConfig(
    kp="0.2 / day",
    ki="0.02 / day**2",
    kd=0.0,
    tau="7 day",
    noise_band=("0.001 USD", "0.003 USD")
)
adapter = ControllerAdapter(config)

# Create simulation
sim = Simulation(controller=adapter)

# Run simulation
errors = [1.0, 0.8, 0.6, 0.4, 0.2]
for error in errors:
    control = sim.step(error, dt=0.1)
    print(f"Error: {error:.1f} → Control: {control:.4f}")

# Check final state
state = sim.get_state()
print(f"Final integral: {state['controller']['integral']:.4f}")

# Reset for next simulation
sim.reset()
```

### Example 5: Stateful Simulation Loop

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig
import numpy as np

# Configuration
config = ControllerConfig(
    kp="0.3 / day",
    ki="0.03 / day**2",
    kd="0.01 day",
    tau="5 day",
    noise_band=("0 USD", "0.005 USD")
)

# Create simulation
adapter = ControllerAdapter(config)
sim = Simulation(controller=adapter)

# Simulate a step response
target = 1.0
measurement = 0.0
history = []

for t in range(200):
    error = target - measurement
    control = sim.step(error, dt=0.1)

    # Simple first-order dynamics
    measurement += (control - measurement) * 0.05

    history.append({
        'time': t * 0.1,
        'error': error,
        'control': control,
        'measurement': measurement
    })

# Analyze results
import pandas as pd
df = pd.DataFrame(history)
print(df.tail())

# Plot (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    df.plot(x='time', y='measurement', ax=axes[0], title='System Response')
    df.plot(x='time', y='error', ax=axes[1], title='Tracking Error')
    df.plot(x='time', y='control', ax=axes[2], title='Control Signal')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install matplotlib for plotting")
```

---

## Direct JAX Access

### Example 6: Using Runtime Directly with JAX

```python
from ethode import ControllerAdapter, ControllerConfig
from ethode.controller.kernel import controller_step
import jax
import jax.numpy as jnp

# Create adapter
config = ControllerConfig(
    kp="0.2 / day",
    ki="0.02 / day**2",
    kd=0.0,
    tau="7 day"
)
adapter = ControllerAdapter(config)

# Extract JAX-ready structures
runtime = adapter.runtime
state = adapter.state

# Create JIT-compiled control function
@jax.jit
def fast_control_step(state, error, dt):
    return controller_step(runtime, state, error, dt)

# Run compiled version
error = jnp.array(1.0)
dt = jnp.array(0.1)
new_state, output = fast_control_step(state, error, dt)

print(f"Output: {output}")
print(f"New integral: {new_state.integral}")
```

### Example 7: Vectorized Control with vmap

```python
from ethode import ControllerAdapter, ControllerConfig
from ethode.controller.kernel import controller_step
import jax
import jax.numpy as jnp

config = ControllerConfig(kp=1.0, ki=0.1, kd=0.01)
adapter = ControllerAdapter(config)

runtime = adapter.runtime
initial_state = adapter.state

# Process multiple errors in parallel
errors = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
dt = jnp.array(0.01)

# Vectorize over errors (each gets independent state)
def single_step(error):
    state, output = controller_step(runtime, initial_state, error, dt)
    return output

vectorized_control = jax.vmap(single_step)
outputs = vectorized_control(errors)

print(f"Errors: {errors}")
print(f"Outputs: {outputs}")
```

---

## Batch Processing with scan()

### Example 8: Efficient Batch Simulation

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig
import jax.numpy as jnp

# Setup
config = ControllerConfig(
    kp="0.2 / day",
    ki="0.02 / day**2",
    kd=0.0,
    tau="7 day"
)
adapter = ControllerAdapter(config)
sim = Simulation(controller=adapter)

# Prepare batch data
n_steps = 1000
errors = jnp.linspace(1.0, 0.0, n_steps)  # Decaying error
dts = jnp.full(n_steps, 0.1)              # Constant timestep

# Process all steps efficiently
outputs, final_state = sim.scan(errors, dts)

print(f"Processed {n_steps} steps")
print(f"Output shape: {outputs.shape}")
print(f"Final integral: {final_state.integral}")
print(f"First 5 outputs: {outputs[:5]}")
print(f"Last 5 outputs: {outputs[-5:]}")
```

### Example 9: Chaining Multiple scan() Calls

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig
import jax.numpy as jnp

config = ControllerConfig(kp="0.3/day", ki="0.03/day**2", kd=0.0, tau="5 day")
adapter = ControllerAdapter(config)
sim = Simulation(controller=adapter)

# Phase 1: Large errors
errors1 = jnp.array([1.0, 0.9, 0.8, 0.7, 0.6])
dts1 = jnp.full(5, 0.1)
outputs1, state1 = sim.scan(errors1, dts1)

print(f"Phase 1 complete, integral: {state1.integral}")

# Phase 2: Small errors (state continues from phase 1)
errors2 = jnp.array([0.1, 0.05, 0.02, 0.01, 0.0])
dts2 = jnp.full(5, 0.1)
outputs2, state2 = sim.scan(errors2, dts2)

print(f"Phase 2 complete, integral: {state2.integral}")

# Verify continuity
print(f"Internal state matches: {sim.controller.state.integral == state2.integral}")
```

### Example 10: Varying Time Steps

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig
import jax.numpy as jnp

config = ControllerConfig(kp=0.5, ki=0.05, kd=0.01)
adapter = ControllerAdapter(config)
sim = Simulation(controller=adapter)

# Non-uniform time steps
errors = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
dts = jnp.array([0.1, 0.05, 0.2, 0.15, 0.1])  # Varying dt

outputs, final_state = sim.scan(errors, dts)

for i, (err, dt, out) in enumerate(zip(errors, dts, outputs)):
    print(f"Step {i}: error={err:.1f}, dt={dt:.2f}, output={out:.4f}")
```

---

## Unit Validation and Debugging

### Example 11: Using step_with_units() for Debugging

```python
from ethode import ControllerAdapter, ControllerConfig
from pint import Quantity

# Create config with financial units
config = ControllerConfig(
    kp="0.2 / day",
    ki="0.02 / day**2",
    kd="0.01 day",
    tau="7 day",
    noise_band=("0.001 USD", "0.003 USD")
)
adapter = ControllerAdapter(config)

# Use pint quantities for debugging
error = Quantity(1.0, "USD")
dt = Quantity(0.1, "second")

output = adapter.step_with_units(error, dt)

print(f"Error: {error}")
print(f"Output: {output}")
print(f"Output units: {output.units}")
print(f"Output in USD/day: {output.to('USD/day')}")
```

### Example 12: Diagnosing Controller Behavior

```python
from ethode import ControllerAdapter, ControllerConfig

config = ControllerConfig(
    kp="0.3 / day",
    ki="0.03 / day**2",
    kd="0.05 day",
    tau="10 day"
)
adapter = ControllerAdapter(config)

# Step with diagnostics
error = 1.0
dt = 0.1

output, diagnostics = adapter.step_with_diagnostics(error, dt)

print(f"Total output: {output:.4f}")
print(f"Diagnostics:")
print(f"  Proportional term: {diagnostics['proportional']:.4f}")
print(f"  Integral term: {diagnostics['integral']:.4f}")
print(f"  Derivative term: {diagnostics['derivative']:.4f}")
print(f"  Noise factor: {diagnostics.get('noise_factor', 1.0):.4f}")

# Run multiple steps to see integral build-up
for i in range(5):
    output, diag = adapter.step_with_diagnostics(error=0.5, dt=0.1)
    print(f"Step {i}: integral={diag['integral']:.4f}")
```

### Example 13: Catching Unit Errors

```python
from ethode import ControllerAdapter, ControllerConfig

try:
    # This should fail - incompatible units
    bad_config = ControllerConfig(
        kp="0.2 / day",        # OK: 1/time
        ki="0.02 meter",       # BAD: should be 1/time²
        kd=0.0,
        tau="7 day"
    )
    adapter = ControllerAdapter(bad_config)  # Validation happens here
except ValueError as e:
    print(f"Caught unit error: {e}")

# Disable validation if needed (not recommended)
adapter = ControllerAdapter(bad_config, check_units=False)
print("Warning: Running with invalid units!")
```

---

## Advanced Patterns

### Example 14: Custom JAX Scan Function

```python
from ethode import ControllerAdapter, ControllerConfig
from ethode.controller.kernel import controller_step
import jax
import jax.numpy as jnp

config = ControllerConfig(kp=0.5, ki=0.05, kd=0.01)
adapter = ControllerAdapter(config)

runtime = adapter.runtime
initial_state = adapter.state

# Custom scan with additional tracking
def step_with_tracking(carry, inputs):
    state, metrics = carry
    error, dt = inputs

    new_state, output = controller_step(runtime, state, error, dt)

    # Track cumulative output
    new_metrics = {
        'cumulative_output': metrics['cumulative_output'] + output,
        'max_output': jnp.maximum(metrics['max_output'], output)
    }

    return (new_state, new_metrics), output

# Initial metrics
initial_metrics = {
    'cumulative_output': jnp.array(0.0),
    'max_output': jnp.array(-jnp.inf)
}

errors = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
dts = jnp.full(5, 0.1)

(final_state, final_metrics), outputs = jax.lax.scan(
    step_with_tracking,
    (initial_state, initial_metrics),
    (errors, dts)
)

print(f"Cumulative output: {final_metrics['cumulative_output']}")
print(f"Max output: {final_metrics['max_output']}")
```

### Example 15: Gradient-Based Tuning

```python
from ethode import ControllerConfig
from ethode.controller.kernel import controller_step
from ethode.runtime import ControllerState
import jax
import jax.numpy as jnp

# Define a loss function for controller performance
def controller_loss(kp_value, target_errors, target_outputs):
    """Compute MSE between controller outputs and target outputs."""
    config = ControllerConfig(kp=kp_value, ki=0.1, kd=0.01)
    runtime = config.to_runtime()

    def step_fn(state, error):
        new_state, output = controller_step(
            runtime, state, error, jnp.array(0.1)
        )
        return new_state, output

    initial_state = ControllerState.zero()
    _, outputs = jax.lax.scan(step_fn, initial_state, target_errors)

    return jnp.mean((outputs - target_outputs) ** 2)

# Target behavior
errors = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
target_outputs = jnp.array([0.5, 0.4, 0.3, 0.2, 0.1])

# Compute gradient
grad_fn = jax.grad(controller_loss)
kp_initial = 0.5
gradient = grad_fn(kp_initial, errors, target_outputs)

print(f"Gradient of loss w.r.t. kp: {gradient}")
print(f"Suggests {'increasing' if gradient < 0 else 'decreasing'} kp")
```

### Example 16: Performance Comparison

```python
from ethode import Simulation, ControllerAdapter, ControllerConfig
import jax.numpy as jnp
import time

config = ControllerConfig(kp="0.2/day", ki="0.02/day**2", kd=0.0, tau="7 day")

# Method 1: Sequential steps
adapter1 = ControllerAdapter(config)
sim1 = Simulation(controller=adapter1)

errors_list = [1.0, 0.9, 0.8, 0.7, 0.6] * 200  # 1000 steps
start = time.time()
for error in errors_list:
    _ = sim1.step(error, dt=0.1)
seq_time = time.time() - start

# Method 2: Batch with scan
adapter2 = ControllerAdapter(config)
sim2 = Simulation(controller=adapter2)

errors_jax = jnp.array(errors_list)
dts_jax = jnp.full(len(errors_list), 0.1)

start = time.time()
outputs, _ = sim2.scan(errors_jax, dts_jax)
# Run twice to account for compilation
outputs, _ = sim2.scan(errors_jax, dts_jax)
scan_time = time.time() - start

print(f"Sequential: {seq_time:.4f}s")
print(f"Scan: {scan_time:.4f}s")
print(f"Speedup: {seq_time / scan_time:.2f}x")
```

---

## Summary

These examples demonstrate:

1. **Basic Usage**: Simple controller setup with units
2. **Simulation**: High-level orchestration API
3. **JAX Integration**: Direct access to runtime for transformations
4. **Batch Processing**: Efficient `.scan()` for large datasets
5. **Debugging**: Unit validation and diagnostic tools
6. **Advanced**: Custom patterns and performance optimization

For more information, see:
- [PID_MIGRATION_GUIDE.md](../PID_MIGRATION_GUIDE.md) - Complete migration guide
- [test_adapters.py](../test_adapters.py) - Adapter test examples
- [test_simulation.py](../test_simulation.py) - Simulation test examples
