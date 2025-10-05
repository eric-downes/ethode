# Adapter & Simulation: Code Examples

This document provides practical examples for using the high-level adapter APIs, including Controller and Fee subsystems.

## Table of Contents
- [Basic Controller Usage](#basic-controller-usage)
- [Using Simulation for Orchestration](#using-simulation-for-orchestration)
- [Direct JAX Access](#direct-jax-access)
- [Batch Processing with scan()](#batch-processing-with-scan)
- [Unit Validation and Debugging](#unit-validation-and-debugging)
- [Advanced Patterns](#advanced-patterns)
- [Fee Subsystem](#fee-subsystem)
- [Liquidity Subsystem](#liquidity-subsystem)
- [Hawkes Process Subsystem](#hawkes-process-subsystem)
- [Multi-Subsystem Integration](#multi-subsystem-integration)

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

---

## Fee Subsystem

### Example 17: Basic Fee Calculation

```python
from ethode import FeeAdapter, FeeConfig

# Configure fee structure
config = FeeConfig(
    base_fee_rate="50 bps",      # 0.5% base fee
    max_fee_rate="200 bps",      # 2% maximum
    fee_decay_time="1 day",
    min_fee_amount="0.01 USD"
)

# Create adapter
adapter = FeeAdapter(config)

# Calculate fees for transactions
fee1 = adapter.step(transaction_amount=100.0, dt=0.1)
print(f"Fee for $100 transaction: ${fee1:.2f}")

fee2 = adapter.step(transaction_amount=500.0, dt=0.1)
print(f"Fee for $500 transaction: ${fee2:.2f}")

# Check accumulated fees
state = adapter.get_state()
print(f"Total fees collected: ${state['accumulated_fees']:.2f}")
```

### Example 18: Stress-Based Fee Adjustment

```python
from ethode import FeeAdapter, FeeConfig

# Configure dynamic fees that respond to market stress
config = FeeConfig(
    base_fee_rate="50 bps",       # 0.5% baseline
    max_fee_rate="500 bps",       # 5% under extreme stress
    fee_growth_rate="10 bps / second",  # How fast fees increase with stress
    fee_decay_time="1 hour",      # How fast fees decay back to baseline
    min_fee_amount="0.01 USD"
)

adapter = FeeAdapter(config)

# Low stress conditions - fees near base rate
adapter.update_stress(volatility=0.1, volume_ratio=1.0)
fee_low = adapter.step(transaction_amount=100.0, dt=0.1)
print(f"Fee (low stress): ${fee_low:.2f}")

# High stress conditions - fees increase
adapter.update_stress(volatility=0.9, volume_ratio=3.0)
# Let stress affect the rate over time
for _ in range(10):
    adapter.step(100.0, dt=1.0)

fee_high = adapter.step(transaction_amount=100.0, dt=0.1)
print(f"Fee (high stress): ${fee_high:.2f}")
print(f"Stress multiplier: {fee_high / fee_low:.2f}x")

# Fees naturally decay back to baseline
adapter.update_stress(volatility=0.0, volume_ratio=1.0)
for i in range(20):
    adapter.step(1.0, dt=60.0)  # 60 second steps
    if i % 5 == 0:
        current_rate = adapter.get_state()['current_fee_rate']
        print(f"After {i*60}s: fee rate = {current_rate*100:.2f}%")
```

### Example 19: Fee Diagnostics and Analysis

```python
from ethode import FeeAdapter, FeeConfig
import pandas as pd

# Configure fees
config = FeeConfig(
    base_fee_rate="50 bps",
    max_fee_rate="300 bps",
    fee_growth_rate="5 bps / second",
    fee_decay_time="30 minutes",
    min_fee_amount="0.01 USD",
    max_fee_amount="100 USD"
)

adapter = FeeAdapter(config)

# Simulate varying market conditions
history = []

for t in range(100):
    # Simulate time-varying stress
    volatility = 0.5 + 0.4 * (t % 20) / 20  # Cycles between 0.5 and 0.9
    volume_ratio = 1.0 + 2.0 * (t % 30) / 30  # Cycles between 1.0 and 3.0

    adapter.update_stress(volatility, volume_ratio)

    # Process transaction with diagnostics
    amount = 1000.0
    fee, diagnostics = adapter.step_with_diagnostics(amount, dt=1.0)

    history.append({
        'time': t,
        'volatility': volatility,
        'volume_ratio': volume_ratio,
        'stress_level': diagnostics['stress_level'],
        'current_rate': diagnostics['current_rate'] * 100,  # as percentage
        'base_fee': diagnostics['base_fee'],
        'stress_adjustment': diagnostics['stress_adjustment'],
        'total_fee': fee,
        'accumulated_fees': diagnostics['accumulated_fees'],
        'rate_at_max': diagnostics['max_rate_applied']
    })

# Analyze with pandas
df = pd.DataFrame(history)
print("\nFee Statistics:")
print(f"Average fee: ${df['total_fee'].mean():.2f}")
print(f"Max fee: ${df['total_fee'].max():.2f}")
print(f"Total collected: ${df['accumulated_fees'].iloc[-1]:.2f}")
print(f"\nRate Statistics:")
print(f"Average rate: {df['current_rate'].mean():.2f}%")
print(f"Max rate: {df['current_rate'].max():.2f}%")
print(f"Times at max rate: {df['rate_at_max'].sum()}")

# Plot if matplotlib available
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Market conditions
    ax1 = axes[0]
    ax1.plot(df['time'], df['volatility'], label='Volatility', color='red', alpha=0.7)
    ax1.plot(df['time'], df['volume_ratio'], label='Volume Ratio', color='blue', alpha=0.7)
    ax1.set_ylabel('Level')
    ax1.set_title('Market Conditions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fee rate evolution
    ax2 = axes[1]
    ax2.plot(df['time'], df['current_rate'], label='Current Rate', color='green')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Base Rate (0.5%)')
    ax2.axhline(y=3.0, color='red', linestyle='--', label='Max Rate (3.0%)')
    ax2.set_ylabel('Fee Rate (%)')
    ax2.set_title('Dynamic Fee Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Cumulative fees
    ax3 = axes[2]
    ax3.plot(df['time'], df['accumulated_fees'], color='purple')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Accumulated Fees ($)')
    ax3.set_title('Total Fees Collected')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fee_dynamics.png', dpi=150)
    print("\nPlot saved to fee_dynamics.png")
except ImportError:
    print("\nInstall matplotlib and pandas for visualization")
```

### Example 20: Direct JAX Access for Fee Calculations

```python
from ethode import FeeAdapter, FeeConfig
from ethode.fee.kernel import calculate_fee
import jax
import jax.numpy as jnp

# Create adapter
config = FeeConfig(
    base_fee_rate="50 bps",
    max_fee_rate="200 bps",
    fee_decay_time="1 day"
)
adapter = FeeAdapter(config)

# Extract JAX-ready structures for power users
runtime = adapter.runtime
state = adapter.state

# Create JIT-compiled fee function
@jax.jit
def fast_fee_calculation(state, amount, dt):
    return calculate_fee(runtime, state, amount, dt)

# Process batch of transactions efficiently
amounts = jnp.array([100.0, 200.0, 150.0, 300.0, 250.0])
dt = jnp.array(0.1)

# Use scan for sequential state updates
def fee_step(state, amount):
    new_state, fee = fast_fee_calculation(state, amount, dt)
    return new_state, fee

final_state, fees = jax.lax.scan(fee_step, state, amounts)

print(f"Fees calculated: {fees}")
print(f"Total fees: ${float(jnp.sum(fees)):.2f}")
print(f"Final accumulated: ${float(final_state.accumulated_fees):.2f}")
```

### Example 21: Reset and Multiple Simulation Runs

```python
from ethode import FeeAdapter, FeeConfig

config = FeeConfig(
    base_fee_rate="50 bps",
    max_fee_rate="300 bps",
    fee_growth_rate="10 bps / second",
    fee_decay_time="1 hour"
)

adapter = FeeAdapter(config)

# Simulation 1: High stress scenario
print("Simulation 1: High Stress")
adapter.update_stress(volatility=0.9, volume_ratio=5.0)
total_fees_1 = 0.0
for _ in range(100):
    fee = adapter.step(transaction_amount=100.0, dt=1.0)
    total_fees_1 += fee

print(f"Total fees collected: ${total_fees_1:.2f}")
print(f"Final state: {adapter.get_state()}")

# Reset for new simulation
adapter.reset()

# Simulation 2: Low stress scenario
print("\nSimulation 2: Low Stress")
adapter.update_stress(volatility=0.1, volume_ratio=1.0)
total_fees_2 = 0.0
for _ in range(100):
    fee = adapter.step(transaction_amount=100.0, dt=1.0)
    total_fees_2 += fee

print(f"Total fees collected: ${total_fees_2:.2f}")
print(f"Final state: {adapter.get_state()}")

# Compare scenarios
print(f"\nHigh stress collected {total_fees_1 / total_fees_2:.2f}x more fees")
```

---

## Liquidity Subsystem

### Example 22: Basic Stochastic Liquidity

```python
from ethode import LiquidityAdapter, LiquiditySDEConfig

# Configure stochastic liquidity dynamics
config = LiquiditySDEConfig(
    initial_liquidity="1M USD",
    mean_liquidity="1M USD",         # Long-term equilibrium
    mean_reversion_rate="0.1 / day", # Speed of mean reversion
    volatility=0.2                    # Volatility coefficient
)

# Create adapter with fixed seed for reproducibility
adapter = LiquidityAdapter(config, seed=42)

# Simulate liquidity evolution
for t in range(100):
    liquidity = adapter.step(dt=0.1)
    if t % 20 == 0:
        print(f"t={t*0.1:.1f}: Liquidity = ${liquidity:,.0f}")

# Check final state
state = adapter.get_state()
print(f"\nFinal state:")
print(f"  Liquidity: ${state['liquidity_level']:,.0f}")
print(f"  Total provision: ${state['cumulative_provision']:,.0f}")
print(f"  Total removal: ${state['cumulative_removal']:,.0f}")
```

### Example 23: Liquidity with External Shocks

```python
from ethode import LiquidityAdapter, LiquiditySDEConfig

config = LiquiditySDEConfig(
    initial_liquidity="1M USD",
    mean_liquidity="1M USD",
    mean_reversion_rate="0.2 / day",
    volatility=0.15,
    min_liquidity="100000 USD",  # Floor
    max_liquidity="5M USD"        # Ceiling
)

adapter = LiquidityAdapter(config, seed=42)

# Normal operation
print("Normal operation:")
for i in range(10):
    liq = adapter.step(dt=0.1)
    print(f"  t={i*0.1:.1f}: ${liq:,.0f}")

# Large liquidity provider enters
print("\nLP adds 500k:")
adapter.apply_shock(amount=500000.0)
state = adapter.get_state()
print(f"  New liquidity: ${state['liquidity_level']:,.0f}")

# Continue evolution
print("\nPost-shock evolution:")
for i in range(10):
    liq = adapter.step(dt=0.1)
    print(f"  t={i*0.1:.1f}: ${liq:,.0f}")

# Large LP exits
print("\nLP removes 700k:")
adapter.apply_shock(amount=-700000.0)
state = adapter.get_state()
print(f"  New liquidity: ${state['liquidity_level']:,.0f}")
print(f"  Cumulative provision: ${state['cumulative_provision']:,.0f}")
print(f"  Cumulative removal: ${state['cumulative_removal']:,.0f}")
```

### Example 24: Mean Reversion Analysis

```python
from ethode import LiquidityAdapter, LiquiditySDEConfig
import numpy as np
import matplotlib.pyplot as plt

# Strong mean reversion
config = LiquiditySDEConfig(
    initial_liquidity="200000 USD",   # Start far from mean
    mean_liquidity="1M USD",
    mean_reversion_rate="0.5 / day",  # Fast reversion
    volatility=0.1                     # Low noise
)

adapter = LiquidityAdapter(config, seed=42)

# Track evolution
times = []
liquidity_levels = []

for t in range(200):
    dt = 0.1
    liq = adapter.step(dt=dt)
    times.append(t * dt)
    liquidity_levels.append(liq)

# Analyze convergence
mean_liq = 1000000.0
distances = [abs(liq - mean_liq) for liq in liquidity_levels]

print(f"Initial distance from mean: ${distances[0]:,.0f}")
print(f"Final distance from mean: ${distances[-1]:,.0f}")
print(f"Reduction: {(distances[0] - distances[-1]) / distances[0] * 100:.1f}%")

# Plot
try:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(times, np.array(liquidity_levels) / 1e6, label='Liquidity')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Mean ($1M)')
    plt.xlabel('Time (days)')
    plt.ylabel('Liquidity ($M)')
    plt.title('Liquidity Evolution (Mean Reversion)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(times, np.array(distances) / 1000, label='Distance from Mean')
    plt.xlabel('Time (days)')
    plt.ylabel('Distance ($k)')
    plt.title('Convergence to Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('liquidity_mean_reversion.png', dpi=150)
    print("\nPlot saved to liquidity_mean_reversion.png")
except ImportError:
    print("\nInstall matplotlib for visualization")
```

### Example 25: Volatility and Uncertainty

```python
from ethode import LiquidityAdapter, LiquiditySDEConfig
import numpy as np

# Run multiple simulations with different volatilities
volatilities = [0.05, 0.15, 0.30]
n_paths = 50
n_steps = 100

results = {}

for vol in volatilities:
    config = LiquiditySDEConfig(
        initial_liquidity="1M USD",
        mean_liquidity="1M USD",
        mean_reversion_rate="0.1 / day",
        volatility=vol
    )

    paths = []
    for seed in range(n_paths):
        adapter = LiquidityAdapter(config, seed=seed)
        path = [adapter.step(dt=0.1) for _ in range(n_steps)]
        paths.append(path[-1])  # Final value

    results[vol] = {
        'mean': np.mean(paths),
        'std': np.std(paths),
        'min': np.min(paths),
        'max': np.max(paths),
        'paths': paths
    }

print("Volatility Analysis (after 10 days):")
print("-" * 70)
for vol in volatilities:
    r = results[vol]
    print(f"\nVolatility = {vol:.2f}:")
    print(f"  Mean:   ${r['mean']:,.0f}")
    print(f"  StdDev: ${r['std']:,.0f}")
    print(f"  Range:  [${r['min']:,.0f}, ${r['max']:,.0f}]")
    print(f"  Coeff of Variation: {r['std'] / r['mean'] * 100:.1f}%")

# Visualize distribution
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, vol in enumerate(volatilities):
        ax = axes[i]
        ax.hist(np.array(results[vol]['paths']) / 1e6, bins=20,
                edgecolor='black', alpha=0.7)
        ax.axvline(x=1.0, color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Final Liquidity ($M)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Volatility = {vol:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('liquidity_volatility_comparison.png', dpi=150)
    print("\nPlot saved to liquidity_volatility_comparison.png")
except ImportError:
    pass
```

### Example 26: Direct JAX Access for Liquidity

```python
from ethode import LiquidityAdapter, LiquiditySDEConfig
from ethode.liquidity.kernel import update_liquidity
import jax
import jax.numpy as jnp

# Create adapter
config = LiquiditySDEConfig(
    initial_liquidity="1M USD",
    mean_liquidity="1M USD",
    mean_reversion_rate="0.1 / day",
    volatility=0.2
)
adapter = LiquidityAdapter(config, seed=42)

# Extract JAX-ready structures for power users
runtime = adapter.runtime
initial_state = adapter.state

# Create JIT-compiled liquidity function
@jax.jit
def fast_liquidity_update(state, key, dt):
    return update_liquidity(runtime, state, key, dt)

# Generate batch of PRNGkeys
master_key = jax.random.PRNGKey(42)
keys = jax.random.split(master_key, 100)
dt = jnp.array(0.1)

# Use scan for efficient sequential updates
def liquidity_step(state, key):
    new_state, liquidity = fast_liquidity_update(state, key, dt)
    return new_state, liquidity

final_state, liquidity_path = jax.lax.scan(liquidity_step, initial_state, keys)

print(f"Simulated {len(liquidity_path)} steps")
print(f"Initial liquidity: ${float(liquidity_path[0]):,.0f}")
print(f"Final liquidity: ${float(liquidity_path[-1]):,.0f}")
print(f"Min liquidity: ${float(jnp.min(liquidity_path)):,.0f}")
print(f"Max liquidity: ${float(jnp.max(liquidity_path)):,.0f}")
print(f"Mean liquidity: ${float(jnp.mean(liquidity_path)):,.0f}")
print(f"Std liquidity: ${float(jnp.std(liquidity_path)):,.0f}")
```

### Example 27: Comparing Deterministic vs Stochastic

```python
from ethode import LiquidityAdapter, LiquiditySDEConfig
from ethode.liquidity.kernel import deterministic_update
import jax.numpy as jnp

# Deterministic version (no noise)
config_det = LiquiditySDEConfig(
    initial_liquidity="500000 USD",
    mean_liquidity="1M USD",
    mean_reversion_rate="0.3 / day",
    volatility=0.0  # No volatility
)

# Stochastic version (with noise)
config_stoch = LiquiditySDEConfig(
    initial_liquidity="500000 USD",
    mean_liquidity="1M USD",
    mean_reversion_rate="0.3 / day",
    volatility=0.2
)

# Run deterministic
adapter_det = LiquidityAdapter(config_det, seed=42)
det_path = []
for _ in range(100):
    liq = adapter_det.step(dt=0.1)
    det_path.append(liq)

# Run stochastic (multiple realizations)
stoch_paths = []
for seed in range(10):
    adapter_stoch = LiquidityAdapter(config_stoch, seed=seed)
    path = [adapter_stoch.step(dt=0.1) for _ in range(100)]
    stoch_paths.append(path)

print("Comparison after 10 days:")
print(f"Deterministic endpoint: ${det_path[-1]:,.0f}")
print(f"\nStochastic endpoints (10 paths):")
for i, path in enumerate(stoch_paths):
    print(f"  Path {i+1}: ${path[-1]:,.0f}")

import numpy as np
stoch_final = [path[-1] for path in stoch_paths]
print(f"\nStochastic statistics:")
print(f"  Mean: ${np.mean(stoch_final):,.0f}")
print(f"  Std: ${np.std(stoch_final):,.0f}")
print(f"  Deterministic is within {abs(det_path[-1] - np.mean(stoch_final)) / np.std(stoch_final):.2f} std devs of mean")

# Plot if available
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot all stochastic paths
    times = [i * 0.1 for i in range(100)]
    for path in stoch_paths:
        plt.plot(times, np.array(path) / 1e6, 'b-', alpha=0.3, linewidth=0.5)

    # Plot deterministic path
    plt.plot(times, np.array(det_path) / 1e6, 'r-', linewidth=2, label='Deterministic')

    # Plot mean of stochastic paths
    mean_stoch = np.mean(stoch_paths, axis=0)
    plt.plot(times, mean_stoch / 1e6, 'g--', linewidth=2, label='Stochastic Mean')

    plt.axhline(y=1.0, color='k', linestyle=':', label='Target Mean')
    plt.xlabel('Time (days)')
    plt.ylabel('Liquidity ($M)')
    plt.title('Deterministic vs Stochastic Liquidity Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('liquidity_deterministic_vs_stochastic.png', dpi=150)
    print("\nPlot saved to liquidity_deterministic_vs_stochastic.png")
except ImportError:
    pass
```

---

## Hawkes Process Subsystem

### Example 28: Basic Self-Exciting Events

```python
from ethode import HawkesAdapter, HawkesConfig

# Configure Hawkes process
# λ(t) = λ₀ + Σ α * exp(-(t - tᵢ) / τ)
config = HawkesConfig(
    jump_rate="100 / hour",          # Base event rate
    excitation_strength=0.3,          # Self-excitation (< 1 for stability)
    excitation_decay="5 minutes"     # Decay time constant
)

# Create adapter
adapter = HawkesAdapter(config, seed=42)

# Check stability
print(f"Branching ratio: {adapter.get_branching_ratio():.3f}")
print(f"Stationary intensity: {adapter.get_stationary_intensity():.4f} events/sec")

# Simulate events
events = []
for t in range(1000):
    dt = 0.01  # Small timestep
    event_occurred = adapter.step(dt=dt)
    if event_occurred:
        events.append(t * dt)

print(f"\nTotal events: {len(events)}")
print(f"First 10 event times: {events[:10]}")
print(f"Current intensity: {adapter.get_intensity():.6f} events/sec")
```

### Example 29: Event Clustering Analysis

```python
from ethode import HawkesAdapter, HawkesConfig
import numpy as np

# High excitation creates clustered events
config = HawkesConfig(
    jump_rate="50 / hour",
    excitation_strength=0.7,  # Strong self-excitation
    excitation_decay="2 minutes"
)

adapter = HawkesAdapter(config, seed=42)

# Simulate and record event times
event_times = []
intensity_history = []

for t in range(10000):
    dt = 0.01
    event = adapter.step(dt=dt)
    intensity_history.append(adapter.get_intensity())

    if event:
        event_times.append(t * dt)

# Analyze inter-event times
if len(event_times) > 1:
    inter_event_times = np.diff(event_times)
    print(f"Total events: {len(event_times)}")
    print(f"Mean inter-event time: {np.mean(inter_event_times):.2f}s")
    print(f"Std inter-event time: {np.std(inter_event_times):.2f}s")
    print(f"Min inter-event time: {np.min(inter_event_times):.2f}s")
    print(f"Max inter-event time: {np.max(inter_event_times):.2f}s")

    # Short inter-event times indicate clustering
    clustered_events = sum(1 for iet in inter_event_times if iet < 5.0)
    print(f"Clustered events (< 5s apart): {clustered_events} ({clustered_events/len(inter_event_times)*100:.1f}%)")

# Plot if available
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Intensity evolution
    times = np.arange(len(intensity_history)) * 0.01
    ax1 = axes[0]
    ax1.plot(times, intensity_history, linewidth=0.5)
    for et in event_times:
        ax1.axvline(x=et, color='r', alpha=0.3, linewidth=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Intensity (events/s)')
    ax1.set_title('Hawkes Process Intensity (red lines = events)')
    ax1.grid(True, alpha=0.3)

    # Inter-event time distribution
    ax2 = axes[1]
    if len(inter_event_times) > 0:
        ax2.hist(inter_event_times, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(inter_event_times), color='r',
                   linestyle='--', label=f'Mean = {np.mean(inter_event_times):.2f}s')
        ax2.set_xlabel('Inter-Event Time (s)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Inter-Event Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hawkes_clustering.png', dpi=150)
    print("\nPlot saved to hawkes_clustering.png")
except ImportError:
    print("\nInstall matplotlib for visualization")
```

### Example 30: External Shocks (News/Market Events)

```python
from ethode import HawkesAdapter, HawkesConfig

config = HawkesConfig(
    jump_rate="100 / hour",
    excitation_strength=0.4,
    excitation_decay="3 minutes"
)

adapter = HawkesAdapter(config, seed=42)

# Normal operation
print("Normal trading:")
events_normal = []
for t in range(500):
    event = adapter.step(dt=0.01)
    if event:
        events_normal.append(t * 0.01)

print(f"Events in normal period: {len(events_normal)}")
print(f"Intensity before shock: {adapter.get_intensity():.6f}")

# News announcement causes intensity spike
print("\nNEWS SHOCK!")
adapter.apply_shock(intensity_boost=2.0)  # Major news
print(f"Intensity after shock: {adapter.get_intensity():.6f}")

# Trading activity increases temporarily
events_after_shock = []
for t in range(500):
    event = adapter.step(dt=0.01)
    if event:
        events_after_shock.append(t * 0.01)

print(f"Events after shock: {len(events_after_shock)}")
print(f"Increase: {len(events_after_shock) / max(len(events_normal), 1):.2f}x")

# Intensity gradually decays
print("\nIntensity decay:")
for i in range(10):
    # Run some steps
    for _ in range(50):
        adapter.step(dt=0.01)
    print(f"  After {(i+1)*50*0.01:.1f}s: {adapter.get_intensity():.6f}")
```

### Example 31: Comparing Different Excitation Levels

```python
from ethode import HawkesAdapter, HawkesConfig
import numpy as np

# Test different excitation strengths
excitations = [0.1, 0.3, 0.5, 0.8]
n_steps = 5000
dt = 0.01

results = {}

for alpha in excitations:
    config = HawkesConfig(
        jump_rate="200 / hour",
        excitation_strength=alpha,
        excitation_decay="2 minutes"
    )

    adapter = HawkesAdapter(config, seed=42)

    # Record events and intensity
    events = []
    intensities = []

    for t in range(n_steps):
        event = adapter.step(dt=dt)
        intensities.append(adapter.get_intensity())
        if event:
            events.append(t * dt)

    results[alpha] = {
        'event_count': len(events),
        'mean_intensity': np.mean(intensities),
        'max_intensity': np.max(intensities),
        'branching_ratio': alpha,
        'stationary': adapter.config.jump_rate[0] / 3600 / (1 - alpha),  # Theoretical
    }

print("Excitation Strength Analysis:")
print("=" * 70)
for alpha, res in results.items():
    print(f"\nExcitation α = {alpha:.1f}:")
    print(f"  Branching ratio: {res['branching_ratio']:.2f}")
    print(f"  Event count: {res['event_count']}")
    print(f"  Mean intensity: {res['mean_intensity']:.6f} events/s")
    print(f"  Max intensity: {res['max_intensity']:.6f} events/s")
    print(f"  Theoretical stationary: {res['stationary']:.6f} events/s")

# Visualize
try:
    import matplotlib.pyplot as plt

    alphas = list(results.keys())
    event_counts = [results[a]['event_count'] for a in alphas]
    mean_intensities = [results[a]['mean_intensity'] for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    ax1.bar(range(len(alphas)), event_counts, edgecolor='black')
    ax1.set_xticks(range(len(alphas)))
    ax1.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax1.set_xlabel('Excitation Strength (α)')
    ax1.set_ylabel('Total Events')
    ax1.set_title('Event Count vs Excitation')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    ax2.bar(range(len(alphas)), mean_intensities, color='orange', edgecolor='black')
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax2.set_xlabel('Excitation Strength (α)')
    ax2.set_ylabel('Mean Intensity (events/s)')
    ax2.set_title('Mean Intensity vs Excitation')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('hawkes_excitation_comparison.png', dpi=150)
    print("\nPlot saved to hawkes_excitation_comparison.png")
except ImportError:
    pass
```

### Example 32: Direct JAX Access for Hawkes

```python
from ethode import HawkesAdapter, HawkesConfig
from ethode.hawkes.kernel import generate_event
import jax
import jax.numpy as jnp

# Create adapter
config = HawkesConfig(
    jump_rate="100 / hour",
    excitation_strength=0.3,
    excitation_decay="5 minutes"
)
adapter = HawkesAdapter(config, seed=42)

# Extract JAX-ready structures
runtime = adapter.runtime
initial_state = adapter.state

# Create JIT-compiled event generation
@jax.jit
def fast_event_generation(state, key, dt):
    return generate_event(runtime, state, key, dt)

# Generate batch of PRNG keys
master_key = jax.random.PRNGKey(42)
keys = jax.random.split(master_key, 1000)
dt = jnp.array(0.01)

# Use scan for efficient sequential simulation
def hawkes_step(carry, key):
    state = carry
    new_state, event_occurred, impact = fast_event_generation(state, key, dt)
    return new_state, (event_occurred, new_state.current_intensity)

final_state, (events, intensities) = jax.lax.scan(hawkes_step, initial_state, keys)

print(f"Simulated {len(events)} steps")
print(f"Total events: {int(jnp.sum(events))}")
print(f"Event rate: {float(jnp.mean(events)) / dt:.2f} events/sec")
print(f"Final intensity: {float(intensities[-1]):.6f}")
print(f"Mean intensity: {float(jnp.mean(intensities)):.6f}")
print(f"Max intensity: {float(jnp.max(intensities)):.6f}")
```

### Example 33: Stability Analysis

```python
from ethode import HawkesAdapter, HawkesConfig

# Subcritical (stable) process
config_stable = HawkesConfig(
    jump_rate="100 / hour",
    excitation_strength=0.3,  # < 1
    excitation_decay="5 minutes"
)

# Near-critical process
config_critical = HawkesConfig(
    jump_rate="100 / hour",
    excitation_strength=0.95,  # Close to 1
    excitation_decay="5 minutes"
)

adapters = {
    'Subcritical (α=0.3)': HawkesAdapter(config_stable, seed=42),
    'Near-critical (α=0.95)': HawkesAdapter(config_critical, seed=42),
}

for name, adapter in adapters.items():
    print(f"\n{name}:")
    print(f"  Branching ratio: {adapter.get_branching_ratio():.3f}")
    print(f"  Stationary intensity: {adapter.get_stationary_intensity():.6f} events/s")

    # Simulate
    events = []
    for _ in range(2000):
        event = adapter.step(dt=0.01)
        if event:
            events.append(1)

    print(f"  Events in simulation: {len(events)}")
    print(f"  Current intensity: {adapter.get_intensity():.6f}")

print("\nNote: Near-critical processes have:")
print("  - Much higher event rates")
print("  - Longer-lasting clustering")
print("  - Greater sensitivity to perturbations")
print("  - α >= 1 would be unstable (explosive)")
```

---

## Multi-Subsystem Integration

The `Simulation` class orchestrates multiple subsystems together, handling dependencies and execution order automatically.

### Example 34: Basic Multi-Subsystem Simulation

```python
from ethode import (
    Simulation,
    ControllerAdapter, ControllerConfig,
    FeeAdapter, FeeConfig,
    LiquidityAdapter, LiquiditySDEConfig,
)

# Configure subsystems
controller_config = ControllerConfig(
    kp="0.2 / day",
    ki="0.02 / day**2",
    kd=0.0,
    tau="7 day",
    noise_band=("0.001 USD", "0.003 USD")
)

fee_config = FeeConfig(
    base_fee_rate="50 bps",      # 0.5%
    max_fee_rate="200 bps",      # 2%
    fee_decay_time="1 day"
)

liquidity_config = LiquiditySDEConfig(
    initial_liquidity="1M USD",
    mean_liquidity="1M USD",
    mean_reversion_rate="0.1 / day",
    volatility=0.15
)

# Create simulation with all subsystems
sim = Simulation(
    controller=ControllerAdapter(controller_config),
    fee=FeeAdapter(fee_config),
    liquidity=LiquidityAdapter(liquidity_config, seed=42)
)

# Run simulation
for t in range(100):
    inputs = {
        'error': 1.0 / (t + 1),         # Decreasing error
        'market_volatility': 0.3 + 0.2 * (t / 100),  # Increasing volatility
        'volume_ratio': 1.0 + 0.5 * (t / 100)
    }

    outputs = sim.step(inputs, dt=0.1)

    if t % 20 == 0:
        print(f"t={t}:")
        print(f"  Control: {outputs['control']:.6f}")
        print(f"  Fee: {outputs['fee']:.6f} USD")
        print(f"  Liquidity: {outputs['liquidity']:.2f} USD")

# Get final states
states = sim.get_state()
print(f"\nFinal accumulated fees: {states['fee']['accumulated_fees']:.2f} USD")
```

### Example 35: Controller + Hawkes for Event-Driven Control

```python
from ethode import (
    Simulation,
    ControllerAdapter, ControllerConfig,
    HawkesAdapter, HawkesConfig,
)

# Controller for gradual adjustments
controller_config = ControllerConfig(
    kp="0.5 / day",
    ki="0.05 / day**2",
    kd=0.0,
    tau="7 day",
    noise_band=("0.001 USD", "0.01 USD")
)

# Hawkes process for event modeling
hawkes_config = HawkesConfig(
    jump_rate="50 / hour",
    excitation_strength=0.6,
    excitation_decay="10 minutes"
)

sim = Simulation(
    controller=ControllerAdapter(controller_config),
    hawkes=HawkesAdapter(hawkes_config, seed=123)
)

# Simulate with event tracking
event_times = []
control_at_events = []

for t in range(500):
    error = 2.0 - 1.8 * (t / 500)  # Error decreasing from 2.0 to 0.2

    outputs = sim.step({'error': error}, dt=0.01)

    if outputs['event_occurred']:
        event_times.append(t * 0.01)
        control_at_events.append(outputs['control'])

print(f"Total events: {len(event_times)}")
print(f"Event clustering: {adapter.get_intensity():.6f} events/s")
print(f"Avg control at events: {sum(control_at_events)/len(control_at_events):.6f}")
```

### Example 36: Full System with Matplotlib Visualization

```python
from ethode import (
    Simulation,
    ControllerAdapter, ControllerConfig,
    FeeAdapter, FeeConfig,
    LiquidityAdapter, LiquiditySDEConfig,
    HawkesAdapter, HawkesConfig,
)
import matplotlib.pyplot as plt

# Configure all subsystems
sim = Simulation(
    controller=ControllerAdapter(ControllerConfig(
        kp="0.3 / day", ki="0.03 / day**2", kd=0.0,
        tau="7 day", noise_band=("0.001 USD", "0.005 USD")
    )),
    fee=FeeAdapter(FeeConfig(
        base_fee_rate="25 bps",
        max_fee_rate="150 bps",
        fee_decay_time="12 hours"
    )),
    liquidity=LiquidityAdapter(LiquiditySDEConfig(
        initial_liquidity="1M USD",
        mean_liquidity="1M USD",
        mean_reversion_rate="0.2 / day",
        volatility=0.1
    ), seed=42),
    hawkes=HawkesAdapter(HawkesConfig(
        jump_rate="100 / hour",
        excitation_strength=0.5,
        excitation_decay="5 minutes"
    ), seed=123)
)

# Run simulation and collect data
n_steps = 1000
times = []
errors = []
controls = []
fees = []
liquidities = []
events = []

for i in range(n_steps):
    t = i * 0.1
    error = 2.0 * (1.0 - t / 100.0) if t < 100 else 0.0  # Step response
    volatility = 0.2 + 0.3 * (i / n_steps)  # Increasing volatility

    outputs = sim.step({
        'error': error,
        'market_volatility': volatility,
        'volume_ratio': 1.0 + 0.5 * volatility
    }, dt=0.1)

    times.append(t)
    errors.append(error)
    controls.append(outputs['control'])
    fees.append(outputs['fee'])
    liquidities.append(outputs['liquidity'])
    events.append(1 if outputs['event_occurred'] else 0)

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(times, errors, label='Error', color='red', alpha=0.7)
axes[0].plot(times, controls, label='Control', color='blue')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].set_title('Controller Response')
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, fees, color='green')
axes[1].set_ylabel('Fee (USD)')
axes[1].set_title('Dynamic Fees')
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, liquidities, color='purple', alpha=0.7)
axes[2].set_ylabel('Liquidity (USD)')
axes[2].set_title('Stochastic Liquidity')
axes[2].grid(True, alpha=0.3)

event_times = [t for t, e in zip(times, events) if e == 1]
axes[3].vlines(event_times, 0, 1, colors='orange', alpha=0.5)
axes[3].set_ylabel('Events')
axes[3].set_xlabel('Time')
axes[3].set_title('Hawkes Events')
axes[3].set_ylim(-0.1, 1.1)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_subsystem_simulation.png', dpi=150)
print(f"Visualization saved to multi_subsystem_simulation.png")
print(f"\nSimulation Summary:")
print(f"  Total events: {sum(events)}")
print(f"  Total fees: {sum(fees):.2f} USD")
print(f"  Final liquidity: {liquidities[-1]:.2f} USD")
print(f"  Final control: {controls[-1]:.6f}")
```

### Example 37: Conditional Subsystems

```python
from ethode import Simulation, FeeAdapter, FeeConfig

# Simulation with only fee subsystem (no controller)
fee_only_sim = Simulation(
    fee=FeeAdapter(FeeConfig(
        base_fee_rate="100 bps",
        max_fee_rate="500 bps"
    ))
)

# Can specify transaction amount explicitly
outputs = fee_only_sim.step({
    'transaction_amount': 1000.0,
    'market_volatility': 0.4
}, dt=0.1)

print(f"Fee for $1000 transaction: ${outputs['fee']:.2f}")

# Only returns fee output (no control, liquidity, etc.)
assert 'fee' in outputs
assert 'control' not in outputs
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
7. **Fee Subsystem**: Dynamic fee calculations with stress-based adjustments
8. **Liquidity Subsystem**: Stochastic liquidity dynamics with mean reversion
9. **Hawkes Subsystem**: Self-exciting point processes for event clustering
10. **Multi-Subsystem**: Orchestrating multiple subsystems together

For more information, see:
- [PID_MIGRATION_GUIDE.md](../PID_MIGRATION_GUIDE.md) - Complete migration guide
- [test_adapters.py](../test_adapters.py) - Adapter test examples
- [test_fee_adapter.py](../test_fee_adapter.py) - Fee adapter test examples
- [test_liquidity_adapter.py](../test_liquidity_adapter.py) - Liquidity adapter test examples
- [test_hawkes_adapter.py](../test_hawkes_adapter.py) - Hawkes adapter test examples
- [test_simulation.py](../test_simulation.py) - Simulation test examples
