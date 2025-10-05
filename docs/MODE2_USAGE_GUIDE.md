# Mode 2 (Online Hawkes) Usage Guide

**Version**: 1.0
**Date**: 2025-10-06
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [When to Use Mode 2](#when-to-use-mode-2)
3. [Basic Usage](#basic-usage)
4. [Custom Callables](#custom-callables)
5. [State-Dependent Base Rate](#state-dependent-base-rate)
6. [Custom Excitation Kernels](#custom-excitation-kernels)
7. [Buffer Management](#buffer-management)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Mode 2 (Online Hawkes) generates Hawkes process events lazily during simulation using a cumulative excitation accumulator. This enables:

- **State-dependent base rates**: λ₀(y) depends on ODE state y(t)
- **Custom excitation kernels**: Non-exponential decay functions
- **Memory efficiency**: Fixed 100-element buffer vs pre-generating all events
- **Dynamic adaptation**: Event rate responds to state evolution

**Key Difference from Mode 1**: Mode 1 pre-generates all events at t=0 with constant λ₀. Mode 2 generates events on-demand and can use state-dependent λ₀(y).

---

## When to Use Mode 2

### Use Mode 2 When:
- ✅ You need state-dependent jump rates: λ₀(y) varies with ODE state
- ✅ You want custom excitation kernels (power-law, stretched exponential, etc.)
- ✅ You have long simulations where pre-generation is impractical
- ✅ Memory is constrained (fixed 100-event buffer)

### Use Mode 1 When:
- ✅ Jump rate is constant (state-independent)
- ✅ You need standard exponential Hawkes kernels
- ✅ Maximum speed is critical (pre-generation is faster)
- ✅ You want to inspect event times before simulation

---

## Basic Usage

### Minimal Example

```python
import jax.numpy as jnp
from ethode.jumpdiffusion.config import JumpDiffusionConfig
from ethode.hawkes.config import HawkesConfig
from ethode.jumpdiffusion.kernel import simulate

# Create Mode 2 configuration
config = JumpDiffusionConfig(
    initial_state=jnp.array([1.0]),
    dynamics_fn=lambda t, y, p: -0.1 * y,  # ODE dynamics
    jump_effect_fn=lambda t, y, p: y + 0.1,  # Jump effect
    jump_process=HawkesConfig(
        jump_rate="100 / hour",
        excitation_strength=0.3,
        excitation_decay="5 minute",
        seed=42
    ),
    hawkes_mode='online',  # ← Mode 2
    hawkes_max_events=500,
    solver='euler',
    dt_max="0.1 second",
    params={}
)

# Run simulation
runtime = config.to_runtime(check_units=True)
times, states = simulate(runtime, jnp.array([1.0]), (0.0, 600.0), max_steps=1000)

# Filter valid events
valid_mask = jnp.isfinite(times)
event_times = times[valid_mask]
event_states = states[valid_mask]
```

**Key Points**:
- Set `hawkes_mode='online'` for Mode 2
- `hawkes_max_events` controls thinning rejection limit (not buffer size)
- `hawkes_dt` is ignored (set to NaN internally)
- Uses default exponential decay and linear intensity

---

## Custom Callables

Mode 2 exposes four customization points:

1. **lambda_0_fn**: State-dependent base rate
2. **excitation_decay_fn**: Custom decay function
3. **excitation_jump_fn**: Excitation increment on jump
4. **intensity_fn**: Custom intensity formula

### Default Behavior

If not provided, Mode 2 uses:

```python
# Default lambda_0_fn: constant base rate
lambda_0_fn = lambda ode_state: jump_rate.value  # From HawkesConfig

# Default excitation_decay_fn: exponential decay
excitation_decay_fn = lambda E, dt, hawkes: E * jnp.exp(-hawkes.excitation_decay.value * dt)

# Default excitation_jump_fn: unit jump
excitation_jump_fn = lambda E, hawkes: E + 1.0

# Default intensity_fn: linear
intensity_fn = lambda lambda_0, E, hawkes: lambda_0 + hawkes.excitation_strength.value * E
```

These match standard Hawkes processes: λ(t) = λ₀ + α Σᵢ exp(-β(t-tᵢ))

---

## State-Dependent Base Rate

### Example: Jump Rate Increases with State

```python
def lambda_0_fn(ode_state):
    """Base rate depends on ODE state."""
    # Higher state → higher jump rate
    return 0.01 + 0.0005 * ode_state[0]

config = JumpDiffusionConfig(
    initial_state=jnp.array([10.0]),
    dynamics_fn=lambda t, y, p: jnp.array([0.02 * y[0]]),  # Exponential growth
    jump_effect_fn=lambda t, y, p: jnp.array([y[0] + 0.5]),
    jump_process=HawkesConfig(
        jump_rate="20 / hour",  # Fallback (overridden by lambda_0_fn)
        excitation_strength=0.2,
        excitation_decay="5 minute",
        seed=42
    ),
    hawkes_mode='online',
    lambda_0_fn=lambda_0_fn,  # ← Custom state-dependent rate
    hawkes_max_events=500,
    solver='euler',
    dt_max="0.1 second",
    params={}
)
```

**Use Cases**:
- Portfolio liquidation: jump rate increases as inventory depletes
- Epidemic models: infection rate depends on susceptible population
- Financial markets: volatility-dependent jump intensity

**Important**: Keep state dependence gentle to avoid buffer overflow. Very high λ₀(y) can exhaust the 100-element buffer.

---

## Custom Excitation Kernels

### Power-Law Decay

```python
def powerlaw_decay(E, dt, hawkes):
    """Power-law decay: E(t+dt) = E(t) / (1 + dt)^gamma."""
    gamma = 0.5
    return E / jnp.power(1.0 + dt, gamma)

def powerlaw_jump(E, hawkes):
    """Add excitation proportional to alpha."""
    alpha = hawkes.excitation_strength.value
    return E + alpha

config = JumpDiffusionConfig(
    initial_state=jnp.array([1.0]),
    dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
    jump_effect_fn=lambda t, y, p: y,
    jump_process=HawkesConfig(
        jump_rate="100 / hour",
        excitation_strength=0.3,
        excitation_decay="5 minute",  # Decay param (used differently)
        seed=42
    ),
    hawkes_mode='online',
    excitation_decay_fn=powerlaw_decay,  # ← Custom decay
    excitation_jump_fn=powerlaw_jump,
    hawkes_max_events=500,
    solver='euler',
    dt_max="0.1 second",
    params={}
)
```

### Stretched Exponential

```python
def stretched_exp_decay(E, dt, hawkes):
    """Stretched exponential: E(t+dt) = E(t) * exp(-(beta*dt)^delta)."""
    beta = hawkes.excitation_decay.value
    delta = 0.7  # Stretching exponent
    return E * jnp.exp(-jnp.power(beta * dt, delta))
```

### Pytree Support

Cumulative excitation `E` can be a pytree (dict, tuple, etc.) for complex kernels:

```python
# E = {"value": 5.0, "elapsed_time": 10.0}

def complex_decay(E, dt, hawkes):
    """Track both excitation and time since last event."""
    return {
        "value": E["value"] * jnp.exp(-hawkes.excitation_decay.value * dt),
        "elapsed_time": E["elapsed_time"] + dt  # Could be used for time-dependent effects
    }
```

Default functions handle both scalar and dict `E`.

---

## Buffer Management

### Buffer Size

Mode 2 uses a **fixed 100-element buffer** for lazy event generation. This is sufficient for most simulations, but can overflow if:

- Jump rate is extremely high
- Excitation is very strong (α → 1)
- State-dependent λ₀(y) grows rapidly

### Buffer Overflow Protection

If the buffer is exhausted, a clear error is raised:

```
RuntimeError: Mode 2 buffer overflow: Simulation exhausted 100-element buffer.
This indicates extremely rapid jumps or pathological simulation.
Possible causes: (1) Intensity too high, (2) Stuck in loop, (3) Bug in kernel.
Consider: Reducing excitation_strength, checking lambda_0_fn, or using Mode 1.
```

### Avoiding Overflow

1. **Reduce excitation**: Lower `excitation_strength` (α < 0.5 recommended)
2. **Check lambda_0_fn**: Ensure state-dependent rate doesn't explode
3. **Shorter simulations**: Break long runs into segments
4. **Use Mode 1**: Pre-generation doesn't have buffer limits

### Example: Safe Parameters

```python
# Safe for 30+ minute simulations
config = JumpDiffusionConfig(
    jump_process=HawkesConfig(
        jump_rate="100 / hour",  # ~1.67/min
        excitation_strength=0.3,  # Stationary λ∞ = 100/(1-0.3) = 143/hour
        excitation_decay="5 minute",
        seed=42
    ),
    hawkes_mode='online',
    # ...
)
```

```python
# Risk of overflow (too aggressive)
config = JumpDiffusionConfig(
    jump_process=HawkesConfig(
        jump_rate="1000 / hour",  # Very high
        excitation_strength=0.8,  # Near-critical
        excitation_decay="1 minute",
        seed=42
    ),
    hawkes_mode='online',
    # ← May exhaust buffer quickly
)
```

---

## Performance Considerations

### JIT Compilation

Mode 2 is fully JIT-compatible:

```python
import jax

runtime = config.to_runtime(check_units=True)

# JIT compile the simulation
simulate_jit = jax.jit(
    lambda: simulate(runtime, jnp.array([1.0]), (0.0, 600.0), max_steps=1000)
)

times, states = simulate_jit()
```

**First call** will compile (slow), subsequent calls are fast.

### Speed vs Mode 1

- **Mode 1**: Pre-generation fast, good for constant λ₀
- **Mode 2**: Lazy generation, overhead from online thinning
- **Crossover**: Mode 2 becomes competitive for very long simulations where pre-generation is expensive

### Custom Callables Overhead

- Default callables are optimized
- Custom callables add overhead (especially if complex)
- JIT compilation helps, but complex Python logic may slow down

---

## Troubleshooting

### Problem: Buffer Overflow

**Symptom**: `RuntimeError: Mode 2 buffer overflow: Simulation exhausted 100-element buffer`

**Solutions**:
1. Reduce `excitation_strength` (try α ≤ 0.3)
2. Check `lambda_0_fn` doesn't grow too fast
3. Use shorter time spans
4. Switch to Mode 1

### Problem: Events Not Matching Expected Rate

**Symptom**: Too few or too many events compared to theoretical rate

**Diagnosis**:
```python
# Expected stationary rate (if lambda_0 constant):
lambda_0 = 100  # events/hour
alpha = 0.3
lambda_inf = lambda_0 / (1 - alpha)  # = 143 events/hour

# Count actual events
valid_times = times[jnp.isfinite(times)]
event_count = len([t for t in valid_times if 0 < t < 3600.0])
print(f"Expected: ~{lambda_inf}, Actual: {event_count}")
```

**Common Causes**:
- State-dependent λ₀ changing the rate
- Custom intensity function altering dynamics
- Buffer overflow truncating events
- Simulation ending before equilibrium

### Problem: Different Results from Mode 1

**Symptom**: Mode 1 and Mode 2 produce different event counts with same parameters

**Expected Behavior**:
- Same seed + state-independent λ₀ → statistically similar (not identical)
- Different algorithms → some variation in event sequences
- Should match within 30-40% on long runs

**Not a Bug**: Mode 1 uses jax.lax.scan with discretized time, Mode 2 uses jax.lax.while_loop with continuous-time thinning. Both are correct Hawkes implementations.

### Problem: JIT Compilation Errors

**Symptom**: `ConcretizationTypeError` or tracing issues

**Solutions**:
- Ensure all custom callables use JAX operations (jnp, not np)
- Avoid Python control flow (if/for) inside callables
- Use `jax.lax.cond` instead of Python `if`
- Check that custom functions don't use unsupported operations

---

## Complete Working Examples

### Example 1: State-Dependent Redemption Rate

```python
"""Portfolio liquidation with state-dependent redemption rate."""

import jax.numpy as jnp
from ethode.jumpdiffusion.config import JumpDiffusionConfig
from ethode.hawkes.config import HawkesConfig
from ethode.jumpdiffusion.kernel import simulate

def redemption_rate(inventory):
    """Redemption rate increases as inventory depletes."""
    # inventory[0] starts at 1000, decreases toward 0
    # Rate increases from 10/hr to 50/hr as inventory drops
    return 0.003 + 0.00005 * (1000.0 - inventory[0])

config = JumpDiffusionConfig(
    initial_state=jnp.array([1000.0]),  # Initial inventory
    dynamics_fn=lambda t, y, p: jnp.array([0.0]),  # No drift
    jump_effect_fn=lambda t, y, p: jnp.array([y[0] - 10.0]),  # Each redemption = 10 units
    jump_process=HawkesConfig(
        jump_rate="10 / hour",  # Fallback
        excitation_strength=0.2,  # Clustering
        excitation_decay="30 minute",
        seed=42
    ),
    hawkes_mode='online',
    lambda_0_fn=redemption_rate,
    hawkes_max_events=500,
    solver='euler',
    dt_max="1 second",
    params={}
)

runtime = config.to_runtime(check_units=True)
times, states = simulate(runtime, jnp.array([1000.0]), (0.0, 3600.0), max_steps=2000)

# Analyze inventory depletion
valid_mask = jnp.isfinite(times)
valid_states = states[valid_mask]
print(f"Final inventory: {valid_states[-1, 0]:.1f}")
```

### Example 2: Custom Non-Linear Intensity

```python
"""Hawkes with quadratic intensity (super-linear excitation)."""

import jax.numpy as jnp
from ethode.jumpdiffusion.config import JumpDiffusionConfig
from ethode.hawkes.config import HawkesConfig
from ethode.jumpdiffusion.kernel import simulate

def quadratic_intensity(lambda_0, E, hawkes):
    """Non-linear intensity: λ = λ₀ + α * E^2."""
    alpha = hawkes.excitation_strength.value
    E_value = E if not isinstance(E, dict) else E["value"]
    return lambda_0 + alpha * jnp.square(E_value)

config = JumpDiffusionConfig(
    initial_state=jnp.array([1.0]),
    dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
    jump_effect_fn=lambda t, y, p: y,
    jump_process=HawkesConfig(
        jump_rate="50 / hour",
        excitation_strength=0.05,  # Lower for stability with E^2
        excitation_decay="5 minute",
        seed=42
    ),
    hawkes_mode='online',
    intensity_fn=quadratic_intensity,  # ← Custom intensity
    hawkes_max_events=500,
    solver='euler',
    dt_max="0.1 second",
    params={}
)

runtime = config.to_runtime(check_units=True)
times, states = simulate(runtime, jnp.array([1.0]), (0.0, 1800.0), max_steps=2000)
```

---

## Summary

Mode 2 (Online Hawkes) provides flexible, state-dependent jump processes with custom excitation kernels. Key takeaways:

✅ **Use Mode 2 for**: State-dependent rates, custom kernels, memory efficiency
✅ **JIT-compatible**: Full support for JAX compilation
✅ **Buffer management**: 100-element limit, overflow protection
✅ **Performance**: Comparable to Mode 1 for many use cases
✅ **Customization**: Four extensibility points (λ₀, decay, jump, intensity)

For simple constant-rate Hawkes, Mode 1 may be faster. For complex state-feedback or custom kernels, Mode 2 is essential.
