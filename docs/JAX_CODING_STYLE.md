# JAX Coding Style Guide for ethode

This guide documents JAX-specific coding patterns and pitfalls discovered while implementing Mode 1 (Pre-generated Hawkes) processes. Following these guidelines ensures code is JIT-compilable, efficient, and compatible with JAX's functional programming model.

## Table of Contents

1. [Background: JAX JIT Compilation](#background-jax-jit-compilation)
2. [Critical Rules](#critical-rules)
3. [Common Pitfalls](#common-pitfalls)
4. [Solutions and Patterns](#solutions-and-patterns)
5. [Testing for JIT Compatibility](#testing-for-jit-compatibility)
6. [Quick Reference](#quick-reference)

---

## Background: JAX JIT Compilation

### What is JIT?

JAX's Just-In-Time (JIT) compilation compiles Python functions to optimized XLA code for CPU/GPU/TPU execution. This happens in two phases:

1. **Tracing Phase**: JAX runs the function with abstract "tracer" values to build a computation graph
2. **Execution Phase**: The compiled graph executes with concrete values

### Key Insight: Tracers vs. Concrete Values

During JIT compilation, JAX values become **tracers** (placeholders representing "some array of shape X and dtype Y"). **You cannot use tracers where Python expects concrete values.**

```python
import jax
import jax.numpy as jnp

# ✓ This works - dt is concrete
def works():
    dt = 0.01
    n_steps = int(100.0 / dt)  # int(10000.0) → 10000
    return n_steps

# ✗ This crashes - dt is traced
@jax.jit
def crashes():
    dt = jnp.array(0.01)  # Becomes a tracer under JIT
    n_steps = int(100.0 / dt)  # ERROR: can't convert tracer to int!
    return n_steps

works()    # ✓ OK
crashes()  # ✗ ConcretizationTypeError
```

---

## Critical Rules

### Rule 1: Never Use Python Type Conversions on JAX Arrays Inside JIT

**❌ WRONG:**
```python
n = int(jnp.ceil(x / y))
result = bool(jax_condition)
text = str(jax_array)
length = len(jax_array)  # OK for shape, but not for data-dependent length
```

**✓ CORRECT:**
```python
n = jnp.ceil(x / y).astype(jnp.int32)  # Keep in JAX domain
result = jax_condition  # Already a JAX boolean array
# For str/debugging: use outside JIT or jax.debug.print()
```

**Why it fails:** Python's `int()`, `bool()`, `str()` are host-side operations that require concrete values. JAX tracers don't have concrete values during compilation.

**Real example from ethode:**
```python
# ethode/hawkes/kernel.py:133 - FIXED
# BEFORE (crashed in JIT):
return new_state, bool(event_occurred), event_impact

# AFTER (JIT-compatible):
return new_state, event_occurred, event_impact
# Return type changed from Tuple[HawkesState, bool, jax.Array]
#                     to Tuple[HawkesState, jax.Array, jax.Array]
```

### Rule 2: Control Flow Lengths Must Be Static

JAX control flow primitives (`jax.lax.scan`, `jax.lax.fori_loop`) require **compile-time constant** loop lengths.

**❌ WRONG:**
```python
@jax.jit
def bad_scan(t_span, dt):
    n_steps = jnp.ceil((t_span[1] - t_span[0]) / dt).astype(jnp.int32)
    # n_steps is traced → ERROR
    result, _ = jax.lax.scan(body_fn, init, None, length=n_steps)
    return result
```

**✓ CORRECT:**
```python
import math

@jax.jit
def good_scan(t_span, dt):
    # Use Python arithmetic to compute static length
    n_steps = int(math.ceil((t_span[1] - t_span[0]) / dt))
    result, _ = jax.lax.scan(body_fn, init, None, length=n_steps)
    return result
```

**Real example from ethode:**
```python
# ethode/hawkes/scheduler.py:48-60 - FIXED

# BEFORE (crashed in JIT):
dt_floor = 1e-6 * (t_end - t_start)  # Traced if t_span is traced
dt_clamped = jnp.maximum(dt, dt_floor)  # JAX operation → traced
n_steps = int(jnp.ceil((t_end - t_start) / dt_clamped))  # ERROR: traced → int

# AFTER (JIT-compatible):
import math
dt_floor = 1e-6 * (t_end - t_start)  # Python arithmetic → static
dt_clamped_static = max(dt, dt_floor)  # Python max() → static
n_steps = int(math.ceil((t_end - t_start) / dt_clamped_static))  # Static int

# Convert to JAX for use in scan body
dt_clamped = jnp.array(dt_clamped_static)
```

**When to use dynamic lengths:** If loop length depends on traced values, use `jax.lax.while_loop` instead of `scan`:

```python
# Dynamic stopping condition
def cond_fn(carry):
    val, _ = carry
    return val < threshold  # Can depend on traced values

def body_fn(carry):
    val, count = carry
    return (val * 2, count + 1)

result = jax.lax.while_loop(cond_fn, body_fn, (init_val, 0))
```

### Rule 3: Separate Static Configuration from Dynamic Computation

**Design Pattern: Extract Static Values Before JIT Boundary**

```python
# ✓ CORRECT: Extract floats before JIT
@jax.jit
def simulate(runtime, initial_state):
    # Extract static configuration values
    dt_value = float(runtime.hawkes_dt)  # Static Python float
    max_events = int(runtime.max_events)  # Static Python int

    # Now use in JAX operations
    n_steps = int(math.ceil(total_time / dt_value))
    result = jax.lax.scan(..., length=n_steps)
    return result
```

**Real example from ethode:**
```python
# ethode/jumpdiffusion/kernel.py:385-395 - FIXED

# BEFORE (dt was JAX array → traced in JIT):
events, _ = generate_schedule(
    runtime.scheduler.hawkes,
    t_span,
    runtime.scheduler.hawkes_dt,  # JAX array → traced
    ...
)

# AFTER (extract float first):
hawkes_dt_value = float(runtime.scheduler.hawkes_dt)  # Static
events, _ = generate_schedule(
    runtime.scheduler.hawkes,
    t_span,
    hawkes_dt_value,  # Python float → static
    ...
)
```

---

## Common Pitfalls

### Pitfall 1: Array Indexing with Traced Indices

**❌ WRONG:**
```python
# If idx is traced, this creates control flow dependency
value = array[idx] if idx < len(array) else default
```

**✓ CORRECT:**
```python
# Use JAX operations
value = jax.lax.cond(
    idx < len(array),
    lambda: array[idx],
    lambda: default
)
# Or use safe indexing with padding
```

### Pitfall 2: Shape-Dependent Logic

**❌ WRONG:**
```python
@jax.jit
def process(x):
    if x.shape[0] > 100:  # Shape is known at compile time, but this is bad style
        return jnp.mean(x)
    else:
        return jnp.sum(x)
```

**✓ CORRECT:**
```python
# Either:
# 1. Make shape a static argument
def process(x):
    return jnp.mean(x) if x.shape[0] > 100 else jnp.sum(x)

process = jax.jit(process, static_argnums=(0,))  # If shape varies

# Or:
# 2. Use JAX control flow (less efficient)
@jax.jit
def process(x):
    return jax.lax.cond(
        x.shape[0] > 100,
        lambda x: jnp.mean(x),
        lambda x: jnp.sum(x),
        x
    )
```

### Pitfall 3: Mixing Python and JAX Math

**⚠️ SUBTLE BUG:**
```python
import math
import jax.numpy as jnp

@jax.jit
def compute(x):
    # Python math.sqrt won't work on JAX arrays
    result = math.sqrt(x)  # ERROR: expects Python float
    return result
```

**✓ CORRECT:**
```python
@jax.jit
def compute(x):
    result = jnp.sqrt(x)  # Use JAX version
    return result
```

**But for static values:**
```python
import math

@jax.jit
def compute_with_static(x, static_dt):
    # OK: static_dt is Python float, use Python math
    n_steps = int(math.ceil(100.0 / static_dt))

    # For JAX arrays, use jnp
    scaled = x * jnp.sqrt(static_dt)
    return scaled
```

### Pitfall 4: Returning Python Types from JIT

**❌ WRONG:**
```python
@jax.jit
def check_threshold(x):
    # Attempting to return Python bool
    return bool(x > 0.5)  # ERROR in JIT
```

**✓ CORRECT:**
```python
@jax.jit
def check_threshold(x):
    # Return JAX array
    return x > 0.5  # Returns jax.Array with dtype bool
```

**Converting after JIT:**
```python
# ✓ Convert AFTER exiting JIT boundary
result_jax = check_threshold(x)
result_python = bool(result_jax)  # OK: outside JIT
```

---

## Solutions and Patterns

### Pattern 1: Static/Dynamic Value Separation

**Problem:** Function needs both compile-time constants and runtime arrays.

**Solution:** Use static arguments or extract before JIT boundary.

```python
# Method 1: static_argnums
@jax.jit(static_argnums=(1,))  # dt_static is static
def simulate(state, dt_static, t_span):
    n_steps = int(math.ceil((t_span[1] - t_span[0]) / dt_static))
    # ...

# Method 2: Extract before JIT
def simulate(runtime, state):
    dt_value = float(runtime.dt)  # Extract static value

    @jax.jit
    def _simulate_jit(state):
        # dt_value is captured as constant
        n_steps = int(math.ceil(100.0 / dt_value))
        # ...

    return _simulate_jit(state)
```

**Real example from ethode:**
```python
# We chose Method 2 for cleaner API
# User doesn't need to know about static_argnums

# ethode/jumpdiffusion/kernel.py
def simulate(runtime, initial_state, t_span, max_steps):
    # Extract static config BEFORE calling JIT-compiled functions
    hawkes_dt_value = float(runtime.scheduler.hawkes_dt)

    # Now pass to function that will be JIT-compiled
    events, _ = generate_schedule(
        runtime.scheduler.hawkes,
        t_span,
        hawkes_dt_value,  # Static Python float
        int(runtime.scheduler.hawkes_max_events),
        int(runtime.scheduler.seed),
        dtype=initial_state.dtype
    )
```

### Pattern 2: Return JAX Arrays, Convert Outside JIT

**Problem:** Need Python types for user API, but JIT requires JAX types.

**Solution:** Keep JIT functions pure (JAX in/out), convert in wrapper.

```python
# ✓ CORRECT layering
@jax.jit
def _simulate_jit(state, params):
    # Pure JAX computation
    new_state = update(state, params)
    success = new_state.value > 0
    return new_state, success  # JAX arrays

def simulate(state, params):
    """User-facing API."""
    new_state_jax, success_jax = _simulate_jit(state, params)

    # Convert to Python types for user
    return {
        'state': np.array(new_state_jax),
        'success': bool(success_jax),
    }
```

**Real example from ethode:**
```python
# ethode/hawkes/kernel.py:133
def generate_event(...) -> Tuple[HawkesState, jax.Array, jax.Array]:
    # ...
    return new_state, event_occurred, event_impact
    # All JAX types - JIT-compatible

# ethode/adapters.py:644
def step(self, dt: float) -> bool:
    """User-facing API with Python bool."""
    new_state, event_occurred, _ = generate_event(...)

    # Convert to Python bool for user
    return bool(event_occurred)
```

### Pattern 3: Compile-Time vs. Runtime Decisions

**Compile-time (static):**
- Configuration choices (solver type, mode)
- Array shapes/dimensions
- Loop lengths for `scan`/`fori_loop`

**Runtime (dynamic):**
- Array values
- Conditional results (via `lax.cond`, `lax.select`)
- Early stopping (via `while_loop`)

```python
# ✓ CORRECT: Static mode selection
def simulate(runtime, state):
    mode = int(runtime.mode)  # Extract static

    if mode == 0:
        return _simulate_mode0(state)
    elif mode == 1:
        return _simulate_mode1(state)
    else:
        return _simulate_mode2(state)
    # Python if/else OK: mode is static

# Each mode can be JIT-compiled separately
_simulate_mode0 = jax.jit(_simulate_mode0_impl)
_simulate_mode1 = jax.jit(_simulate_mode1_impl)
```

**Real example from ethode:**
```python
# ethode/jumpdiffusion/kernel.py:358-407
mode = runtime.scheduler.mode  # Static int

if mode == 0:
    # Mode 0: Poisson
    # ... specific logic
elif mode == 1:
    # Mode 1: Pre-gen Hawkes
    # ... specific logic
else:
    # Mode 2: Online Hawkes
    # ... specific logic

# This if/else is fine because mode is determined
# before JIT boundary in simulate()
```

---

## Testing for JIT Compatibility

### Test Pattern 1: Direct JIT Test

```python
import jax
import pytest

def test_jit_compilation():
    """Test that function can be JIT compiled."""
    config = create_config()
    runtime = config.to_runtime()

    # JIT compile the function
    simulate_jit = jax.jit(
        lambda: simulate(runtime, initial_state, t_span, max_steps)
    )

    # Should not crash
    result = simulate_jit()
    assert result is not None
```

**Real example:**
```python
# test_mode1_hawkes.py:457
def test_mode1_jit_compilation(self):
    """Test that Mode 1 simulation can be JIT compiled."""
    config = JumpDiffusionConfig(...)
    runtime = config.to_runtime(check_units=True)

    simulate_jit = jax.jit(
        lambda: simulate(runtime, jnp.array([1.0]), (0.0, 600.0), max_steps=1000)
    )

    times, states = simulate_jit()
    assert jnp.sum(jnp.isfinite(times)) > 0
```

### Test Pattern 2: Check Error Messages

```python
def test_helpful_error_on_wrong_usage():
    """Verify that wrong usage gives clear error."""

    with pytest.raises(jax.errors.ConcretizationTypeError,
                      match="Abstract tracer value encountered"):
        @jax.jit
        def bad_function(x):
            # This should fail with helpful message
            return int(jnp.sum(x))

        bad_function(jnp.array([1, 2, 3]))
```

### Test Pattern 3: Compare JIT vs. Non-JIT

```python
def test_jit_matches_non_jit():
    """Verify JIT and non-JIT produce same results."""
    config = create_config(seed=42)
    runtime = config.to_runtime()

    # Non-JIT
    result1 = simulate(runtime, state, t_span)

    # JIT
    simulate_jit = jax.jit(lambda: simulate(runtime, state, t_span))
    result2 = simulate_jit()

    # Should be identical
    np.testing.assert_array_equal(result1, result2)
```

---

## Quick Reference

### Checklist: Is My Code JIT-Compatible?

- [ ] No `int()`, `bool()`, `str()` on JAX arrays inside JIT
- [ ] `scan`/`fori_loop` lengths are static Python ints
- [ ] Functions return JAX types, convert outside JIT boundary
- [ ] Static configuration extracted before JIT (or via `static_argnums`)
- [ ] Use `jnp.*` not `math.*` for JAX arrays
- [ ] Use `jax.lax.cond` not `if/else` for traced conditions
- [ ] All pytree types registered with JAX (via Penzai `@struct.pytree_dataclass`)
- [ ] Test includes `jax.jit(my_function)()` call

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ConcretizationTypeError: Abstract tracer value encountered` | Using Python ops (`int`, `bool`, `if/else`) on traced values | Use JAX ops (`astype`, array, `lax.cond`) |
| `TypeError: not a valid JAX type` | Returning non-pytree types from JIT | Use Penzai `@struct.pytree_dataclass` or register pytree |
| `scan` length error | Using traced value for `length=` | Compute length with Python arithmetic before scan |
| `UnexpectedTracerError` | Leaked tracer from JIT context | Check for Python ops on JAX arrays |

### JAX vs. Python Equivalents

| Python | JAX Alternative | When to Use Each |
|--------|----------------|------------------|
| `int(x)` | `x.astype(jnp.int32)` | JAX for traced, Python for static |
| `bool(x)` | Keep as `jax.Array` | JAX inside JIT, Python outside |
| `if cond:` | `jax.lax.cond(cond, ...)` | JAX for traced cond, Python for static |
| `for i in range(n):` | `jax.lax.fori_loop(0, n, ...)` | JAX when needs JIT, Python when n is static |
| `while cond:` | `jax.lax.while_loop(cond_fn, ...)` | JAX for traced cond |
| `len(array)` | `array.shape[0]` | Both OK, but shape is compile-time constant |
| `math.ceil(x)` | `jnp.ceil(x)` | Python for static x, JAX for traced |
| `max(a, b)` | `jnp.maximum(a, b)` | Python for static, JAX for traced |

### Debugging JIT Issues

1. **Enable full tracebacks:**
   ```python
   import jax
   jax.config.update('jax_traceback_filtering', 'off')
   ```

2. **Print during JIT (for debugging only):**
   ```python
   @jax.jit
   def debug_fn(x):
       jax.debug.print("x = {}", x)  # Works in JIT
       # print("x =", x)  # Would fail
       return x * 2
   ```

3. **Check if value is traced:**
   ```python
   import jax.core

   def is_traced(x):
       return isinstance(x, jax.core.Tracer)

   # Use during development to identify issue
   ```

4. **Isolate the JIT boundary:**
   ```python
   # If unsure where JIT breaks, test progressively:

   # Step 1: Test inner function
   result1 = inner_fn(x)  # Works?

   # Step 2: JIT inner function
   result2 = jax.jit(inner_fn)(x)  # Still works?

   # Step 3: Test outer function
   result3 = outer_fn(x)  # Works?

   # Step 4: JIT outer function
   result4 = jax.jit(outer_fn)(x)  # Fails here → issue in outer_fn
   ```

---

## References

- [JAX Documentation: JIT Compilation](https://jax.readthedocs.io/en/latest/jit-compilation.html)
- [JAX Sharp Bits: Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow)
- [JAX Errors: ConcretizationTypeError](https://docs.jax.dev/en/latest/errors.html#jax.errors.ConcretizationTypeError)
- [Penzai pytree documentation](https://penzai.readthedocs.io/en/stable/notebooks/how_to_think_in_pytrees.html)

---

## Case Studies from ethode

### Case Study 1: Hawkes Event Generation (bool → JAX Array)

**Issue:** `ethode/hawkes/kernel.py:133`

**Problem:**
```python
def generate_event(...) -> Tuple[HawkesState, bool, jax.Array]:
    # ...
    return new_state, bool(event_occurred), event_impact
    #                  ^^^^ Python bool conversion on traced array
```

**Error:**
```
ConcretizationTypeError: Attempted boolean conversion of traced array
```

**Fix:**
```python
def generate_event(...) -> Tuple[HawkesState, jax.Array, jax.Array]:
    # ...
    return new_state, event_occurred, event_impact
    # Keep as JAX array - let caller convert if needed
```

**Lesson:** Return JAX types from JIT-compiled functions. Let the adapter layer convert to Python types for user API.

### Case Study 2: Scan Length Must Be Static

**Issue:** `ethode/hawkes/scheduler.py:59`

**Problem:**
```python
def generate_schedule(runtime, t_span, dt: jax.Array, ...):
    dt_floor = 1e-6 * (t_end - t_start)
    dt_clamped = jnp.maximum(dt, dt_floor)  # JAX op → traced
    n_steps = int(jnp.ceil((t_end - t_start) / dt_clamped))  # Traced!

    jax.lax.scan(scan_fn, init, None, length=n_steps)  # ERROR
```

**Error:**
```
ConcretizationTypeError: The `length` argument to `scan` expects a concrete `int` value
```

**Root Causes:**
1. `dt` was JAX array → traced under JIT
2. `dt_floor` depended on traced `t_span`
3. `n_steps` computed from traced values → traced

**Fix:**
```python
import math

def generate_schedule(runtime, t_span, dt: float, ...):  # dt is now Python float
    # Use Python arithmetic → static values
    dt_floor = 1e-6 * (t_end - t_start)
    dt_clamped_static = max(dt, dt_floor)  # Python max()
    n_steps = int(math.ceil((t_end - t_start) / dt_clamped_static))  # Static!

    # Convert to JAX for scan body
    dt_clamped = jnp.array(dt_clamped_static)

    jax.lax.scan(scan_fn, init, None, length=n_steps)  # ✓ Works
```

**Lesson:** Extract configuration values to Python types before JIT boundary. Use Python arithmetic for compile-time constants.

### Case Study 3: Excitation Formula Units Bug

**Issue:** `ethode/hawkes/kernel.py:113-120`

**Problem:**
```python
# Incorrect dimensional analysis
excitation = float(runtime.excitation_strength.value)  # Dimensionless
base_rate = float(runtime.jump_rate.value)  # Events/time

# WRONG: α * λ₀ has dimensions of events/time, but should be events/time²
new_intensity = state.current_intensity + excitation * base_rate
```

**Fix:**
```python
# Correct dimensional analysis
excitation = float(runtime.excitation_strength.value)  # Dimensionless
decay_time = float(runtime.excitation_decay.value)  # Time

# CORRECT: α / τ has dimensions of 1/time
# When integrated: ∫(α/τ)dt = α·events gives dimensionless
new_intensity = state.current_intensity + excitation / decay_time
```

**Lesson:** Not JAX-specific, but important. Even with automatic unit checking, verify formulas match mathematical specifications. The stationary intensity formula `λ_∞ = λ₀ / (1 - α)` should match simulation results.

---

## Contributing Guidelines

When submitting code to ethode:

1. **Test JIT compatibility** if your code is in the kernel layer (`ethode/*/kernel.py`)
2. **Document static requirements** if your function requires static arguments
3. **Separate layers properly:**
   - Config layer: Pydantic models, Python types
   - Runtime layer: Penzai pytrees, JAX arrays
   - Kernel layer: Pure JAX functions (JIT-compatible)
   - Adapter layer: Python types for user API
4. **Add JIT test** for any new kernel function using `jax.jit`
5. **Follow patterns** documented here for static/dynamic value separation

**Example PR checklist:**
- [ ] Code follows JAX coding style (no Python ops on traced arrays)
- [ ] Added JIT compatibility test
- [ ] Return types are JAX arrays in kernel layer
- [ ] Static configuration extracted before JIT boundary
- [ ] Pytree types registered with Penzai
- [ ] Documentation updated if adding new patterns

---

*This document reflects lessons learned during Mode 1 Hawkes implementation (2024-10). Update as new patterns emerge.*
