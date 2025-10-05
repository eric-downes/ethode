# Ethode Development Plan

**Last updated:** 2025-10-05
**Architecture status:** JAX+penzai rebase complete ✓

## Current State

The ethode framework has been fully migrated to a modern JAX+penzai architecture with complete unit awareness. All core subsystems are implemented and tested.

### ✅ Completed (v2.0)

**Core Architecture:**
- JAX-native computation with full JIT compilation support
- Penzai pytree structures for all runtime state
- Unit-aware configuration with pint integration
- Pydantic validation for all config classes
- Four-layer pattern: Config → Runtime → Kernel → Adapter

**Subsystems Implemented:**
- Controller (PID with configurable deadzone, saturation, rate limits)
- Fee (stress-based dynamic fee calculation)
- Liquidity (SDE-based stochastic liquidity modeling)
- Hawkes (self-exciting point process)
- JumpProcess (Poisson and deterministic jump processes)
- JumpDiffusion (ODE+Jump hybrid simulation with diffrax)

**Infrastructure:**
- UnitManager singleton for centralized unit handling
- QuantityNode for JAX-compatible unit-aware arrays
- Simulation orchestration class for multi-subsystem coordination
- Comprehensive test suite (300+ tests, optimized for speed)
- Migration guide with deprecation warnings

**Integration:**
- diffrax for ODE solving (JAX-native)
- jax.lax.scan for efficient sequence processing
- Full support for jax.jit, jax.vmap, jax.grad

## Immediate Priorities

### P0: Legacy Code Removal

**Blocking:** Clean codebase needed before external release

Remove deprecated files:
```
ethode.py                      # Legacy Params/Sim/AutoDefault/DEFAULTS
ethode_extended.py             # Extensions on legacy patterns
stochastic_extensions.py       # Duplicates new jumpprocess/hawkes
ethode/controller/legacy.py    # Legacy PIDController/PIDParams
test_ethode.py                 # Tests legacy code
test_deprecation.py            # Tests warnings (no longer needed)
test_stochastic_extensions.py  # Tests legacy implementations
test_implemented_extensions.py # Tests ethode_extended.py
```

**Action items:**
1. Audit test_module_configs.py for dependencies on stochastic_extensions
2. Remove all legacy files in single commit
3. Update ethode/__init__.py to remove lazy-loading of legacy attributes
4. Remove ethode/legacy.py (already moved legacy imports there)
5. Update migration_guide.md to mark legacy code as removed in v2.5

**Timeline:** Complete before any external release

### P1: Documentation Completeness

**Current gap:** Migration guide exists but lacks comprehensive user documentation

**Needed:**
1. **User Guide** (`docs/user_guide.md`)
   - Getting started tutorial
   - Conceptual overview of Config/Runtime/Kernel/Adapter layers
   - Common patterns and best practices
   - Multi-subsystem simulation examples

2. **API Reference** (auto-generated from docstrings)
   - Set up Sphinx or mkdocs
   - Ensure all public APIs have complete docstrings
   - Generate HTML documentation

3. **Example Notebooks**
   - Single subsystem examples (one per subsystem)
   - Multi-subsystem orchestration
   - JAX transformations (jit, vmap, scan)
   - Custom subsystem implementation guide

4. **Performance Guide** (`docs/performance.md`)
   - When to use jit vs scan
   - Batch processing patterns
   - Memory management tips
   - Common pitfalls

**Timeline:** Complete within 2-4 weeks

### P2: Test Infrastructure Hardening

**Current state:** Good coverage but some gaps

**Improvements needed:**
1. **CI/CD pipeline**
   - GitHub Actions for automated testing
   - Run tests on push/PR
   - Separate jobs for fast tests vs slow tests
   - mypy type checking in CI

2. **Coverage reporting**
   - Set up pytest-cov in CI
   - Target: 90%+ coverage for ethode/ package
   - Identify untested edge cases

3. **Performance regression tests**
   - Benchmark suite for critical paths
   - Track performance over commits
   - Alert on significant regressions

**Timeline:** Set up basic CI within 1 week

## Near-term Enhancements (v2.1)

### Feature: Batch Simulation API

**Motivation:** Support parameter sweeps efficiently

**Proposed Design** (not yet implemented):
```python
from ethode import Simulation, ControllerConfig
import jax.numpy as jnp

# Define parameter variations
configs = [
    ControllerConfig(kp=kp, ki=0.1, kd=0.01, tau=7.0)
    for kp in [0.1, 0.2, 0.3, 0.4, 0.5]
]

# Batch simulation with vmap (PROPOSED API)
results = Simulation.batch_run(
    configs=configs,
    inputs={'error': jnp.linspace(-1, 1, 100)},
    dt=0.1
)
```

**Implementation effort:** ~1-2 days

### Feature: Observation/Witness System

**Motivation:** Declarative event detection for simulations

**Design:**
```python
from ethode import Simulation, Observer

# Define observers
observers = [
    Observer.threshold('control', above=1.0, label='saturation'),
    Observer.zero_crossing('error', direction='down', label='convergence'),
    Observer.custom(lambda state: state['liquidity'] < 1000, label='low_liquidity')
]

sim = Simulation(controller=adapter, observers=observers)
outputs = sim.step(inputs, dt=0.1)

# Access triggered events
if outputs.events:
    for event in outputs.events:
        print(f"{event.label} at t={event.time}")
```

**Implementation effort:** ~2-3 days

### Feature: Plotting Utilities

**Motivation:** Common visualization patterns

**Design:**
```python
from ethode.plotting import plot_timeseries, plot_phase_portrait

# Simple timeseries
plot_timeseries(
    data=outputs,
    variables=['error', 'control'],
    title='Controller Response'
)

# Phase portrait
plot_phase_portrait(
    x='error',
    y='control',
    data=outputs,
    annotate_events=True
)
```

**Implementation effort:** ~1 day (optional matplotlib dependency)

## Longer-term Roadmap (v2.2+)

### Advanced Features

1. **Adaptive time-stepping**
   - Event-driven simulation with variable dt
   - Automatic refinement near discontinuities
   - Integration with diffrax event detection

2. **State estimation**
   - Kalman filter for noisy observations
   - Particle filter for nonlinear systems
   - Integration with controller subsystem

3. **Optimization integration**
   - Parameter optimization with jaxopt
   - MPC (model predictive control) support
   - Gradient-based tuning of controller params

4. **Extended process types**
   - Cox processes (doubly stochastic)
   - Marked point processes (jumps with sizes)
   - Renewal processes with general inter-arrival distributions

### Performance Optimization

1. **Memory profiling**
   - Identify allocation hotspots
   - Optimize array reuse patterns
   - Consider in-place updates where safe

2. **Compilation caching**
   - Cache JIT-compiled functions
   - Faster startup for repeated runs
   - AOT compilation for deployment

3. **Multi-device support**
   - Leverage pmap for parallel execution
   - GPU acceleration for large batches
   - Distributed simulation support

## Release Checklist

Before v2.5 (next minor release):
- [ ] Remove all legacy code (P0)
- [ ] Complete user guide (P1)
- [ ] Set up CI/CD (P2)
- [ ] API reference documentation (P1)
- [ ] At least 2 example notebooks (P1)
- [ ] Performance guide (P1)
- [ ] Changelog for v2.5

Before v3.0 (major release):
- [ ] All v2.1 features implemented
- [ ] 95%+ test coverage
- [ ] Full API stability guarantee
- [ ] External beta testing complete
- [ ] Migration path for v2.x → v3.0 documented

## Contributing

For contributors working on this codebase:

1. **Code style:**
   - Follow existing patterns (Config/Runtime/Kernel/Adapter)
   - Use Pydantic for configuration
   - Use Penzai structs for runtime state
   - All kernel functions must be pure (no side effects)

2. **Testing:**
   - Add tests for all new features
   - Mark slow tests with `@pytest.mark.slow`
   - Aim for <30s total test runtime (excluding slow tests)

3. **Documentation:**
   - Docstrings for all public APIs
   - Update relevant guides
   - Add examples to docs/adapter_examples.md

4. **Units:**
   - All physical quantities must have units
   - Use UnitManager.instance() for conversions
   - Validate units in Pydantic field_validators
