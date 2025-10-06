# ethode

JAX-based dynamical systems simulation with automatic unit checking
and dimensional analysis.  While built for studying Ethereum
Macroeconomics and smart contract microeconomic dynamics, it is
general purpose.

`ethode` is a Python framework for simulating stochastic dynamical
systems with full support for physical units, dimensional analysis,
and JIT compilation. Built on [JAX](https://github.com/google/jax) for
high-performance numerical computing and
[Penzai](https://github.com/google-deepmind/penzai) for pytree
structures allowing a user-friendly unit specification.

To see usage examples look at the [guide](./guide/)

## Key Features

- **JAX-native**: Full support for JIT compilation, automatic differentiation, and vectorization
- **Automatic unit checking**: Catch dimensional errors at configuration time, not runtime
- **Penzai pytrees**: All runtime structures are JAX-compatible pytrees for seamless transformation
- **Stochastic processes**: Jump-diffusion, Hawkes processes, SDEs with units preserved
- **Type-safe configuration**: Pydantic-based configs with automatic unit conversion and validation
- **No silent failures**: Explicit errors is a design philosophy.

## Core Dependencies

- **JAX ≥0.4.35**: High-performance numerical computing with automatic differentiation
- **Penzai ≥0.2.2**: Critical for pytree structures and runtime state management
- **Pint ≥0.24.3**: Physical unit handling and dimensional analysis
- **Pydantic ≥2.8.2**: Type-safe configuration with validation

## Installation

### Using `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) handles both Python version management and package installation automatically:

```bash
# Clone the repository
git clone <repo-url>
cd ethode
uv install -e .
```

### Using pip

Requires Python ≥3.12:

```bash
# Clone the repository
git clone <repo-url>
cd ethode

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### Jump-Diffusion with Units

```python
from ethode import JumpDiffusionConfig, JumpDiffusionAdapter
from ethode import JumpProcessConfig
import jax.numpy as jnp

# Configure with physical units
config = JumpDiffusionConfig(
    dynamics_fn=lambda t, y, p: -y / p["tau"],
    jump_effect_fn=lambda t, y, p: y + p["jump_size"],
    jump_process=JumpProcessConfig(
        jump_rate=(10.0, "1/second"),  # 10 jumps per second
        jump_times="poisson"
    ),
    solver="euler",
    dt_max=(0.01, "second"),
    t_span=(0.0, 10.0, "second"),
    params={"tau": 1.0, "jump_size": 0.5}
)

# Create adapter and simulate
adapter = JumpDiffusionAdapter(config)
times, states = adapter.simulate(initial_state=jnp.array([1.0]))
```

### Hawkes Process (Self-Exciting)

```python
from ethode import HawkesConfig, HawkesAdapter

# Configure Hawkes process with clustering
config = HawkesConfig(
    jump_rate=(100.0, "1/hour"),           # Base rate
    excitation_strength=0.3,                # 30% self-excitation
    excitation_decay=(5.0, "1/minute")     # 5-minute decay
)

# Simulate event clustering
adapter = HawkesAdapter(config, seed=42)
for _ in range(1000):
    event_occurred = adapter.step(dt=0.1)
    if event_occurred:
        intensity = adapter.get_intensity()
        print(f"Event at t={adapter.get_state()['time']:.2f}, λ={intensity:.4f}")
```

## Architecture

### Configuration Layer
- **Pydantic models** with unit-aware field validators
- **Automatic conversion** to canonical units (e.g., meters, seconds)
- **No silent defaults**: Validators warn when default values are used

### Runtime Layer
- **Penzai pytree structs** for all runtime state and parameters
- **JAX-compatible**: Works seamlessly with `jax.jit`, `jax.lax.scan`, `jax.grad`
- **QuantityNode**: Preserves units through JAX transformations

### Adapter Layer
- **High-level API** for common use cases
- **NumPy-friendly**: Returns standard Python types by default
- **Low-level access**: Direct access to runtime structs for custom JAX code

## Design Principles

1. **JAX-first**: All code is designed for JIT compilation and automatic differentiation
2. **Penzai for state**: Runtime structures use Penzai's `@struct.pytree_dataclass` for automatic pytree registration
3. **Units everywhere**: Physical units are first-class citizens, not afterthoughts
4. **No mocks in production**: Dependencies are required, not optional with fallbacks
5. **Fail fast**: Dimensional errors caught at configuration time, not during simulation

## Testing

```bash
# Run fast tests (default)
pytest

# Run all tests including slow ones
pytest -m ""

# Run with coverage
pytest --cov=ethode --cov-report=html

# Run specific test file
pytest test_hawkes_adapter.py -v
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Type checking
mypy ethode

# Format code (if using formatter)
# black ethode tests
```

## Project Structure

```
ethode/
├── ethode/
│   ├── adapters.py          # High-level adapter API
│   ├── runtime.py            # Core runtime structures (QuantityNode)
│   ├── units.py              # Unit management and canonical conversions
│   ├── fields.py             # Pydantic field validators for units
│   ├── hawkes/               # Hawkes process implementation
│   │   ├── config.py         # Pydantic configuration
│   │   ├── runtime.py        # Penzai pytree runtime structs
│   │   ├── kernel.py         # JAX kernel functions
│   │   └── scheduler.py      # Pre-generation for Mode 1
│   ├── jumpdiffusion/        # Jump-diffusion processes
│   ├── jumpprocess/          # Poisson/deterministic jumps
│   ├── liquidity/            # Liquidity modeling (SDE)
│   └── fee/                  # Fee calculations with units
├── docs/                     # Documentation and specs
├── guide/                    # Pedagogical guide to using ethode
├── tests/                    # Test suite
└── pyproject.toml           # Project configuration
```

## Why JAX + Penzai?

- **JAX** provides high-performance numerical computing with automatic differentiation
- **Penzai's pytree structures** handle the complex state management required for unit-aware simulations
- **Automatic pytree registration** eliminates boilerplate and prevents subtle bugs
- **JIT compatibility** ensures performance without sacrificing correctness

## Examples

See the following for detailed examples:
- `test_hawkes_adapter.py` - Hawkes process usage patterns
- `docs/adapter_examples.md` - Adapter API examples
- Individual module tests for low-level usage

## Contributing

This is a research project. Before making significant changes:
1. Ensure all tests pass: `pytest`
2. Check type annotations: `mypy ethode`
3. No silent failures or mock fallbacks in production code

