# Migration Guide: Moving to ethode v2.0

This guide helps you migrate from the legacy ethode patterns to the new unit-aware Config/Runtime architecture introduced in v2.0.

## Overview of Changes

### Old Pattern (Deprecated)
- `Params` base class for parameters
- Direct manipulation of numeric values
- Limited unit support
- Stateful controllers

### New Pattern (v2.0+)
- **Config classes**: Pydantic models with unit validation
- **Runtime structures**: Penzai/JAX compatible structures
- **Full unit awareness**: pint integration throughout
- **Functional design**: JAX-friendly, pure functions

## Migration Examples

### 1. PID Controller Migration

#### Old Code (Deprecated)
```python
from ethode.controller.legacy import PIDController, PIDParams

# Using PIDParams
params = PIDParams(
    kp=1.0,
    ki=0.1,
    kd=0.01,
    integral_leak=0.001,
    noise_threshold=0.003
)
controller = PIDController(params)

# Or directly
controller = PIDController(kp=1.0, ki=0.1, kd=0.01)

# Update step
output = controller.update(1.0, dt=0.1)
```

#### New Code (Recommended)
```python
from ethode.controller import PIDController, ControllerConfig

# Using ControllerConfig with units
config = ControllerConfig(
    kp=1.0,                          # Can use units: "1.0 1/hour"
    ki="0.1 / hour",                 # String with units
    kd=0.01,
    tau=1000.0,                      # Time constant (replaces integral_leak)
    noise_band=(0.001, 0.003),       # Tuple for dead zone
    rate_limit="5 USD/second"        # Units supported everywhere
)

# Create controller
controller = PIDController(config)

# Update step (same API)
output = controller.update(1.0, dt=0.1)
```

### 2. Custom Params Migration

#### Old Code (Deprecated)
```python
from ethode import Params, U

@dataclass
class MyParams(Params):
    base_fee: float = 0.005
    max_fee: float = 0.02
    decay_time: float = 3600  # seconds

    def get_fee_in_bps(self):
        return self.base_fee * 10000  # Manual conversion
```

#### New Code (Recommended)
```python
from pydantic import BaseModel, Field
from ethode.fields import quantity_field
from ethode.runtime import QuantityNode

class FeeConfig(BaseModel):
    base_fee: Tuple[float, UnitSpec] = Field(
        description="Base fee rate"
    )
    max_fee: Tuple[float, UnitSpec] = Field(
        description="Maximum fee rate"
    )
    decay_time: Tuple[float, UnitSpec] = Field(
        description="Fee decay time constant"
    )

    # Validators ensure correct units
    _validate_base = field_validator("base_fee", mode="before")(
        quantity_field("dimensionless", "dimensionless")
    )

    _validate_decay = field_validator("decay_time", mode="before")(
        quantity_field("time", "second")
    )

    def to_runtime(self) -> FeeRuntime:
        """Convert to JAX-compatible runtime structure."""
        return FeeRuntime(
            base_fee=QuantityNode.from_float(self.base_fee[0], self.base_fee[1]),
            max_fee=QuantityNode.from_float(self.max_fee[0], self.max_fee[1]),
            decay_time=QuantityNode.from_float(self.decay_time[0], self.decay_time[1])
        )

# Usage with automatic unit conversion
config = FeeConfig(
    base_fee="50 bps",      # Basis points automatically converted
    max_fee="200 bps",
    decay_time="1 hour"      # Automatically converts to seconds
)
```

### 3. Unit-Aware Computations

#### Old Code (Manual Units)
```python
# Manual unit tracking
rate_usd_per_sec = 5.0
time_hours = 2.0
time_sec = time_hours * 3600  # Manual conversion
amount = rate_usd_per_sec * time_sec
```

#### New Code (Automatic Units)
```python
from ethode.units import UnitManager

manager = UnitManager.instance()

# Parse with units
rate = manager.ensure_quantity("5 USD/second")
time = manager.ensure_quantity("2 hours")

# Automatic unit conversion in calculations
amount = rate * time  # Result: 36000 USD (automatic!)
print(amount.to("USD"))  # 36000 USD
```

## Key Differences

### 1. Configuration
- **Old**: Plain dataclasses with numeric fields
- **New**: Pydantic models with unit validation

### 2. Units
- **Old**: Manual conversions, prone to errors
- **New**: Automatic unit tracking with pint

### 3. Validation
- **Old**: Runtime checks or no validation
- **New**: Compile-time + runtime validation with Pydantic

### 4. JAX Integration
- **Old**: Limited or no JAX support
- **New**: Full JAX compatibility with Penzai structures

## Module-Specific Migrations

### Fee Module
```python
# Old
base_fee = 0.005  # What unit? Fraction? Percent?

# New
from ethode.fee import FeeConfig
config = FeeConfig(base_fee="50 bps")  # Clear unit specification
```

### Liquidity Module
```python
# Old
liquidity = 1000000  # USD? ETH?

# New
from ethode.liquidity import LiquiditySDEConfig
config = LiquiditySDEConfig(
    initial_liquidity="1M USD",  # Clear and concise
    mean_reversion_rate="0.1 / day"
)
```

### Hawkes Process
```python
# Old
jump_rate = 0.01  # Per what time unit?

# New
from ethode.hawkes import HawkesConfig
config = HawkesConfig(
    jump_rate="100 / hour",  # Explicit time unit
    excitation_strength=0.3   # Dimensionless OK
)
```

## Deprecation Timeline

- **v2.0** (Current): Deprecation warnings added
- **v2.5**: Legacy modules moved to `ethode.legacy` namespace
- **v3.0**: Legacy code removed entirely

## Getting Help

### Suppressing Warnings (Temporary)
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ethode")
```

### Automated Migration
For large codebases, consider using the migration script:
```bash
python -m ethode.migrate_code path/to/your/code
```
(Note: Script coming in v2.1)

### Common Issues

1. **Import errors after upgrading**
   - Check if you're importing from `ethode.controller.legacy`
   - Update to `from ethode.controller import ...`

2. **Unit conversion errors**
   - Ensure all numeric inputs specify units
   - Use strings like "5 USD/second" not just `5`

3. **Missing attributes**
   - `integral_leak` → use `tau` (inverse relationship)
   - `noise_threshold` → use `noise_band` tuple

## Examples Repository

Full migration examples available at:
https://github.com/yourusername/ethode-migration-examples

## Support

For migration assistance:
- Open an issue on GitHub
- Check the FAQ at docs/faq.md
- Join our Discord community