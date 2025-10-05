# Multi-Subsystem Simulation Design

## Overview

This document describes the design for orchestrating multiple subsystems (controller, fee, liquidity, hawkes, etc.) within the `Simulation` facade. It covers dependency management, state synchronization, and execution ordering.

## Current Status

**Implemented (Phase 3):**
- Single-subsystem simulation (controller only)
- Stateful `.step()` API
- Functional `.scan()` wrapper
- State management for controller

**Future (This Document):**
- Multi-subsystem orchestration
- Dependency ordering
- Cross-subsystem data flow
- State synchronization strategies

---

## Design Principles

### 1. Explicit Dependencies

Subsystems may depend on each other's outputs. Make these dependencies explicit:

```
Controller → produces control signal
    ↓
Fee → calculates fees based on control signal (transaction amount)
    ↓
Liquidity → updates based on control signal and fees
    ↓
Market → processes overall market dynamics
```

### 2. Stateful Facade, Functional Core

- `Simulation` maintains stateful adapters (convenience)
- Each adapter wraps a functional kernel (JAX-friendly)
- Power users can access kernels directly

### 3. Composable Architecture

- Each subsystem is independent
- Simulation orchestrates but doesn't couple subsystems
- Easy to add/remove subsystems

---

## Execution Order Strategies

### Strategy 1: Sequential Pipeline (Recommended)

Execute subsystems in dependency order, each seeing outputs from previous steps.

**Pros:**
- Simple to reason about
- Clear data flow
- Easy to debug
- Natural for dependencies

**Cons:**
- Can't run subsystems in parallel
- Order matters

**Implementation:**

```python
class Simulation:
    def __init__(
        self,
        *,
        controller: Optional[ControllerAdapter] = None,
        fee: Optional[FeeAdapter] = None,
        liquidity: Optional[LiquidityAdapter] = None,
        market: Optional[MarketAdapter] = None
    ):
        self.controller = controller
        self.fee = fee
        self.liquidity = liquidity
        self.market = market

    def step(self, inputs: SimulationInputs, dt: float) -> SimulationOutputs:
        """Execute one simulation step with sequential subsystem updates.

        Execution order:
        1. Controller computes control signal from error
        2. Fee calculates transaction fees
        3. Liquidity updates based on trading activity
        4. Market processes overall dynamics

        Args:
            inputs: SimulationInputs(error, market_price, ...)
            dt: Time step

        Returns:
            SimulationOutputs with all subsystem results
        """
        outputs = SimulationOutputs()

        # Step 1: Controller
        if self.controller:
            control_signal = self.controller.step(inputs.error, dt)
            outputs.control = control_signal
        else:
            outputs.control = 0.0

        # Step 2: Fee (depends on control signal)
        if self.fee:
            transaction_amount = abs(outputs.control)
            fee_amount = self.fee.step(transaction_amount, dt)
            outputs.fee = fee_amount
        else:
            outputs.fee = 0.0

        # Step 3: Liquidity (depends on control + fees)
        if self.liquidity:
            net_trade_size = outputs.control - outputs.fee
            liquidity_change = self.liquidity.step(net_trade_size, dt)
            outputs.liquidity_change = liquidity_change
        else:
            outputs.liquidity_change = 0.0

        # Step 4: Market (depends on all previous)
        if self.market:
            market_output = self.market.step(
                control=outputs.control,
                liquidity=outputs.liquidity_change,
                dt=dt
            )
            outputs.market_price = market_output
        else:
            outputs.market_price = inputs.market_price

        return outputs
```

### Strategy 2: Parallel with Shared State

All subsystems read from shared state, update independently, then combine.

**Pros:**
- Can parallelize with `jax.vmap`
- No ordering dependencies
- More flexible

**Cons:**
- More complex state management
- Potential circular dependencies
- Harder to debug

**Implementation:**

```python
class Simulation:
    def step(self, shared_state: SharedState, dt: float) -> SimulationOutputs:
        """Execute one step with parallel subsystem updates.

        All subsystems read from shared_state, update independently.
        """
        outputs = SimulationOutputs()

        # All subsystems run independently
        if self.controller:
            outputs.control = self.controller.step(shared_state.error, dt)

        if self.fee:
            outputs.fee = self.fee.step(shared_state.volume, dt)

        if self.liquidity:
            outputs.liquidity = self.liquidity.step(shared_state.trades, dt)

        # Combine outputs → new shared state
        new_shared_state = self._combine_outputs(shared_state, outputs, dt)

        return new_shared_state, outputs
```

### Strategy 3: Hybrid (Conditional Dependencies)

Some subsystems run in parallel, others in sequence where needed.

**Example:**
```python
# Parallel: Fee and liquidity don't depend on each other
fee_output, liquidity_output = parallel_run(
    self.fee, self.liquidity, inputs, dt
)

# Sequential: Market depends on both
market_output = self.market.step(
    fee=fee_output,
    liquidity=liquidity_output,
    dt=dt
)
```

---

## Data Flow Patterns

### Pattern 1: Input/Output Structs

Use structured inputs and outputs for clarity:

```python
@dataclass
class SimulationInputs:
    """Inputs for one simulation step."""
    error: float                    # Controller input
    market_price: float             # Current market price
    external_demand: float          # External market demand
    timestamp: float                # Current time

@dataclass
class SimulationOutputs:
    """Outputs from one simulation step."""
    control: float                  # Controller output
    fee: float                      # Transaction fee
    liquidity_change: float         # Change in liquidity
    market_price: float             # Updated market price
    diagnostics: Dict[str, Any]     # Optional diagnostics

class Simulation:
    def step(self, inputs: SimulationInputs, dt: float) -> SimulationOutputs:
        """Type-safe simulation step."""
        ...
```

### Pattern 2: Dictionary-Based (Flexible)

Use dictionaries for dynamic subsystems:

```python
class Simulation:
    def step(self, inputs: dict, dt: float) -> dict:
        """Flexible simulation step.

        Args:
            inputs: {
                'error': float,
                'market_price': float,
                ...
            }

        Returns:
            outputs: {
                'control': float,
                'fee': float,
                ...
            }
        """
        outputs = {}

        if self.controller and 'error' in inputs:
            outputs['control'] = self.controller.step(inputs['error'], dt)

        # ... other subsystems

        return outputs
```

---

## State Synchronization

### Challenge

Each adapter maintains its own state. How do we coordinate?

### Solution 1: Independent States (Current)

Each adapter owns its state independently:

```python
class Simulation:
    def __init__(self, *, controller, fee, liquidity):
        self.controller = controller  # Has .state
        self.fee = fee                # Has .state
        self.liquidity = liquidity    # Has .state

    def get_state(self) -> dict:
        """Get all subsystem states."""
        return {
            'controller': self.controller.get_state() if self.controller else None,
            'fee': self.fee.get_state() if self.fee else None,
            'liquidity': self.liquidity.get_state() if self.liquidity else None,
        }

    def reset(self):
        """Reset all subsystem states."""
        if self.controller:
            self.controller.reset()
        if self.fee:
            self.fee.reset()
        if self.liquidity:
            self.liquidity.reset()
```

**Pros:**
- Simple
- No coupling
- Each subsystem controls its own state

**Cons:**
- No shared state
- Can't easily serialize entire simulation state

### Solution 2: Unified State Container

Simulation owns all states:

```python
@dataclass
class SimulationState:
    """Unified state for all subsystems."""
    controller: ControllerState
    fee: FeeState
    liquidity: LiquidityState
    time: float

class Simulation:
    def __init__(self, *, controller, fee, liquidity):
        # Adapters don't own state
        self.controller_adapter = controller
        self.fee_adapter = fee
        self.liquidity_adapter = liquidity

        # Simulation owns unified state
        self.state = SimulationState(
            controller=ControllerState.zero(),
            fee=FeeState.zero(),
            liquidity=LiquidityState.zero(),
            time=0.0
        )

    def step(self, inputs, dt):
        """Step updates unified state."""
        # Pass state to each subsystem, get new state back
        new_controller_state, control = controller_step(
            self.controller_adapter.runtime,
            self.state.controller,
            inputs.error,
            dt
        )

        new_fee_state, fee = fee_step(
            self.fee_adapter.runtime,
            self.state.fee,
            control,
            dt
        )

        # Update unified state
        self.state = SimulationState(
            controller=new_controller_state,
            fee=new_fee_state,
            liquidity=...,
            time=self.state.time + dt
        )

        return outputs
```

**Pros:**
- Single source of truth
- Easy to serialize/deserialize
- Clear state ownership

**Cons:**
- More complex
- Adapters can't be used independently
- Tight coupling

---

## Multi-Subsystem scan() Implementation

### Goal

Extend `.scan()` to work with multiple subsystems.

### Implementation

```python
class Simulation:
    def scan(
        self,
        inputs: SimulationInputsArray,  # Batch of inputs
        dts: jax.Array                   # Array of timesteps
    ) -> Tuple[SimulationOutputsArray, SimulationState]:
        """Batch process multiple steps with all subsystems.

        Args:
            inputs: Batched inputs (error, prices, etc.)
            dts: Array of time steps [n_steps]

        Returns:
            outputs: Batched outputs from all subsystems
            final_state: Final unified state
        """
        def multi_step_fn(unified_state, step_inputs):
            """Single step for all subsystems."""
            inputs, dt = step_inputs

            # Extract individual states
            controller_state = unified_state['controller']
            fee_state = unified_state['fee']
            liquidity_state = unified_state['liquidity']

            # Controller step
            new_controller_state, control = controller_step(
                self.controller.runtime,
                controller_state,
                inputs['error'],
                dt
            )

            # Fee step (depends on control)
            new_fee_state, fee = fee_step(
                self.fee.runtime,
                fee_state,
                control,
                dt
            )

            # Liquidity step (depends on control + fee)
            new_liquidity_state, liquidity = liquidity_step(
                self.liquidity.runtime,
                liquidity_state,
                control - fee,
                dt
            )

            # Combine new states
            new_unified_state = {
                'controller': new_controller_state,
                'fee': new_fee_state,
                'liquidity': new_liquidity_state
            }

            # Package outputs
            step_outputs = {
                'control': control,
                'fee': fee,
                'liquidity': liquidity
            }

            return new_unified_state, step_outputs

        # Initial unified state
        initial_state = {
            'controller': self.controller.state,
            'fee': self.fee.state,
            'liquidity': self.liquidity.state
        }

        # Run scan
        final_state, outputs = jax.lax.scan(
            multi_step_fn,
            initial_state,
            (inputs, dts)
        )

        # Update all adapter states
        self.controller.state = final_state['controller']
        self.fee.state = final_state['fee']
        self.liquidity.state = final_state['liquidity']

        return outputs, final_state
```

---

## Dependency Injection

### Pattern: Configurable Dependencies

Allow users to specify subsystem dependencies explicitly:

```python
class Simulation:
    def __init__(
        self,
        subsystems: Dict[str, Any],
        execution_order: Optional[List[str]] = None
    ):
        """Initialize with dynamic subsystem configuration.

        Args:
            subsystems: {
                'controller': ControllerAdapter(...),
                'fee': FeeAdapter(...),
                'liquidity': LiquidityAdapter(...)
            }
            execution_order: ['controller', 'fee', 'liquidity']
                If None, uses default order
        """
        self.subsystems = subsystems
        self.execution_order = execution_order or list(subsystems.keys())

    def step(self, inputs, dt):
        """Execute subsystems in specified order."""
        outputs = {}

        for name in self.execution_order:
            subsystem = self.subsystems[name]
            # Execute subsystem, making previous outputs available
            output = subsystem.step(inputs, outputs, dt)
            outputs[name] = output

        return outputs
```

---

## Error Handling

### Cross-Subsystem Validation

```python
class Simulation:
    def __init__(self, *, controller, fee, liquidity):
        self.controller = controller
        self.fee = fee
        self.liquidity = liquidity

        # Validate compatibility
        self._validate_subsystem_compatibility()

    def _validate_subsystem_compatibility(self):
        """Check that subsystems have compatible units/dimensions."""
        if self.controller and self.fee:
            # Check that controller output units match fee input units
            controller_output_dim = "price/time"  # From controller config
            fee_input_dim = "price"               # From fee config

            if not self._dimensions_compatible(controller_output_dim, fee_input_dim):
                raise ValueError(
                    f"Incompatible dimensions: controller output ({controller_output_dim}) "
                    f"cannot be used as fee input ({fee_input_dim})"
                )

    @staticmethod
    def _dimensions_compatible(dim1: str, dim2: str) -> bool:
        """Check if two dimensions are compatible."""
        # Implementation depends on unit system
        ...
```

---

## Recommended Implementation Sequence

### Phase 1: Extend for Fee Subsystem

1. Implement `FeeAdapter` following adapter pattern
2. Add `fee` parameter to `Simulation.__init__()`
3. Update `.step()` to include fee calculation
4. Update `.scan()` to handle controller + fee
5. Update `.get_state()` and `.reset()` for fee
6. Add tests for controller + fee orchestration

### Phase 2: Add Liquidity Subsystem

1. Implement `LiquidityAdapter`
2. Add to `Simulation.__init__()`
3. Update `.step()` with liquidity calculation
4. Update `.scan()` for all three subsystems
5. Update state management
6. Add tests for multi-subsystem coordination

### Phase 3: Generalize Pattern

1. Refactor to support arbitrary subsystems
2. Add configurable execution order
3. Add cross-subsystem validation
4. Document multi-subsystem usage

---

## Testing Strategy

### Test Levels

1. **Unit Tests**: Each adapter independently
   ```python
   def test_fee_adapter():
       adapter = FeeAdapter(config)
       output = adapter.step(amount=100.0, dt=0.1)
       assert output > 0
   ```

2. **Integration Tests**: Subsystem pairs
   ```python
   def test_controller_fee_integration():
       controller = ControllerAdapter(controller_config)
       fee = FeeAdapter(fee_config)
       sim = Simulation(controller=controller, fee=fee)
       # Test coordination
   ```

3. **System Tests**: Full simulation
   ```python
   def test_full_simulation():
       sim = Simulation(
           controller=ControllerAdapter(...),
           fee=FeeAdapter(...),
           liquidity=LiquidityAdapter(...)
       )
       # Test complete workflow
   ```

### Key Test Cases

- [ ] Sequential execution order maintained
- [ ] State synchronization across subsystems
- [ ] `.scan()` works with multiple subsystems
- [ ] `.reset()` resets all subsystems
- [ ] `.get_state()` returns all states
- [ ] Cross-subsystem data flow correct
- [ ] Unit compatibility validation works
- [ ] Optional subsystems (None) handled correctly

---

## Summary

**Recommended Approach:**

✅ **Sequential Pipeline** - Simple, predictable, easy to debug
✅ **Independent States** - Each adapter owns its state
✅ **Input/Output Structs** - Type-safe interfaces
✅ **Explicit Dependencies** - Clear execution order
✅ **Gradual Extension** - Add subsystems one at a time

**Implementation Order:**

1. Implement `FeeAdapter` (adapter pattern)
2. Extend `Simulation` for controller + fee
3. Implement `LiquidityAdapter`
4. Extend `Simulation` for all three
5. Add `.scan()` for multi-subsystem
6. Generalize pattern (optional)

**Next Steps:**

See `docs/ADAPTER_PATTERN.md` for creating new adapters, then extend `Simulation` to orchestrate multiple subsystems.
