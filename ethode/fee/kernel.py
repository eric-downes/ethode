"""Fee calculation kernels for JAX.

This module provides JIT-compiled functions for fee calculations.
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from .runtime import FeeRuntime, FeeState


def calculate_fee(
    runtime: FeeRuntime,
    state: FeeState,
    transaction_amount: jax.Array,
    dt: jax.Array
) -> Tuple[FeeState, jax.Array]:
    """Calculate fee for a transaction.

    Args:
        runtime: Fee configuration
        state: Current fee state
        transaction_amount: Transaction amount in USD
        dt: Time since last update

    Returns:
        Tuple of (updated_state, fee_amount)
    """
    # Update time
    new_time = state.last_update_time + dt

    # Apply fee decay if configured
    current_rate = state.current_fee_rate
    if runtime.fee_decay_time is not None:
        decay_rate = 1.0 / float(runtime.fee_decay_time.value)
        base_rate = float(runtime.base_fee_rate.value)

        # Exponential decay toward base rate
        current_rate = base_rate + (current_rate - base_rate) * jnp.exp(-decay_rate * dt)

    # Apply stress-based adjustment if configured
    if runtime.fee_growth_rate is not None:
        growth_rate = float(runtime.fee_growth_rate.value)
        max_rate = float(runtime.max_fee_rate.value)

        # Increase fee based on stress level
        stress_adjustment = state.stress_level * growth_rate * dt
        current_rate = jnp.minimum(current_rate + stress_adjustment, max_rate)

    # Ensure rate is within bounds
    current_rate = jnp.clip(
        current_rate,
        float(runtime.min_fee_rate.value) if runtime.min_fee_rate else 0.0,
        float(runtime.max_fee_rate.value)
    )

    # Calculate fee amount
    fee_amount = transaction_amount * current_rate

    # Apply min/max fee amounts if configured
    if runtime.min_fee_amount is not None:
        fee_amount = jnp.maximum(fee_amount, float(runtime.min_fee_amount.value))
    if runtime.max_fee_amount is not None:
        fee_amount = jnp.minimum(fee_amount, float(runtime.max_fee_amount.value))

    # Update state
    new_state = FeeState(
        current_fee_rate=current_rate,
        accumulated_fees=state.accumulated_fees + fee_amount,
        last_update_time=new_time,
        stress_level=state.stress_level,  # Stress level updated elsewhere
    )

    return new_state, fee_amount


def update_stress_level(
    state: FeeState,
    market_volatility: jax.Array,
    volume_ratio: jax.Array
) -> FeeState:
    """Update stress level based on market conditions.

    Args:
        state: Current fee state
        market_volatility: Market volatility indicator [0, 1]
        volume_ratio: Volume relative to normal [0, inf)

    Returns:
        Updated state with new stress level
    """
    # Simple stress calculation (can be made more sophisticated)
    stress = jnp.clip(
        0.5 * market_volatility + 0.5 * jnp.tanh(volume_ratio - 1.0),
        0.0,
        1.0
    )

    return FeeState(
        current_fee_rate=state.current_fee_rate,
        accumulated_fees=state.accumulated_fees,
        last_update_time=state.last_update_time,
        stress_level=stress,
    )


def calculate_fee_with_diagnostics(
    runtime: FeeRuntime,
    state: FeeState,
    transaction_amount: jax.Array,
    dt: jax.Array
) -> Tuple[FeeState, jax.Array, dict]:
    """Calculate fee with diagnostic information.

    Args:
        runtime: Fee configuration
        state: Current fee state
        transaction_amount: Transaction amount in USD
        dt: Time since last update

    Returns:
        Tuple of (updated_state, fee_amount, diagnostics)

    Diagnostics include:
        - base_fee: Fee at base rate (no stress adjustment)
        - stress_adjustment: Additional fee due to stress
        - current_rate: Effective fee rate used
        - stress_level: Current market stress level
        - accumulated_fees: Total fees accumulated
        - rate_decayed: Whether rate decayed toward base
        - rate_grew: Whether rate grew due to stress
        - min_rate_applied: Whether min rate bound was hit
        - max_rate_applied: Whether max rate bound was hit
        - min_amount_applied: Whether min amount bound was hit
        - max_amount_applied: Whether max amount bound was hit

    Example:
        >>> runtime = fee_config.to_runtime()
        >>> state = FeeState.from_base_rate(0.005)
        >>> new_state, fee, diag = calculate_fee_with_diagnostics(
        ...     runtime, state, jnp.array(1000.0), jnp.array(0.1)
        ... )
        >>> print(f"Fee: ${fee:.2f}")
        >>> print(f"Stress adjustment: ${diag['stress_adjustment']:.2f}")
    """
    # Store initial rate for comparison
    initial_rate = state.current_fee_rate

    # Call main fee calculation
    new_state, fee_amount = calculate_fee(runtime, state, transaction_amount, dt)

    # Calculate base fee (what it would be at base rate, no stress)
    base_rate = float(runtime.base_fee_rate.value)
    base_fee = float(transaction_amount) * base_rate

    # Calculate stress adjustment
    stress_adjustment = float(fee_amount) - base_fee

    # Determine if bounds were applied
    final_rate = float(new_state.current_fee_rate)

    # Check rate bounds
    min_rate_applied = False
    max_rate_applied = False
    if runtime.min_fee_rate is not None:
        min_rate_applied = abs(final_rate - float(runtime.min_fee_rate.value)) < 1e-7
    if runtime.max_fee_rate is not None:
        max_rate_applied = abs(final_rate - float(runtime.max_fee_rate.value)) < 1e-7

    # Check amount bounds
    min_amount_applied = False
    max_amount_applied = False
    if runtime.min_fee_amount is not None:
        expected_fee = float(transaction_amount) * final_rate
        min_amount_applied = float(fee_amount) > expected_fee and abs(float(fee_amount) - float(runtime.min_fee_amount.value)) < 1e-6
    if runtime.max_fee_amount is not None:
        expected_fee = float(transaction_amount) * final_rate
        max_amount_applied = float(fee_amount) < expected_fee and abs(float(fee_amount) - float(runtime.max_fee_amount.value)) < 1e-6

    # Check if rate changed
    rate_decayed = final_rate < float(initial_rate) and runtime.fee_decay_time is not None
    rate_grew = final_rate > float(initial_rate) and runtime.fee_growth_rate is not None

    # Build diagnostics dictionary
    diagnostics = {
        'base_fee': float(base_fee),
        'stress_adjustment': float(stress_adjustment),
        'current_rate': float(final_rate),
        'stress_level': float(new_state.stress_level),
        'accumulated_fees': float(new_state.accumulated_fees),
        'rate_decayed': bool(rate_decayed),
        'rate_grew': bool(rate_grew),
        'min_rate_applied': bool(min_rate_applied),
        'max_rate_applied': bool(max_rate_applied),
        'min_amount_applied': bool(min_amount_applied),
        'max_amount_applied': bool(max_amount_applied),
    }

    return new_state, fee_amount, diagnostics


def reset_accumulated_fees(state: FeeState) -> Tuple[FeeState, jax.Array]:
    """Reset accumulated fees and return the amount.

    Args:
        state: Current fee state

    Returns:
        Tuple of (state_with_reset_fees, accumulated_amount)
    """
    accumulated = state.accumulated_fees

    new_state = FeeState(
        current_fee_rate=state.current_fee_rate,
        accumulated_fees=jnp.array(0.0),
        last_update_time=state.last_update_time,
        stress_level=state.stress_level,
    )

    return new_state, accumulated