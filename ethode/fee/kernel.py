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