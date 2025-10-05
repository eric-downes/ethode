"""JAX-compatible PID controller kernel.

This module provides the core controller computation that operates on
runtime structures with pure JAX operations.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import dataclasses

from ..runtime import ControllerRuntime, ControllerState


def apply_noise_band(error: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    """Apply noise band filtering to error signal.

    Args:
        error: Error signal
        low: Lower threshold (errors below this are zeroed)
        high: Upper threshold (errors above this pass through)

    Returns:
        Filtered error signal
    """
    abs_error = jnp.abs(error)

    # Three cases:
    # 1. Below low: zero out
    # 2. Between low and high: linear interpolation
    # 3. Above high: pass through

    # Calculate interpolation factor for the band
    band_factor = (abs_error - low) / jnp.maximum(high - low, 1e-10)
    band_factor = jnp.clip(band_factor, 0.0, 1.0)

    # Apply factor: 0 below low, interpolated in band, 1 above high
    filtered_abs = abs_error * band_factor

    # Preserve sign
    result = jnp.sign(error) * filtered_abs

    # Handle the case where high is very large (effectively no upper limit)
    result = jnp.where(high > 1e9,
                      jnp.where(abs_error < low, 0.0, error),
                      result)

    return result


def apply_integral_leak(integral: jax.Array, tau: jax.Array, dt: jax.Array) -> jax.Array:
    """Apply exponential decay to integral term (anti-windup).

    Args:
        integral: Current integral value
        tau: Time constant for decay
        dt: Time step

    Returns:
        Decayed integral value
    """
    decay_factor = jnp.exp(-dt / (tau + 1e-10))
    return integral * decay_factor


def apply_limits(
    value: jax.Array,
    min_val: jax.Array = None,
    max_val: jax.Array = None
) -> jax.Array:
    """Apply min/max limits to a value.

    Args:
        value: Value to limit
        min_val: Minimum allowed value (None for no limit)
        max_val: Maximum allowed value (None for no limit)

    Returns:
        Limited value
    """
    if min_val is not None:
        value = jnp.maximum(value, min_val)
    if max_val is not None:
        value = jnp.minimum(value, max_val)
    return value


def apply_rate_limit(
    value: jax.Array,
    last_value: jax.Array,
    rate_limit: jax.Array,
    dt: jax.Array
) -> jax.Array:
    """Apply rate limiting to prevent excessive changes.

    Args:
        value: Desired value
        last_value: Previous value
        rate_limit: Maximum rate of change per unit time
        dt: Time step

    Returns:
        Rate-limited value
    """
    max_change = rate_limit * dt
    change = value - last_value
    limited_change = jnp.clip(change, -max_change, max_change)
    return last_value + limited_change


@jax.jit
def controller_step(
    runtime: ControllerRuntime,
    state: ControllerState,
    error: jax.Array,
    dt: jax.Array
) -> Tuple[ControllerState, jax.Array]:
    """Execute one step of PID control.

    This is a pure function suitable for JAX compilation. All parameters
    must be in canonical units (the runtime structure handles this).

    Args:
        runtime: Controller parameters (gains, limits, etc.)
        state: Current controller state (integral, last_error, etc.)
        error: Current error signal (setpoint - measurement)
        dt: Time step size

    Returns:
        Tuple of (updated_state, control_output)
    """
    # Apply noise band filtering
    filtered_error = apply_noise_band(
        error,
        runtime.noise_band_low.value,
        runtime.noise_band_high.value
    )

    # Update integral with anti-windup
    leaked_integral = apply_integral_leak(state.integral, runtime.tau.value, dt)

    # Calculate derivative (with filtering)
    derivative = (filtered_error - state.last_error) / (dt + 1e-10)

    # Calculate P and D terms
    p_term = runtime.kp.value * filtered_error
    d_term = runtime.kd.value * derivative

    # Calculate what the integral would be if we integrate
    tentative_integral = leaked_integral + filtered_error * dt
    tentative_i_term = runtime.ki.value * tentative_integral

    # Calculate what the output would be with the new integral
    tentative_output = p_term + tentative_i_term + d_term

    # Check if we need anti-windup
    if runtime.output_min is not None or runtime.output_max is not None:
        min_val = runtime.output_min.value if runtime.output_min else -jnp.inf
        max_val = runtime.output_max.value if runtime.output_max else jnp.inf

        # Check if output would saturate
        would_saturate_high = tentative_output > max_val
        would_saturate_low = tentative_output < min_val

        # Simple anti-windup: don't integrate if output would saturate
        # This is the most common form of anti-windup
        should_prevent = would_saturate_high | would_saturate_low

        # Don't integrate if we would saturate
        new_integral = jnp.where(should_prevent, leaked_integral, tentative_integral)
    else:
        # No limits, always integrate
        new_integral = tentative_integral

    # Calculate final output with the actual integral
    i_term = runtime.ki.value * new_integral
    raw_output = p_term + i_term + d_term

    # Apply output limits if specified
    if runtime.output_min is not None or runtime.output_max is not None:
        min_val = runtime.output_min.value if runtime.output_min else None
        max_val = runtime.output_max.value if runtime.output_max else None
        limited_output = apply_limits(raw_output, min_val, max_val)
    else:
        limited_output = raw_output

    # Apply rate limiting if specified
    if runtime.rate_limit is not None:
        final_output = apply_rate_limit(
            limited_output,
            state.last_output,
            runtime.rate_limit.value,
            dt
        )
    else:
        final_output = limited_output

    # Update state
    new_state = dataclasses.replace(
        state,
        integral=new_integral,
        last_error=filtered_error,
        last_output=final_output,
        time=state.time + dt
    )

    return new_state, final_output


def controller_step_with_diagnostics(
    runtime: ControllerRuntime,
    state: ControllerState,
    error: jax.Array,
    dt: jax.Array
) -> Tuple[ControllerState, jax.Array, dict]:
    """Execute controller step with diagnostic information.

    Like controller_step but also returns intermediate values for analysis.

    Args:
        runtime: Controller parameters
        state: Current controller state
        error: Error signal
        dt: Time step

    Returns:
        Tuple of (updated_state, control_output, diagnostics_dict)
    """
    # Apply noise band
    filtered_error = apply_noise_band(
        error,
        runtime.noise_band_low.value,
        runtime.noise_band_high.value
    )

    # Calculate terms
    leaked_integral = apply_integral_leak(state.integral, runtime.tau.value, dt)
    new_integral = leaked_integral + filtered_error * dt
    derivative = (filtered_error - state.last_error) / (dt + 1e-10)

    p_term = runtime.kp.value * filtered_error
    i_term = runtime.ki.value * new_integral
    d_term = runtime.kd.value * derivative

    raw_output = p_term + i_term + d_term

    # Apply limits
    if runtime.output_min is not None or runtime.output_max is not None:
        min_val = runtime.output_min.value if runtime.output_min else None
        max_val = runtime.output_max.value if runtime.output_max else None
        limited_output = apply_limits(raw_output, min_val, max_val)
    else:
        limited_output = raw_output

    if runtime.rate_limit is not None:
        final_output = apply_rate_limit(
            limited_output,
            state.last_output,
            runtime.rate_limit.value,
            dt
        )
    else:
        final_output = limited_output

    # Update state
    new_state = dataclasses.replace(
        state,
        integral=new_integral,
        last_error=filtered_error,
        last_output=final_output,
        time=state.time + dt
    )

    # Collect diagnostics
    diagnostics = {
        "raw_error": error,
        "filtered_error": filtered_error,
        "p_term": p_term,
        "i_term": i_term,
        "d_term": d_term,
        "raw_output": raw_output,
        "limited_output": limited_output,
        "final_output": final_output,
        "integral": new_integral,
        "derivative": derivative,
    }

    return new_state, final_output, diagnostics