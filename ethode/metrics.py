"""Metrics and monitoring utilities for control systems.

This module provides functions to compute various control system
metrics including stability margins and performance indicators.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
import numpy as np

from .runtime import ControllerRuntime, ControllerState
from .types import DimensionlessScalar, to_dimensionless_scalar


def guardrail(
    runtime: ControllerRuntime,
    market_stiffness: float,
    effective_fee: float = 0.0
) -> float:
    """Compute guardrail stability metric.

    Calculates the metric: M = κ * k_m - (1 - f_eff)

    Where:
        - κ (kappa) is the integral gain (ki)
        - k_m is the market stiffness
        - f_eff is the effective fee rate

    This metric indicates system stability margins. Positive values
    indicate stable operation, while negative values suggest potential
    instability.

    Args:
        runtime: Controller runtime configuration
        market_stiffness: Market stiffness parameter (k_m)
        effective_fee: Effective fee rate (0 to 1)

    Returns:
        Guardrail metric value

    Example:
        >>> from ethode.controller import ControllerConfig
        >>> config = ControllerConfig(
        ...     kp=0.2, ki=0.02, kd=0.0,
        ...     tau=3600.0, noise_band=(0.001, 0.003)
        ... )
        >>> runtime = config.to_runtime()
        >>> metric = guardrail(runtime, market_stiffness=10.0, effective_fee=0.005)
    """
    # Extract integral gain (ki) from runtime
    ki_value = float(runtime.ki.value)

    # Compute metric: M = κ * k_m - (1 - f_eff)
    metric = ki_value * market_stiffness - (1.0 - effective_fee)

    return metric


@jax.jit
def guardrail_jax(
    ki: jax.Array,
    market_stiffness: jax.Array,
    effective_fee: jax.Array
) -> jax.Array:
    """JAX-compatible guardrail metric computation.

    Args:
        ki: Integral gain
        market_stiffness: Market stiffness parameter
        effective_fee: Effective fee rate

    Returns:
        Guardrail metric
    """
    return ki * market_stiffness - (1.0 - effective_fee)


def stability_margin(
    runtime: ControllerRuntime,
    market_stiffness: float,
    damping_ratio: float = 1.0
) -> float:
    """Compute stability margin for the control system.

    Estimates how far the system is from instability based on
    controller gains and market parameters.

    Args:
        runtime: Controller runtime configuration
        market_stiffness: Market stiffness parameter
        damping_ratio: System damping ratio (default 1.0 for critical damping)

    Returns:
        Stability margin (positive = stable, negative = unstable)
    """
    kp_value = float(runtime.kp.value)
    ki_value = float(runtime.ki.value)
    kd_value = float(runtime.kd.value)

    # Natural frequency
    omega_n = np.sqrt(ki_value * market_stiffness)

    # Damping coefficient
    zeta = (kp_value + kd_value * market_stiffness) / (2 * omega_n) if omega_n > 0 else 0

    # Stability margin: positive if damped, negative if unstable
    margin = zeta - damping_ratio

    return margin


def settling_time(
    runtime: ControllerRuntime,
    market_stiffness: float,
    tolerance: float = 0.02
) -> float:
    """Estimate settling time for the control system.

    Calculates the approximate time for the system to settle
    within a specified tolerance of the final value.

    Args:
        runtime: Controller runtime configuration
        market_stiffness: Market stiffness parameter
        tolerance: Settling tolerance (default 2%)

    Returns:
        Estimated settling time in seconds
    """
    kp_value = float(runtime.kp.value)
    ki_value = float(runtime.ki.value)

    if ki_value <= 0:
        # No integral action, use P-only settling time
        if kp_value <= 0:
            return float('inf')
        return -np.log(tolerance) / (kp_value * market_stiffness)

    # Natural frequency
    omega_n = np.sqrt(ki_value * market_stiffness)

    # Damping ratio
    kd_value = float(runtime.kd.value)
    zeta = (kp_value + kd_value * market_stiffness) / (2 * omega_n)

    if zeta >= 1:
        # Overdamped or critically damped
        return -np.log(tolerance) / (zeta * omega_n)
    else:
        # Underdamped
        return -np.log(tolerance * np.sqrt(1 - zeta**2)) / (zeta * omega_n)


def control_effort(
    state: ControllerState,
    runtime: ControllerRuntime,
    error: float
) -> Dict[str, float]:
    """Compute control effort breakdown by component.

    Args:
        state: Current controller state
        runtime: Controller runtime configuration
        error: Current error signal

    Returns:
        Dictionary with P, I, D components and total effort
    """
    kp_value = float(runtime.kp.value)
    ki_value = float(runtime.ki.value)
    kd_value = float(runtime.kd.value)

    # Proportional component
    p_effort = kp_value * error

    # Integral component
    i_effort = ki_value * float(state.integral)

    # Derivative component (approximation)
    d_effort = 0.0
    if state.last_error is not None:
        error_diff = error - float(state.last_error)
        d_effort = kd_value * error_diff  # Note: needs dt for proper derivative

    # Total effort
    total = p_effort + i_effort + d_effort

    return {
        "proportional": p_effort,
        "integral": i_effort,
        "derivative": d_effort,
        "total": total
    }


def performance_metrics(
    errors: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """Compute performance metrics from error history.

    Args:
        errors: Array of error values
        dt: Time step between samples

    Returns:
        Dictionary with performance metrics:
            - mae: Mean Absolute Error
            - rmse: Root Mean Square Error
            - iae: Integral of Absolute Error
            - ise: Integral of Square Error
            - itae: Integral of Time-weighted Absolute Error
    """
    if len(errors) == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "iae": 0.0,
            "ise": 0.0,
            "itae": 0.0
        }

    # Mean Absolute Error
    mae = np.mean(np.abs(errors))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean(errors**2))

    # Integral of Absolute Error
    iae = np.sum(np.abs(errors)) * dt

    # Integral of Square Error
    ise = np.sum(errors**2) * dt

    # Integral of Time-weighted Absolute Error
    time_weights = np.arange(len(errors)) * dt
    itae = np.sum(time_weights * np.abs(errors)) * dt

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "iae": float(iae),
        "ise": float(ise),
        "itae": float(itae)
    }


@jax.jit
def performance_metrics_jax(
    errors: jax.Array,
    dt: jax.Array
) -> Dict[str, jax.Array]:
    """JAX-compatible performance metrics computation.

    Args:
        errors: Array of error values
        dt: Time step

    Returns:
        Dictionary with performance metrics
    """
    # Mean Absolute Error
    mae = jnp.mean(jnp.abs(errors))

    # Root Mean Square Error
    rmse = jnp.sqrt(jnp.mean(errors**2))

    # Integral of Absolute Error
    iae = jnp.sum(jnp.abs(errors)) * dt

    # Integral of Square Error
    ise = jnp.sum(errors**2) * dt

    # Integral of Time-weighted Absolute Error
    time_weights = jnp.arange(len(errors)) * dt
    itae = jnp.sum(time_weights * jnp.abs(errors)) * dt

    return {
        "mae": mae,
        "rmse": rmse,
        "iae": iae,
        "ise": ise,
        "itae": itae
    }