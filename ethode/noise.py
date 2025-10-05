"""Noise filtering utilities for control systems.

This module provides noise filtering functions including dead-zone
and ramp functions for signal processing.
"""

from __future__ import annotations
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np

from .types import PriceScalar, to_price_scalar
from .runtime import QuantityNode


def noise_barrier(
    error: float,
    band: Tuple[float, float]
) -> float:
    """Apply noise barrier with dead zone and linear ramp.

    Filters small errors to zero and applies linear scaling
    in the transition band.

    Args:
        error: Input error signal
        band: (low, high) thresholds
            - Below low: output is 0
            - Above high: output equals input
            - Between: linear interpolation

    Returns:
        Filtered error signal

    Example:
        >>> noise_barrier(0.0005, (0.001, 0.003))  # Small error
        0.0
        >>> noise_barrier(0.002, (0.001, 0.003))   # In ramp zone
        0.001  # Scaled
        >>> noise_barrier(0.005, (0.001, 0.003))   # Large error
        0.005  # Unfiltered
    """
    low, high = band
    abs_error = abs(error)

    if abs_error < low:
        return 0.0
    elif abs_error > high:
        return error
    else:
        # Linear interpolation in band
        factor = (abs_error - low) / (high - low)
        return error * factor


@jax.jit
def noise_barrier_jax(
    error: jax.Array,
    low: jax.Array,
    high: jax.Array
) -> jax.Array:
    """JAX-compatible noise barrier for JIT compilation.

    Args:
        error: Input error signal
        low: Lower threshold
        high: Upper threshold

    Returns:
        Filtered error signal
    """
    abs_error = jnp.abs(error)

    # Compute scaling factor
    factor = jnp.where(
        abs_error < low,
        0.0,
        jnp.where(
            abs_error > high,
            1.0,
            (abs_error - low) / (high - low)
        )
    )

    return error * factor


def noise_barrier_with_units(
    error: PriceScalar,
    band: Tuple[QuantityNode, QuantityNode]
) -> PriceScalar:
    """Apply noise barrier with unit-aware thresholds.

    Args:
        error: Input error in USD
        band: Tuple of QuantityNodes with price units

    Returns:
        Filtered error in USD

    Example:
        >>> from ethode.runtime import QuantityNode, UnitSpec
        >>> band = (
        ...     QuantityNode(
        ...         value=jnp.array(0.001),
        ...         units=UnitSpec("price", "USD", 1.0)
        ...     ),
        ...     QuantityNode(
        ...         value=jnp.array(0.003),
        ...         units=UnitSpec("price", "USD", 1.0)
        ...     )
        ... )
        >>> noise_barrier_with_units(0.002, band)
    """
    # Extract values from QuantityNodes
    low_value = float(band[0].value)
    high_value = float(band[1].value)

    # Apply noise barrier
    filtered = noise_barrier(float(error), (low_value, high_value))

    return to_price_scalar(filtered)


def smooth_ramp(
    error: float,
    thresholds: Tuple[float, float, float, float]
) -> float:
    """Apply smooth ramp function with four-point control.

    Creates a smooth transition with flat regions and ramps:
    - Below t1: output is 0
    - t1 to t2: ramp up from 0 to error
    - t2 to t3: output equals error
    - t3 to t4: ramp to amplified error
    - Above t4: amplified by max factor

    Args:
        error: Input error signal
        thresholds: (t1, t2, t3, t4) threshold values

    Returns:
        Filtered error signal
    """
    t1, t2, t3, t4 = thresholds
    abs_error = abs(error)

    if abs_error < t1:
        return 0.0
    elif abs_error < t2:
        # Ramp from 0 to 1
        factor = (abs_error - t1) / (t2 - t1)
        return error * factor
    elif abs_error < t3:
        # Pass through
        return error
    elif abs_error < t4:
        # Ramp from 1 to amplification
        max_factor = 2.0  # Maximum amplification
        factor = 1.0 + (max_factor - 1.0) * (abs_error - t3) / (t4 - t3)
        return error * factor
    else:
        # Maximum amplification
        return error * 2.0


@jax.jit
def smooth_ramp_jax(
    error: jax.Array,
    t1: jax.Array,
    t2: jax.Array,
    t3: jax.Array,
    t4: jax.Array,
    max_factor: jax.Array = jnp.array(2.0)
) -> jax.Array:
    """JAX-compatible smooth ramp function.

    Args:
        error: Input error signal
        t1, t2, t3, t4: Threshold values
        max_factor: Maximum amplification factor

    Returns:
        Filtered error signal
    """
    abs_error = jnp.abs(error)

    # Compute factor for each region
    factor = jnp.where(
        abs_error < t1,
        0.0,
        jnp.where(
            abs_error < t2,
            (abs_error - t1) / (t2 - t1),
            jnp.where(
                abs_error < t3,
                1.0,
                jnp.where(
                    abs_error < t4,
                    1.0 + (max_factor - 1.0) * (abs_error - t3) / (t4 - t3),
                    max_factor
                )
            )
        )
    )

    return error * factor


def dead_band(error: float, threshold: float) -> float:
    """Simple dead band filter.

    Args:
        error: Input signal
        threshold: Dead band threshold

    Returns:
        0 if |error| < threshold, else error
    """
    return 0.0 if abs(error) < threshold else error


@jax.jit
def dead_band_jax(error: jax.Array, threshold: jax.Array) -> jax.Array:
    """JAX-compatible dead band filter.

    Args:
        error: Input signal
        threshold: Dead band threshold

    Returns:
        Filtered signal
    """
    return jnp.where(jnp.abs(error) < threshold, 0.0, error)


def saturation(value: float, limits: Tuple[float, float]) -> float:
    """Apply saturation limits to a value.

    Args:
        value: Input value
        limits: (min, max) saturation limits

    Returns:
        Saturated value
    """
    min_val, max_val = limits
    return max(min_val, min(max_val, value))


@jax.jit
def saturation_jax(
    value: jax.Array,
    min_val: jax.Array,
    max_val: jax.Array
) -> jax.Array:
    """JAX-compatible saturation function.

    Args:
        value: Input value
        min_val: Minimum limit
        max_val: Maximum limit

    Returns:
        Saturated value
    """
    return jnp.clip(value, min_val, max_val)