"""Time-Weighted Average Price (TWAP) calculators with unit awareness.

This module provides TWAP calculators that maintain sliding windows
of price observations and compute time-weighted averages.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import penzai as pz
from penzai.core import struct

from .types import TimeScalar, PriceScalar, to_time_scalar, to_price_scalar
from .units import UnitManager
from .runtime import UnitSpec


@dataclass
class FlatWindowTWAP:
    """Flat window TWAP calculator with unit-aware inputs.

    Maintains a sliding window of price observations and computes
    the time-weighted average price over the window.

    Attributes:
        window_size: Window duration in seconds (canonical time)
        observations: List of (time, price) tuples
    """

    window_size: float  # Window size in seconds
    observations: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        """Validate window size."""
        if self.window_size <= 0:
            raise ValueError(f"Window size must be positive, got {self.window_size}")

    def update(self, price: PriceScalar, dt: TimeScalar) -> PriceScalar:
        """Add new price observation and return current TWAP.

        Args:
            price: Price value in USD (canonical)
            dt: Time since last update in seconds (canonical)

        Returns:
            Current TWAP in USD
        """
        # Calculate current time based on last observation
        current_time = 0.0
        if self.observations:
            current_time = self.observations[-1][0] + float(dt)

        # Add new observation
        self.observations.append((current_time, float(price)))

        # Remove old observations outside window
        cutoff_time = current_time - self.window_size
        self.observations = [
            (t, p) for t, p in self.observations
            if t >= cutoff_time
        ]

        # Calculate TWAP
        return to_price_scalar(self._calculate_twap())

    def _calculate_twap(self) -> float:
        """Calculate current TWAP from observations.

        Returns:
            Time-weighted average price
        """
        if not self.observations:
            return 0.0

        if len(self.observations) == 1:
            return self.observations[0][1]

        # Trapezoidal integration for time-weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for i in range(1, len(self.observations)):
            t0, p0 = self.observations[i-1]
            t1, p1 = self.observations[i]
            dt = t1 - t0

            # Trapezoidal area: average price * time interval
            avg_price = (p0 + p1) / 2.0
            weighted_sum += avg_price * dt
            total_weight += dt

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def reset(self) -> None:
        """Clear all observations."""
        self.observations = []

    @property
    def value(self) -> PriceScalar:
        """Get current TWAP value."""
        return to_price_scalar(self._calculate_twap())

    @property
    def num_observations(self) -> int:
        """Get number of observations in window."""
        return len(self.observations)

    @property
    def time_coverage(self) -> TimeScalar:
        """Get time span covered by observations."""
        if len(self.observations) < 2:
            return to_time_scalar(0.0)

        time_span = self.observations[-1][0] - self.observations[0][0]
        return to_time_scalar(time_span)


# ============================================================================
# JAX Functional TWAP Implementation
# ============================================================================

@struct.pytree_dataclass
class TWAPState(struct.Struct):
    """Immutable TWAP state for JAX transformations.

    Attributes:
        times: Array of observation timestamps [shape: (max_obs,)]
        prices: Array of observed prices [shape: (max_obs,)]
        count: Number of valid observations stored [shape: ()]
        head: Next insertion index in circular buffer [shape: ()]
    """
    times: jax.Array
    prices: jax.Array
    count: jax.Array
    head: jax.Array

    @classmethod
    def zeros(cls, max_observations: int) -> 'TWAPState':
        """Create an empty TWAP state."""
        return cls(
            times=jnp.zeros(max_observations),
            prices=jnp.zeros(max_observations),
            count=jnp.array(0, dtype=jnp.int32),
            head=jnp.array(0, dtype=jnp.int32)
        )


@struct.pytree_dataclass
class TWAPRuntime(struct.Struct):
    """Immutable TWAP configuration for JAX operations.

    Attributes:
        window_size: Window duration in seconds
        max_observations: Maximum buffer size
    """
    window_size: float
    max_observations: int


def twap_update(
    state: TWAPState,
    runtime: TWAPRuntime,
    price: jax.Array,
    dt: jax.Array
) -> tuple[TWAPState, jax.Array]:
    """Pure functional TWAP update.

    Args:
        state: Current TWAP state
        runtime: TWAP configuration
        price: New price observation
        dt: Time since last observation

    Returns:
        Tuple of (new_state, current_twap)
    """
    # Calculate current time
    current_time = jnp.where(
        state.count > 0,
        state.times[(state.head - 1) % runtime.max_observations] + dt,
        0.0
    )

    # Update circular buffer
    new_times = state.times.at[state.head].set(current_time)
    new_prices = state.prices.at[state.head].set(price)
    new_head = (state.head + 1) % runtime.max_observations
    new_count = jnp.minimum(state.count + 1, runtime.max_observations)

    # Create new state
    new_state = TWAPState(
        times=new_times,
        prices=new_prices,
        count=new_count,
        head=new_head
    )

    # Calculate TWAP
    twap = _calculate_twap(new_state, runtime, current_time)

    return new_state, twap


def _calculate_twap(
    state: TWAPState,
    runtime: TWAPRuntime,
    current_time: jax.Array
) -> jax.Array:
    """Calculate TWAP from state (pure function).

    Args:
        state: TWAP state
        runtime: TWAP configuration
        current_time: Current timestamp

    Returns:
        Time-weighted average price
    """
    # Determine valid observations within window
    cutoff_time = current_time - runtime.window_size

    # Create mask for valid observations
    # An observation is valid if:
    # 1. Its index is < count (it contains real data)
    # 2. Its time is >= cutoff_time (it's within the window)
    indices = jnp.arange(runtime.max_observations)
    has_data = indices < state.count
    in_window = state.times >= cutoff_time
    valid_mask = has_data & in_window

    num_valid = jnp.sum(valid_mask)

    # Handle edge cases
    return lax.cond(
        num_valid == 0,
        lambda: jnp.array(0.0),
        lambda: lax.cond(
            num_valid == 1,
            lambda: _single_observation_twap(state, valid_mask),
            lambda: _multi_observation_twap(state, valid_mask, runtime.max_observations)
        )
    )


def _single_observation_twap(state: TWAPState, valid_mask: jax.Array) -> jax.Array:
    """TWAP for single observation (just return that price)."""
    return jnp.sum(state.prices * valid_mask) / jnp.sum(valid_mask)


def _multi_observation_twap(
    state: TWAPState,
    valid_mask: jax.Array,
    max_obs: int
) -> jax.Array:
    """Calculate weighted average for multiple observations.

    Uses trapezoidal integration over time intervals.
    """
    # Get sorted indices by time (for valid observations)
    # This ensures we process observations in chronological order
    sorted_indices = jnp.argsort(state.times)

    # Apply mask to sorted indices
    sorted_times = state.times[sorted_indices]
    sorted_prices = state.prices[sorted_indices]
    sorted_valid = valid_mask[sorted_indices]

    # Calculate weighted sum using scan
    def scan_fn(carry, inputs):
        prev_time, prev_price, total_weighted, total_time = carry
        curr_time, curr_price, is_valid = inputs

        # Calculate interval (only if both observations are valid)
        dt = jnp.where(
            is_valid,
            curr_time - prev_time,
            0.0
        )

        # Average price in interval
        avg_price = (prev_price + curr_price) / 2.0

        # Update totals (only if current is valid)
        new_weighted = total_weighted + jnp.where(is_valid, avg_price * dt, 0.0)
        new_time = total_time + jnp.where(is_valid, dt, 0.0)

        # Update previous (only if current is valid)
        new_prev_time = jnp.where(is_valid, curr_time, prev_time)
        new_prev_price = jnp.where(is_valid, curr_price, prev_price)

        return (new_prev_time, new_prev_price, new_weighted, new_time), None

    # Find first valid observation to initialize
    first_valid_idx = jnp.argmax(sorted_valid)
    init_time = sorted_times[first_valid_idx]
    init_price = sorted_prices[first_valid_idx]

    # Run scan from second observation onwards
    (_, _, total_weighted, total_time), _ = lax.scan(
        scan_fn,
        (init_time, init_price, 0.0, 0.0),
        (sorted_times[1:], sorted_prices[1:], sorted_valid[1:])
    )

    # Return weighted average
    return jnp.where(
        total_time > 0,
        total_weighted / total_time,
        init_price  # If no time intervals, return the single price
    )


def twap_scan(
    runtime: TWAPRuntime,
    init_state: TWAPState,
    prices: jax.Array,
    dts: jax.Array
) -> tuple[TWAPState, jax.Array]:
    """Apply TWAP update over a sequence of observations using scan.

    Args:
        runtime: TWAP configuration
        init_state: Initial state
        prices: Array of prices [shape: (n,)]
        dts: Array of time deltas [shape: (n,)]

    Returns:
        Tuple of (final_state, twap_values)
    """
    def scan_fn(state, inputs):
        price, dt = inputs
        new_state, twap = twap_update(state, runtime, price, dt)
        return new_state, twap

    return lax.scan(scan_fn, init_state, (prices, dts))


# Vectorized version for multiple independent TWAP calculations
twap_update_vmap = jax.vmap(
    twap_update,
    in_axes=(0, None, 0, 0),  # Vectorize over state, price, dt
    out_axes=(0, 0)
)


@dataclass
class JAXFlatWindowTWAP:
    """JAX-compatible flat window TWAP wrapper.

    This class provides a convenient object-oriented interface to the
    functional TWAP implementation. The underlying implementation is
    pure functional and can be JIT-compiled.

    Attributes:
        window_size: Window duration in seconds
        max_observations: Maximum number of observations to store
        runtime: Immutable configuration
        state: Current TWAP state (PyTree)
    """

    window_size: float
    max_observations: int = 100
    runtime: TWAPRuntime = field(init=False)
    state: TWAPState = field(init=False)

    def __post_init__(self):
        """Initialize runtime and state."""
        self.runtime = TWAPRuntime(
            window_size=self.window_size,
            max_observations=self.max_observations
        )
        self.state = TWAPState.zeros(self.max_observations)

    def update(self, price: jax.Array, dt: jax.Array) -> jax.Array:
        """Add observation and return TWAP.

        Args:
            price: Price value
            dt: Time since last update

        Returns:
            Current TWAP

        Note:
            This method updates internal state. For pure functional
            usage, use twap_update() directly.
        """
        self.state, twap = twap_update(self.state, self.runtime, price, dt)
        return twap

    def reset(self):
        """Reset to empty state."""
        self.state = TWAPState.zeros(self.max_observations)

    def get_state(self) -> TWAPState:
        """Get current state (for functional usage)."""
        return self.state

    def set_state(self, state: TWAPState):
        """Set state (for restoring from functional usage)."""
        self.state = state


def create_twap(window_duration: str, use_jax: bool = False) -> FlatWindowTWAP | JAXFlatWindowTWAP:
    """Create a TWAP calculator with unit-aware window duration.

    Args:
        window_duration: Window size as a string with units (e.g., "5 minutes")
        use_jax: If True, return JAX-compatible version

    Returns:
        TWAP calculator instance

    Example:
        >>> twap = create_twap("5 minutes")
        >>> price = twap.update(100.0, 1.0)  # $100, 1 second dt
    """
    manager = UnitManager.instance()

    # Convert window duration to seconds
    qty = manager.ensure_quantity(window_duration, "second")
    window_seconds = float(qty.to(manager.registry.second).magnitude)

    if use_jax:
        return JAXFlatWindowTWAP(window_size=window_seconds)
    else:
        return FlatWindowTWAP(window_size=window_seconds)


__all__ = [
    # Python implementation
    'FlatWindowTWAP',
    # JAX functional implementation
    'TWAPState',
    'TWAPRuntime',
    'twap_update',
    'twap_scan',
    'twap_update_vmap',
    # JAX wrapper class
    'JAXFlatWindowTWAP',
    # Factory function
    'create_twap',
]