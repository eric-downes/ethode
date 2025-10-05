"""Time-Weighted Average Price (TWAP) calculators with unit awareness.

This module provides TWAP calculators that maintain sliding windows
of price observations and compute time-weighted averages.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np

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


@dataclass
class JAXFlatWindowTWAP:
    """JAX-compatible flat window TWAP for use in JIT-compiled functions.

    This version uses fixed-size arrays suitable for JAX operations.

    Attributes:
        window_size: Window duration in seconds
        max_observations: Maximum number of observations to store
        times: Array of observation times
        prices: Array of observation prices
        count: Number of valid observations
    """

    window_size: float
    max_observations: int = 100
    times: jax.Array = field(init=False)
    prices: jax.Array = field(init=False)
    count: jax.Array = field(init=False)

    def __post_init__(self):
        """Initialize JAX arrays."""
        self.times = jnp.zeros(self.max_observations)
        self.prices = jnp.zeros(self.max_observations)
        self.count = jnp.array(0)

    def update(self, price: jax.Array, dt: jax.Array) -> jax.Array:
        """Add observation and return TWAP (JAX-compatible).

        Args:
            price: Price value
            dt: Time since last update

        Returns:
            Current TWAP
        """
        # Calculate current time
        current_time = jnp.where(
            self.count > 0,
            self.times[self.count - 1] + dt,
            0.0
        )

        # Add new observation (circular buffer)
        idx = self.count % self.max_observations
        self.times = self.times.at[idx].set(current_time)
        self.prices = self.prices.at[idx].set(price)
        self.count = jnp.minimum(self.count + 1, self.max_observations)

        # Filter observations within window
        cutoff_time = current_time - self.window_size
        valid_mask = self.times >= cutoff_time

        # Calculate TWAP using valid observations
        return self._calculate_twap_jax(valid_mask)

    def _calculate_twap_jax(self, valid_mask: jax.Array) -> jax.Array:
        """Calculate TWAP using JAX operations.

        Args:
            valid_mask: Boolean mask for valid observations

        Returns:
            Time-weighted average price
        """
        # Count valid observations
        num_valid = jnp.sum(valid_mask)

        # Handle edge cases
        return jax.lax.cond(
            num_valid == 0,
            lambda: 0.0,
            lambda: jax.lax.cond(
                num_valid == 1,
                lambda: jnp.sum(self.prices * valid_mask) / num_valid,
                lambda: self._compute_weighted_average(valid_mask)
            )
        )

    def _compute_weighted_average(self, valid_mask: jax.Array) -> jax.Array:
        """Compute weighted average for multiple observations.

        Args:
            valid_mask: Boolean mask for valid observations

        Returns:
            Weighted average price
        """
        # Create shifted arrays for pairwise operations
        times_shifted = jnp.roll(self.times, -1)
        prices_shifted = jnp.roll(self.prices, -1)

        # Mask for valid pairs (both current and next must be valid)
        pair_mask = valid_mask & jnp.roll(valid_mask, -1)

        # Calculate time intervals
        dt_array = times_shifted - self.times

        # Average prices for each interval
        avg_prices = (self.prices + prices_shifted) / 2.0

        # Weighted sum
        weighted_sum = jnp.sum(avg_prices * dt_array * pair_mask)
        total_weight = jnp.sum(dt_array * pair_mask)

        return jnp.where(total_weight > 0, weighted_sum / total_weight, 0.0)


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