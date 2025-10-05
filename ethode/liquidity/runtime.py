"""Runtime structures for liquidity SDE with Penzai/JAX.

This module provides JAX-compatible runtime structures for
stochastic liquidity dynamics.
"""

from __future__ import annotations
from typing import Optional
import dataclasses
import jax
import jax.numpy as jnp

try:
    import penzai
    from penzai import struct
except ImportError:
    # Fallback for Penzai import
    import sys

    class struct:
        """Mock struct module for fallback."""
        @staticmethod
        def pytree_dataclass(cls):
            return dataclasses.dataclass(frozen=True)(cls)

        Struct = object


from ..runtime import QuantityNode


@struct.pytree_dataclass
class LiquidityRuntime(struct.Struct):
    """Runtime liquidity SDE parameters for JAX computation.

    All fields are QuantityNodes containing JAX arrays with unit metadata.
    """

    initial_liquidity: QuantityNode
    mean_liquidity: QuantityNode
    mean_reversion_rate: QuantityNode
    volatility: QuantityNode
    min_liquidity: Optional[QuantityNode] = None
    max_liquidity: Optional[QuantityNode] = None
    jump_intensity: Optional[QuantityNode] = None
    jump_size_mean: Optional[QuantityNode] = None
    jump_size_std: Optional[QuantityNode] = None
    provision_rate: Optional[QuantityNode] = None
    removal_threshold: Optional[QuantityNode] = None


@struct.pytree_dataclass
class LiquidityState(struct.Struct):
    """Runtime state for liquidity dynamics.

    Tracks current liquidity level and related statistics.
    """

    liquidity_level: jax.Array     # Current liquidity in USD
    time: jax.Array                 # Current time
    cumulative_provision: jax.Array # Total liquidity provided
    cumulative_removal: jax.Array   # Total liquidity removed
    jump_count: jax.Array           # Number of jumps occurred

    @classmethod
    def initialize(cls, initial_liquidity: float) -> LiquidityState:
        """Create initial state with given liquidity.

        Args:
            initial_liquidity: Starting liquidity level

        Returns:
            LiquidityState initialized
        """
        return cls(
            liquidity_level=jnp.array(initial_liquidity),
            time=jnp.array(0.0),
            cumulative_provision=jnp.array(0.0),
            cumulative_removal=jnp.array(0.0),
            jump_count=jnp.array(0),
        )