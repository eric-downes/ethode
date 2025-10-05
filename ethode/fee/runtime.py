"""Runtime structures for fee calculations with Penzai/JAX.

This module provides JAX-compatible runtime structures for fees
that preserve unit metadata through transformations.
"""

from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from penzai.core import struct

from ..runtime import QuantityNode


@struct.pytree_dataclass
class FeeRuntime(struct.Struct):
    """Runtime fee parameters for JAX computation.

    All fields are QuantityNodes containing JAX arrays with unit metadata.
    Penzai's @struct.pytree_dataclass automatically registers this as a JAX pytree.
    """

    base_fee_rate: QuantityNode
    max_fee_rate: QuantityNode
    min_fee_rate: Optional[QuantityNode] = None
    fee_decay_time: Optional[QuantityNode] = None
    fee_growth_rate: Optional[QuantityNode] = None
    min_fee_amount: Optional[QuantityNode] = None
    max_fee_amount: Optional[QuantityNode] = None
    accumulation_period: Optional[QuantityNode] = None


@struct.pytree_dataclass
class FeeState(struct.Struct):
    """Runtime state for fee calculations.

    Tracks accumulated fees and dynamic fee adjustments.
    Penzai's @struct.pytree_dataclass automatically registers this as a JAX pytree.
    """

    current_fee_rate: jax.Array  # Current effective fee rate
    accumulated_fees: jax.Array  # Total fees accumulated
    last_update_time: jax.Array  # Time of last fee update
    stress_level: jax.Array      # Market stress indicator [0, 1]

    @classmethod
    def zero(cls) -> FeeState:
        """Create zero-initialized state.

        Returns:
            FeeState with all values at zero
        """
        return cls(
            current_fee_rate=jnp.array(0.0),
            accumulated_fees=jnp.array(0.0),
            last_update_time=jnp.array(0.0),
            stress_level=jnp.array(0.0),
        )

    @classmethod
    def from_base_rate(cls, base_rate: float) -> FeeState:
        """Create state initialized with base fee rate.

        Args:
            base_rate: Base fee rate to start with

        Returns:
            FeeState with current rate set to base
        """
        return cls(
            current_fee_rate=jnp.array(base_rate),
            accumulated_fees=jnp.array(0.0),
            last_update_time=jnp.array(0.0),
            stress_level=jnp.array(0.0),
        )