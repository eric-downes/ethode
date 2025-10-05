"""Runtime structures for Hawkes processes with Penzai/JAX.

This module provides JAX-compatible runtime structures for
self-exciting Hawkes point processes.
"""

from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from penzai.core import struct

from ..runtime import QuantityNode


@struct.pytree_dataclass
class HawkesRuntime(struct.Struct):
    """Runtime Hawkes process parameters for JAX computation.

    All fields are QuantityNodes containing JAX arrays with unit metadata.
    Penzai's @struct.pytree_dataclass automatically registers this as a JAX pytree.
    """

    jump_rate: QuantityNode
    excitation_strength: QuantityNode
    excitation_decay: QuantityNode
    max_intensity: Optional[QuantityNode] = None
    min_intensity: Optional[QuantityNode] = None
    event_impact_mean: Optional[QuantityNode] = None
    event_impact_std: Optional[QuantityNode] = None
    cluster_decay_rate: Optional[QuantityNode] = None


@struct.pytree_dataclass
class HawkesState(struct.Struct):
    """Runtime state for Hawkes process.

    Tracks event history and current intensity.
    Penzai's @struct.pytree_dataclass automatically registers this as a JAX pytree.
    """

    current_intensity: jax.Array    # Current process intensity
    time: jax.Array                  # Current time
    event_count: jax.Array          # Total number of events
    last_event_time: jax.Array     # Time of most recent event
    cumulative_impact: jax.Array    # Sum of all event impacts

    @classmethod
    def initialize(cls, base_rate: float) -> HawkesState:
        """Create initial state with base intensity.

        Args:
            base_rate: Base jump rate (events/time)

        Returns:
            HawkesState initialized
        """
        return cls(
            current_intensity=jnp.array(base_rate),
            time=jnp.array(0.0),
            event_count=jnp.array(0),
            last_event_time=jnp.array(-1000.0),  # Far in past
            cumulative_impact=jnp.array(0.0),
        )