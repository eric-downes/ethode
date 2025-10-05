"""Runtime structures using Penzai for JAX-compatible unit-aware computations.

This module provides Penzai structs that maintain unit metadata while being
compatible with JAX transformations like jit, grad, and vmap.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple
import dataclasses
import jax
import jax.numpy as jnp
import penzai as pz
from penzai.core import struct

from .units import UnitSpec


# Register JAX pytree for UnitSpec so it can be used as metadata
jax.tree_util.register_static(UnitSpec)


@struct.pytree_dataclass
class QuantityNode(struct.Struct):
    """A Penzai struct that holds a value with unit metadata.

    This struct is designed to carry unit information through JAX transformations.
    The value field is treated as a pytree node (participates in transformations),
    while the units field is treated as metadata (static).

    Attributes:
        value: JAX array containing the numerical value in canonical units
        units: UnitSpec metadata describing the units
    """
    value: jax.Array
    units: UnitSpec = dataclasses.field(metadata={'pytree_node': False})

    @classmethod
    def from_float(
        cls,
        value: float,
        units: UnitSpec,
        dtype: jnp.dtype = jnp.float64
    ) -> QuantityNode:
        """Create a QuantityNode from a float value.

        Args:
            value: Numerical value in canonical units
            units: Unit specification
            dtype: JAX array dtype (default float64)

        Returns:
            QuantityNode instance
        """
        return cls(
            value=jnp.array(value, dtype=dtype),
            units=units
        )

    def to_float(self) -> float:
        """Extract the float value from the node.

        Returns:
            Float value (assumes scalar array)
        """
        return float(self.value)

    def __repr__(self) -> str:
        """Pretty representation showing value and units."""
        return f"QuantityNode({self.value}, {self.units.symbol})"


@struct.pytree_dataclass
class ControllerRuntime(struct.Struct):
    """Runtime structure for PID controller parameters.

    All fields are QuantityNodes containing canonical values with unit metadata.
    This structure is designed to be used directly in JAX-compiled functions.

    Attributes:
        kp: Proportional gain (dimension: 1/time or dimensionless)
        ki: Integral gain (dimension: 1/time^2 or 1/time)
        kd: Derivative gain (dimension: dimensionless or time)
        tau: Time constant for integral leak (dimension: time)
        noise_band_low: Lower threshold for noise filtering (dimension: price)
        noise_band_high: Upper threshold for noise filtering (dimension: price)
        output_min: Optional minimum output limit
        output_max: Optional maximum output limit
        rate_limit: Optional maximum rate of change per time unit
    """
    kp: QuantityNode
    ki: QuantityNode
    kd: QuantityNode
    tau: QuantityNode
    noise_band_low: QuantityNode
    noise_band_high: QuantityNode
    output_min: Optional[QuantityNode] = None
    output_max: Optional[QuantityNode] = None
    rate_limit: Optional[QuantityNode] = None

    def get_kp(self) -> jax.Array:
        """Get proportional gain value for computation."""
        return self.kp.value

    def get_ki(self) -> jax.Array:
        """Get integral gain value for computation."""
        return self.ki.value

    def get_kd(self) -> jax.Array:
        """Get derivative gain value for computation."""
        return self.kd.value

    def get_tau(self) -> jax.Array:
        """Get time constant value for computation."""
        return self.tau.value

    def get_noise_band(self) -> Tuple[jax.Array, jax.Array]:
        """Get noise band limits for computation."""
        return self.noise_band_low.value, self.noise_band_high.value


@struct.pytree_dataclass
class ControllerState(struct.Struct):
    """Internal state for PID controller.

    This structure maintains the controller's internal state between updates.
    All values are in canonical units without unit metadata (since they're
    intermediate computational values).

    Attributes:
        integral: Accumulated integral of error over time
        last_error: Error from previous timestep (for derivative)
        last_output: Output from previous timestep (for rate limiting)
        time: Current simulation time
    """
    integral: jax.Array
    last_error: jax.Array
    last_output: jax.Array
    time: jax.Array

    @classmethod
    def zero(cls, dtype: jnp.dtype = jnp.float32) -> ControllerState:
        """Create a zero-initialized controller state.

        Args:
            dtype: JAX array dtype

        Returns:
            ControllerState with all fields initialized to zero
        """
        return cls(
            integral=jnp.array(0.0, dtype=dtype),
            last_error=jnp.array(0.0, dtype=dtype),
            last_output=jnp.array(0.0, dtype=dtype),
            time=jnp.array(0.0, dtype=dtype)
        )

    def reset_integral(self) -> ControllerState:
        """Reset the integral accumulator to zero.

        Returns:
            New ControllerState with integral reset
        """
        return dataclasses.replace(self, integral=jnp.zeros_like(self.integral))


@struct.pytree_dataclass
class SimulationOutput(struct.Struct):
    """Output from a simulation step.

    Attributes:
        control: Control output value
        state: Updated controller state
        error: Current error value
        metrics: Optional dict of additional metrics
    """
    control: jax.Array
    state: ControllerState
    error: jax.Array
    metrics: Optional[dict] = dataclasses.field(default=None, metadata={'pytree_node': False})


# Additional runtime structures can be added here as needed
@struct.pytree_dataclass
class MarketRuntime(struct.Struct):
    """Runtime structure for market parameters.

    Placeholder for future market-related runtime parameters.
    """
    liquidity: QuantityNode
    volatility: QuantityNode
    base_fee: QuantityNode


@struct.pytree_dataclass
class SystemRuntime(struct.Struct):
    """Complete system runtime combining all subsystem parameters.

    This is the top-level runtime structure that contains all parameters
    needed for a complete simulation.

    Attributes:
        controller: Controller parameters
        market: Market parameters (optional)
    """
    controller: ControllerRuntime
    market: Optional[MarketRuntime] = None


def tree_info(pytree) -> str:
    """Get information about a pytree structure.

    Useful for debugging and understanding the structure of runtime objects.

    Args:
        pytree: Any pytree structure

    Returns:
        String description of the pytree structure
    """
    flat, treedef = jax.tree_util.tree_flatten(pytree)
    return (
        f"PyTree with {len(flat)} leaves:\n"
        f"  Structure: {treedef}\n"
        f"  Leaf shapes: {[getattr(x, 'shape', type(x).__name__) for x in flat]}"
    )