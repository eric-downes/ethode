"""Configuration for jump processes.

This module provides Pydantic configuration classes for jump processes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional, Tuple
import pint
import jax.numpy as jnp

from ..units import UnitManager, UnitSpec, QuantityInput


class JumpProcessConfig(BaseModel):
    """Configuration for jump (point) processes.

    Supports Poisson (constant rate) and deterministic (periodic) processes.
    For self-exciting (Hawkes) processes, use HawkesConfig.

    Example:
        >>> config = JumpProcessConfig(
        ...     process_type='poisson',
        ...     rate="100 / day",
        ...     seed=42
        ... )
        >>> adapter = JumpProcessAdapter(config)

    Attributes:
        process_type: Type of jump process ('poisson' or 'deterministic')
        rate: Event rate (events per time unit)
        seed: Random seed for reproducibility
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Process type
    process_type: Literal['poisson', 'deterministic'] = Field(
        default='poisson',
        description="Type of jump process"
    )

    # Rate parameter (unit-aware)
    rate: Tuple[float, UnitSpec] = Field(
        description="Event rate (events per time unit)"
    )

    # Random seed for reproducibility
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for stochastic processes"
    )

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, v: QuantityInput, info) -> Tuple[float, UnitSpec]:
        """Validate rate has dimension 1/time (frequency).

        Uses manager.to_canonical() to properly handle pint's dimensionality
        representation (e.g., "1 / [time]").
        """
        manager = UnitManager.instance()

        # Parse input to quantity
        if isinstance(v, str):
            q = manager.ensure_quantity(v)
        elif isinstance(v, pint.Quantity):
            q = v
        else:
            # Bare number - assume 1/year
            q = manager.ensure_quantity(f"{v} 1/year", "1/year")

        # Validate and convert to canonical form using to_canonical
        # This properly handles pint's "1 / [time]" dimensionality format
        try:
            rate_value, rate_spec = manager.to_canonical(q, "1/time")
        except (ValueError, pint.DimensionalityError) as e:
            raise ValueError(
                f"Rate must have frequency dimension (1/time), got {q.dimensionality}. "
                f"Error: {e}"
            )

        # Validate rate is non-negative
        if rate_value < 0:
            raise ValueError(f"Rate must be non-negative, got {rate_value}")

        return (rate_value, rate_spec)

    def to_runtime(self, check_units: bool = True) -> 'JumpProcessRuntime':
        """Convert config to JAX-ready runtime structure."""
        from ..runtime import QuantityNode
        from .runtime import JumpProcessRuntime

        rate_value, rate_spec = self.rate

        # Store rate as QuantityNode for consistency with rest of ethode
        rate_node = QuantityNode.from_float(rate_value, rate_spec)

        return JumpProcessRuntime(
            process_type=0 if self.process_type == 'poisson' else 1,  # Enum for JAX
            rate=rate_node,
            seed=self.seed if self.seed is not None else 0,
        )


class JumpProcessConfigOutput(BaseModel):
    """Output wrapper for JumpProcessConfig (for introspection).

    Attributes:
        config: The JumpProcessConfig used
        runtime: The JAX-ready runtime structure
        expected_events_per_unit_time: Expected event rate
        process_type_name: Human-readable process type
    """

    config: JumpProcessConfig
    runtime: 'JumpProcessRuntime'

    # Diagnostic info
    expected_events_per_unit_time: float
    process_type_name: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
