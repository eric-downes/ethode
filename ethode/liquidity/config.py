"""Liquidity SDE configuration with unit-aware Pydantic models.

This module provides configuration for stochastic differential equation
models of liquidity dynamics.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
import pint

from ..units import UnitManager
from ..fields import quantity_field
from ..runtime import UnitSpec, QuantityNode
from .runtime import LiquidityRuntime, LiquidityState


class LiquiditySDEConfig(BaseModel):
    """Configuration for liquidity stochastic differential equation.

    Models liquidity dynamics as:
        dL = μ(L,t) dt + σ(L,t) dW

    Where:
        - L is liquidity level
        - μ is drift (mean reversion)
        - σ is volatility
        - W is Brownian motion

    Example:
        >>> config = LiquiditySDEConfig(
        ...     initial_liquidity="1000000 USD",
        ...     mean_liquidity="1000000 USD",
        ...     mean_reversion_rate="0.1 / day",
        ...     volatility="0.2",
        ...     min_liquidity="10000 USD"
        ... )
    """

    # Liquidity levels
    initial_liquidity: Tuple[float, UnitSpec] = Field(
        description="Initial liquidity level in USD"
    )

    mean_liquidity: Tuple[float, UnitSpec] = Field(
        description="Long-term mean liquidity level"
    )

    min_liquidity: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Minimum liquidity floor"
    )

    max_liquidity: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Maximum liquidity ceiling"
    )

    # SDE parameters
    mean_reversion_rate: Tuple[float, UnitSpec] = Field(
        description="Rate of mean reversion (1/time)"
    )

    volatility: Tuple[float, UnitSpec] = Field(
        description="Volatility coefficient (dimensionless or sqrt(1/time))"
    )

    # Jump parameters (for jump-diffusion)
    jump_intensity: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Poisson jump intensity (events/time)"
    )

    jump_size_mean: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Mean jump size in USD"
    )

    jump_size_std: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Jump size standard deviation in USD"
    )

    # Liquidity provision/removal rates
    provision_rate: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Rate of liquidity provision (USD/time)"
    )

    removal_threshold: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Threshold for liquidity removal"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Validators
    _validate_initial = field_validator("initial_liquidity", mode="before")(
        quantity_field("price", "USD")
    )

    _validate_mean = field_validator("mean_liquidity", mode="before")(
        quantity_field("price", "USD")
    )

    @field_validator("min_liquidity", mode="before")
    def _validate_min(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("max_liquidity", mode="before")
    def _validate_max(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    _validate_reversion = field_validator("mean_reversion_rate", mode="before")(
        quantity_field("1/time", "1/second")
    )

    _validate_volatility = field_validator("volatility", mode="before")(
        quantity_field("dimensionless", "dimensionless")
    )

    @field_validator("jump_intensity", mode="before")
    def _validate_jump_intensity(cls, v):
        if v is None:
            return None
        return quantity_field("1/time", "1/second")(v, None)

    @field_validator("jump_size_mean", mode="before")
    def _validate_jump_mean(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("jump_size_std", mode="before")
    def _validate_jump_std(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("provision_rate", mode="before")
    def _validate_provision(cls, v):
        if v is None:
            return None
        return quantity_field("price/time", "USD/second")(v, None)

    @field_validator("removal_threshold", mode="before")
    def _validate_threshold(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("initial_liquidity", "mean_liquidity", "min_liquidity", mode="after")
    def _validate_positive_liquidity(cls, value: Optional[Tuple[float, UnitSpec]]) -> Optional[Tuple[float, UnitSpec]]:
        """Ensure liquidity values are positive."""
        if value is None:
            return value

        if value[0] <= 0:
            raise ValueError(f"Liquidity must be positive, got {value[0]}")

        return value

    @field_validator("max_liquidity", mode="after")
    def _validate_max_greater_than_min(cls, max_liq: Optional[Tuple[float, UnitSpec]], info) -> Optional[Tuple[float, UnitSpec]]:
        """Ensure max liquidity > min liquidity if both specified."""
        if max_liq is None:
            return max_liq

        min_liq = info.data.get("min_liquidity")
        if min_liq and max_liq[0] <= min_liq[0]:
            raise ValueError(f"Max liquidity {max_liq[0]} must be > min liquidity {min_liq[0]}")

        return max_liq

    def to_runtime(self, manager: Optional[UnitManager] = None) -> LiquidityRuntime:
        """Convert to runtime structure for JAX.

        Args:
            manager: Optional UnitManager instance

        Returns:
            LiquidityRuntime structure with QuantityNodes
        """
        def to_node(value_spec: Optional[Tuple[float, UnitSpec]]) -> Optional[QuantityNode]:
            if value_spec is None:
                return None
            return QuantityNode.from_float(value_spec[0], value_spec[1])

        return LiquidityRuntime(
            initial_liquidity=to_node(self.initial_liquidity),
            mean_liquidity=to_node(self.mean_liquidity),
            min_liquidity=to_node(self.min_liquidity),
            max_liquidity=to_node(self.max_liquidity),
            mean_reversion_rate=to_node(self.mean_reversion_rate),
            volatility=to_node(self.volatility),
            jump_intensity=to_node(self.jump_intensity),
            jump_size_mean=to_node(self.jump_size_mean),
            jump_size_std=to_node(self.jump_size_std),
            provision_rate=to_node(self.provision_rate),
            removal_threshold=to_node(self.removal_threshold),
        )

    @staticmethod
    def from_runtime(runtime: LiquidityRuntime, manager: Optional[UnitManager] = None) -> LiquiditySDEConfigOutput:
        """Create output config from runtime structure.

        Args:
            runtime: LiquidityRuntime to convert
            manager: Optional UnitManager instance

        Returns:
            LiquiditySDEConfigOutput with pint quantities
        """
        if manager is None:
            manager = UnitManager.instance()

        def to_quantity(node: Optional[QuantityNode]) -> Optional[pint.Quantity]:
            if node is None:
                return None
            return manager.from_canonical(float(node.value), node.units)

        return LiquiditySDEConfigOutput(
            initial_liquidity=to_quantity(runtime.initial_liquidity),
            mean_liquidity=to_quantity(runtime.mean_liquidity),
            min_liquidity=to_quantity(runtime.min_liquidity),
            max_liquidity=to_quantity(runtime.max_liquidity),
            mean_reversion_rate=to_quantity(runtime.mean_reversion_rate),
            volatility=to_quantity(runtime.volatility),
            jump_intensity=to_quantity(runtime.jump_intensity),
            jump_size_mean=to_quantity(runtime.jump_size_mean),
            jump_size_std=to_quantity(runtime.jump_size_std),
            provision_rate=to_quantity(runtime.provision_rate),
            removal_threshold=to_quantity(runtime.removal_threshold),
        )

    def summary(self, format: str = "markdown") -> str:
        """Generate summary of liquidity SDE configuration.

        Args:
            format: Output format ('markdown', 'text', or 'dict')

        Returns:
            Formatted summary string
        """
        if format == "dict":
            return str(self.model_dump())

        manager = UnitManager.instance()
        lines = []

        if format == "markdown":
            lines.append("# Liquidity SDE Configuration\n")
            lines.append("| Parameter | Value | Units |")
            lines.append("|-----------|--------|-------|")

            def format_row(name: str, value: Optional[Tuple[float, UnitSpec]]) -> str:
                if value is None:
                    return f"| {name} | - | - |"
                qty = manager.from_canonical(value[0], value[1])
                return f"| {name} | {qty.magnitude:.4g} | {qty.units} |"

        else:  # text format
            lines.append("Liquidity SDE Configuration")
            lines.append("-" * 40)

            def format_row(name: str, value: Optional[Tuple[float, UnitSpec]]) -> str:
                if value is None:
                    return f"  {name}: -"
                qty = manager.from_canonical(value[0], value[1])
                return f"  {name}: {qty.magnitude:.4g} {qty.units}"

        lines.append(format_row("Initial liquidity", self.initial_liquidity))
        lines.append(format_row("Mean liquidity", self.mean_liquidity))
        lines.append(format_row("Mean reversion rate", self.mean_reversion_rate))
        lines.append(format_row("Volatility", self.volatility))

        if self.min_liquidity:
            lines.append(format_row("Min liquidity", self.min_liquidity))
        if self.max_liquidity:
            lines.append(format_row("Max liquidity", self.max_liquidity))
        if self.jump_intensity:
            lines.append(format_row("Jump intensity", self.jump_intensity))
        if self.provision_rate:
            lines.append(format_row("Provision rate", self.provision_rate))

        return "\n".join(lines)


class LiquiditySDEConfigOutput(BaseModel):
    """Output format for liquidity SDE configuration with pint quantities."""

    initial_liquidity: pint.Quantity
    mean_liquidity: pint.Quantity
    min_liquidity: Optional[pint.Quantity] = None
    max_liquidity: Optional[pint.Quantity] = None
    mean_reversion_rate: pint.Quantity
    volatility: pint.Quantity
    jump_intensity: Optional[pint.Quantity] = None
    jump_size_mean: Optional[pint.Quantity] = None
    jump_size_std: Optional[pint.Quantity] = None
    provision_rate: Optional[pint.Quantity] = None
    removal_threshold: Optional[pint.Quantity] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)