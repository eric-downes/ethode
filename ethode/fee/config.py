"""Fee configuration with unit-aware Pydantic models.

This module provides configuration for system fees including
base fees, dynamic fees, and fee accumulation.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
import pint

from ..units import UnitManager
from ..fields import quantity_field, tuple_quantity_field
from ..runtime import UnitSpec, QuantityNode
from .runtime import FeeRuntime, FeeState


class FeeConfig(BaseModel):
    """Configuration for system fees.

    All fee parameters support unit-aware inputs:
    - Percentages: "0.5%", "50 bps", 0.005
    - Rates: "0.1% / day", "10 bps / hour"
    - Amounts: "1 USD", "0.001 ETH"
    - Time periods: "1 day", "24 hours", "1 week"

    Example:
        >>> config = FeeConfig(
        ...     base_fee_rate="50 bps",  # 0.5%
        ...     max_fee_rate="200 bps",   # 2%
        ...     fee_decay_time="1 week",
        ...     min_fee_amount="0.01 USD"
        ... )
    """

    # Fee rates (stored as fractions, e.g., 0.005 for 0.5%)
    base_fee_rate: Tuple[float, UnitSpec] = Field(
        description="Base fee rate as fraction (0.005 = 0.5%)"
    )

    max_fee_rate: Tuple[float, UnitSpec] = Field(
        description="Maximum fee rate as fraction"
    )

    min_fee_rate: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Minimum fee rate (optional)"
    )

    # Dynamic fee parameters
    fee_decay_time: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Time constant for fee decay"
    )

    fee_growth_rate: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Rate of fee growth under stress"
    )

    # Fee amounts
    min_fee_amount: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Minimum fee amount in USD"
    )

    max_fee_amount: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Maximum fee amount in USD"
    )

    # Fee accumulation
    accumulation_period: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Period for fee accumulation"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Validators
    _validate_base_fee = field_validator("base_fee_rate", mode="before")(
        quantity_field("dimensionless", "dimensionless")
    )

    _validate_max_fee = field_validator("max_fee_rate", mode="before")(
        quantity_field("dimensionless", "dimensionless")
    )

    @field_validator("min_fee_rate", mode="before")
    def _validate_min_fee(cls, v):
        if v is None:
            return None
        return quantity_field("dimensionless", "dimensionless")(v, None)

    @field_validator("fee_decay_time", mode="before")
    def _validate_decay_time(cls, v):
        if v is None:
            return None
        return quantity_field("time", "second")(v, None)

    @field_validator("fee_growth_rate", mode="before")
    def _validate_growth_rate(cls, v):
        if v is None:
            return None
        return quantity_field("1/time", "1/second")(v, None)

    @field_validator("min_fee_amount", mode="before")
    def _validate_min_amount(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("max_fee_amount", mode="before")
    def _validate_max_amount(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("accumulation_period", mode="before")
    def _validate_accumulation(cls, v):
        if v is None:
            return None
        return quantity_field("time", "second")(v, None)

    @field_validator("base_fee_rate", "max_fee_rate", "min_fee_rate", mode="after")
    def _validate_fee_rates(cls, value: Optional[Tuple[float, UnitSpec]]) -> Optional[Tuple[float, UnitSpec]]:
        """Ensure fee rates are in valid range [0, 1]."""
        if value is None:
            return value

        fee_value = value[0]
        if not (0 <= fee_value <= 1):
            raise ValueError(f"Fee rate must be between 0 and 1, got {fee_value}")

        return value

    @field_validator("max_fee_rate", mode="after")
    def _validate_max_greater_than_base(cls, max_fee: Tuple[float, UnitSpec], info) -> Tuple[float, UnitSpec]:
        """Ensure max fee rate >= base fee rate."""
        base_fee = info.data.get("base_fee_rate")
        if base_fee and max_fee[0] < base_fee[0]:
            raise ValueError(f"Max fee rate {max_fee[0]} must be >= base fee rate {base_fee[0]}")
        return max_fee

    def to_runtime(self, manager: Optional[UnitManager] = None) -> FeeRuntime:
        """Convert to runtime structure for JAX.

        Args:
            manager: Optional UnitManager instance

        Returns:
            FeeRuntime structure with QuantityNodes
        """
        def to_node(value_spec: Optional[Tuple[float, UnitSpec]]) -> Optional[QuantityNode]:
            if value_spec is None:
                return None
            return QuantityNode.from_float(value_spec[0], value_spec[1])

        return FeeRuntime(
            base_fee_rate=to_node(self.base_fee_rate),
            max_fee_rate=to_node(self.max_fee_rate),
            min_fee_rate=to_node(self.min_fee_rate),
            fee_decay_time=to_node(self.fee_decay_time),
            fee_growth_rate=to_node(self.fee_growth_rate),
            min_fee_amount=to_node(self.min_fee_amount),
            max_fee_amount=to_node(self.max_fee_amount),
            accumulation_period=to_node(self.accumulation_period),
        )

    @staticmethod
    def from_runtime(runtime: FeeRuntime, manager: Optional[UnitManager] = None) -> FeeConfigOutput:
        """Create output config from runtime structure.

        Args:
            runtime: FeeRuntime to convert
            manager: Optional UnitManager instance

        Returns:
            FeeConfigOutput with pint quantities
        """
        if manager is None:
            manager = UnitManager.instance()

        def to_quantity(node: Optional[QuantityNode]) -> Optional[pint.Quantity]:
            if node is None:
                return None
            return manager.from_canonical(float(node.value), node.units)

        return FeeConfigOutput(
            base_fee_rate=to_quantity(runtime.base_fee_rate),
            max_fee_rate=to_quantity(runtime.max_fee_rate),
            min_fee_rate=to_quantity(runtime.min_fee_rate),
            fee_decay_time=to_quantity(runtime.fee_decay_time),
            fee_growth_rate=to_quantity(runtime.fee_growth_rate),
            min_fee_amount=to_quantity(runtime.min_fee_amount),
            max_fee_amount=to_quantity(runtime.max_fee_amount),
            accumulation_period=to_quantity(runtime.accumulation_period),
        )

    def summary(self, format: str = "markdown") -> str:
        """Generate summary of fee configuration.

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
            lines.append("# Fee Configuration\n")
            lines.append("| Parameter | Value | Units |")
            lines.append("|-----------|--------|-------|")

            def format_row(name: str, value: Optional[Tuple[float, UnitSpec]]) -> str:
                if value is None:
                    return f"| {name} | - | - |"
                qty = manager.from_canonical(value[0], value[1])
                # Format percentages nicely
                if value[1].dimension == "dimensionless":
                    return f"| {name} | {value[0]*100:.2f}% | - |"
                return f"| {name} | {qty.magnitude:.4g} | {qty.units} |"

        else:  # text format
            lines.append("Fee Configuration")
            lines.append("-" * 40)

            def format_row(name: str, value: Optional[Tuple[float, UnitSpec]]) -> str:
                if value is None:
                    return f"  {name}: -"
                qty = manager.from_canonical(value[0], value[1])
                if value[1].dimension == "dimensionless":
                    return f"  {name}: {value[0]*100:.2f}%"
                return f"  {name}: {qty.magnitude:.4g} {qty.units}"

        lines.append(format_row("Base fee rate", self.base_fee_rate))
        lines.append(format_row("Max fee rate", self.max_fee_rate))

        if self.min_fee_rate:
            lines.append(format_row("Min fee rate", self.min_fee_rate))
        if self.fee_decay_time:
            lines.append(format_row("Fee decay time", self.fee_decay_time))
        if self.fee_growth_rate:
            lines.append(format_row("Fee growth rate", self.fee_growth_rate))
        if self.min_fee_amount:
            lines.append(format_row("Min fee amount", self.min_fee_amount))
        if self.max_fee_amount:
            lines.append(format_row("Max fee amount", self.max_fee_amount))
        if self.accumulation_period:
            lines.append(format_row("Accumulation period", self.accumulation_period))

        return "\n".join(lines)


class FeeConfigOutput(BaseModel):
    """Output format for fee configuration with pint quantities.

    Used when converting from runtime back to user-friendly format.
    """

    base_fee_rate: pint.Quantity
    max_fee_rate: pint.Quantity
    min_fee_rate: Optional[pint.Quantity] = None
    fee_decay_time: Optional[pint.Quantity] = None
    fee_growth_rate: Optional[pint.Quantity] = None
    min_fee_amount: Optional[pint.Quantity] = None
    max_fee_amount: Optional[pint.Quantity] = None
    accumulation_period: Optional[pint.Quantity] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)