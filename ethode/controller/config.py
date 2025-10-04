"""Pydantic configuration models for PID controller.

This module provides user-friendly configuration interfaces that validate
inputs and convert them to runtime structures.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, Any
from dataclasses import dataclass
import jax.numpy as jnp

from pydantic import BaseModel, ConfigDict, field_validator, Field
import pint

from ..units import UnitManager, UnitSpec, QuantityInput
from ..fields import quantity_field, tuple_quantity_field
from ..runtime import QuantityNode, ControllerRuntime


class ControllerConfig(BaseModel):
    """Configuration for PID controller with unit validation.

    Accepts user-friendly inputs (strings, floats, pint Quantities) and
    validates them against expected dimensions. Converts to canonical
    units internally while preserving original unit information.

    Attributes:
        kp: Proportional gain (1/time or dimensionless)
        ki: Integral gain (1/time^2 or 1/time or dimensionless)
        kd: Derivative gain (time or dimensionless)
        tau: Time constant for integral leak (time)
        noise_band: Tuple of (low, high) thresholds for error filtering
        output_min: Optional minimum output value
        output_max: Optional maximum output value
        rate_limit: Optional maximum rate of change per time unit
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required fields
    kp: Tuple[float, UnitSpec] = Field(
        description="Proportional gain"
    )
    ki: Tuple[float, UnitSpec] = Field(
        description="Integral gain"
    )
    kd: Tuple[float, UnitSpec] = Field(
        description="Derivative gain"
    )
    tau: Tuple[float, UnitSpec] = Field(
        description="Integral leak time constant"
    )
    noise_band: Tuple[Tuple[float, UnitSpec], Tuple[float, UnitSpec]] = Field(
        description="Error noise filtering band (low, high)"
    )

    # Optional fields
    output_min: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Minimum output limit"
    )
    output_max: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Maximum output limit"
    )
    rate_limit: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Maximum rate of change"
    )

    @field_validator("kp", "ki", "kd", mode="before")
    @classmethod
    def validate_gain(cls, v: QuantityInput, info) -> Tuple[float, UnitSpec]:
        """Validate gain parameters (accept various dimensions)."""
        field_name = info.field_name
        manager = UnitManager.instance()

        # For string inputs like "0.2 / day", parse and check dimensionality
        if isinstance(v, str) and "/" in v:
            # Parse as a rate/frequency
            q = manager.ensure_quantity(v)
            # Check if it's a rate (1/time dimension)
            try:
                # Try to convert to Hz (1/second) to verify it's a frequency
                hz_value = q.to("Hz")
                return manager.to_canonical(q, "frequency")
            except pint.DimensionalityError:
                pass

        if field_name in ("kp", "ki"):
            # These are typically rates (1/time) but can be dimensionless
            try:
                return quantity_field("frequency", default_unit="1/second")(v)
            except (ValueError, pint.DimensionalityError):
                # Fall back to dimensionless
                return quantity_field("dimensionless", default_unit="")(v)
        else:  # kd
            # Derivative gain can be time or dimensionless
            try:
                return quantity_field("time", default_unit="second")(v)
            except (ValueError, pint.DimensionalityError):
                return quantity_field("dimensionless", default_unit="")(v)

    @field_validator("tau", mode="before")
    @classmethod
    def validate_tau(cls, v: QuantityInput) -> Tuple[float, UnitSpec]:
        """Validate time constant."""
        return quantity_field("time", default_unit="second", min_value=0)(v)

    @field_validator("noise_band", mode="before")
    @classmethod
    def validate_noise_band(cls, v: Any) -> Tuple[Tuple[float, UnitSpec], Tuple[float, UnitSpec]]:
        """Validate noise band as tuple of values."""
        # Accept various dimensions - typically price/error units
        try:
            return tuple_quantity_field("price", default_unit="USD")(v)
        except (ValueError, pint.DimensionalityError):
            # Fall back to dimensionless
            return tuple_quantity_field("dimensionless", default_unit="")(v)

    @field_validator("output_min", "output_max", "rate_limit", mode="before")
    @classmethod
    def validate_optional_limits(cls, v: Optional[QuantityInput]) -> Optional[Tuple[float, UnitSpec]]:
        """Validate optional limit parameters."""
        if v is None:
            return None
        # These are typically dimensionless or in output units
        try:
            return quantity_field("dimensionless", default_unit="")(v)
        except (ValueError, pint.DimensionalityError):
            # Could be in specific units depending on application
            return quantity_field("price", default_unit="USD")(v)

    def to_runtime(self, dtype: jnp.dtype = jnp.float32) -> ControllerRuntime:
        """Convert configuration to runtime structure.

        Args:
            dtype: JAX array data type for values

        Returns:
            ControllerRuntime with QuantityNodes
        """
        return ControllerRuntime(
            kp=QuantityNode.from_float(self.kp[0], self.kp[1], dtype),
            ki=QuantityNode.from_float(self.ki[0], self.ki[1], dtype),
            kd=QuantityNode.from_float(self.kd[0], self.kd[1], dtype),
            tau=QuantityNode.from_float(self.tau[0], self.tau[1], dtype),
            noise_band_low=QuantityNode.from_float(
                self.noise_band[0][0], self.noise_band[0][1], dtype
            ),
            noise_band_high=QuantityNode.from_float(
                self.noise_band[1][0], self.noise_band[1][1], dtype
            ),
            output_min=(
                QuantityNode.from_float(self.output_min[0], self.output_min[1], dtype)
                if self.output_min else None
            ),
            output_max=(
                QuantityNode.from_float(self.output_max[0], self.output_max[1], dtype)
                if self.output_max else None
            ),
            rate_limit=(
                QuantityNode.from_float(self.rate_limit[0], self.rate_limit[1], dtype)
                if self.rate_limit else None
            ),
        )

    @staticmethod
    def from_runtime(
        runtime: ControllerRuntime,
        manager: Optional[UnitManager] = None
    ) -> ControllerConfigOutput:
        """Create configuration output from runtime structure.

        Args:
            runtime: ControllerRuntime to convert
            manager: Optional UnitManager for conversions

        Returns:
            ControllerConfigOutput with pint Quantities
        """
        if manager is None:
            manager = UnitManager.instance()

        return ControllerConfigOutput(
            kp=manager.from_canonical(runtime.kp.value.item(), runtime.kp.units),
            ki=manager.from_canonical(runtime.ki.value.item(), runtime.ki.units),
            kd=manager.from_canonical(runtime.kd.value.item(), runtime.kd.units),
            tau=manager.from_canonical(runtime.tau.value.item(), runtime.tau.units),
            noise_band_low=manager.from_canonical(
                runtime.noise_band_low.value.item(), runtime.noise_band_low.units
            ),
            noise_band_high=manager.from_canonical(
                runtime.noise_band_high.value.item(), runtime.noise_band_high.units
            ),
            output_min=(
                manager.from_canonical(runtime.output_min.value.item(), runtime.output_min.units)
                if runtime.output_min else None
            ),
            output_max=(
                manager.from_canonical(runtime.output_max.value.item(), runtime.output_max.units)
                if runtime.output_max else None
            ),
            rate_limit=(
                manager.from_canonical(runtime.rate_limit.value.item(), runtime.rate_limit.units)
                if runtime.rate_limit else None
            ),
        )

    def summary(self, format: str = "markdown") -> str:
        """Generate a formatted summary of the configuration.

        Args:
            format: Output format ("markdown", "text", or "dict")

        Returns:
            Formatted string representation
        """
        manager = UnitManager.instance()

        # Reconstruct quantities for display
        values = {
            "kp": manager.from_canonical(self.kp[0], self.kp[1]),
            "ki": manager.from_canonical(self.ki[0], self.ki[1]),
            "kd": manager.from_canonical(self.kd[0], self.kd[1]),
            "tau": manager.from_canonical(self.tau[0], self.tau[1]),
            "noise_band": (
                manager.from_canonical(self.noise_band[0][0], self.noise_band[0][1]),
                manager.from_canonical(self.noise_band[1][0], self.noise_band[1][1])
            ),
        }

        if self.output_min:
            values["output_min"] = manager.from_canonical(self.output_min[0], self.output_min[1])
        if self.output_max:
            values["output_max"] = manager.from_canonical(self.output_max[0], self.output_max[1])
        if self.rate_limit:
            values["rate_limit"] = manager.from_canonical(self.rate_limit[0], self.rate_limit[1])

        if format == "dict":
            return values

        if format == "markdown":
            lines = [
                "## Controller Configuration",
                "",
                "| Parameter | Value | Units |",
                "|-----------|-------|-------|",
            ]
            for key, value in values.items():
                if key == "noise_band":
                    low, high = value
                    lines.append(f"| {key} | ({low.magnitude:.4g}, {high.magnitude:.4g}) | {low.units} |")
                elif value is not None:
                    lines.append(f"| {key} | {value.magnitude:.4g} | {value.units} |")
            return "\n".join(lines)

        else:  # text format
            lines = ["Controller Configuration:"]
            for key, value in values.items():
                if key == "noise_band":
                    low, high = value
                    lines.append(f"  {key}: ({low.magnitude:.4g}, {high.magnitude:.4g}) {low.units}")
                elif value is not None:
                    lines.append(f"  {key}: {value.magnitude:.4g} {value.units}")
            return "\n".join(lines)


@dataclass
class ControllerConfigOutput:
    """Output structure with pint Quantities for display.

    This is returned by from_runtime() for user-friendly display.
    """
    kp: pint.Quantity
    ki: pint.Quantity
    kd: pint.Quantity
    tau: pint.Quantity
    noise_band_low: pint.Quantity
    noise_band_high: pint.Quantity
    output_min: Optional[pint.Quantity] = None
    output_max: Optional[pint.Quantity] = None
    rate_limit: Optional[pint.Quantity] = None

    @property
    def noise_band(self) -> Tuple[pint.Quantity, pint.Quantity]:
        """Get noise band as tuple."""
        return (self.noise_band_low, self.noise_band_high)