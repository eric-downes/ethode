"""Pydantic configuration models for PID controller.

This module provides user-friendly configuration interfaces that validate
inputs and convert them to runtime structures.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, Any, Dict
from dataclasses import dataclass
import jax.numpy as jnp

from pydantic import BaseModel, ConfigDict, field_validator, Field
import pint

from ..units import UnitManager, UnitSpec, QuantityInput
from ..fields import quantity_field, tuple_quantity_field
from ..runtime import QuantityNode, ControllerRuntime
from .dimensions import ControllerDimensions, FINANCIAL


class ControllerConfig(BaseModel):
    """Configuration for PID controller with unit validation.

    Accepts user-friendly inputs (strings, floats, pint Quantities) and
    validates them against expected dimensions. Converts to canonical
    units internally while preserving original unit information.

    Attributes:
        dimensions: Schema for signal dimensions (defaults to financial: price->price/time)
        kp: Proportional gain (dimension depends on error/output schema)
        ki: Integral gain (dimension depends on error/output schema)
        kd: Derivative gain (dimension depends on error/output schema)
        tau: Time constant for integral leak (time)
        noise_band: Tuple of (low, high) thresholds for error filtering
        output_min: Optional minimum output value
        output_max: Optional maximum output value
        rate_limit: Optional maximum rate of change per time unit
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Dimension schema (default to financial for backward compatibility)
    dimensions: ControllerDimensions = Field(
        default_factory=lambda: FINANCIAL,
        description="Schema defining signal dimensions"
    )

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

        # Parse input to a quantity
        if isinstance(v, str):
            q = manager.ensure_quantity(v)
        elif isinstance(v, pint.Quantity):
            q = v
        else:
            # Numeric value - assume dimensionless
            q = manager.ensure_quantity(v, default_unit="")

        # Get the actual dimensions and conversion factor
        dimension_str = str(q.dimensionality)

        # Convert to base units to get canonical value
        try:
            base_q = q.to_base_units()
            canonical_value = float(base_q.magnitude)
            to_canonical = float(base_q.magnitude / q.magnitude) if q.magnitude != 0 else 1.0
        except:
            canonical_value = float(q.magnitude)
            to_canonical = 1.0

        # Create a UnitSpec that preserves the actual dimensions
        # Return CANONICAL value (not raw magnitude) for correct runtime scaling
        return (canonical_value, UnitSpec(
            dimension=dimension_str if dimension_str != "dimensionless" else "dimensionless",
            symbol=str(q.units),
            to_canonical=to_canonical
        ))

    @field_validator("tau", mode="before")
    @classmethod
    def validate_tau(cls, v: QuantityInput) -> Tuple[float, UnitSpec]:
        """Validate time constant."""
        return quantity_field("time", default_unit="second", min_value=0)(v)

    @field_validator("noise_band", mode="before")
    @classmethod
    def validate_noise_band(cls, v: Any, info) -> Tuple[Tuple[float, UnitSpec], Tuple[float, UnitSpec]]:
        """Validate noise band as tuple of values."""
        manager = UnitManager.instance()

        # Parse the tuple
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError("noise_band must be a tuple of two values")

        low, high = v

        # Get the dimensions from the config if available
        dimensions_val = info.data.get("dimensions")
        if dimensions_val and hasattr(dimensions_val, 'error_dim'):
            # Map dimension to default unit
            dim_unit_map = {
                "dimensionless": "",
                "price": "USD",
                "temperature": "kelvin",
                "length": "meter",
            }
            default_unit = dim_unit_map.get(dimensions_val.error_dim, "USD")
        else:
            default_unit = "USD"

        # Parse each value
        low_q = manager.ensure_quantity(low, default_unit=default_unit)
        high_q = manager.ensure_quantity(high, default_unit=default_unit)

        # Check ordering (allow equal values for disabling noise band)
        if low_q.magnitude > high_q.magnitude:
            raise ValueError("noise_band[0] must be less than or equal to noise_band[1]")

        # Convert to canonical form, preserving dimensions
        dimension_str = str(low_q.dimensionality)

        # Try standard dimensions first
        for dim_name in ["price", "dimensionless", "price/time"]:
            try:
                low_canonical = manager.to_canonical(low_q, dim_name)
                high_canonical = manager.to_canonical(high_q, dim_name)
                return (low_canonical, high_canonical)
            except (ValueError, pint.DimensionalityError):
                continue

        # Accept any dimension if standard ones don't work
        # Get proper conversion factor for the units
        try:
            # Get conversion to base SI units to get canonical values
            low_base = low_q.to_base_units()
            high_base = high_q.to_base_units()
            low_canonical = float(low_base.magnitude)
            high_canonical = float(high_base.magnitude)
            low_factor = low_base.magnitude / low_q.magnitude if low_q.magnitude != 0 else 1.0
            high_factor = high_base.magnitude / high_q.magnitude if high_q.magnitude != 0 else 1.0
        except:
            low_canonical = float(low_q.magnitude)
            high_canonical = float(high_q.magnitude)
            low_factor = 1.0
            high_factor = 1.0

        # Return CANONICAL values (not raw magnitudes) for correct runtime scaling
        low_spec = (low_canonical, UnitSpec(
            dimension=dimension_str,
            symbol=str(low_q.units),
            to_canonical=low_factor
        ))
        high_spec = (high_canonical, UnitSpec(
            dimension=dimension_str,
            symbol=str(high_q.units),
            to_canonical=high_factor
        ))
        return (low_spec, high_spec)

    @field_validator("output_min", "output_max", mode="before")
    @classmethod
    def validate_optional_output_limits(cls, v: Optional[QuantityInput]) -> Optional[Tuple[float, UnitSpec]]:
        """Validate optional output limit parameters."""
        if v is None:
            return None

        manager = UnitManager.instance()

        # Parse the input value
        if isinstance(v, str):
            q = manager.ensure_quantity(v)
        elif isinstance(v, pint.Quantity):
            q = v
        else:
            # Numeric value - assume dimensionless
            q = manager.ensure_quantity(v, default_unit="")

        # Accept any dimension - validation will check compatibility later
        dimension_str = str(q.dimensionality)

        # Try common dimensions first
        for dim_name in ["dimensionless", "price", "price/time"]:
            try:
                return manager.to_canonical(q, dim_name)
            except (ValueError, pint.DimensionalityError):
                continue

        # Accept any dimension - convert to canonical (base SI units)
        try:
            base_q = q.to_base_units()
            canonical_value = float(base_q.magnitude)
            to_canonical = float(base_q.magnitude / q.magnitude) if q.magnitude != 0 else 1.0
        except:
            canonical_value = float(q.magnitude)
            to_canonical = 1.0

        # Return CANONICAL value (not raw magnitude) for correct runtime scaling
        return (canonical_value, UnitSpec(
            dimension=dimension_str,
            symbol=str(q.units),
            to_canonical=to_canonical
        ))

    @field_validator("rate_limit", mode="before")
    @classmethod
    def validate_rate_limit(cls, v: Optional[QuantityInput]) -> Optional[Tuple[float, UnitSpec]]:
        """Validate rate limit parameter."""
        if v is None:
            return None
        # Rate limit should be output_units/time
        manager = UnitManager.instance()

        # Parse the input
        if isinstance(v, str) and "/" in v:
            # It's a rate like "10 USD/hour"
            q = manager.ensure_quantity(v)
            # Try to convert to price/time dimension
            try:
                return manager.to_canonical(q, "price/time")
            except (ValueError, pint.DimensionalityError):
                # Fall back to dimensionless/time
                try:
                    return manager.to_canonical(q, "1/time")
                except:
                    # Just treat as given
                    return manager.to_canonical(q, str(q.dimensionality))
        else:
            # Default to price/time dimension
            return quantity_field("price/time", default_unit="USD/second")(v)

    def validate_units(self, verbose: bool = False) -> 'ValidationReport':
        """Validate unit consistency using pint.

        Runs a dry-run with pint quantities to catch dimension mismatches
        before building JAX runtime structures.

        Args:
            verbose: If True, print validation details

        Returns:
            ValidationReport with results
        """
        from ..validation import validate_config_units
        return validate_config_units(self, verbose=verbose)

    def to_runtime(
        self,
        dtype: jnp.dtype = jnp.float32,
        check_units: bool = False
    ) -> ControllerRuntime:
        """Convert configuration to runtime structure.

        Args:
            dtype: JAX array data type for values
            check_units: If True, validate units before building runtime (default: False for backward compatibility)

        Returns:
            ControllerRuntime with QuantityNodes

        Raises:
            ValueError: If check_units=True and validation fails
        """
        # Optionally validate units first
        if check_units:
            report = self.validate_units()
            if not report.success:
                raise ValueError(
                    f"Unit validation failed:\n{report}"
                )

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

    def summary(self, format: str = "markdown") -> Union[str, Dict[str, pint.Quantity]]:
        """Generate a formatted summary of the configuration.

        Args:
            format: Output format ("markdown", "text", or "dict")

        Returns:
            Formatted representation (string or dict of pint quantities)
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
