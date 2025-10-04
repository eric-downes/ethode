"""Pydantic field validators for unit-aware configurations.

Provides field validators that parse user-friendly unit inputs
and convert them to canonical floats with metadata.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union, Tuple
from functools import wraps
import pint
from pydantic import field_validator, ValidationError
from pydantic_core import core_schema

from .units import UnitManager, UnitSpec, QuantityInput


class QuantityFieldInfo:
    """Information about a quantity field for validation.

    Attributes:
        dimension: Expected physical dimension
        default_unit: Unit to use if none specified
        min_value: Optional minimum value (in canonical units)
        max_value: Optional maximum value (in canonical units)
    """

    def __init__(
        self,
        dimension: str,
        default_unit: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.dimension = dimension
        self.default_unit = default_unit
        self.min_value = min_value
        self.max_value = max_value


def quantity_field(
    dimension: str,
    default_unit: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Callable:
    """Create a Pydantic field validator for quantity inputs.

    This validator accepts strings, numbers, or pint Quantities and
    converts them to canonical floats with metadata.

    Args:
        dimension: Expected physical dimension (e.g., "time", "length")
        default_unit: Unit to apply to dimensionless numbers
        min_value: Optional minimum value in canonical units
        max_value: Optional maximum value in canonical units

    Returns:
        Field validator function for Pydantic models

    Example:
        class MyConfig(BaseModel):
            duration = quantity_field("time", default_unit="second")

            @field_validator("duration")
            def _validate_duration(cls, v, info):
                return quantity_field("time", "second")(v, info)
    """
    def validator(value: Any, info: Optional[Any] = None) -> tuple[float, UnitSpec]:
        """Validate and convert quantity input.

        Args:
            value: Input value to validate
            info: Pydantic validation info (unused but required by signature)

        Returns:
            Tuple of (canonical_float, unit_spec)

        Raises:
            ValueError: If validation fails
        """
        manager = UnitManager.instance()

        # Convert to pint quantity
        try:
            quantity = manager.ensure_quantity(value, default_unit)
        except Exception as e:
            raise ValueError(f"Cannot parse quantity: {e}")

        # Convert to canonical units
        try:
            canonical_value, spec = manager.to_canonical(quantity, dimension)
        except ValueError as e:
            raise ValueError(f"Dimension mismatch: {e}")

        # Validate bounds if specified
        if min_value is not None and canonical_value < min_value:
            raise ValueError(
                f"Value {canonical_value} below minimum {min_value} "
                f"(in canonical {dimension} units)"
            )
        if max_value is not None and canonical_value > max_value:
            raise ValueError(
                f"Value {canonical_value} above maximum {max_value} "
                f"(in canonical {dimension} units)"
            )

        return canonical_value, spec

    return validator


def tuple_quantity_field(
    dimension: str,
    default_unit: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Callable:
    """Create a validator for tuple quantity inputs (e.g., ranges).

    This validator accepts tuples of (min, max) values where each
    element can be a string, number, or pint Quantity.

    Args:
        dimension: Expected physical dimension
        default_unit: Unit to apply to dimensionless numbers
        min_value: Optional minimum for both tuple elements
        max_value: Optional maximum for both tuple elements

    Returns:
        Field validator function for tuple inputs

    Example:
        class MyConfig(BaseModel):
            noise_band = tuple_quantity_field("price", "USD")

            @field_validator("noise_band")
            def _validate_noise_band(cls, v, info):
                return tuple_quantity_field("price", "USD")(v, info)
    """
    base_validator = quantity_field(dimension, default_unit, min_value, max_value)

    def validator(
        value: Union[tuple, list], info: Optional[Any] = None
    ) -> tuple[tuple[float, UnitSpec], tuple[float, UnitSpec]]:
        """Validate tuple of quantities.

        Args:
            value: Tuple/list of two quantity inputs
            info: Pydantic validation info

        Returns:
            Tuple of ((min_float, min_spec), (max_float, max_spec))

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("Expected tuple of two values")

        # Validate each element
        result1 = base_validator(value[0], info)
        result2 = base_validator(value[1], info)

        # Ensure first is less than second
        if result1[0] > result2[0]:
            raise ValueError(
                f"First value {result1[0]} must be less than second {result2[0]} "
                f"(in canonical units)"
            )

        return result1, result2

    return validator


# QuantityAnnotation removed for now - needs proper core schema implementation
# TODO: Implement proper Annotated type support for Pydantic v2 if needed


def create_quantity_validator(
    field_name: str,
    dimension: str,
    default_unit: Optional[str] = None,
    **kwargs
) -> classmethod:
    """Create a field validator method for a Pydantic model.

    This is a convenience function for adding validators to models:

    Example:
        class MyConfig(BaseModel):
            duration: QuantityInput

            _validate_duration = create_quantity_validator(
                "duration", "time", "second"
            )

    Args:
        field_name: Name of field to validate
        dimension: Expected dimension
        default_unit: Default unit if none specified
        **kwargs: Additional arguments for quantity_field

    Returns:
        Class method suitable for use as field_validator
    """
    validator_func = quantity_field(dimension, default_unit, **kwargs)

    @field_validator(field_name, mode='before')
    @classmethod
    def field_validator_method(cls, v, info):
        return validator_func(v, info)

    return field_validator_method