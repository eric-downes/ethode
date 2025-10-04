"""Static typing helpers for dimension-aware scalars.

This module provides NewType aliases and helper functions to enable
type-safe handling of dimensioned quantities at the type level.
"""

from typing import NewType, Annotated, TypeVar, Generic, Optional, cast
from typing_extensions import TypeAlias
import pint

# NewType aliases for dimension-specific scalars
TimeScalar = NewType("TimeScalar", float)
PriceScalar = NewType("PriceScalar", float)
VolumeScalar = NewType("VolumeScalar", float)
FrequencyScalar = NewType("FrequencyScalar", float)
RateScalar = NewType("RateScalar", float)
DimensionlessScalar = NewType("DimensionlessScalar", float)

# Additional financial types
MoneyScalar = NewType("MoneyScalar", float)  # Alias for PriceScalar
PercentScalar = NewType("PercentScalar", float)  # For percentages (0-100)
FractionScalar = NewType("FractionScalar", float)  # For fractions (0-1)

# Compound types
PriceRateScalar = NewType("PriceRateScalar", float)  # USD/second, etc.
VolumeRateScalar = NewType("VolumeRateScalar", float)  # tokens/second, etc.

# Helper functions for type conversion with validation
def to_time_scalar(value: float) -> TimeScalar:
    """Convert a float to a TimeScalar.

    Args:
        value: Time value in seconds (canonical unit)

    Returns:
        TimeScalar wrapping the value

    Raises:
        ValueError: If value is negative (time should be positive)
    """
    if value < 0:
        raise ValueError(f"Time value must be non-negative, got {value}")
    return TimeScalar(value)


def to_price_scalar(value: float) -> PriceScalar:
    """Convert a float to a PriceScalar.

    Args:
        value: Price value in USD (canonical unit)

    Returns:
        PriceScalar wrapping the value

    Raises:
        ValueError: If value is negative (prices should be non-negative)
    """
    if value < 0:
        raise ValueError(f"Price value must be non-negative, got {value}")
    return PriceScalar(value)


def to_volume_scalar(value: float) -> VolumeScalar:
    """Convert a float to a VolumeScalar.

    Args:
        value: Volume value in tokens (canonical unit)

    Returns:
        VolumeScalar wrapping the value
    """
    return VolumeScalar(value)


def to_frequency_scalar(value: float) -> FrequencyScalar:
    """Convert a float to a FrequencyScalar.

    Args:
        value: Frequency value in Hz (canonical unit)

    Returns:
        FrequencyScalar wrapping the value

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"Frequency value must be non-negative, got {value}")
    return FrequencyScalar(value)


def to_rate_scalar(value: float) -> RateScalar:
    """Convert a float to a RateScalar (1/time).

    Args:
        value: Rate value in 1/second (canonical unit)

    Returns:
        RateScalar wrapping the value
    """
    return RateScalar(value)


def to_dimensionless_scalar(value: float) -> DimensionlessScalar:
    """Convert a float to a DimensionlessScalar.

    Args:
        value: Dimensionless value

    Returns:
        DimensionlessScalar wrapping the value
    """
    return DimensionlessScalar(value)


def to_percent_scalar(value: float) -> PercentScalar:
    """Convert a float to a PercentScalar (0-100).

    Args:
        value: Percentage value (0-100)

    Returns:
        PercentScalar wrapping the value

    Raises:
        ValueError: If value is not in [0, 100]
    """
    if not (0 <= value <= 100):
        raise ValueError(f"Percent value must be in [0, 100], got {value}")
    return PercentScalar(value)


def to_fraction_scalar(value: float) -> FractionScalar:
    """Convert a float to a FractionScalar (0-1).

    Args:
        value: Fractional value (0-1)

    Returns:
        FractionScalar wrapping the value

    Raises:
        ValueError: If value is not in [0, 1]
    """
    if not (0 <= value <= 1):
        raise ValueError(f"Fraction value must be in [0, 1], got {value}")
    return FractionScalar(value)


# Annotated aliases with dimension metadata
class Unit:
    """Metadata class for dimension annotations."""

    def __init__(self, dimension: str):
        self.dimension = dimension

    def __repr__(self) -> str:
        return f"Unit({self.dimension!r})"


# Annotated type aliases for users who want dimension metadata
TimeValue: TypeAlias = Annotated[float, Unit("time")]
PriceValue: TypeAlias = Annotated[float, Unit("price")]
VolumeValue: TypeAlias = Annotated[float, Unit("volume")]
FrequencyValue: TypeAlias = Annotated[float, Unit("frequency")]
RateValue: TypeAlias = Annotated[float, Unit("1/time")]
DimensionlessValue: TypeAlias = Annotated[float, Unit("dimensionless")]

# Compound annotated types
PriceRateValue: TypeAlias = Annotated[float, Unit("price/time")]
VolumeRateValue: TypeAlias = Annotated[float, Unit("volume/time")]

# Generic type for quantities with units
T = TypeVar('T')


class Quantity(Generic[T]):
    """Generic type wrapper for quantities with units.

    This is primarily for static type checking and doesn't perform
    runtime unit conversions. Use pint.Quantity for actual unit math.
    """

    def __init__(self, value: float, dimension: str):
        self.value = value
        self.dimension = dimension

    def __repr__(self) -> str:
        return f"Quantity({self.value}, {self.dimension!r})"

    def to_scalar(self) -> float:
        """Extract the scalar value."""
        return self.value


# Utility functions for working with pint quantities
def from_pint_quantity(qty: pint.Quantity, expected_dimension: str) -> float:
    """Extract scalar value from pint Quantity after dimension check.

    Args:
        qty: pint Quantity to convert
        expected_dimension: Expected dimensionality string

    Returns:
        Scalar value in canonical units

    Raises:
        ValueError: If dimensions don't match
    """
    from .units import UnitManager

    manager = UnitManager.instance()

    # This will raise ValueError if dimensions don't match
    canonical_value, spec = manager.to_canonical(qty, expected_dimension)

    return canonical_value


def to_pint_quantity(value: float, dimension: str) -> pint.Quantity:
    """Convert scalar value to pint Quantity with canonical units.

    Args:
        value: Scalar value in canonical units
        dimension: Dimension string

    Returns:
        pint Quantity with appropriate units
    """
    from .units import UnitManager

    manager = UnitManager.instance()
    canonical_unit = manager.canonical_units.get(dimension)

    if canonical_unit is None:
        raise ValueError(f"Unknown dimension: {dimension}")

    return manager.registry.Quantity(value, canonical_unit)


# Type guards for runtime checking
def is_time_scalar(value: float) -> bool:
    """Check if value can be treated as a TimeScalar."""
    return isinstance(value, (int, float)) and value >= 0


def is_price_scalar(value: float) -> bool:
    """Check if value can be treated as a PriceScalar."""
    return isinstance(value, (int, float)) and value >= 0


def is_percent_scalar(value: float) -> bool:
    """Check if value can be treated as a PercentScalar."""
    return isinstance(value, (int, float)) and 0 <= value <= 100


def is_fraction_scalar(value: float) -> bool:
    """Check if value can be treated as a FractionScalar."""
    return isinstance(value, (int, float)) and 0 <= value <= 1


# Dimension mapping for automatic type selection
DIMENSION_TO_TYPE = {
    "time": TimeScalar,
    "price": PriceScalar,
    "volume": VolumeScalar,
    "frequency": FrequencyScalar,
    "1/time": RateScalar,
    "dimensionless": DimensionlessScalar,
    "price/time": PriceRateScalar,
    "volume/time": VolumeRateScalar,
}


def scalar_for_dimension(value: float, dimension: str) -> float:
    """Get typed scalar for a given dimension.

    Args:
        value: Scalar value
        dimension: Dimension string

    Returns:
        Appropriately typed scalar

    Note:
        Returns plain float if dimension is unknown, to maintain
        compatibility with existing code.
    """
    scalar_type = DIMENSION_TO_TYPE.get(dimension)
    if scalar_type is None:
        return value
    return cast(float, scalar_type(value))