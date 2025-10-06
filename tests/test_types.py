#!/usr/bin/env python
"""Test static typing helpers for dimension-aware scalars."""

import pytest
from typing import get_args, get_origin
from typing_extensions import Annotated

from ethode.types import (
    # NewType aliases
    TimeScalar,
    PriceScalar,
    VolumeScalar,
    FrequencyScalar,
    RateScalar,
    DimensionlessScalar,
    MoneyScalar,
    PercentScalar,
    FractionScalar,
    PriceRateScalar,
    VolumeRateScalar,
    # Helper functions
    to_time_scalar,
    to_price_scalar,
    to_volume_scalar,
    to_frequency_scalar,
    to_rate_scalar,
    to_dimensionless_scalar,
    to_percent_scalar,
    to_fraction_scalar,
    # Annotated aliases
    TimeValue,
    PriceValue,
    VolumeValue,
    FrequencyValue,
    RateValue,
    DimensionlessValue,
    PriceRateValue,
    VolumeRateValue,
    Unit,
    # Utility functions
    from_pint_quantity,
    to_pint_quantity,
    scalar_for_dimension,
    # Type guards
    is_time_scalar,
    is_price_scalar,
    is_percent_scalar,
    is_fraction_scalar,
    # Generic quantity
    Quantity,
    DIMENSION_TO_TYPE,
)


class TestNewTypes:
    """Test NewType aliases behave correctly."""

    def test_newtype_creation(self):
        """Test that NewTypes are created properly."""
        # NewTypes should be callable
        t = TimeScalar(1.0)
        assert t == 1.0
        assert isinstance(t, float)

        p = PriceScalar(100.0)
        assert p == 100.0
        assert isinstance(p, float)

    def test_newtype_distinct(self):
        """Test that different NewTypes are distinct at type level."""
        # At runtime they're all floats
        t = TimeScalar(1.0)
        p = PriceScalar(1.0)
        assert t == p  # Runtime equality

        # But type checkers would flag TimeScalar != PriceScalar
        # (This is validated by mypy, not runtime)

    def test_all_newtypes_defined(self):
        """Test all expected NewTypes are defined."""
        expected_types = [
            TimeScalar,
            PriceScalar,
            VolumeScalar,
            FrequencyScalar,
            RateScalar,
            DimensionlessScalar,
            MoneyScalar,
            PercentScalar,
            FractionScalar,
            PriceRateScalar,
            VolumeRateScalar,
        ]
        for scalar_type in expected_types:
            # Should be callable and return float
            value = scalar_type(1.0)
            assert isinstance(value, float)


class TestHelperFunctions:
    """Test helper functions for type conversion."""

    def test_to_time_scalar(self):
        """Test time scalar conversion."""
        # Valid values
        assert to_time_scalar(0.0) == 0.0
        assert to_time_scalar(1.5) == 1.5
        assert to_time_scalar(100.0) == 100.0

        # Invalid values
        with pytest.raises(ValueError, match="non-negative"):
            to_time_scalar(-1.0)

    def test_to_price_scalar(self):
        """Test price scalar conversion."""
        # Valid values
        assert to_price_scalar(0.0) == 0.0
        assert to_price_scalar(99.99) == 99.99
        assert to_price_scalar(1000000.0) == 1000000.0

        # Invalid values
        with pytest.raises(ValueError, match="non-negative"):
            to_price_scalar(-0.01)

    def test_to_volume_scalar(self):
        """Test volume scalar conversion."""
        # Volume can be negative (e.g., net flow)
        assert to_volume_scalar(-100.0) == -100.0
        assert to_volume_scalar(0.0) == 0.0
        assert to_volume_scalar(1000.0) == 1000.0

    def test_to_frequency_scalar(self):
        """Test frequency scalar conversion."""
        # Valid values
        assert to_frequency_scalar(0.0) == 0.0
        assert to_frequency_scalar(60.0) == 60.0

        # Invalid values
        with pytest.raises(ValueError, match="non-negative"):
            to_frequency_scalar(-1.0)

    def test_to_rate_scalar(self):
        """Test rate scalar conversion."""
        # Rates can be negative
        assert to_rate_scalar(-0.1) == -0.1
        assert to_rate_scalar(0.0) == 0.0
        assert to_rate_scalar(0.5) == 0.5

    def test_to_dimensionless_scalar(self):
        """Test dimensionless scalar conversion."""
        # Any value is valid
        assert to_dimensionless_scalar(-1.0) == -1.0
        assert to_dimensionless_scalar(0.0) == 0.0
        assert to_dimensionless_scalar(1.0) == 1.0

    def test_to_percent_scalar(self):
        """Test percent scalar conversion."""
        # Valid values
        assert to_percent_scalar(0.0) == 0.0
        assert to_percent_scalar(50.0) == 50.0
        assert to_percent_scalar(100.0) == 100.0

        # Invalid values
        with pytest.raises(ValueError, match="must be in"):
            to_percent_scalar(-1.0)
        with pytest.raises(ValueError, match="must be in"):
            to_percent_scalar(101.0)

    def test_to_fraction_scalar(self):
        """Test fraction scalar conversion."""
        # Valid values
        assert to_fraction_scalar(0.0) == 0.0
        assert to_fraction_scalar(0.5) == 0.5
        assert to_fraction_scalar(1.0) == 1.0

        # Invalid values
        with pytest.raises(ValueError, match="must be in"):
            to_fraction_scalar(-0.1)
        with pytest.raises(ValueError, match="must be in"):
            to_fraction_scalar(1.1)

    def test_helpers_return_floats(self):
        """Test that all helper functions return actual floats."""
        # Runtime check that helpers return raw floats
        assert type(to_time_scalar(1.0)) is float
        assert type(to_price_scalar(1.0)) is float
        assert type(to_volume_scalar(1.0)) is float
        assert type(to_frequency_scalar(1.0)) is float
        assert type(to_rate_scalar(1.0)) is float
        assert type(to_dimensionless_scalar(1.0)) is float
        assert type(to_percent_scalar(50.0)) is float
        assert type(to_fraction_scalar(0.5)) is float


class TestAnnotatedTypes:
    """Test Annotated type aliases."""

    def test_unit_metadata(self):
        """Test Unit metadata class."""
        unit = Unit("time")
        assert unit.dimension == "time"
        assert repr(unit) == "Unit('time')"

    def test_annotated_aliases(self):
        """Test that Annotated aliases are properly formed."""
        # Check TimeValue structure
        assert get_origin(TimeValue) is Annotated
        args = get_args(TimeValue)
        assert args[0] is float
        assert isinstance(args[1], Unit)
        assert args[1].dimension == "time"

        # Check PriceValue structure
        assert get_origin(PriceValue) is Annotated
        args = get_args(PriceValue)
        assert args[0] is float
        assert isinstance(args[1], Unit)
        assert args[1].dimension == "price"

        # Check compound types
        assert get_origin(PriceRateValue) is Annotated
        args = get_args(PriceRateValue)
        assert args[0] is float
        assert isinstance(args[1], Unit)
        assert args[1].dimension == "price/time"


class TestQuantityGeneric:
    """Test generic Quantity type."""

    def test_quantity_creation(self):
        """Test creating Quantity instances."""
        q = Quantity(10.0, "time")
        assert q.value == 10.0
        assert q.dimension == "time"
        assert repr(q) == "Quantity(10.0, 'time')"

    def test_quantity_to_scalar(self):
        """Test extracting scalar from Quantity."""
        q = Quantity(5.5, "price")
        assert q.to_scalar() == 5.5
        assert isinstance(q.to_scalar(), float)


class TestPintIntegration:
    """Test integration with pint quantities."""

    def test_from_pint_quantity(self):
        """Test converting from pint quantity."""
        from ethode.units import UnitManager

        manager = UnitManager.instance()

        # Create a pint quantity
        qty = manager.registry.Quantity(5.0, "hour")

        # Convert to scalar (should convert to seconds)
        scalar = from_pint_quantity(qty, "time")
        assert scalar == 5.0 * 3600  # 5 hours in seconds

        # Wrong dimension should raise
        with pytest.raises(ValueError, match="Cannot convert"):
            from_pint_quantity(qty, "price")

    def test_to_pint_quantity(self):
        """Test converting to pint quantity."""
        from ethode.units import UnitManager

        manager = UnitManager.instance()

        # Convert scalar to pint quantity
        qty = to_pint_quantity(3600.0, "time")
        assert qty.magnitude == 3600.0
        assert qty.units == manager.registry.second

        # Unknown dimension should raise
        with pytest.raises(ValueError, match="Unknown dimension"):
            to_pint_quantity(1.0, "unknown_dimension")


class TestTypeGuards:
    """Test type guard functions."""

    def test_is_time_scalar(self):
        """Test time scalar type guard."""
        assert is_time_scalar(0.0) is True
        assert is_time_scalar(1.0) is True
        assert is_time_scalar(-1.0) is False
        assert is_time_scalar("1.0") is False

    def test_is_price_scalar(self):
        """Test price scalar type guard."""
        assert is_price_scalar(0.0) is True
        assert is_price_scalar(100.0) is True
        assert is_price_scalar(-1.0) is False
        assert is_price_scalar(None) is False

    def test_is_percent_scalar(self):
        """Test percent scalar type guard."""
        assert is_percent_scalar(0.0) is True
        assert is_percent_scalar(50.0) is True
        assert is_percent_scalar(100.0) is True
        assert is_percent_scalar(-1.0) is False
        assert is_percent_scalar(101.0) is False

    def test_is_fraction_scalar(self):
        """Test fraction scalar type guard."""
        assert is_fraction_scalar(0.0) is True
        assert is_fraction_scalar(0.5) is True
        assert is_fraction_scalar(1.0) is True
        assert is_fraction_scalar(-0.1) is False
        assert is_fraction_scalar(1.1) is False


class TestDimensionMapping:
    """Test dimension to type mapping."""

    def test_dimension_to_type_mapping(self):
        """Test DIMENSION_TO_TYPE dictionary."""
        assert DIMENSION_TO_TYPE["time"] is TimeScalar
        assert DIMENSION_TO_TYPE["price"] is PriceScalar
        assert DIMENSION_TO_TYPE["volume"] is VolumeScalar
        assert DIMENSION_TO_TYPE["frequency"] is FrequencyScalar
        assert DIMENSION_TO_TYPE["1/time"] is RateScalar
        assert DIMENSION_TO_TYPE["dimensionless"] is DimensionlessScalar
        assert DIMENSION_TO_TYPE["price/time"] is PriceRateScalar
        assert DIMENSION_TO_TYPE["volume/time"] is VolumeRateScalar

    def test_scalar_for_dimension(self):
        """Test automatic type selection by dimension."""
        # Known dimensions return typed scalars
        t = scalar_for_dimension(1.0, "time")
        assert t == 1.0
        assert isinstance(t, float)

        p = scalar_for_dimension(100.0, "price")
        assert p == 100.0
        assert isinstance(p, float)

        # Unknown dimension returns plain float
        unknown = scalar_for_dimension(42.0, "unknown")
        assert unknown == 42.0
        assert isinstance(unknown, float)


class TestTypeSafety:
    """Test that types provide safety (for mypy checking)."""

    def test_type_annotations_example(self):
        """Example of how type annotations would be used."""
        # This function would be type-checked by mypy
        def process_time(t: TimeScalar) -> TimeScalar:
            # Process time value
            return TimeScalar(t * 2)

        # This would pass type checking
        result = process_time(to_time_scalar(5.0))
        assert result == 10.0

        # This would fail mypy (but works at runtime)
        # result = process_time(to_price_scalar(5.0))  # mypy error

    def test_annotated_usage_example(self):
        """Example using Annotated types."""
        def calculate_rate(distance: float, time: TimeValue) -> RateValue:
            # In practice, mypy plugins could validate dimensions
            return distance / time

        # Usage
        rate = calculate_rate(100.0, 10.0)
        assert rate == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])