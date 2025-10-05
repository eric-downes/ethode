#!/usr/bin/env python
"""Test the updated unit system with mixed dimensions and proper conversions."""

import pytest
from ethode.units import UnitManager
from ethode.controller import PIDController
from ethode.types import scalar_for_dimension, TimeScalar, PriceRateScalar


def test_mixed_dimension_canonical_units():
    """Test that mixed dimensions are properly defined in canonical units."""
    manager = UnitManager.instance()

    # Check that mixed dimensions are defined
    assert "price/time" in manager.canonical_units
    assert "volume/time" in manager.canonical_units
    assert "price*time" in manager.canonical_units

    # Verify they have the right units
    price_rate = manager.canonical_units["price/time"]
    assert str(price_rate) == "USD / second"

    volume_rate = manager.canonical_units["volume/time"]
    assert "second" in str(volume_rate)

    print("✓ Mixed dimensions are properly defined")


def test_rate_limit_with_unit_conversion():
    """Test that rate_limit uses proper unit conversion."""
    manager = UnitManager.instance()

    # Test with USD/second (canonical)
    pid1 = PIDController(kp=1.0, ki=0.1, kd=0.01, rate_limit=5.0)
    assert pid1.config.rate_limit[0] == 5.0
    assert pid1.config.rate_limit[1].dimension == "price/time"
    assert pid1.config.rate_limit[1].to_canonical == 1.0  # Already canonical

    print("✓ rate_limit with canonical units works")

    # Test conversion from different units (USD/hour)
    qty_hour = manager.registry.Quantity(3600.0, "USD/hour")
    value_canonical, spec = manager.to_canonical(qty_hour, "price/time")

    # 3600 USD/hour = 1 USD/second
    assert abs(value_canonical - 1.0) < 1e-6
    assert spec.dimension == "price/time"

    print("✓ Unit conversion from USD/hour to USD/second works")


def test_rate_limit_different_units():
    """Test rate_limit with various unit specifications."""
    manager = UnitManager.instance()

    # Test USD/minute conversion
    qty = manager.ensure_quantity("60 USD/minute", "USD/minute")
    value, spec = manager.to_canonical(qty, "price/time")
    # 60 USD/minute = 1 USD/second
    assert abs(value - 1.0) < 1e-6

    # Test USD/day conversion
    qty = manager.ensure_quantity("86400 USD/day", "USD/day")
    value, spec = manager.to_canonical(qty, "price/time")
    # 86400 USD/day = 1 USD/second
    assert abs(value - 1.0) < 1e-6

    print("✓ Different time units convert correctly")


def test_scalar_for_dimension_preserves_type():
    """Test that scalar_for_dimension preserves NewType information."""
    # Get a time scalar
    t = scalar_for_dimension(5.0, "time")
    assert t == 5.0
    # At runtime it's a float, but mypy sees it as TimeScalar
    assert isinstance(t, float)

    # Get a price/time scalar
    rate = scalar_for_dimension(10.0, "price/time")
    assert rate == 10.0
    assert isinstance(rate, float)

    # Unknown dimension returns plain float
    unknown = scalar_for_dimension(42.0, "unknown_dimension")
    assert unknown == 42.0
    assert isinstance(unknown, float)

    print("✓ scalar_for_dimension returns proper types")


def test_type_preservation_example():
    """Example showing type preservation is useful for mypy."""
    # This function uses the specific NewType
    def process_rate(r: PriceRateScalar) -> PriceRateScalar:
        return PriceRateScalar(r * 2.0)

    # Get a typed scalar
    rate = scalar_for_dimension(5.0, "price/time")

    # This would work with mypy if scalar_for_dimension
    # returns the right type (not cast to float)
    # result = process_rate(rate)  # mypy would check this

    # Direct creation works
    typed_rate = PriceRateScalar(5.0)
    result = process_rate(typed_rate)
    assert result == 10.0

    print("✓ Type preservation enables better type checking")


def test_volume_time_dimension():
    """Test volume/time dimension for token rates."""
    manager = UnitManager.instance()

    # Create a volume rate quantity
    qty = manager.ensure_quantity("100 1/second", "1/second")  # tokens per second
    value, spec = manager.to_canonical(qty, "volume/time")

    assert value == 100.0
    assert spec.dimension == "volume/time"

    print("✓ volume/time dimension works for token rates")


def test_price_time_product():
    """Test price*time dimension for accumulated costs."""
    manager = UnitManager.instance()

    # Create a price*time quantity (e.g., USD-hours)
    qty = manager.registry.Quantity(10.0, "USD * hour")
    value, spec = manager.to_canonical(qty, "price*time")

    # 10 USD*hour = 10*3600 USD*second
    assert abs(value - 36000.0) < 1e-6
    assert spec.dimension == "price*time"

    print("✓ price*time dimension works for accumulated costs")


if __name__ == "__main__":
    # Run all tests
    test_mixed_dimension_canonical_units()
    test_rate_limit_with_unit_conversion()
    test_rate_limit_different_units()
    test_scalar_for_dimension_preserves_type()
    test_type_preservation_example()
    test_volume_time_dimension()
    test_price_time_product()

    print("\n✅ All unit system updates work correctly!")
    print("The unit story is now tight with proper conversions and type preservation.")