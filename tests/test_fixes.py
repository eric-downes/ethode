"""Tests for specific bug fixes in the unit system."""

import pytest
from pathlib import Path
import tempfile

from ethode.units import UnitManager, UnitSpec


class TestPackageRelativeLoading:
    """Test that unit files are loaded from package directory."""

    def test_loads_from_package_directory(self):
        """Test that default unit files are loaded from ethode package directory."""
        # Create a fresh manager
        UnitManager._instance = None
        manager = UnitManager()

        # Check that ETH units were loaded (these are in ethode/eth_units.txt)
        q_eth = manager.ensure_quantity("1 ETH")
        assert q_eth.magnitude == 1

        # Check that wei conversion works
        q_wei = manager.ensure_quantity("1e18 wei")
        assert q_wei.to("ETH").magnitude == pytest.approx(1)

    def test_explicit_paths_still_work(self):
        """Test that explicitly provided paths still work."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test_unit = [test_dimension]\n")
            f.write("kilo_test = 1000 * test_unit\n")
            temp_path = Path(f.name)

        try:
            manager = UnitManager()
            manager.load_aliases(temp_path)

            q = manager.ensure_quantity("5 kilo_test")
            assert q.to("test_unit").magnitude == 5000
        finally:
            temp_path.unlink()


class TestCanonicalConversionWithZero:
    """Test canonical conversion doesn't depend on magnitude."""

    def test_zero_magnitude_conversion(self):
        """Test that conversion works correctly with zero magnitude."""
        manager = UnitManager()

        # Create a zero-valued quantity in hours
        q_zero = manager.registry.Quantity(0, "hour")
        value, spec = manager.to_canonical(q_zero, "time")

        # Value should be 0 in canonical units (seconds)
        assert value == 0

        # Conversion factor should still be correct (1 hour = 3600 seconds)
        assert spec.to_canonical == 3600

        # Reconstruction should work
        reconstructed = manager.from_canonical(0, spec)
        assert reconstructed.magnitude == 0
        assert str(reconstructed.units) == "hour"

    def test_non_zero_after_zero(self):
        """Test that non-zero values work correctly after zero."""
        manager = UnitManager()

        # First convert zero
        q_zero = manager.registry.Quantity(0, "minute")
        value_zero, spec_zero = manager.to_canonical(q_zero, "time")
        assert value_zero == 0
        assert spec_zero.to_canonical == 60  # 1 minute = 60 seconds

        # Now convert a non-zero value with the same spec
        # This tests that the spec from zero conversion works correctly
        reconstructed = manager.from_canonical(120, spec_zero)  # 120 seconds
        assert reconstructed.magnitude == 2  # Should be 2 minutes
        assert str(reconstructed.units) == "minute"

    def test_various_zero_units(self):
        """Test zero conversion across different unit types."""
        manager = UnitManager()

        test_cases = [
            ("0 meter", "length", 1.0),  # meter is canonical
            ("0 kilometer", "length", 1000.0),
            ("0 day", "time", 86400.0),
            ("0 USD", "price", 1.0),  # USD is canonical for price
        ]

        for quantity_str, dimension, expected_factor in test_cases:
            q = manager.ensure_quantity(quantity_str)
            value, spec = manager.to_canonical(q, dimension)

            assert value == 0, f"Failed for {quantity_str}"
            assert spec.to_canonical == pytest.approx(expected_factor), \
                f"Wrong factor for {quantity_str}: got {spec.to_canonical}, expected {expected_factor}"


class TestRemovedQuantityAnnotation:
    """Test that removing QuantityAnnotation doesn't break imports."""

    def test_fields_module_imports(self):
        """Test that fields module still imports without QuantityAnnotation."""
        # This should not raise ImportError
        from ethode.fields import quantity_field, tuple_quantity_field

        # These should still work
        validator = quantity_field("time", "second")
        assert callable(validator)

        tuple_validator = tuple_quantity_field("price", "USD")
        assert callable(tuple_validator)

    def test_no_annotation_in_namespace(self):
        """Verify QuantityAnnotation is not in module namespace."""
        import ethode.fields as fields

        # Should not have QuantityAnnotation
        assert not hasattr(fields, 'QuantityAnnotation')

        # But should have the other functions
        assert hasattr(fields, 'quantity_field')
        assert hasattr(fields, 'tuple_quantity_field')
        assert hasattr(fields, 'create_quantity_validator')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])