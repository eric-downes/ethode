"""Unit tests for the units module."""

import pytest
import pint
from pathlib import Path
from typing import Annotated, Union

from pydantic import BaseModel, ValidationError, field_validator

from ethode.units import UnitManager, UnitSpec, QuantityInput
from ethode.fields import (
    quantity_field,
    tuple_quantity_field,
    create_quantity_validator,
)


class TestUnitSpec:
    """Test UnitSpec dataclass."""

    def test_creation(self):
        """Test basic UnitSpec creation."""
        spec = UnitSpec(dimension="time", symbol="s", to_canonical=1.0)
        assert spec.dimension == "time"
        assert spec.symbol == "s"
        assert spec.to_canonical == 1.0

    def test_immutability(self):
        """Test that UnitSpec is immutable."""
        spec = UnitSpec(dimension="time", symbol="s")
        with pytest.raises(AttributeError):
            spec.dimension = "length"

    def test_hashable(self):
        """Test that UnitSpec can be used in sets/dicts."""
        spec1 = UnitSpec(dimension="time", symbol="s", to_canonical=1.0)
        spec2 = UnitSpec(dimension="time", symbol="s", to_canonical=1.0)
        spec3 = UnitSpec(dimension="time", symbol="min", to_canonical=60.0)

        # Same specs should have same hash
        assert hash(spec1) == hash(spec2)
        # Different specs should (likely) have different hash
        assert hash(spec1) != hash(spec3)

        # Can use in set
        spec_set = {spec1, spec2, spec3}
        assert len(spec_set) == 2  # spec1 and spec2 are equal


class TestUnitManager:
    """Test UnitManager functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnitManager._instance = None

    def test_singleton_pattern(self):
        """Test that UnitManager.instance() returns singleton."""
        manager1 = UnitManager.instance()
        manager2 = UnitManager.instance()
        assert manager1 is manager2

    def test_custom_registry(self):
        """Test creation with custom registry."""
        custom_reg = pint.UnitRegistry()
        manager = UnitManager(registry=custom_reg)
        assert manager.registry is custom_reg

    def test_canonical_units(self):
        """Test that canonical units are defined."""
        manager = UnitManager()
        assert "time" in manager.canonical_units
        assert "length" in manager.canonical_units
        assert "price" in manager.canonical_units

    def test_ensure_quantity_from_string(self):
        """Test converting string to quantity."""
        manager = UnitManager()

        # Parse unit string
        q = manager.ensure_quantity("5 meters")
        assert q.magnitude == 5
        assert str(q.units) == "meter"

        # Parse with math
        q = manager.ensure_quantity("2 * 3.5 seconds")
        assert q.magnitude == 7.0

    def test_ensure_quantity_from_number(self):
        """Test converting number to quantity."""
        manager = UnitManager()

        # Without default unit
        q = manager.ensure_quantity(42)
        assert q.magnitude == 42
        assert q.dimensionless

        # With default unit
        q = manager.ensure_quantity(42, default_unit="second")
        assert q.magnitude == 42
        assert str(q.units) == "second"

    def test_ensure_quantity_from_quantity(self):
        """Test that quantities pass through unchanged."""
        manager = UnitManager()
        original = manager.registry.Quantity(5, "meter")
        result = manager.ensure_quantity(original)
        assert result is original

    def test_ensure_quantity_invalid(self):
        """Test error handling for invalid inputs."""
        manager = UnitManager()
        with pytest.raises(ValueError, match="Cannot parse"):
            manager.ensure_quantity("not a valid unit string")

    def test_to_canonical_time(self):
        """Test conversion to canonical time units."""
        manager = UnitManager()

        # Convert hours to seconds (canonical)
        q = manager.registry.Quantity(2, "hour")
        value, spec = manager.to_canonical(q, "time")

        assert value == 7200  # 2 * 3600 seconds
        assert spec.dimension == "time"
        assert spec.symbol == "hour"
        assert spec.to_canonical == pytest.approx(3600)  # 1 hour = 3600 seconds

    def test_to_canonical_custom_dimension(self):
        """Test handling of unknown dimensions."""
        manager = UnitManager()
        q = manager.registry.Quantity(100, "volt")
        value, spec = manager.to_canonical(q, "voltage")

        assert value == 100
        assert spec.dimension == "voltage"
        assert spec.to_canonical == 1.0

    def test_to_canonical_dimension_mismatch(self):
        """Test error when dimensions don't match."""
        manager = UnitManager()
        q = manager.registry.Quantity(5, "meter")
        with pytest.raises(ValueError, match="Cannot convert"):
            manager.to_canonical(q, "time")

    def test_from_canonical(self):
        """Test reconstruction from canonical value."""
        manager = UnitManager()

        # Round trip: quantity -> canonical -> quantity
        original = manager.registry.Quantity(3, "hour")
        canonical_value, spec = manager.to_canonical(original, "time")
        reconstructed = manager.from_canonical(canonical_value, spec)

        assert reconstructed.magnitude == pytest.approx(3)
        assert str(reconstructed.units) == "hour"

    def test_infer_dimension(self):
        """Test dimension inference from quantity."""
        manager = UnitManager()

        # Known dimensions
        q = manager.registry.Quantity(5, "meter")
        assert manager.infer_dimension(q) == "length"

        q = manager.registry.Quantity(10, "second")
        assert manager.infer_dimension(q) == "time"

        # Unknown dimension returns dimensionality string
        q = manager.registry.Quantity(100, "volt")
        dim = manager.infer_dimension(q)
        assert "length" in dim.lower() or "[" in dim  # Volt = kg⋅m²⋅s⁻³⋅A⁻¹


class TestQuantityField:
    """Test quantity field validators."""

    def test_quantity_field_basic(self):
        """Test basic quantity field validation."""
        validator = quantity_field("time", default_unit="second")

        # String input
        value, spec = validator("5 minutes")
        assert value == 300  # 5 * 60 seconds
        assert spec.dimension == "time"

        # Number with default unit
        value, spec = validator(10)
        assert value == 10
        assert spec.symbol == "second"

    def test_quantity_field_bounds(self):
        """Test min/max value validation."""
        validator = quantity_field(
            "time",
            default_unit="second",
            min_value=0,
            max_value=3600
        )

        # Valid value
        value, spec = validator("30 minutes")
        assert value == 1800

        # Below minimum
        with pytest.raises(ValueError, match="below minimum"):
            validator("-5 seconds")

        # Above maximum
        with pytest.raises(ValueError, match="above maximum"):
            validator("2 hours")

    def test_tuple_quantity_field(self):
        """Test tuple quantity validation."""
        validator = tuple_quantity_field("price", default_unit="USD")

        # Valid tuple
        result = validator(("1 USD", "5 USD"))
        assert result[0][0] == 1  # First value
        assert result[1][0] == 5  # Second value

        # List also works
        result = validator([0.5, 2.5])
        assert result[0][0] == 0.5
        assert result[1][0] == 2.5

        # Invalid: not a tuple/list
        with pytest.raises(ValueError, match="Expected tuple"):
            validator("not a tuple")

        # Invalid: wrong length
        with pytest.raises(ValueError, match="Expected tuple of two"):
            validator((1, 2, 3))

        # Invalid: first > second
        with pytest.raises(ValueError, match="must be less than"):
            validator((10, 5))


class TestPydanticIntegration:
    """Test integration with Pydantic models."""

    def test_model_with_field_validator(self):
        """Test Pydantic model with field validators."""

        class TestConfig(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            duration: QuantityInput
            temperature: QuantityInput

            @field_validator("duration", mode='before')
            @classmethod
            def validate_duration(cls, v):
                value, spec = quantity_field("time", "second")(v)
                return value  # Store just the float

            @field_validator("temperature", mode='before')
            @classmethod
            def validate_temperature(cls, v):
                value, spec = quantity_field("temperature", "kelvin")(v)
                return value

        # Valid config
        config = TestConfig(
            duration="5 minutes",
            temperature="298.15 kelvin"  # Use kelvin directly to avoid offset unit issues
        )
        assert config.duration == 300  # seconds
        assert config.temperature == pytest.approx(298.15)  # kelvin

        # Invalid dimension
        with pytest.raises(ValidationError):
            TestConfig(
                duration="5 meters",  # Wrong dimension
                temperature="298.15 kelvin"
            )

    def test_model_with_tuple_field(self):
        """Test model with tuple quantity field."""

        class TestConfig(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            price_range: tuple[QuantityInput, QuantityInput]

            @field_validator("price_range", mode='before')
            @classmethod
            def validate_price_range(cls, v):
                result = tuple_quantity_field("price", "USD")(v)
                # Return just the values for storage
                return (result[0][0], result[1][0])

        config = TestConfig(price_range=("10 USD", "50 USD"))
        assert config.price_range == (10, 50)

    def test_model_with_annotated_types(self):
        """Test using Annotated types (if supported)."""

        class TestConfig(BaseModel):
            # Store both value and spec
            duration: tuple[float, UnitSpec]

            @field_validator("duration", mode='before')
            @classmethod
            def validate_duration(cls, v):
                return quantity_field("time", "second")(v)

        config = TestConfig(duration="2 hours")
        assert config.duration[0] == 7200  # seconds
        assert config.duration[1].dimension == "time"
        assert config.duration[1].symbol == "hour"

    def test_create_quantity_validator_helper(self):
        """Test the create_quantity_validator helper."""

        class TestConfig(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

            speed: QuantityInput

            # Use helper to create validator
            _validate_speed = create_quantity_validator(
                "speed",
                "length",  # Will fail for speed units
                "meter"
            )

        # This will fail because speed has dimension length/time
        # but we specified just "length"
        with pytest.raises(ValidationError):
            TestConfig(speed="10 meter/second")


class TestCustomUnits:
    """Test loading custom unit definitions from files."""

    def test_load_eth_units(self):
        """Test loading Ethereum units from file."""
        manager = UnitManager()

        # ETH units should be loaded automatically
        q = manager.ensure_quantity("1 ETH")
        assert q.magnitude == 1

        # Test wei conversion
        wei = manager.ensure_quantity("1e18 wei")
        eth = manager.ensure_quantity("1 ETH")
        assert wei.to("ETH").magnitude == pytest.approx(1)

        # Test Gwei
        gwei = manager.ensure_quantity("1 Gwei")
        assert gwei.to("wei").magnitude == pytest.approx(1e9)

    def test_custom_registry_passed_in(self):
        """Test using a custom pint registry."""
        # Create custom registry with special units
        custom_reg = pint.UnitRegistry()
        custom_reg.define("banana = [fruit]")
        custom_reg.define("bunch = 6 * banana")

        manager = UnitManager(registry=custom_reg)

        # Custom units should work
        q = manager.ensure_quantity("2 bunch")
        assert q.to("banana").magnitude == 12

        # Standard units should still work
        q = manager.ensure_quantity("5 meter")
        assert q.magnitude == 5

    def test_load_definitions_method(self):
        """Test loading additional definitions after initialization."""
        from tempfile import NamedTemporaryFile

        manager = UnitManager()

        # Create a temporary unit definition file
        with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("RAI = [StableCoin]\n")
            f.write("RD = [RAIDollar] = RAI_Dollar\n")
            f.write("kRD = 1000 * RD\n")
            temp_path = Path(f.name)

        try:
            # Load the custom definitions
            manager.load_aliases(temp_path)

            # Should be able to use the new units
            q = manager.ensure_quantity("5 kRD")
            assert q.to("RD").magnitude == 5000
        finally:
            # Clean up
            temp_path.unlink()


class TestRealWorldUnits:
    """Test with real-world unit scenarios."""

    def test_financial_units(self):
        """Test financial unit handling."""
        manager = UnitManager()

        # USD should be defined
        q = manager.ensure_quantity("100 USD")
        assert q.magnitude == 100

        value, spec = manager.to_canonical(q, "price")
        assert value == 100
        assert spec.dimension == "price"

    def test_time_aliases(self):
        """Test common time aliases."""
        manager = UnitManager()

        # Test various time units
        for unit_str, expected_seconds in [
            ("1 second", 1),
            ("1 minute", 60),
            ("1 hour", 3600),
            ("1 day", 86400),
            ("1 week", 604800),
        ]:
            q = manager.ensure_quantity(unit_str)
            value, spec = manager.to_canonical(q, "time")
            assert value == pytest.approx(expected_seconds, rel=1e-6)

    def test_controller_typical_units(self):
        """Test units typical for PID controller."""
        validator_kp = quantity_field("dimensionless")  # or 1/time
        validator_tau = quantity_field("time", "second")

        # Proportional gain (often dimensionless or 1/time)
        value, spec = validator_kp("0.2")
        assert value == 0.2

        # Time constant
        value, spec = validator_tau("24 hours")
        assert value == 86400