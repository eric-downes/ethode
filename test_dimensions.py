"""Tests for controller dimension schema system."""

import pytest
from ethode import (
    ControllerConfig,
    ControllerDimensions,
    DIMENSIONLESS,
    FINANCIAL,
    TEMPERATURE,
    VELOCITY,
    POSITION,
)


class TestControllerDimensions:
    """Test ControllerDimensions schema class."""

    def test_default_financial_schema(self):
        """Test default financial schema."""
        dims = FINANCIAL
        assert dims.error_dim == "price"
        assert dims.output_dim == "price/time"
        assert dims.time_dim == "time"

    def test_dimensionless_schema(self):
        """Test dimensionless controller schema."""
        dims = DIMENSIONLESS
        assert dims.error_dim == "dimensionless"
        assert dims.output_dim == "dimensionless"

    def test_temperature_schema(self):
        """Test temperature controller schema."""
        dims = TEMPERATURE
        assert dims.error_dim == "temperature"
        assert dims.output_dim == "dimensionless"

    def test_expected_gain_dimensions_financial(self):
        """Test expected gain dimensions for financial controller."""
        dims = FINANCIAL  # price -> price/time

        # kp: (price/time) / price = 1/time
        assert "1/time" in dims.expected_gain_dimension("kp").lower() or \
               "frequency" in dims.expected_gain_dimension("kp").lower()

        # ki: (price/time) / (price * time) = 1/time²
        assert "1/time" in dims.expected_gain_dimension("ki").lower() and \
               "2" in dims.expected_gain_dimension("ki")

        # kd: (price/time * time) / price = dimensionless
        assert dims.expected_gain_dimension("kd") == "dimensionless"

    def test_expected_gain_dimensions_dimensionless(self):
        """Test expected gain dimensions for dimensionless controller."""
        dims = DIMENSIONLESS

        # All dimensionless -> gains should be dimensionless, 1/time, time
        assert dims.expected_gain_dimension("kp") == "dimensionless"
        assert dims.expected_gain_dimension("ki") == "1/time"
        assert dims.expected_gain_dimension("kd") == "time"

    def test_expected_gain_dimensions_temperature(self):
        """Test expected gain dimensions for temperature controller."""
        dims = TEMPERATURE  # temperature -> dimensionless

        # kp: dimensionless / temperature = 1/temperature
        assert "1/temperature" in dims.expected_gain_dimension("kp")

        # ki: dimensionless / (temperature * time) = 1/(temperature*time)
        assert "temperature" in dims.expected_gain_dimension("ki").lower()
        assert "time" in dims.expected_gain_dimension("ki").lower()

        # kd: (dimensionless * time) / temperature = time/temperature
        assert "time/temperature" in dims.expected_gain_dimension("kd").lower()

    def test_from_string(self):
        """Test creating dimensions from string specification."""
        # Full spec
        dims = ControllerDimensions.from_string("length->force")
        assert dims.error_dim == "length"
        assert dims.output_dim == "force"

        # Just error dimension (uses default output)
        dims = ControllerDimensions.from_string("temperature")
        assert dims.error_dim == "temperature"
        assert dims.output_dim == "price/time"  # Default

    def test_expected_dimensions_dict(self):
        """Test getting all expected dimensions."""
        dims = FINANCIAL
        expected = dims.expected_dimensions()

        assert "kp" in expected
        assert "ki" in expected
        assert "kd" in expected
        assert "tau" in expected
        assert "noise_band" in expected
        assert "output" in expected
        assert "rate_limit" in expected

        # Check specific values
        assert expected["tau"] == "time"
        assert expected["noise_band"] == "price"
        assert expected["output"] == "price/time"
        assert expected["rate_limit"] == "price/time/time"


class TestControllerConfigWithDimensions:
    """Test ControllerConfig with dimension schemas."""

    def test_default_dimensions(self):
        """Test that default dimensions are financial."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day / day",
            kd="0.0",
            tau="7 day",
            noise_band=("0.001 USD", "0.003 USD"),
        )

        assert config.dimensions.error_dim == "price"
        assert config.dimensions.output_dim == "price/time"

    def test_explicit_dimensionless(self):
        """Test explicit dimensionless schema."""
        config = ControllerConfig(
            dimensions=DIMENSIONLESS,
            kp=1.0,               # dimensionless
            ki="0.1 / second",    # 1/time
            kd="0.01 second",     # time
            tau=100.0,
            noise_band=(0.001, 0.003),
        )

        assert config.dimensions == DIMENSIONLESS

        # Validate units should pass
        report = config.validate_units()
        assert report.success
        assert len(report.errors) == 0

    def test_temperature_controller(self):
        """Test temperature controller with proper schema."""
        config = ControllerConfig(
            dimensions=TEMPERATURE,
            kp="0.5 / kelvin",           # 1/temperature
            ki="0.05 / kelvin / minute", # 1/(temperature*time)
            kd="0.01 minute / kelvin",   # time/temperature
            tau="10 minute",
            noise_band=("0.1 kelvin", "0.5 kelvin"),
            output_min=0.0,              # dimensionless (heater %)
            output_max=100.0,
        )

        assert config.dimensions == TEMPERATURE

        # This should now validate successfully
        report = config.validate_units()
        assert report.success
        assert len(report.errors) == 0

    def test_financial_controller_with_schema(self):
        """Test financial controller with explicit schema."""
        config = ControllerConfig(
            dimensions=FINANCIAL,
            kp="0.2 / day",           # 1/time
            ki="0.02 / day / day",    # 1/time²
            kd="0.0",                 # dimensionless
            tau="1 week",
            noise_band=("1 milliUSD", "3 milliUSD"),
            output_min="-100 USD/day",
            output_max="100 USD/day",
            rate_limit="10 USD/day/hour",
        )

        assert config.dimensions == FINANCIAL

        # Should validate successfully
        report = config.validate_units()
        assert report.success

    def test_velocity_controller(self):
        """Test velocity controller with custom schema."""
        config = ControllerConfig(
            dimensions=VELOCITY,  # length -> length/time
            kp="2 / second",      # 1/time
            ki="0.1 / second / second",  # 1/time²
            kd="0.05",           # dimensionless
            tau="10 second",
            noise_band=("0.01 meter", "0.05 meter"),
            output_min="-10 meter/second",
            output_max="10 meter/second",
        )

        assert config.dimensions == VELOCITY

        # Should validate
        report = config.validate_units()
        assert report.success

    def test_custom_dimensions(self):
        """Test custom dimension schema."""
        # Create custom schema for pressure control
        pressure_dims = ControllerDimensions(
            error_dim="pressure",
            output_dim="flow_rate"
        )

        config = ControllerConfig(
            dimensions=pressure_dims,
            kp="0.1 liter/second/pascal",    # flow_rate/pressure
            ki="0.01 liter/second/pascal/second",  # flow_rate/(pressure*time)
            kd="0.001 liter*second/second/pascal",  # (flow_rate*time)/pressure
            tau="5 second",
            noise_band=("10 pascal", "50 pascal"),
        )

        assert config.dimensions.error_dim == "pressure"
        assert config.dimensions.output_dim == "flow_rate"

    def test_runtime_with_check_units(self):
        """Test that check_units uses dimension schema."""
        # This would fail without proper schema
        config = ControllerConfig(
            dimensions=TEMPERATURE,
            kp="0.5 / kelvin",
            ki="0.05 / kelvin / minute",
            kd="0.01 minute / kelvin",
            tau="10 minute",
            noise_band=("0.1 kelvin", "0.5 kelvin"),
        )

        # Should work with check_units=True now
        runtime = config.to_runtime(check_units=True)
        assert runtime is not None

    def test_incompatible_gains_detected(self):
        """Test that incompatible gains are still detected."""
        config = ControllerConfig(
            dimensions=FINANCIAL,  # Expects specific dimensions
            kp="2 kilogram",       # Wrong dimension!
            ki="0.02 / day / day",
            kd="0.0",
            tau="7 day",
            noise_band=("0.001 USD", "0.003 USD"),
        )

        # Should fail validation
        with pytest.raises(ValueError) as exc_info:
            config.to_runtime(check_units=True)

        assert "validation failed" in str(exc_info.value).lower()


class TestDimensionValidation:
    """Test dimension validation with schemas."""

    def test_dimensionless_no_warning(self):
        """Test that dimensionless schema doesn't produce warnings."""
        config = ControllerConfig(
            dimensions=DIMENSIONLESS,
            kp=1.0,
            ki="0.1 / second",
            kd="0.01 second",
            tau=100.0,
            noise_band=(0.001, 0.003),
        )

        report = config.validate_units()
        assert report.success

        # Should have no warnings about dimensionless
        for warning in report.warnings:
            assert "dimensionless" not in warning.lower()

    def test_mixed_units_with_schema(self):
        """Test that schema allows mixed unit systems."""
        # Mix metric and imperial with proper schema
        custom_dims = ControllerDimensions(
            error_dim="length",
            output_dim="force"
        )

        config = ControllerConfig(
            dimensions=custom_dims,
            kp="10 pound_force / inch",      # force/length
            ki="1 pound_force / inch / second",  # force/(length*time)
            kd="0.1 pound_force * second / inch",  # (force*time)/length
            tau="10 second",
            noise_band=("0.01 inch", "0.05 inch"),
            output_min="-100 pound_force",
            output_max="100 pound_force",
        )

        # Should validate despite mixed units
        report = config.validate_units()
        assert report.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])