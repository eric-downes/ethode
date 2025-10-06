"""Tests for unit validation system."""

import pytest
import jax.numpy as jnp
import pint

from ethode.units import UnitManager
from ethode.runtime import QuantityNode, ControllerRuntime, ControllerState
from ethode.controller import ControllerConfig
from ethode.validation import (
    controller_step_units,
    validate_controller_dimensions,
    validate_config_units,
    ValidationReport
)


class TestQuantityNodeHelpers:
    """Test QuantityNode to_quantity and from_quantity helpers."""

    def test_to_quantity(self):
        """Test converting QuantityNode to pint Quantity."""
        manager = UnitManager.instance()

        # Create a QuantityNode
        node = QuantityNode.from_float(
            3600.0,  # 1 hour in seconds
            units=manager.to_canonical(
                manager.ensure_quantity("1 hour"), "time"
            )[1]
        )

        # Convert back to quantity
        q = node.to_quantity(manager)

        # Should recover original units
        assert q.magnitude == pytest.approx(1.0)  # 1 hour
        assert str(q.units) == "hour"

    def test_from_quantity(self):
        """Test creating QuantityNode from pint Quantity."""
        manager = UnitManager.instance()

        # Create a pint quantity
        q = manager.ensure_quantity("7 day")

        # Convert to QuantityNode
        node = QuantityNode.from_quantity(q, "time", manager)

        # Check canonical value (7 days in seconds)
        assert float(node.value) == pytest.approx(7 * 86400)
        assert node.units.dimension == "time"
        assert node.units.symbol == "day"

    def test_round_trip(self):
        """Test round-trip conversion."""
        manager = UnitManager.instance()

        # Start with a quantity
        q_orig = manager.ensure_quantity("0.2 / hour")

        # Convert to node and back
        node = QuantityNode.from_quantity(q_orig, "frequency", manager)
        q_recovered = node.to_quantity(manager)

        # Should be equivalent
        assert q_recovered.magnitude == pytest.approx(q_orig.magnitude)
        assert q_recovered.dimensionality == q_orig.dimensionality


class TestControllerStepUnits:
    """Test pint-based controller step validation."""

    def test_basic_step(self):
        """Test basic controller step with units."""
        manager = UnitManager.instance()

        # Create config and runtime
        # For dimensional consistency with USD error:
        # - P: kp * error → (1/time) * USD = USD/time
        # - I: ki * (error*time) → (1/time²) * (USD*time) = USD/time
        # - D: kd * (error/time) → dimensionless * (USD/time) = USD/time
        config = ControllerConfig(
            kp="0.2 / hour",         # 1/time: produces USD/hour
            ki="0.02 / hour / hour", # 1/time²: produces USD/hour
            kd="0.0",                # dimensionless: produces USD/hour
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )
        runtime = config.to_runtime(check_units=False)

        # Create test inputs
        error = manager.ensure_quantity("5 milliUSD")
        dt = manager.ensure_quantity("1 hour")
        state = ControllerState.zero()

        # Run validation step
        state_dict, output = controller_step_units(runtime, state, error, dt, manager)

        # Check output has correct dimensions (USD/time since gains produce rates)
        expected_dim = (manager.registry.USD / manager.registry.second).dimensionality
        assert output.dimensionality == expected_dim

        # Check state updates have correct dimensions
        assert state_dict['integral'].dimensionality == (
            manager.registry.USD * manager.registry.second
        ).dimensionality
        assert state_dict['last_error'].dimensionality == manager.registry.USD.dimensionality
        assert state_dict['time'].dimensionality == manager.registry.second.dimensionality

    def test_dimension_mismatch_caught(self):
        """Test that dimension mismatches are caught."""
        manager = UnitManager.instance()

        # Create config with incompatible units
        # kp with time dimension (wrong - should be 1/time)
        config = ControllerConfig(
            kp="2 hour",  # Wrong dimension!
            ki="0.02 / hour",
            kd="0.1 second",
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )
        runtime = config.to_runtime(check_units=False)

        error = manager.ensure_quantity("5 milliUSD")
        dt = manager.ensure_quantity("1 hour")
        state = ControllerState.zero()

        # This should raise a dimensionality error
        with pytest.raises(pint.DimensionalityError):
            controller_step_units(runtime, state, error, dt, manager)


class TestValidateControllerDimensions:
    """Test controller dimension validation."""

    def test_valid_dimensions(self):
        """Test validation with correct dimensions."""
        # For dimensional consistency:
        # kp: 1/time, ki: 1/time², kd: dimensionless
        config = ControllerConfig(
            kp="0.2 / day",           # 1/time
            ki="0.02 / day / day",    # 1/time²
            kd="0.0",                 # dimensionless
            tau="7 day",
            noise_band=("0.001 USD", "0.003 USD"),
        )
        runtime = config.to_runtime(check_units=False)

        # Should validate successfully
        dimensions = validate_controller_dimensions(runtime)

        # Check returned dimensions
        assert 'kp' in dimensions
        assert 'ki' in dimensions
        assert 'kd' in dimensions
        assert 'tau' in dimensions
        assert 'output' in dimensions

    def test_invalid_dimensions(self):
        """Test validation catches dimension errors."""
        # Create config with wrong dimensions
        config = ControllerConfig(
            kp="2 meter",  # Completely wrong dimension
            ki="0.02 / hour",
            kd="0.1 second",
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )
        runtime = config.to_runtime(check_units=False)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            validate_controller_dimensions(runtime)

        assert "dimension validation failed" in str(exc_info.value).lower()


class TestControllerConfigValidation:
    """Test ControllerConfig validation methods."""

    def test_validate_units_method(self):
        """Test the validate_units() method."""
        config = ControllerConfig(
            kp="0.2 / hour",            # 1/time
            ki="0.02 / hour / hour",    # 1/time²
            kd="0.0",                   # dimensionless
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )

        # Validate units
        report = config.validate_units()

        assert isinstance(report, ValidationReport)
        assert report.success
        assert len(report.errors) == 0
        assert 'kp' in report.dimensions

    def test_to_runtime_with_check(self):
        """Test to_runtime with unit checking enabled."""
        # Good config
        config = ControllerConfig(
            kp="0.2 / hour",           # 1/time
            ki="0.02 / hour / hour",   # 1/time²
            kd="0.0",                  # dimensionless
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )

        # Should work with check_units=True
        runtime = config.to_runtime(check_units=True)
        assert isinstance(runtime, ControllerRuntime)

        # Bad config (wrong dimensions)
        bad_config = ControllerConfig(
            kp="2 kilogram",  # Mass instead of frequency
            ki="0.02 / hour",
            kd="0.1 second",
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )

        # Should fail with check_units=True
        with pytest.raises(ValueError) as exc_info:
            bad_config.to_runtime(check_units=True)

        assert "unit validation failed" in str(exc_info.value).lower()

        # But should work with check_units=False
        runtime = bad_config.to_runtime(check_units=False)
        assert isinstance(runtime, ControllerRuntime)

    def test_validation_report_formatting(self):
        """Test ValidationReport string formatting."""
        report = ValidationReport(
            success=True,
            dimensions={
                'kp': '[frequency]',
                'ki': '[frequency] ** 2',
                'tau': '[time]'
            },
            warnings=['Test warning'],
            errors=[]
        )

        report_str = str(report)
        assert "PASS" in report_str
        assert "kp: [frequency]" in report_str
        assert "Test warning" in report_str

        # Failed report
        failed_report = ValidationReport(
            success=False,
            dimensions={},
            warnings=[],
            errors=['Test error']
        )

        failed_str = str(failed_report)
        assert "FAIL" in failed_str
        assert "Test error" in failed_str


class TestValidateConfigUnits:
    """Test the general config validation function."""

    def test_validate_controller_config(self):
        """Test validating a controller config."""
        config = ControllerConfig(
            kp="0.2 / hour",          # 1/time
            ki="0.02 / hour / hour",  # 1/time²
            kd="0.0",                 # dimensionless
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )

        report = validate_config_units(config)
        assert report.success
        assert len(report.dimensions) > 0

    def test_dimensionally_consistent_config(self):
        """Test validation passes for dimensionally consistent configuration."""
        # Use dimensionally consistent gains for USD error
        # When error is USD:
        # - P: kp * error → (1/time) * USD = USD/time
        # - I: ki * integral → (1/time²) * (USD*time) = USD/time
        # - D: kd * derivative → dimensionless * (USD/time) = USD/time
        config = ControllerConfig(
            kp="1 / second",          # 1/time: produces USD/second output
            ki="0.1 / second / second",  # 1/time²: produces USD/second output
            kd="0.01",                # dimensionless
            tau=100.0,
            noise_band=("0.001 USD", "0.003 USD"),  # USD error
        )

        report = validate_config_units(config)
        assert report.success

        # Should NOT have a warning anymore since gains are properly dimensioned
        # The test name is misleading - we're actually testing proper dimensions now
        assert len(report.errors) == 0


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_financial_controller(self):
        """Test a financial trading controller."""
        # For dimensional consistency with USD error and USD/time output:
        # - P: kp * error → (1/time) * USD = USD/time
        # - I: ki * integral → (1/time²) * (USD*time) = USD/time
        # - D: kd * derivative → dimensionless * (USD/time) = USD/time
        config = ControllerConfig(
            kp="0.2 / day",           # 1/time: produces USD/day output
            ki="0.02 / day / day",    # 1/time²: produces USD/day output
            kd="0.0",                 # dimensionless
            tau="1 week",
            noise_band=("1 milliUSD", "3 milliUSD"),
            output_min="-100 USD/day",     # Rate output in USD/day
            output_max="100 USD/day",      # Rate output in USD/day
            rate_limit="10 USD/day/hour",  # Change rate: (USD/day)/hour
        )

        # Validate units
        report = config.validate_units()
        assert report.success

        # Build runtime with checking
        runtime = config.to_runtime(check_units=True)
        assert runtime is not None

        # Verify rate limit has correct units
        if runtime.rate_limit:
            manager = UnitManager.instance()
            rate_q = runtime.rate_limit.to_quantity(manager)
            # Should be USD/time² (rate of change of USD/time output)
            expected_dim = (
                manager.registry.USD / manager.registry.second / manager.registry.second
            ).dimensionality
            assert rate_q.dimensionality == expected_dim

    def test_metric_controller(self):
        """Test a controller with metric units."""
        from ethode import VELOCITY
        manager = UnitManager.instance()

        # For dimensional consistency with meter error and meter/second output:
        # - P: kp * error → (1/second) * meter = meter/second
        # - I: ki * integral → (1/second²) * (meter*second) = meter/second
        # - D: kd * derivative → dimensionless * (meter/second) = meter/second
        config = ControllerConfig(
            dimensions=VELOCITY,  # length -> length/time schema
            kp="2 / second",          # 1/time: produces meter/second output
            ki="0.1 / second / second",  # 1/time²: produces meter/second output
            kd="0.05",                # dimensionless
            tau="10 second",
            noise_band=("0.01 meter", "0.05 meter"),
            output_min="-10 meter/second",  # Velocity output
            output_max="10 meter/second",   # Velocity output
        )

        # Validate
        report = config.validate_units()
        assert report.success

        # Build runtime
        runtime = config.to_runtime(check_units=True)
        assert runtime is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])