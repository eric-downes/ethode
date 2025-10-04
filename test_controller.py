"""Tests for controller configuration and kernel."""

import pytest
import jax
import jax.numpy as jnp
from pydantic import ValidationError

from ethode import UnitManager, UnitSpec
from ethode.controller import ControllerConfig, ControllerConfigOutput, controller_step
from ethode.runtime import ControllerState


class TestControllerConfig:
    """Test ControllerConfig Pydantic model."""

    def test_create_with_strings(self):
        """Test creating config with string inputs."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.2 / day / 7 day",
            kd="0.0",
            tau="7 day",
            noise_band=("0.001 USD", "0.003 USD"),
        )

        # Check values were parsed and converted
        # "0.2 / day" should be converted to Hz (canonical for frequency)
        expected_kp = 0.2 / 86400  # 0.2 per day in Hz
        assert config.kp[0] == pytest.approx(expected_kp, rel=1e-6)
        assert config.tau[0] == pytest.approx(7 * 86400)  # 7 days in seconds

    def test_create_with_floats(self):
        """Test creating config with float inputs."""
        config = ControllerConfig(
            kp=0.2,
            ki=0.02,
            kd=0.0,
            tau=86400.0,
            noise_band=(0.001, 0.003),
        )

        # Floats should use default units
        assert config.kp[0] == 0.2
        assert config.tau[0] == 86400.0

    def test_create_with_pint_quantities(self):
        """Test creating config with pint Quantity inputs."""
        manager = UnitManager.instance()

        config = ControllerConfig(
            kp=manager.ensure_quantity("0.2 / hour"),
            ki=manager.ensure_quantity("0.02 / hour"),
            kd=manager.ensure_quantity("0 second"),
            tau=manager.ensure_quantity("24 hour"),
            noise_band=(
                manager.ensure_quantity("1 milliUSD"),
                manager.ensure_quantity("3 milliUSD"),
            ),
        )

        # Check conversions
        assert config.tau[0] == pytest.approx(24 * 3600)  # 24 hours in seconds
        assert config.noise_band[0][0] == pytest.approx(0.001)  # milliUSD to USD

    def test_mixed_input_types(self):
        """Test mixing different input types."""
        manager = UnitManager.instance()

        config = ControllerConfig(
            kp="0.2 / day",  # String
            ki=0.02,  # Float
            kd=manager.ensure_quantity("0.1 second"),  # pint Quantity
            tau="1 week",  # String
            noise_band=(0.001, "0.003 USD"),  # Mixed tuple
        )

        assert config.tau[0] == pytest.approx(7 * 86400)  # 1 week in seconds

    def test_optional_fields(self):
        """Test optional limit fields."""
        config = ControllerConfig(
            kp=0.2,
            ki=0.02,
            kd=0.0,
            tau=86400.0,
            noise_band=(0.001, 0.003),
            output_min=-10.0,
            output_max=10.0,
            rate_limit=1.0,
        )

        assert config.output_min[0] == -10.0
        assert config.output_max[0] == 10.0
        assert config.rate_limit[0] == 1.0

    def test_noise_band_validation(self):
        """Test that noise band validates ordering."""
        with pytest.raises(ValidationError) as exc_info:
            ControllerConfig(
                kp=0.2,
                ki=0.02,
                kd=0.0,
                tau=86400.0,
                noise_band=(0.003, 0.001),  # Wrong order
            )

        assert "must be less than" in str(exc_info.value)

    def test_negative_tau_rejected(self):
        """Test that negative time constant is rejected."""
        with pytest.raises(ValidationError):
            ControllerConfig(
                kp=0.2,
                ki=0.02,
                kd=0.0,
                tau=-1.0,  # Negative time constant
                noise_band=(0.001, 0.003),
            )

    def test_to_runtime_conversion(self):
        """Test conversion to runtime structure."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day",
            kd="0.0",
            tau="7 day",
            noise_band=("0.001 USD", "0.003 USD"),
        )

        runtime = config.to_runtime()

        # Check runtime structure
        assert runtime.kp.value.shape == ()  # Scalar
        assert runtime.kp.units.dimension in ["frequency", "1/time", "dimensionless"]
        assert runtime.tau.value == pytest.approx(7 * 86400)
        assert runtime.noise_band_low.value == pytest.approx(0.001)
        assert runtime.noise_band_high.value == pytest.approx(0.003)

    def test_from_runtime_conversion(self):
        """Test conversion from runtime back to config output."""
        config = ControllerConfig(
            kp="0.2 / hour",
            ki="0.02 / hour",
            kd="0.1 second",
            tau="24 hour",
            noise_band=("1 milliUSD", "3 milliUSD"),
        )

        runtime = config.to_runtime()
        output = ControllerConfig.from_runtime(runtime)

        # Check output is in original units
        assert output.kp.units == config.kp[1].symbol
        assert output.tau.magnitude == pytest.approx(24)
        assert str(output.tau.units) == "hour"
        assert output.noise_band_low.magnitude == pytest.approx(1)
        assert "milli" in str(output.noise_band_low.units).lower()

    def test_summary_methods(self):
        """Test summary output methods."""
        config = ControllerConfig(
            kp=0.2,
            ki=0.02,
            kd=0.0,
            tau=86400.0,
            noise_band=(0.001, 0.003),
        )

        # Test markdown format
        md_summary = config.summary(format="markdown")
        assert "## Controller Configuration" in md_summary
        assert "| kp |" in md_summary
        assert "| tau |" in md_summary

        # Test text format
        text_summary = config.summary(format="text")
        assert "Controller Configuration:" in text_summary
        assert "kp:" in text_summary
        assert "tau:" in text_summary

        # Test dict format
        dict_summary = config.summary(format="dict")
        assert isinstance(dict_summary, dict)
        assert "kp" in dict_summary
        assert "tau" in dict_summary


class TestControllerKernel:
    """Test controller kernel functions."""

    def test_basic_pid_step(self):
        """Test basic PID controller step."""
        config = ControllerConfig(
            kp=1.0,
            ki=0.1,
            kd=0.01,
            tau=100.0,
            noise_band=(0.0, 1e10),  # Effectively disabled
        )

        runtime = config.to_runtime()
        state = ControllerState.zero()

        # Apply a step error
        error = jnp.array(1.0)
        dt = jnp.array(0.1)

        new_state, output = controller_step(runtime, state, error, dt)

        # Check P term dominates initially
        expected_p = 1.0 * 1.0  # kp * error
        expected_i = 0.1 * (1.0 * 0.1)  # ki * (error * dt)
        expected_d = 0.01 * (1.0 / 0.1)  # kd * (error / dt) for first step

        expected_output = expected_p + expected_i + expected_d
        assert output == pytest.approx(expected_output, rel=1e-4)

        # Check state update
        assert new_state.integral == pytest.approx(0.1)  # error * dt
        assert new_state.last_error == pytest.approx(1.0)
        assert new_state.time == pytest.approx(0.1)

    def test_noise_band_filtering(self):
        """Test noise band filters small errors."""
        config = ControllerConfig(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            tau=100.0,
            noise_band=(0.1, 0.5),  # Filter below 0.1, interpolate 0.1-0.5
        )

        runtime = config.to_runtime()
        state = ControllerState.zero()
        dt = jnp.array(0.1)

        # Small error should be filtered
        small_error = jnp.array(0.05)
        _, output_small = controller_step(runtime, state, small_error, dt)
        assert output_small == pytest.approx(0.0)

        # Medium error should be interpolated
        medium_error = jnp.array(0.3)
        _, output_medium = controller_step(runtime, state, medium_error, dt)
        # At 0.3, we're (0.3-0.1)/(0.5-0.1) = 0.5 through the band
        expected = 1.0 * 0.3 * 0.5  # kp * error * band_factor
        assert output_medium == pytest.approx(expected, rel=1e-4)

        # Large error should pass through
        large_error = jnp.array(1.0)
        _, output_large = controller_step(runtime, state, large_error, dt)
        assert output_large == pytest.approx(1.0)  # kp * error

    def test_integral_leak(self):
        """Test integral leak (anti-windup)."""
        config = ControllerConfig(
            kp=0.0,
            ki=1.0,
            kd=0.0,
            tau=1.0,  # 1 second decay time
            noise_band=(0.0, 1e10),
        )

        runtime = config.to_runtime()
        dt = jnp.array(0.5)

        # Build up integral
        state = ControllerState.zero()
        error = jnp.array(1.0)
        state, _ = controller_step(runtime, state, error, dt)

        # Integral should accumulate
        assert state.integral == pytest.approx(0.5)  # error * dt

        # With zero error, integral should decay
        zero_error = jnp.array(0.0)
        state, _ = controller_step(runtime, state, zero_error, dt)

        # Integral should decay by exp(-dt/tau) = exp(-0.5/1.0) ≈ 0.606
        expected_integral = 0.5 * jnp.exp(-0.5)
        assert state.integral == pytest.approx(expected_integral, rel=1e-4)

    def test_output_limits(self):
        """Test output limiting."""
        config = ControllerConfig(
            kp=10.0,
            ki=0.0,
            kd=0.0,
            tau=100.0,
            noise_band=(0.0, 1e10),
            output_min=-1.0,
            output_max=1.0,
        )

        runtime = config.to_runtime()
        state = ControllerState.zero()
        dt = jnp.array(0.1)

        # Large positive error should saturate at max
        large_error = jnp.array(10.0)
        _, output_high = controller_step(runtime, state, large_error, dt)
        assert output_high == pytest.approx(1.0)

        # Large negative error should saturate at min
        large_neg_error = jnp.array(-10.0)
        _, output_low = controller_step(runtime, state, large_neg_error, dt)
        assert output_low == pytest.approx(-1.0)

    def test_rate_limiting(self):
        """Test rate limiting."""
        config = ControllerConfig(
            kp=10.0,
            ki=0.0,
            kd=0.0,
            tau=100.0,
            noise_band=(0.0, 1e10),
            rate_limit=1.0,  # Max 1 unit/second change
        )

        runtime = config.to_runtime()
        state = ControllerState.zero()
        dt = jnp.array(0.1)

        # Large error would produce large output
        large_error = jnp.array(10.0)
        new_state, output = controller_step(runtime, state, large_error, dt)

        # But rate limit constrains it
        max_change = 1.0 * 0.1  # rate_limit * dt
        assert output == pytest.approx(max_change)
        assert new_state.last_output == pytest.approx(max_change)

        # Next step can change by at most max_change again
        new_state2, output2 = controller_step(runtime, new_state, large_error, dt)
        assert output2 == pytest.approx(2 * max_change)

    def test_jit_compilation(self):
        """Test that kernel can be JIT compiled."""
        config = ControllerConfig(
            kp=1.0,
            ki=0.1,
            kd=0.01,
            tau=100.0,
            noise_band=(0.001, 0.003),
        )

        runtime = config.to_runtime()
        state = ControllerState.zero()

        # The function is already decorated with @jax.jit
        # Just verify it works
        error = jnp.array(0.01)
        dt = jnp.array(0.1)

        # First call triggers compilation
        new_state, output = controller_step(runtime, state, error, dt)

        # Second call should use compiled version
        new_state2, output2 = controller_step(runtime, new_state, error, dt)

        # Results should be deterministic
        assert new_state.time == pytest.approx(0.1)
        assert new_state2.time == pytest.approx(0.2)


class TestIntegration:
    """Integration tests for the complete controller system."""

    def test_complete_workflow(self):
        """Test complete workflow from config to control."""
        # User creates config with friendly units
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.2 / day / 7 day",
            kd="0.0 hour",
            tau="1 week",
            noise_band=("1 milliUSD", "3 milliUSD"),
            output_min="-100 USD",
            output_max="100 USD",
        )

        # Display summary
        summary = config.summary()
        assert "Controller Configuration" in summary

        # Convert to runtime
        runtime = config.to_runtime(dtype=jnp.float32)
        assert runtime.kp.value.dtype == jnp.float32

        # Initialize state
        state = ControllerState.zero(dtype=jnp.float32)

        # Run control loop
        errors = jnp.array([0.002, 0.0015, 0.001, 0.0005])  # USD errors
        dt = jnp.array(3600.0)  # 1 hour time steps

        for error in errors:
            state, output = controller_step(runtime, state, error, dt)
            assert jnp.isfinite(output)

        # Convert back for display
        final_config = ControllerConfig.from_runtime(runtime)
        assert str(final_config.tau.units) == "week"  # Should preserve original units

    def test_vmap_over_parameters(self):
        """Test vectorizing over different parameter values."""
        # Create a single config and runtime
        config = ControllerConfig(
            kp=1.0,
            ki=0.01,
            kd=0.0,
            tau=100.0,
            noise_band=(0.0, 1.0),
        )
        runtime = config.to_runtime()

        # Create different error values to test
        errors = jnp.array([0.1, 0.2, 0.3, 0.4])

        def run_controller(error_value):
            state = ControllerState.zero()
            dt = jnp.array(0.1)
            _, output = controller_step(runtime, state, error_value, dt)
            return output

        # Vectorize over error values
        vmapped_run = jax.vmap(run_controller)
        outputs = vmapped_run(errors)

        assert outputs.shape == (4,)
        assert jnp.all(outputs[1:] > outputs[:-1])  # Monotonic in error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])