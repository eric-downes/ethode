"""Tests for high-level adapter classes.

This module tests the ControllerAdapter and future adapter classes
that provide stateful, user-friendly wrappers around JAX runtime structures.
"""

import pytest
import warnings
import jax.numpy as jnp
import pint

from ethode.adapters import ControllerAdapter
from ethode.controller.config import ControllerConfig
from ethode.controller.legacy import PIDParams
from ethode.runtime import ControllerState


class TestControllerAdapter:
    """Tests for ControllerAdapter class."""

    def test_init_with_controller_config(self):
        """Test initialization with ControllerConfig (new API)."""
        config = ControllerConfig(
            kp="0.2 / day",        # 1/time for FINANCIAL schema
            ki="0.02 / day**2",    # 1/time² for FINANCIAL schema
            kd=0.0,                 # dimensionless for FINANCIAL schema
            tau="7 day",
            noise_band=("0.001 USD", "0.003 USD")
        )

        adapter = ControllerAdapter(config)

        assert adapter.config is config
        assert adapter.runtime is not None
        assert isinstance(adapter.state, ControllerState)
        assert float(adapter.state.integral) == 0.0

    def test_init_with_pid_params_warns(self):
        """Test initialization with PIDParams emits deprecation warning."""
        params = PIDParams(kp=1.0, ki=0.1, kd=0.01)

        # Legacy PIDParams creates dimensionless gains that fail validation
        # We skip validation for backward compatibility
        with pytest.warns(DeprecationWarning, match="Using PIDParams with ControllerAdapter is deprecated"):
            adapter = ControllerAdapter(params, check_units=False)

        assert adapter.config is not None
        assert adapter.runtime is not None

    def test_init_with_unit_validation_enabled(self):
        """Test that unit validation runs by default."""
        config = ControllerConfig(
            kp="1.0 / hour",      # 1/time for FINANCIAL schema
            ki="0.1 / hour**2",   # 1/time² for FINANCIAL schema
            kd=0.0,                # dimensionless for FINANCIAL schema
            tau=100.0,
            noise_band=("0.01 USD", "1e9 USD")
        )

        # Should not raise (valid config)
        adapter = ControllerAdapter(config, check_units=True)
        assert adapter.runtime is not None

    def test_init_with_unit_validation_disabled(self):
        """Test that unit validation can be disabled."""
        config = ControllerConfig(
            kp=1.0,  # Dimensionless - would fail validation
            ki=0.1,
            kd=0.01,
            tau=100.0,
            noise_band=(0.01, 1e9)
        )

        adapter = ControllerAdapter(config, check_units=False)
        assert adapter.runtime is not None

    def test_step_basic(self):
        """Test basic step functionality."""
        config = ControllerConfig(
            kp="1.0 / hour",
            ki="0.1 / hour**2",
            kd=0.0,
            tau=100.0,
            noise_band=("0.0 USD", "1e9 USD")
        )

        adapter = ControllerAdapter(config)
        output = adapter.step(error=1.0, dt=0.1)

        assert isinstance(output, float)
        assert output != 0.0  # Should have non-zero response to error

    def test_step_updates_state(self):
        """Test that step updates internal state."""
        config = ControllerConfig(
            kp="1.0 / hour",
            ki="0.1 / hour**2",
            kd=0.0,
            tau=100.0,
            noise_band=("0.0 USD", "1e9 USD")
        )

        adapter = ControllerAdapter(config)
        initial_integral = float(adapter.state.integral)

        adapter.step(error=1.0, dt=0.1)

        # Integral should have changed
        assert float(adapter.state.integral) != initial_integral

    def test_step_with_diagnostics(self):
        """Test step_with_diagnostics returns output and diagnostics."""
        config = ControllerConfig(
            kp="1.0 / hour",
            ki="0.1 / hour**2",
            kd=0.0,
            tau=100.0,
            noise_band=("0.0 USD", "1e9 USD")
        )

        adapter = ControllerAdapter(config)
        output, diag = adapter.step_with_diagnostics(error=1.0, dt=0.1)

        assert isinstance(output, float)
        assert isinstance(diag, dict)
        assert 'p_term' in diag
        assert 'i_term' in diag
        assert 'd_term' in diag

    def test_step_with_units(self):
        """Test step_with_units for pint quantity debugging."""
        from ethode.units import UnitManager

        config = ControllerConfig(
            kp="1.0 / hour",
            ki="0.1 / hour**2",
            kd=0.0,
            tau=100.0,
            noise_band=("0.0 USD", "1e9 USD")
        )

        adapter = ControllerAdapter(config)
        manager = UnitManager.instance()

        error_qty = manager.ensure_quantity(1.0, "USD")
        dt_qty = manager.ensure_quantity(0.1, "second")

        output = adapter.step_with_units(error_qty, dt_qty)

        assert isinstance(output, pint.Quantity)

    def test_reset(self):
        """Test reset() clears controller state."""
        config = ControllerConfig(
            kp="1.0 / hour",
            ki="0.1 / hour**2",
            kd=0.0,
            tau=100.0,
            noise_band=("0.0 USD", "1e9 USD")
        )

        adapter = ControllerAdapter(config)

        # Run a step to accumulate state
        adapter.step(error=1.0, dt=0.1)
        assert float(adapter.state.integral) != 0.0

        # Reset should clear state
        adapter.reset()
        assert float(adapter.state.integral) == 0.0
        assert float(adapter.state.last_error) == 0.0

    def test_get_state(self):
        """Test get_state returns dict of state values."""
        config = ControllerConfig(
            kp="1.0 / hour",
            ki="0.1 / hour**2",
            kd=0.0,
            tau=100.0,
            noise_band=("0.0 USD", "1e9 USD")
        )

        adapter = ControllerAdapter(config)
        adapter.step(error=1.0, dt=0.1)

        state = adapter.get_state()

        assert isinstance(state, dict)
        assert 'integral' in state
        assert 'last_error' in state
        assert 'last_output' in state
        assert 'time' in state
        assert all(isinstance(v, float) for v in state.values())

    def test_runtime_state_direct_access(self):
        """Test that runtime and state can be accessed directly for JAX usage."""
        config = ControllerConfig(
            kp="1.0 / hour",
            ki="0.1 / hour**2",
            kd=0.0,
            tau=100.0,
            noise_band=("0.0 USD", "1e9 USD")
        )

        adapter = ControllerAdapter(config)

        # Should be able to access runtime and state for JAX transformations
        from ethode.controller.kernel import controller_step

        error = jnp.array(1.0)
        dt = jnp.array(0.1)

        new_state, output = controller_step(
            adapter.runtime,
            adapter.state,
            error,
            dt
        )

        assert new_state is not None
        assert output is not None


class TestControllerAdapterRegression:
    """Regression tests comparing ControllerAdapter to legacy PIDController."""

    def test_matches_pid_controller_output(self):
        """Test that ControllerAdapter produces same output as legacy PIDController."""
        from ethode.controller.legacy import PIDController

        # Create equivalent configs
        params = PIDParams(kp=1.0, ki=0.1, kd=0.01, integral_leak=0.01)

        with pytest.warns(DeprecationWarning):
            legacy_controller = PIDController(params)

        # Legacy PIDParams creates dimensionless gains - skip validation
        with pytest.warns(DeprecationWarning):
            adapter = ControllerAdapter(params, check_units=False)

        # Run same sequence of steps
        errors = [1.0, 0.5, 0.2, 0.0, -0.1]
        dt = 0.1

        legacy_outputs = []
        adapter_outputs = []

        for error in errors:
            legacy_outputs.append(legacy_controller.update(error, dt))
            adapter_outputs.append(adapter.step(error, dt))

        # Should produce very similar outputs (allow small numerical differences)
        for legacy, adapted in zip(legacy_outputs, adapter_outputs):
            assert abs(legacy - adapted) < 1e-6


class TestControllerAdapterUnitValidation:
    """Tests for unit validation in ControllerAdapter."""

    def test_validation_catches_dimension_mismatch(self):
        """Test that validation catches incompatible dimensions."""
        # Dimensionless gains will fail validation with FINANCIAL schema
        config = ControllerConfig(
            kp=1.0,  # Dimensionless
            ki=0.1,
            kd=0.01,
            tau=100.0,
            noise_band=(0.01, 1e9)
        )

        # Should raise validation error
        with pytest.raises(ValueError, match="Unit validation failed"):
            adapter = ControllerAdapter(config, check_units=True)

    def test_validation_passes_with_correct_units(self):
        """Test that validation passes with properly dimensioned gains."""
        config = ControllerConfig(
            kp="1.0 / hour",    # 1/time for FINANCIAL schema
            ki="0.1 / hour**2", # 1/time² for FINANCIAL schema
            kd=0.0,              # dimensionless for FINANCIAL schema
            tau=100.0,
            noise_band=("0.01 USD", "1e9 USD")
        )

        # Should work with correct dimensions
        adapter = ControllerAdapter(config, check_units=True)
        assert adapter is not None

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled for performance."""
        config = ControllerConfig(
            kp=1.0,  # Dimensionless - would fail validation
            ki=0.1,
            kd=0.01,
            tau=100.0,
            noise_band=(0.01, 1e9)
        )

        # Should work even if we skip validation
        adapter = ControllerAdapter(config, check_units=False)
        assert adapter is not None
        assert adapter.runtime is not None
