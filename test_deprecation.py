#!/usr/bin/env python
"""Test deprecated APIs and ensure proper deprecation warnings."""

import pytest
import warnings
import numpy as np
from dataclasses import dataclass

# Test the base ethode Params deprecation
def test_base_params_deprecation():
    """Test that base Params class emits deprecation warning."""
    from ethode import Params, U

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create a Params instance - should trigger warning
        params = Params(
            init_conds=(("x", 1.0 * U.dimensionless),),
            tspan=(0 * U.second, 10 * U.second)
        )

        # Check that a deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Params base class is deprecated" in str(w[0].message)
        assert "v3.0" in str(w[0].message)

    print("✓ Base Params deprecation warning works")


def test_pid_params_deprecation():
    """Test that PIDParams emits deprecation warning."""
    from ethode.controller.legacy import PIDParams

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create PIDParams - should trigger warning
        params = PIDParams(kp=1.0, ki=0.1, kd=0.01)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "PIDParams is deprecated" in str(w[0].message)
        assert "ControllerConfig" in str(w[0].message)

    print("✓ PIDParams deprecation warning works")


def test_legacy_pidcontroller_deprecation():
    """Test that legacy PIDController emits deprecation warning."""
    from ethode.controller.legacy import PIDController

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create legacy controller - should trigger multiple warnings
        # One for PIDParams (if created internally) and one for controller
        controller = PIDController(kp=1.0, ki=0.1, kd=0.01)

        # Should have at least the controller deprecation warning
        controller_warnings = [warning for warning in w
                              if "Legacy PIDController" in str(warning.message)]
        assert len(controller_warnings) >= 1
        assert "v3.0" in str(controller_warnings[0].message)

    print("✓ Legacy PIDController deprecation warning works")


def test_legacy_pidcontroller_still_functions():
    """Test that legacy controller still works despite deprecation."""
    from ethode.controller.legacy import PIDController, PIDParams

    # Suppress warnings for functional test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # Test with PIDParams
        params = PIDParams(kp=1.0, ki=0.1, kd=0.01)
        controller1 = PIDController(params)

        # Test with kwargs
        controller2 = PIDController(kp=1.0, ki=0.1, kd=0.01)

        # Both should work
        output1 = controller1.update(1.0, dt=0.1)
        output2 = controller2.update(1.0, dt=0.1)

        assert isinstance(output1, float)
        assert isinstance(output2, float)
        assert abs(output1 - output2) < 1e-10  # Should be identical

    print("✓ Legacy PIDController still functions correctly")


def test_migration_path():
    """Test that migration from old to new pattern works."""
    from ethode.controller.legacy import PIDParams
    from ethode.controller import ControllerConfig, PIDController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # Old pattern
        old_params = PIDParams(
            kp=1.0,
            ki=0.1,
            kd=0.01,
            integral_leak=0.001,
            noise_threshold=0.003
        )

        # Convert to new pattern
        new_config = old_params.to_config()

        # Verify conversion
        assert isinstance(new_config, ControllerConfig)
        assert new_config.kp == (1.0, new_config.kp[1])
        assert new_config.ki == (0.1, new_config.ki[1])
        assert new_config.kd == (0.01, new_config.kd[1])

        # Tau should be inverse of integral_leak
        assert abs(new_config.tau[0] - 1000.0) < 0.1

        # Noise band should use threshold as lower bound
        assert new_config.noise_band[0] == (0.003, new_config.noise_band[0][1])

    print("✓ Migration path from old to new works")


def test_backward_compatibility_with_rate_limit():
    """Test backward compatibility with various rate_limit types."""
    from ethode.controller.legacy import PIDController

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # Test all supported input types for rate_limit
        test_cases = [
            5.0,                    # float
            "5",                    # numeric string
            "5.5",                  # decimal string
            "300 USD/minute",       # string with units
        ]

        for rate_limit in test_cases:
            controller = PIDController(kp=0.1, rate_limit=rate_limit)

            # Should have rate limit configured
            assert controller.config.rate_limit is not None
            rate_value, rate_spec = controller.config.rate_limit

            # All should convert to price/time dimension
            assert rate_spec.dimension == "price/time"

            # Controller should still function
            output = controller.update(1.0, dt=0.1)
            assert isinstance(output, float)

    print("✓ Backward compatibility with rate_limit preserved")


def test_deprecation_in_subclasses():
    """Test that subclassing deprecated classes also triggers warnings."""
    from ethode import Params, U

    # Create a subclass of deprecated Params
    @dataclass
    class CustomParams(Params):
        my_param: float = 1.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Creating instance of subclass should still warn
        params = CustomParams(
            init_conds=(("x", 1.0 * U.dimensionless),),
            tspan=(0 * U.second, 10 * U.second),
            my_param=2.0
        )

        # Should get deprecation warning from parent class
        assert len(w) >= 1
        assert any("Params base class is deprecated" in str(warning.message)
                  for warning in w)

    print("✓ Subclasses of deprecated classes warned properly")


def test_new_api_no_warnings():
    """Test that new API doesn't emit deprecation warnings."""
    from ethode.controller import PIDController, ControllerConfig
    from ethode.fee import FeeConfig
    from ethode.liquidity import LiquiditySDEConfig
    from ethode.hawkes import HawkesConfig

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Use new APIs - should NOT trigger warnings
        controller_config = ControllerConfig(
            kp=1.0, ki=0.1, kd=0.01,
            tau=1000.0,
            noise_band=(0.001, 0.003)
        )
        controller = PIDController(controller_config)

        fee_config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )

        liquidity_config = LiquiditySDEConfig(
            initial_liquidity="1000000 USD",
            mean_liquidity="1000000 USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )

        hawkes_config = HawkesConfig(
            jump_rate="100 / hour",
            excitation_strength=0.3,
            excitation_decay="5 minutes"
        )

        # Filter to only deprecation warnings
        deprecation_warnings = [warning for warning in w
                               if issubclass(warning.category, DeprecationWarning)]

        # Should have no deprecation warnings
        assert len(deprecation_warnings) == 0

    print("✓ New API produces no deprecation warnings")


def test_warning_stacklevel():
    """Test that deprecation warnings point to user code, not library."""
    from ethode.controller.legacy import PIDParams

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # This line should be identified as the source
        params = PIDParams(kp=1.0)  # Line that triggers warning

        assert len(w) >= 1
        warning = w[0]

        # The warning should point to this test file, not the library
        assert __file__ in warning.filename or "test_deprecation" in warning.filename
        assert "legacy.py" not in warning.filename

    print("✓ Warning stacklevel correctly points to user code")


if __name__ == "__main__":
    # Run all tests
    test_base_params_deprecation()
    test_pid_params_deprecation()
    test_legacy_pidcontroller_deprecation()
    test_legacy_pidcontroller_still_functions()
    test_migration_path()
    test_backward_compatibility_with_rate_limit()
    test_deprecation_in_subclasses()
    test_new_api_no_warnings()
    test_warning_stacklevel()

    print("\n✅ All deprecation tests passed!")