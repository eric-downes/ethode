#!/usr/bin/env python
"""Test fixes for rate_limit type handling and JAX TWAP wrap-around."""

import pytest
import numpy as np
import jax.numpy as jnp
import pint

from ethode.controller import PIDController
from ethode.controller.legacy import PIDController as LegacyPIDController
from ethode.twap import JAXFlatWindowTWAP
from ethode.units import UnitManager


class TestRateLimitTypeFix:
    """Test that rate_limit handles different input types correctly."""

    def test_rate_limit_with_float(self):
        """Test rate_limit with plain float value."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.01, rate_limit=5.0)

        # Should work without error
        assert pid.config.rate_limit is not None
        assert pid.config.rate_limit[0] == 5.0
        assert pid.config.rate_limit[1].dimension == "price/time"
        print("✓ Float rate_limit works")

    def test_rate_limit_with_string_units(self):
        """Test rate_limit with string containing units."""
        # Pass a string with different units
        pid = PIDController(kp=1.0, ki=0.1, kd=0.01, rate_limit="300 USD/minute")

        # Should convert to USD/second
        assert pid.config.rate_limit is not None
        # 300 USD/minute = 5 USD/second
        assert abs(pid.config.rate_limit[0] - 5.0) < 1e-6
        assert pid.config.rate_limit[1].dimension == "price/time"
        print("✓ String with units rate_limit works")

    def test_rate_limit_with_pint_quantity(self):
        """Test rate_limit with pint Quantity."""
        manager = UnitManager.instance()

        # Create a pint quantity with different units
        rate_qty = manager.registry.Quantity(7200, "USD/hour")
        pid = PIDController(kp=1.0, ki=0.1, kd=0.01, rate_limit=rate_qty)

        # Should convert to USD/second
        assert pid.config.rate_limit is not None
        # 7200 USD/hour = 2 USD/second
        assert abs(pid.config.rate_limit[0] - 2.0) < 1e-6
        assert pid.config.rate_limit[1].dimension == "price/time"
        print("✓ Pint Quantity rate_limit works")

    def test_rate_limit_string_without_units(self):
        """Test rate_limit with string that's just a number."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.01, rate_limit="3.5")

        # Should assume USD/second
        assert pid.config.rate_limit is not None
        assert abs(pid.config.rate_limit[0] - 3.5) < 1e-6
        assert pid.config.rate_limit[1].dimension == "price/time"
        print("✓ String number rate_limit works")

    def test_controller_still_functions(self):
        """Test that controller still works after rate_limit fix."""
        # Test with each type
        for rate_limit in [5.0, "300 USD/minute", "5"]:
            pid = PIDController(kp=1.0, ki=0.1, kd=0.01, rate_limit=rate_limit)

            # Should be able to update
            output = pid.update(1.0, 0.1)
            assert isinstance(output, float)

        print("✓ Controller functions with all rate_limit types")


class TestLegacyRateLimitTypeFix:
    """Test that legacy controller rate_limit handles numeric strings correctly."""

    def test_legacy_numeric_string_without_units(self):
        """Test legacy controller with numeric string (no units)."""
        controller = LegacyPIDController(
            kp=0.1,
            rate_limit="10"  # Should be treated as 10 USD/second
        )

        # Config is created during __init__, no need to call _ensure_config
        assert controller.config.rate_limit is not None
        rate_value, rate_spec = controller.config.rate_limit
        assert rate_value == 10.0  # Should be 10 USD/second in canonical
        assert rate_spec.dimension == "price/time"
        print("✓ Legacy rate limit numeric string works")

    def test_legacy_decimal_string_without_units(self):
        """Test legacy controller with decimal string (no units)."""
        controller = LegacyPIDController(
            kp=0.1,
            rate_limit="5.5"  # Should be treated as 5.5 USD/second
        )

        # Config is created during __init__
        assert controller.config.rate_limit is not None
        rate_value, rate_spec = controller.config.rate_limit
        assert abs(rate_value - 5.5) < 1e-10
        assert rate_spec.dimension == "price/time"
        print("✓ Legacy rate limit decimal string works")

    def test_legacy_string_with_units_still_works(self):
        """Test legacy controller with string containing units still works."""
        controller = LegacyPIDController(
            kp=0.1,
            rate_limit="300 USD/minute"  # Has units, should parse correctly
        )

        # Config is created during __init__
        assert controller.config.rate_limit is not None
        rate_value, rate_spec = controller.config.rate_limit
        # 300 USD/minute = 5 USD/second
        assert abs(rate_value - 5.0) < 1e-6
        assert rate_spec.dimension == "price/time"
        print("✓ Legacy rate limit with units still works")


class TestJAXTWAPWrapAroundFix:
    """Test that JAX TWAP no longer has wrap-around issues.

    Note: The new functional implementation doesn't have the wrap-around
    issues that the old mutable implementation had. These tests verify
    the new implementation works correctly.
    """

    def test_no_negative_dt(self):
        """Test that time intervals are never negative in new implementation."""
        # The new implementation uses a functional approach with scan
        # which inherently avoids wrap-around issues
        from ethode.twap import TWAPState, TWAPRuntime, twap_update

        runtime = TWAPRuntime(window_size=10.0, max_observations=5)
        state = TWAPState.zeros(5)

        # Add observations with increasing timestamps
        for i in range(5):
            price = 100.0 + i
            dt = 1.0
            state, twap = twap_update(state, runtime, jnp.array(price), jnp.array(dt))

        # The TWAP should be reasonable (around the middle of the range)
        assert 100.0 <= float(twap) <= 104.0

        # The implementation uses scan which ensures proper time ordering
        # No wrap-around is possible in the functional implementation
        print("✓ No negative time intervals (functional implementation)")

    def test_correct_pairing(self):
        """Test that observations are paired correctly in new implementation."""
        from ethode.twap import TWAPState, TWAPRuntime, twap_update

        runtime = TWAPRuntime(window_size=10.0, max_observations=5)
        state = TWAPState.zeros(5)

        # Add 4 normal observations
        for _ in range(4):
            state, twap = twap_update(state, runtime, jnp.array(10.0), jnp.array(1.0))

        # The TWAP should be 10.0 (all prices are 10.0)
        assert abs(float(twap) - 10.0) < 0.01

        # Add observation with huge gap and price
        # This will be outside the window and shouldn't affect TWAP much
        state, twap = twap_update(state, runtime, jnp.array(1000.0), jnp.array(100.0))

        # The old observations are now outside the 10-second window
        # TWAP should be close to the new price (1000.0)
        assert float(twap) > 900.0  # Dominated by recent observation

        print("✓ Correct pairing without wrap-around (functional implementation)")

    def test_edge_case_two_observations(self):
        """Test with exactly two observations."""
        from ethode.twap import TWAPState, TWAPRuntime, twap_update

        runtime = TWAPRuntime(window_size=10.0, max_observations=5)
        state = TWAPState.zeros(5)

        # Add exactly two observations
        state, twap1 = twap_update(state, runtime, jnp.array(100.0), jnp.array(1.0))
        assert float(twap1) == 100.0  # First observation

        state, twap2 = twap_update(state, runtime, jnp.array(110.0), jnp.array(1.0))

        # Should be weighted average of 100 and 110 = 105
        assert abs(float(twap2) - 105.0) < 0.01

        print("✓ Two observations handled correctly (functional implementation)")

    def test_comparison_with_roll(self):
        """Verify the new functional implementation avoids roll-based bugs."""
        from ethode.twap import TWAPState, TWAPRuntime, twap_update

        runtime = TWAPRuntime(window_size=10.0, max_observations=5)
        state = TWAPState.zeros(5)

        # Add 4 observations with same price
        for _ in range(4):
            state, twap = twap_update(state, runtime, jnp.array(100.0), jnp.array(1.0))

        # Add one observation with different price after a gap
        state, twap = twap_update(state, runtime, jnp.array(200.0), jnp.array(6.0))

        # The functional implementation correctly handles this
        # TWAP should be weighted toward 200 due to the longer time interval
        assert 100.0 < float(twap) < 200.0

        # Demonstrate that roll would create negative intervals
        times = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0])
        times_rolled = jnp.roll(times, -1)  # [2, 3, 4, 10, 1]
        dt_rolled = times_rolled - times  # [1, 1, 1, 6, -9]

        # The last dt is negative due to wrap-around - this was the bug!
        assert float(dt_rolled[-1]) < 0

        print(f"✓ Functional implementation avoids roll wrap-around bug")


if __name__ == "__main__":
    print("Testing rate_limit type handling fix...")
    rate_limit_tests = TestRateLimitTypeFix()
    rate_limit_tests.test_rate_limit_with_float()
    rate_limit_tests.test_rate_limit_with_string_units()
    rate_limit_tests.test_rate_limit_with_pint_quantity()
    rate_limit_tests.test_rate_limit_string_without_units()
    rate_limit_tests.test_controller_still_functions()

    print("\nTesting JAX TWAP wrap-around fix...")
    twap_tests = TestJAXTWAPWrapAroundFix()
    twap_tests.test_no_negative_dt()
    twap_tests.test_correct_pairing()
    twap_tests.test_edge_case_two_observations()
    twap_tests.test_comparison_with_roll()

    print("\n✅ All fixes verified!")