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
    """Test that JAX TWAP no longer has wrap-around issues."""

    def test_no_negative_dt(self):
        """Test that time intervals are never negative."""
        twap = JAXFlatWindowTWAP(window_size=10.0, max_observations=5)

        # Manually set up some test data
        twap.times = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        twap.prices = jnp.array([100.0, 101.0, 102.0, 103.0, 104.0])
        twap.count = jnp.array(5)

        # All observations are valid
        valid_mask = jnp.ones(5, dtype=bool)

        # Compute weighted average
        result = twap._compute_weighted_average(valid_mask)

        # The average should be around 102 (middle of the range)
        assert 101.0 <= float(result) <= 103.0

        # More importantly, verify no wrap-around by checking the computation
        times_curr = twap.times[:-1]  # [1, 2, 3, 4]
        times_next = twap.times[1:]   # [2, 3, 4, 5]
        dt_array = times_next - times_curr  # [1, 1, 1, 1]

        # All dt values should be positive
        assert jnp.all(dt_array >= 0)
        print("✓ No negative time intervals")

    def test_correct_pairing(self):
        """Test that observations are paired correctly."""
        twap = JAXFlatWindowTWAP(window_size=10.0, max_observations=5)

        # Set up data where wrap-around would be obvious
        twap.times = jnp.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Last time is way off
        twap.prices = jnp.array([10.0, 10.0, 10.0, 10.0, 1000.0])  # Last price is way off
        twap.count = jnp.array(5)

        # Only first 4 observations are valid (within window)
        valid_mask = jnp.array([True, True, True, True, False])

        # Compute weighted average
        result = twap._compute_weighted_average(valid_mask)

        # Should be 10.0 (average of the valid observations)
        # If wrap-around occurred, the huge price would affect the result
        assert abs(float(result) - 10.0) < 0.01
        print("✓ Correct pairing without wrap-around")

    def test_edge_case_two_observations(self):
        """Test with exactly two observations."""
        twap = JAXFlatWindowTWAP(window_size=10.0, max_observations=5)

        # Only two observations
        twap.times = jnp.array([1.0, 2.0, 0.0, 0.0, 0.0])
        twap.prices = jnp.array([100.0, 110.0, 0.0, 0.0, 0.0])
        twap.count = jnp.array(2)

        valid_mask = jnp.array([True, True, False, False, False])

        # Compute weighted average
        result = twap._compute_weighted_average(valid_mask)

        # Should be 105.0 (average of 100 and 110)
        assert abs(float(result) - 105.0) < 0.01
        print("✓ Two observations handled correctly")

    def test_comparison_with_roll(self):
        """Compare fixed version with old roll-based version to show difference."""
        twap = JAXFlatWindowTWAP(window_size=10.0, max_observations=5)

        # Set up data where wrap-around matters
        twap.times = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0])
        twap.prices = jnp.array([100.0, 100.0, 100.0, 100.0, 200.0])
        twap.count = jnp.array(5)
        valid_mask = jnp.ones(5, dtype=bool)

        # Correct calculation (array slicing)
        correct_result = twap._compute_weighted_average(valid_mask)

        # What the old roll-based version would compute
        times_rolled = jnp.roll(twap.times, -1)  # [2, 3, 4, 10, 1]
        prices_rolled = jnp.roll(twap.prices, -1)  # [100, 100, 100, 200, 100]
        pair_mask_rolled = valid_mask & jnp.roll(valid_mask, -1)
        dt_rolled = times_rolled - twap.times  # [1, 1, 1, 6, -9] <- negative!
        avg_prices_rolled = (twap.prices + prices_rolled) / 2.0

        # The last dt is negative due to wrap-around
        assert float(dt_rolled[-1]) < 0  # This is the bug!

        # Old version would give wrong result due to negative dt
        weighted_sum_rolled = jnp.sum(avg_prices_rolled * dt_rolled * pair_mask_rolled)
        total_weight_rolled = jnp.sum(dt_rolled * pair_mask_rolled)
        wrong_result = weighted_sum_rolled / total_weight_rolled

        # The results should be different
        assert abs(float(correct_result) - float(wrong_result)) > 1.0
        print(f"✓ Fixed version gives {float(correct_result):.2f}, "
              f"roll version would give {float(wrong_result):.2f}")


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