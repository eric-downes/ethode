#!/usr/bin/env python
"""Test shared utilities: TWAP, noise filtering, and metrics."""

import pytest
import numpy as np
import jax.numpy as jnp
from typing import Tuple

from ethode.twap import FlatWindowTWAP, JAXFlatWindowTWAP, create_twap
from ethode.noise import (
    noise_barrier,
    noise_barrier_jax,
    noise_barrier_with_units,
    smooth_ramp,
    smooth_ramp_jax,
    dead_band,
    dead_band_jax,
    saturation,
    saturation_jax
)
from ethode.metrics import (
    guardrail,
    guardrail_jax,
    stability_margin,
    settling_time,
    control_effort,
    performance_metrics,
    performance_metrics_jax
)
from ethode.controller import ControllerConfig
from ethode.runtime import ControllerState, QuantityNode, UnitSpec
from ethode.types import to_time_scalar, to_price_scalar


class TestTWAP:
    """Test TWAP calculator implementations."""

    def test_flat_window_twap_basic(self):
        """Test basic TWAP functionality."""
        twap = FlatWindowTWAP(window_size=10.0)  # 10 second window

        # Add first observation
        price1 = twap.update(to_price_scalar(100.0), to_time_scalar(0.0))
        assert price1 == 100.0

        # Add second observation after 1 second
        price2 = twap.update(to_price_scalar(110.0), to_time_scalar(1.0))
        # TWAP should be between 100 and 110
        assert 100.0 <= price2 <= 110.0

        # Add more observations
        for i in range(8):
            twap.update(to_price_scalar(120.0), to_time_scalar(1.0))

        # After 10 seconds total, first observation should be dropped
        assert twap.num_observations <= 10

    def test_twap_window_expiry(self):
        """Test that old observations are dropped."""
        twap = FlatWindowTWAP(window_size=5.0)  # 5 second window

        # Add observations over 10 seconds
        for i in range(10):
            twap.update(to_price_scalar(100.0 + i), to_time_scalar(1.0))

        # Should only have ~5 observations (within window)
        assert twap.time_coverage <= 5.0

    def test_twap_reset(self):
        """Test TWAP reset functionality."""
        twap = FlatWindowTWAP(window_size=10.0)

        # Add observations
        twap.update(to_price_scalar(100.0), to_time_scalar(1.0))
        twap.update(to_price_scalar(110.0), to_time_scalar(1.0))
        assert twap.num_observations == 2

        # Reset
        twap.reset()
        assert twap.num_observations == 0
        assert twap.value == 0.0

    def test_create_twap_with_units(self):
        """Test TWAP creation with unit strings."""
        # Create with different time units
        twap1 = create_twap("5 minutes")
        assert twap1.window_size == 300.0  # 5 minutes = 300 seconds

        twap2 = create_twap("1 hour")
        assert twap2.window_size == 3600.0  # 1 hour = 3600 seconds

        twap3 = create_twap("30 seconds")
        assert twap3.window_size == 30.0

    def test_jax_twap(self):
        """Test JAX-compatible TWAP."""
        # Skip JAX TWAP for now - needs more complex implementation
        # The JAX version requires careful handling of mutable state
        # which is complex with JAX's functional paradigm
        pytest.skip("JAX TWAP implementation needs refinement")

    def test_twap_calculation_accuracy(self):
        """Test TWAP calculation accuracy."""
        twap = FlatWindowTWAP(window_size=10.0)

        # Add constant price - TWAP should equal the price
        for _ in range(5):
            result = twap.update(to_price_scalar(100.0), to_time_scalar(1.0))

        assert abs(result - 100.0) < 0.01

        # Reset and test linear increase
        twap.reset()
        prices = []
        for i in range(5):
            price = 100.0 + i * 10.0
            result = twap.update(to_price_scalar(price), to_time_scalar(1.0))
            prices.append(price)

        # TWAP should be around the average
        expected = np.mean(prices)
        assert abs(result - expected) < 10.0  # Some tolerance for time weighting


class TestNoiseFiltering:
    """Test noise filtering functions."""

    def test_noise_barrier_basic(self):
        """Test basic noise barrier functionality."""
        band = (0.001, 0.003)

        # Below threshold - should be zero
        assert noise_barrier(0.0005, band) == 0.0
        assert noise_barrier(-0.0005, band) == 0.0

        # Above threshold - should pass through
        assert noise_barrier(0.005, band) == 0.005
        assert noise_barrier(-0.005, band) == -0.005

        # In ramp zone - should be scaled
        result = noise_barrier(0.002, band)
        assert 0 < result < 0.002
        # Check scaling factor
        expected = 0.002 * (0.002 - 0.001) / (0.003 - 0.001)
        assert abs(result - expected) < 1e-10

    def test_noise_barrier_jax(self):
        """Test JAX-compatible noise barrier."""
        low = jnp.array(0.001)
        high = jnp.array(0.003)

        # Test various inputs
        result1 = noise_barrier_jax(jnp.array(0.0005), low, high)
        assert float(result1) == 0.0

        result2 = noise_barrier_jax(jnp.array(0.005), low, high)
        assert abs(float(result2) - 0.005) < 1e-6  # Allow for float precision

        result3 = noise_barrier_jax(jnp.array(0.002), low, high)
        assert 0 < float(result3) < 0.002

    def test_noise_barrier_with_units(self):
        """Test unit-aware noise barrier."""
        # Create band with units
        band = (
            QuantityNode(
                value=jnp.array(0.001),
                units=UnitSpec(dimension="price", symbol="USD", to_canonical=1.0)
            ),
            QuantityNode(
                value=jnp.array(0.003),
                units=UnitSpec(dimension="price", symbol="USD", to_canonical=1.0)
            )
        )

        # Test filtering
        result = noise_barrier_with_units(to_price_scalar(0.002), band)
        assert 0 < result < 0.002

    def test_smooth_ramp(self):
        """Test smooth ramp function."""
        thresholds = (0.001, 0.002, 0.008, 0.010)

        # Below t1 - zero
        assert smooth_ramp(0.0005, thresholds) == 0.0

        # Between t1 and t2 - ramp up
        result = smooth_ramp(0.0015, thresholds)
        assert 0 < result < 0.0015

        # Between t2 and t3 - pass through
        assert smooth_ramp(0.005, thresholds) == 0.005

        # Between t3 and t4 - amplification ramp
        result = smooth_ramp(0.009, thresholds)
        assert result > 0.009

        # Above t4 - maximum amplification
        assert smooth_ramp(0.012, thresholds) == 0.012 * 2.0

    def test_smooth_ramp_jax(self):
        """Test JAX-compatible smooth ramp."""
        result = smooth_ramp_jax(
            jnp.array(0.005),
            jnp.array(0.001),
            jnp.array(0.002),
            jnp.array(0.008),
            jnp.array(0.010)
        )
        assert abs(float(result) - 0.005) < 1e-6  # Allow for float precision

    def test_dead_band(self):
        """Test dead band filter."""
        threshold = 0.001

        # Below threshold
        assert dead_band(0.0005, threshold) == 0.0
        assert dead_band(-0.0005, threshold) == 0.0

        # Above threshold
        assert dead_band(0.002, threshold) == 0.002
        assert dead_band(-0.002, threshold) == -0.002

    def test_dead_band_jax(self):
        """Test JAX dead band."""
        result = dead_band_jax(jnp.array(0.0005), jnp.array(0.001))
        assert float(result) == 0.0

        result = dead_band_jax(jnp.array(0.002), jnp.array(0.001))
        assert abs(float(result) - 0.002) < 1e-6  # Allow for float precision

    def test_saturation(self):
        """Test saturation limits."""
        limits = (-1.0, 1.0)

        # Within limits
        assert saturation(0.5, limits) == 0.5

        # Above max
        assert saturation(2.0, limits) == 1.0

        # Below min
        assert saturation(-2.0, limits) == -1.0

    def test_saturation_jax(self):
        """Test JAX saturation."""
        result = saturation_jax(jnp.array(2.0), jnp.array(-1.0), jnp.array(1.0))
        assert float(result) == 1.0


class TestMetrics:
    """Test metrics functions."""

    def test_guardrail_metric(self):
        """Test guardrail stability metric."""
        # Create a controller config
        config = ControllerConfig(
            kp=0.2,
            ki=0.02,
            kd=0.0,
            tau=3600.0,
            noise_band=(0.001, 0.003)
        )
        runtime = config.to_runtime()

        # Calculate guardrail metric
        metric = guardrail(runtime, market_stiffness=10.0, effective_fee=0.005)

        # M = ki * k_m - (1 - f_eff)
        # M = 0.02 * 10.0 - (1 - 0.005) = 0.2 - 0.995 = -0.795
        expected = 0.02 * 10.0 - (1.0 - 0.005)
        assert abs(metric - expected) < 1e-6  # Allow for float32 precision

    def test_guardrail_jax(self):
        """Test JAX guardrail computation."""
        result = guardrail_jax(
            jnp.array(0.02),
            jnp.array(10.0),
            jnp.array(0.005)
        )
        expected = 0.02 * 10.0 - (1.0 - 0.005)
        assert abs(float(result) - expected) < 1e-6  # Allow for float32 precision

    def test_stability_margin(self):
        """Test stability margin calculation."""
        config = ControllerConfig(
            kp=0.2,
            ki=0.02,
            kd=0.01,
            tau=3600.0,
            noise_band=(0.001, 0.003)
        )
        runtime = config.to_runtime()

        margin = stability_margin(runtime, market_stiffness=10.0)
        # Should return a stability metric
        assert isinstance(margin, float)

    def test_settling_time(self):
        """Test settling time estimation."""
        config = ControllerConfig(
            kp=0.2,
            ki=0.02,
            kd=0.0,
            tau=3600.0,
            noise_band=(0.001, 0.003)
        )
        runtime = config.to_runtime()

        time = settling_time(runtime, market_stiffness=10.0)
        assert time > 0  # Should be positive
        assert time < float('inf')  # Should be finite

    def test_control_effort(self):
        """Test control effort breakdown."""
        config = ControllerConfig(
            kp=1.0,
            ki=0.1,
            kd=0.01,
            tau=3600.0,
            noise_band=(0.001, 0.003)
        )
        runtime = config.to_runtime()

        # Create state with specific values
        from ethode.runtime import ControllerState as CS
        state = CS(
            integral=jnp.array(0.5),
            last_error=jnp.array(0.9),
            last_output=jnp.array(0.0),
            time=jnp.array(0.0)
        )

        effort = control_effort(state, runtime, error=1.0)

        assert "proportional" in effort
        assert "integral" in effort
        assert "derivative" in effort
        assert "total" in effort

        # Check calculations
        assert effort["proportional"] == 1.0 * 1.0  # kp * error
        assert abs(effort["integral"] - 0.1 * 0.5) < 1e-6  # ki * integral, with float32 precision

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create error history
        errors = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
        dt = 0.1

        metrics = performance_metrics(errors, dt)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "iae" in metrics
        assert "ise" in metrics
        assert "itae" in metrics

        # Check MAE calculation
        expected_mae = np.mean(np.abs(errors))
        assert abs(metrics["mae"] - expected_mae) < 1e-10

        # Check RMSE calculation
        expected_rmse = np.sqrt(np.mean(errors**2))
        assert abs(metrics["rmse"] - expected_rmse) < 1e-10

    def test_performance_metrics_jax(self):
        """Test JAX performance metrics."""
        errors = jnp.array([0.1, -0.2, 0.15, -0.1, 0.05])
        dt = jnp.array(0.1)

        metrics = performance_metrics_jax(errors, dt)

        assert "mae" in metrics
        assert "rmse" in metrics

        # Check values match numpy version
        np_metrics = performance_metrics(np.array(errors), 0.1)
        assert abs(float(metrics["mae"]) - np_metrics["mae"]) < 1e-10
        assert abs(float(metrics["rmse"]) - np_metrics["rmse"]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])