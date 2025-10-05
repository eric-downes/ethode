#!/usr/bin/env python
"""Test new unit-aware config modules: Fee, Liquidity, Hawkes."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pint

from ethode.fee import FeeConfig, FeeRuntime, FeeState
from ethode.fee.kernel import calculate_fee, update_stress_level
from ethode.liquidity import LiquiditySDEConfig, LiquidityRuntime, LiquidityState
from ethode.hawkes import HawkesConfig, HawkesRuntime, HawkesState
from ethode.units import UnitManager


class TestFeeConfig:
    """Test Fee configuration and runtime."""

    def test_fee_config_with_strings(self):
        """Test creating fee config with string units."""
        config = FeeConfig(
            base_fee_rate="50 bps",  # 50 basis points = 0.5%
            max_fee_rate="200 bps",   # 2%
            fee_decay_time="1 week",
            min_fee_amount="0.01 USD"
        )

        # Check conversions
        assert config.base_fee_rate[0] == 0.005  # 50 bps = 0.005
        assert config.max_fee_rate[0] == 0.02    # 200 bps = 0.02
        assert config.fee_decay_time[0] == 7 * 24 * 3600  # 1 week in seconds

        print("✓ Fee config with string units works")

    def test_fee_config_with_floats(self):
        """Test creating fee config with plain floats."""
        config = FeeConfig(
            base_fee_rate=0.005,
            max_fee_rate=0.02,
            min_fee_rate=0.001,
            fee_decay_time=3600.0  # 1 hour in seconds
        )

        assert config.base_fee_rate[0] == 0.005
        assert config.min_fee_rate[0] == 0.001
        print("✓ Fee config with floats works")

    def test_fee_validation(self):
        """Test fee config validation."""
        # Fee rate must be between 0 and 1
        with pytest.raises(ValueError, match="between 0 and 1"):
            FeeConfig(base_fee_rate=1.5, max_fee_rate=2.0)

        # Max must be >= base
        with pytest.raises(ValueError, match="Max fee rate.*must be >= base"):
            FeeConfig(base_fee_rate=0.02, max_fee_rate=0.01)

        print("✓ Fee validation works")

    def test_fee_to_runtime(self):
        """Test conversion to runtime structure."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps",
            fee_growth_rate="0.1 / hour",
            min_fee_amount="0.01 USD"
        )

        runtime = config.to_runtime()

        assert isinstance(runtime, FeeRuntime)
        assert abs(float(runtime.base_fee_rate.value) - 0.005) < 1e-6  # Use approx equality for float32
        assert runtime.base_fee_rate.units.dimension == "dimensionless"

        # Growth rate should be converted to 1/second
        assert runtime.fee_growth_rate is not None
        expected_rate = 0.1 / 3600  # per hour to per second
        assert abs(float(runtime.fee_growth_rate.value) - expected_rate) < 1e-6  # Relaxed tolerance for float32

        print("✓ Fee to_runtime conversion works")

    def test_fee_calculation(self):
        """Test fee calculation kernel."""
        config = FeeConfig(
            base_fee_rate=0.01,  # 1%
            max_fee_rate=0.05,   # 5%
            min_fee_amount="1 USD",
            max_fee_amount="1000 USD"
        )

        runtime = config.to_runtime()
        state = FeeState.from_base_rate(0.01)

        # Calculate fee on $10,000 transaction
        new_state, fee = calculate_fee(
            runtime, state,
            transaction_amount=jnp.array(10000.0),
            dt=jnp.array(1.0)
        )

        # Should be 1% of 10,000 = $100
        assert abs(float(fee) - 100.0) < 0.01

        # But capped at max_fee_amount if set
        assert float(fee) <= 1000.0

        print("✓ Fee calculation works")

    def test_fee_stress_update(self):
        """Test stress level update."""
        state = FeeState.zero()

        # Update with high stress indicators
        new_state = update_stress_level(
            state,
            market_volatility=jnp.array(0.8),
            volume_ratio=jnp.array(2.0)
        )

        # Stress should be elevated
        assert float(new_state.stress_level) > 0.5

        print("✓ Fee stress update works")

    def test_fee_summary(self):
        """Test fee config summary generation."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps",
            fee_decay_time="1 day"
        )

        summary = config.summary("text")
        assert "Base fee rate: 0.50%" in summary
        assert "Max fee rate: 2.00%" in summary

        print("✓ Fee summary works")


class TestLiquiditySDEConfig:
    """Test Liquidity SDE configuration and runtime."""

    def test_liquidity_config_with_strings(self):
        """Test creating liquidity config with string units."""
        config = LiquiditySDEConfig(
            initial_liquidity="1000000 USD",
            mean_liquidity="1 million USD",
            mean_reversion_rate="0.1 / day",
            volatility="0.2",  # 20% volatility
            min_liquidity="10000 USD",
            provision_rate="1000 USD/hour"
        )

        assert config.initial_liquidity[0] == 1_000_000
        assert config.mean_liquidity[0] == 1_000_000
        assert config.min_liquidity[0] == 10_000

        # Mean reversion converted to 1/second
        expected_rate = 0.1 / (24 * 3600)
        assert abs(config.mean_reversion_rate[0] - expected_rate) < 1e-10

        print("✓ Liquidity config with string units works")

    def test_liquidity_validation(self):
        """Test liquidity config validation."""
        # Liquidity must be positive
        with pytest.raises(ValueError, match="must be positive"):
            LiquiditySDEConfig(
                initial_liquidity=-1000,
                mean_liquidity=1000,
                mean_reversion_rate=0.1,
                volatility=0.2
            )

        # Max must be > min if both specified
        with pytest.raises(ValueError, match="Max liquidity.*must be > min"):
            LiquiditySDEConfig(
                initial_liquidity=1000,
                mean_liquidity=1000,
                mean_reversion_rate=0.1,
                volatility=0.2,
                min_liquidity=2000,
                max_liquidity=1000
            )

        print("✓ Liquidity validation works")

    def test_liquidity_to_runtime(self):
        """Test conversion to runtime structure."""
        config = LiquiditySDEConfig(
            initial_liquidity="1 million USD",
            mean_liquidity="1 million USD",
            mean_reversion_rate="0.5 / week",
            volatility=0.3,
            jump_intensity="10 / day",
            jump_size_mean="50000 USD"
        )

        runtime = config.to_runtime()

        assert isinstance(runtime, LiquidityRuntime)
        assert float(runtime.initial_liquidity.value) == 1_000_000
        assert runtime.initial_liquidity.units.dimension == "price"

        # Jump intensity should be in events/second
        assert runtime.jump_intensity is not None
        expected_intensity = 10 / (24 * 3600)
        assert abs(float(runtime.jump_intensity.value) - expected_intensity) < 1e-10

        print("✓ Liquidity to_runtime conversion works")

    def test_liquidity_state(self):
        """Test liquidity state initialization."""
        state = LiquidityState.initialize(1_000_000)

        assert float(state.liquidity_level) == 1_000_000
        assert float(state.time) == 0.0
        assert float(state.cumulative_provision) == 0.0

        print("✓ Liquidity state initialization works")

    def test_liquidity_summary(self):
        """Test liquidity config summary generation."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )

        summary = config.summary("markdown")
        assert "Initial liquidity" in summary
        assert "1e+06" in summary or "1000000" in summary

        print("✓ Liquidity summary works")


class TestHawkesConfig:
    """Test Hawkes process configuration and runtime."""

    def test_hawkes_config_with_strings(self):
        """Test creating Hawkes config with string units."""
        config = HawkesConfig(
            jump_rate="100 / hour",
            excitation_strength=0.3,  # 30% excitation
            excitation_decay="5 minutes",
            max_intensity="1000 / hour"
        )

        # Jump rate converted to events/second
        expected_rate = 100 / 3600
        assert abs(config.jump_rate[0] - expected_rate) < 1e-10

        assert config.excitation_strength[0] == 0.3
        assert config.excitation_decay[0] == 5 * 60  # 5 minutes in seconds

        print("✓ Hawkes config with string units works")

    def test_hawkes_stability_validation(self):
        """Test Hawkes stability validation."""
        # Excitation strength must be < 1 for stability
        with pytest.raises(ValueError, match="must be < 1 for stability"):
            HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=1.5,  # Too high!
                excitation_decay="1 minute"
            )

        # Must be non-negative
        with pytest.raises(ValueError, match="must be >= 0"):
            HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=-0.1,
                excitation_decay="1 minute"
            )

        print("✓ Hawkes stability validation works")

    def test_hawkes_to_runtime(self):
        """Test conversion to runtime structure."""
        config = HawkesConfig(
            jump_rate="500 / day",
            excitation_strength=0.4,
            excitation_decay="30 seconds",
            event_impact_mean="10 USD",
            cluster_decay_rate="0.1 / minute"
        )

        runtime = config.to_runtime()

        assert isinstance(runtime, HawkesRuntime)

        # Jump rate in events/second
        expected_rate = 500 / (24 * 3600)
        assert abs(float(runtime.jump_rate.value) - expected_rate) < 1e-6  # Relaxed tolerance for float32

        # Event impact
        assert runtime.event_impact_mean is not None
        assert abs(float(runtime.event_impact_mean.value) - 10.0) < 1e-6  # Use approx equality

        print("✓ Hawkes to_runtime conversion works")

    def test_hawkes_state(self):
        """Test Hawkes state initialization."""
        from conftest import assert_close

        base_rate = 0.01  # events/second
        state = HawkesState.initialize(base_rate)

        assert_close(state.current_intensity, base_rate)
        assert_close(state.event_count, 0)
        assert_close(state.time, 0.0)

        print("✓ Hawkes state initialization works")

    def test_hawkes_summary(self):
        """Test Hawkes config summary generation."""
        config = HawkesConfig(
            jump_rate="100 / hour",
            excitation_strength=0.8,  # High but stable
            excitation_decay="1 minute",
            max_intensity="500 / hour"
        )

        summary = config.summary("text")
        assert "Hawkes Process Configuration" in summary
        assert "STABLE" in summary  # α < 1
        assert "0.8" in summary or "0.800" in summary

        print("✓ Hawkes summary works")

    def test_migration_from_old_params(self):
        """Test that we can migrate from old HawkesParams style."""
        # Old style parameters (from stochastic_extensions.py)
        old_params = {
            'jump_rate': 100.0,  # events per time unit
            'excitation_strength': 0.3,
            'excitation_decay': 1.0  # decay timescale
        }

        # Convert to new config (with units)
        config = HawkesConfig(
            jump_rate=f"{old_params['jump_rate']} / second",
            excitation_strength=old_params['excitation_strength'],
            excitation_decay=f"{old_params['excitation_decay']} second"
        )

        assert config.jump_rate[0] == 100.0
        assert config.excitation_strength[0] == 0.3
        assert config.excitation_decay[0] == 1.0

        print("✓ Migration from old HawkesParams works")


class TestConfigInteroperability:
    """Test that configs work together."""

    def test_all_configs_with_pint(self):
        """Test all configs with pint quantities."""
        manager = UnitManager.instance()

        # Create pint quantities
        rate = manager.registry.Quantity(0.01, "1/second")
        amount = manager.registry.Quantity(1000, "USD")
        time = manager.registry.Quantity(1, "hour")

        # Fee config with pint
        fee_config = FeeConfig(
            base_fee_rate=0.005,
            max_fee_rate=0.02,
            min_fee_amount=amount
        )
        assert fee_config.min_fee_amount[0] == 1000

        # Liquidity config with mixed inputs
        liq_config = LiquiditySDEConfig(
            initial_liquidity=amount * 1000,
            mean_liquidity="1 million USD",
            mean_reversion_rate=rate,
            volatility=0.2
        )
        assert liq_config.initial_liquidity[0] == 1_000_000

        # Hawkes config
        hawkes_config = HawkesConfig(
            jump_rate=rate * 100,
            excitation_strength=0.3,
            excitation_decay=time
        )
        assert hawkes_config.excitation_decay[0] == 3600

        print("✓ All configs work with pint quantities")

    def test_round_trip_conversion(self):
        """Test config -> runtime -> config_output round trip."""
        # Fee round trip
        fee_config = FeeConfig(
            base_fee_rate="75 bps",
            max_fee_rate="300 bps",
            fee_decay_time="2 hours"
        )
        fee_runtime = fee_config.to_runtime()
        fee_output = FeeConfig.from_runtime(fee_runtime)

        # Check basis points are preserved
        assert abs(fee_output.base_fee_rate.magnitude - 75) < 1e-5  # Relaxed for float32
        assert str(fee_output.base_fee_rate.units) == "basis_point"
        # Or convert to dimensionless for comparison
        assert abs(fee_output.base_fee_rate.to("dimensionless").magnitude - 0.0075) < 1e-6  # Relaxed for float32
        assert abs(fee_output.fee_decay_time.to("second").magnitude - 7200) < 0.01

        # Liquidity round trip
        liq_config = LiquiditySDEConfig(
            initial_liquidity="500k USD",
            mean_liquidity="500k USD",
            mean_reversion_rate="0.2 / day",
            volatility=0.15
        )
        liq_runtime = liq_config.to_runtime()
        liq_output = LiquiditySDEConfig.from_runtime(liq_runtime)

        assert abs(liq_output.initial_liquidity.magnitude - 500_000) < 0.01

        # Hawkes round trip
        hawkes_config = HawkesConfig(
            jump_rate="50 / hour",
            excitation_strength=0.25,
            excitation_decay="2 minutes"
        )
        hawkes_runtime = hawkes_config.to_runtime()
        hawkes_output = HawkesConfig.from_runtime(hawkes_runtime)

        assert abs(hawkes_output.excitation_strength.magnitude - 0.25) < 1e-6  # Relaxed for float32
        assert abs(hawkes_output.excitation_decay.to("second").magnitude - 120) < 0.01

        print("✓ Round trip conversions work")


if __name__ == "__main__":
    print("Testing Fee Config...")
    fee_tests = TestFeeConfig()
    fee_tests.test_fee_config_with_strings()
    fee_tests.test_fee_config_with_floats()
    fee_tests.test_fee_validation()
    fee_tests.test_fee_to_runtime()
    fee_tests.test_fee_calculation()
    fee_tests.test_fee_stress_update()
    fee_tests.test_fee_summary()

    print("\nTesting Liquidity SDE Config...")
    liq_tests = TestLiquiditySDEConfig()
    liq_tests.test_liquidity_config_with_strings()
    liq_tests.test_liquidity_validation()
    liq_tests.test_liquidity_to_runtime()
    liq_tests.test_liquidity_state()
    liq_tests.test_liquidity_summary()

    print("\nTesting Hawkes Config...")
    hawkes_tests = TestHawkesConfig()
    hawkes_tests.test_hawkes_config_with_strings()
    hawkes_tests.test_hawkes_stability_validation()
    hawkes_tests.test_hawkes_to_runtime()
    hawkes_tests.test_hawkes_state()
    hawkes_tests.test_hawkes_summary()
    hawkes_tests.test_migration_from_old_params()

    print("\nTesting Interoperability...")
    interop_tests = TestConfigInteroperability()
    interop_tests.test_all_configs_with_pint()
    interop_tests.test_round_trip_conversion()

    print("\n✅ All module config tests passed!")