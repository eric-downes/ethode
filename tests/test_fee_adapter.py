"""Tests for FeeAdapter class.

This module tests the high-level FeeAdapter API following the adapter pattern.
"""

import pytest
import jax
import jax.numpy as jnp

from ethode import FeeAdapter, FeeConfig
from ethode.fee.kernel import calculate_fee


class TestFeeAdapter:
    """Tests for FeeAdapter initialization and basic functionality."""

    def test_init_with_config(self):
        """Test initialization with FeeConfig."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        assert adapter.config is config
        assert adapter.runtime is not None
        assert adapter.state is not None

    def test_init_sets_base_rate(self):
        """Test that initialization sets current rate to base rate."""
        config = FeeConfig(
            base_fee_rate="75 bps",
            max_fee_rate="300 bps"
        )
        adapter = FeeAdapter(config)

        state = adapter.get_state()
        # 75 bps = 0.0075
        assert abs(state['current_fee_rate'] - 0.0075) < 1e-6

    def test_step_basic(self):
        """Test basic fee calculation."""
        config = FeeConfig(
            base_fee_rate="100 bps",  # 1%
            max_fee_rate="500 bps"
        )
        adapter = FeeAdapter(config)

        # 1% of 100 = 1.0
        fee = adapter.step(transaction_amount=100.0, dt=0.1)
        assert abs(fee - 1.0) < 0.01

    def test_step_returns_float(self):
        """Test that step returns Python float, not JAX array."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        fee = adapter.step(100.0, dt=0.1)
        assert isinstance(fee, float)
        assert not isinstance(fee, jax.Array)

    def test_step_updates_state(self):
        """Test that step updates internal state."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        initial_state = adapter.get_state()
        adapter.step(100.0, dt=0.5)
        updated_state = adapter.get_state()

        # Time should advance
        assert updated_state['last_update_time'] > initial_state['last_update_time']
        # Fees should accumulate
        assert updated_state['accumulated_fees'] > initial_state['accumulated_fees']


class TestFeeAccumulation:
    """Tests for fee accumulation over multiple steps."""

    def test_fee_accumulation(self):
        """Test that fees accumulate correctly."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        # Process multiple transactions
        fee1 = adapter.step(100.0, dt=0.1)
        fee2 = adapter.step(200.0, dt=0.1)

        state = adapter.get_state()
        expected_total = fee1 + fee2
        assert abs(state['accumulated_fees'] - expected_total) < 0.01

    def test_multiple_steps_accumulate(self):
        """Test that multiple steps accumulate fees."""
        config = FeeConfig(
            base_fee_rate="100 bps",
            max_fee_rate="500 bps"
        )
        adapter = FeeAdapter(config)

        total_fees = 0.0
        for _ in range(10):
            fee = adapter.step(50.0, dt=0.1)
            total_fees += fee

        state = adapter.get_state()
        assert abs(state['accumulated_fees'] - total_fees) < 0.01


class TestStressAdjustment:
    """Tests for stress-based fee adjustment."""

    def test_update_stress_changes_state(self):
        """Test that update_stress modifies state."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="500 bps",
            fee_growth_rate="10 bps / second"
        )
        adapter = FeeAdapter(config)

        initial_stress = adapter.get_state()['stress_level']
        adapter.update_stress(volatility=0.8, volume_ratio=2.5)
        updated_stress = adapter.get_state()['stress_level']

        assert updated_stress != initial_stress
        assert updated_stress > 0.0

    def test_stress_increases_fees(self):
        """Test that stress level increases fees over time."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="500 bps",
            fee_growth_rate="100 bps / second"
        )
        adapter = FeeAdapter(config)

        # Low stress baseline
        adapter.update_stress(volatility=0.0, volume_ratio=1.0)
        fee_low = adapter.step(100.0, dt=0.1)

        # Same adapter, let stress grow the rate
        adapter.update_stress(volatility=0.9, volume_ratio=3.0)
        # Wait some time for stress to affect rate
        adapter.step(100.0, dt=1.0)

        # Reset accumulated fees for fair comparison
        adapter.state = adapter.state.__class__(
            current_fee_rate=adapter.state.current_fee_rate,
            accumulated_fees=jnp.array(0.0),
            last_update_time=adapter.state.last_update_time,
            stress_level=adapter.state.stress_level,
        )

        fee_high = adapter.step(100.0, dt=0.1)

        # High stress should result in higher fees
        assert fee_high > fee_low

    def test_high_volatility_increases_stress(self):
        """Test that high volatility increases stress level."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="500 bps"
        )
        adapter = FeeAdapter(config)

        adapter.update_stress(volatility=0.9, volume_ratio=1.0)
        stress_high_vol = adapter.get_state()['stress_level']

        adapter.update_stress(volatility=0.1, volume_ratio=1.0)
        stress_low_vol = adapter.get_state()['stress_level']

        assert stress_high_vol > stress_low_vol

    def test_high_volume_increases_stress(self):
        """Test that high volume ratio increases stress level."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="500 bps"
        )
        adapter = FeeAdapter(config)

        adapter.update_stress(volatility=0.5, volume_ratio=5.0)
        stress_high_vol = adapter.get_state()['stress_level']

        adapter.update_stress(volatility=0.5, volume_ratio=0.5)
        stress_low_vol = adapter.get_state()['stress_level']

        assert stress_high_vol > stress_low_vol


class TestFeeDecay:
    """Tests for fee rate decay over time."""

    def test_fee_rate_decays_toward_base(self):
        """Test that fee rate decays toward base rate."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="500 bps",
            fee_decay_time="1 second"
        )
        adapter = FeeAdapter(config)

        # Set high fee rate by increasing stress and waiting
        adapter.update_stress(volatility=1.0, volume_ratio=10.0)
        adapter.state = adapter.state.__class__(
            current_fee_rate=jnp.array(0.03),  # 300 bps, well above base
            accumulated_fees=adapter.state.accumulated_fees,
            last_update_time=adapter.state.last_update_time,
            stress_level=adapter.state.stress_level,
        )

        initial_rate = adapter.get_state()['current_fee_rate']

        # Let time pass without updating stress
        adapter.update_stress(volatility=0.0, volume_ratio=1.0)  # Low stress
        for _ in range(10):
            adapter.step(1.0, dt=0.5)

        final_rate = adapter.get_state()['current_fee_rate']
        base_rate = 0.005  # 50 bps

        # Rate should decay toward base
        assert final_rate < initial_rate
        assert abs(final_rate - base_rate) < abs(initial_rate - base_rate)


class TestFeeBounds:
    """Tests for fee rate and amount bounds."""

    def test_max_fee_rate_bound(self):
        """Test that fee rate does not exceed maximum."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps",
            fee_growth_rate="1000 bps / second"
        )
        adapter = FeeAdapter(config)

        # Try to push rate very high
        adapter.update_stress(volatility=1.0, volume_ratio=100.0)
        for _ in range(100):
            adapter.step(100.0, dt=1.0)

        state = adapter.get_state()
        max_rate = 0.02  # 200 bps
        assert state['current_fee_rate'] <= max_rate + 1e-6

    def test_min_fee_amount(self):
        """Test minimum fee amount is enforced."""
        config = FeeConfig(
            base_fee_rate="1 bps",  # Very low
            max_fee_rate="100 bps",
            min_fee_amount="1 USD"
        )
        adapter = FeeAdapter(config)

        # Small transaction that would normally generate tiny fee
        fee = adapter.step(transaction_amount=10.0, dt=0.1)

        # Should be at least min_fee_amount
        assert fee >= 1.0

    def test_max_fee_amount(self):
        """Test maximum fee amount is enforced."""
        config = FeeConfig(
            base_fee_rate="1000 bps",  # 10%
            max_fee_rate="2000 bps",
            max_fee_amount="5 USD"
        )
        adapter = FeeAdapter(config)

        # Large transaction that would normally generate huge fee
        fee = adapter.step(transaction_amount=1000.0, dt=0.1)

        # Should not exceed max_fee_amount
        assert fee <= 5.0 + 1e-6


class TestDiagnostics:
    """Tests for step_with_diagnostics."""

    def test_diagnostics_returns_dict(self):
        """Test that diagnostics returns a dictionary."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        fee, diag = adapter.step_with_diagnostics(100.0, dt=0.1)

        assert isinstance(diag, dict)
        assert 'base_fee' in diag
        assert 'stress_adjustment' in diag
        assert 'current_rate' in diag

    def test_diagnostics_base_fee_correct(self):
        """Test that base_fee in diagnostics is correct."""
        config = FeeConfig(
            base_fee_rate="100 bps",  # 1%
            max_fee_rate="500 bps"
        )
        adapter = FeeAdapter(config)

        fee, diag = adapter.step_with_diagnostics(200.0, dt=0.1)

        # Base fee should be 1% of 200 = 2.0
        assert abs(diag['base_fee'] - 2.0) < 0.01

    def test_diagnostics_matches_step(self):
        """Test that fee from diagnostics matches step."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        # Get fee from regular step
        fee1 = adapter.step(100.0, dt=0.1)

        # Reset and get from diagnostics
        adapter.reset()
        fee2, diag = adapter.step_with_diagnostics(100.0, dt=0.1)

        assert abs(fee1 - fee2) < 1e-6


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_accumulated_fees(self):
        """Test that reset clears accumulated fees."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        # Accumulate some fees
        adapter.step(100.0, dt=0.1)
        adapter.step(200.0, dt=0.1)
        assert adapter.get_state()['accumulated_fees'] > 0

        # Reset
        adapter.reset()

        # Fees should be zero
        assert adapter.get_state()['accumulated_fees'] == 0.0

    def test_reset_restores_base_rate(self):
        """Test that reset restores base fee rate."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="500 bps",
            fee_growth_rate="100 bps / second"
        )
        adapter = FeeAdapter(config)

        # Push rate higher
        adapter.update_stress(volatility=1.0, volume_ratio=10.0)
        adapter.step(100.0, dt=5.0)

        initial_rate = adapter.get_state()['current_fee_rate']
        base_rate = 0.005  # 50 bps

        # Reset
        adapter.reset()

        reset_rate = adapter.get_state()['current_fee_rate']
        assert abs(reset_rate - base_rate) < 1e-6

    def test_reset_clears_stress(self):
        """Test that reset clears stress level."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        adapter.update_stress(volatility=0.8, volume_ratio=3.0)
        assert adapter.get_state()['stress_level'] > 0

        adapter.reset()
        assert adapter.get_state()['stress_level'] == 0.0

    def test_reset_clears_time(self):
        """Test that reset clears time."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        adapter.step(100.0, dt=5.0)
        assert adapter.get_state()['last_update_time'] > 0

        adapter.reset()
        assert adapter.get_state()['last_update_time'] == 0.0


class TestGetState:
    """Tests for get_state functionality."""

    def test_get_state_returns_dict(self):
        """Test that get_state returns a dictionary."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        state = adapter.get_state()
        assert isinstance(state, dict)

    def test_get_state_has_required_keys(self):
        """Test that get_state has all required keys."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        state = adapter.get_state()
        assert 'current_fee_rate' in state
        assert 'accumulated_fees' in state
        assert 'last_update_time' in state
        assert 'stress_level' in state

    def test_get_state_returns_floats(self):
        """Test that get_state values are Python floats."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        state = adapter.get_state()
        for key, value in state.items():
            assert isinstance(value, float), f"{key} is not a float"
            assert not isinstance(value, jax.Array), f"{key} is a JAX array"


class TestJAXCompatibility:
    """Tests for JAX compatibility (direct runtime access)."""

    def test_runtime_state_direct_access(self):
        """Test that runtime and state can be accessed directly."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        # Should be able to access runtime and state
        runtime = adapter.runtime
        state = adapter.state

        assert runtime is not None
        assert state is not None

    def test_direct_kernel_usage(self):
        """Test that adapter state works with direct kernel calls."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        # Use kernel directly
        amount = jnp.array(100.0)
        dt = jnp.array(0.1)

        new_state, fee = calculate_fee(
            adapter.runtime,
            adapter.state,
            amount,
            dt
        )

        assert fee > 0
        assert new_state is not None


class TestRegressionVsKernel:
    """Regression tests against direct kernel usage."""

    def test_adapter_matches_kernel_output(self):
        """Test that adapter produces same results as direct kernel."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps",
            fee_decay_time="1 day"
        )
        adapter = FeeAdapter(config)

        # Adapter approach
        fee_adapter = adapter.step(100.0, dt=0.1)

        # Direct kernel approach
        adapter.reset()
        amount = jnp.array(100.0)
        dt = jnp.array(0.1)
        new_state, fee_kernel = calculate_fee(
            adapter.runtime,
            adapter.state,
            amount,
            dt
        )

        # Should match
        assert abs(fee_adapter - float(fee_kernel)) < 1e-6


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_transaction_amount(self):
        """Test fee calculation with zero transaction amount."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        fee = adapter.step(transaction_amount=0.0, dt=0.1)
        assert fee == 0.0

    def test_very_large_transaction(self):
        """Test fee calculation with very large transaction."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        fee = adapter.step(transaction_amount=1e9, dt=0.1)
        # Should be approximately 0.5% of 1 billion
        assert fee > 1e6

    def test_zero_dt(self):
        """Test that zero dt doesn't cause errors."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )
        adapter = FeeAdapter(config)

        # Should not crash
        fee = adapter.step(100.0, dt=0.0)
        assert fee >= 0

    def test_very_small_dt(self):
        """Test with very small time step."""
        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps",
            fee_decay_time="1 day"
        )
        adapter = FeeAdapter(config)

        fee = adapter.step(100.0, dt=1e-9)
        # Should still calculate reasonable fee
        assert fee > 0
        assert fee < 2.0  # Should be around 0.5% of 100
