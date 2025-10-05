"""Tests for LiquidityAdapter class.

This module tests the high-level LiquidityAdapter API following the adapter pattern.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ethode import LiquidityAdapter, LiquiditySDEConfig
from ethode.liquidity.kernel import update_liquidity


class TestLiquidityAdapter:
    """Tests for LiquidityAdapter initialization and basic functionality."""

    def test_init_with_config(self):
        """Test initialization with LiquiditySDEConfig."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        assert adapter.config is config
        assert adapter.runtime is not None
        assert adapter.state is not None
        assert adapter.key is not None

    def test_init_sets_initial_liquidity(self):
        """Test that initialization sets correct initial liquidity."""
        config = LiquiditySDEConfig(
            initial_liquidity="500000 USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.1
        )
        adapter = LiquidityAdapter(config)

        state = adapter.get_state()
        assert abs(state['liquidity_level'] - 500000.0) < 1.0

    def test_step_returns_float(self):
        """Test that step returns Python float, not JAX array."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        liquidity = adapter.step(dt=0.1)
        assert isinstance(liquidity, float)
        assert not isinstance(liquidity, jax.Array)

    def test_step_updates_state(self):
        """Test that step updates internal state."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.1
        )
        adapter = LiquidityAdapter(config, seed=42)

        initial_state = adapter.get_state()
        adapter.step(dt=0.5)
        updated_state = adapter.get_state()

        # Time should advance
        assert updated_state['time'] > initial_state['time']

    def test_reproducible_with_seed(self):
        """Test that same seed produces same results."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )

        adapter1 = LiquidityAdapter(config, seed=42)
        adapter2 = LiquidityAdapter(config, seed=42)

        liq1 = adapter1.step(dt=0.1)
        liq2 = adapter2.step(dt=0.1)

        assert abs(liq1 - liq2) < 0.01


class TestMeanReversion:
    """Tests for mean reversion behavior."""

    def test_reverts_toward_mean_from_below(self):
        """Test that liquidity below mean trends upward."""
        config = LiquiditySDEConfig(
            initial_liquidity="500000 USD",  # Below mean
            mean_liquidity="1000000 USD",
            mean_reversion_rate="0.5 / day",
            volatility=0.01  # Low volatility for predictability
        )
        adapter = LiquidityAdapter(config, seed=42)

        initial = adapter.get_state()['liquidity_level']

        # Run for some time
        for _ in range(100):
            adapter.step(dt=0.1)

        final = adapter.get_state()['liquidity_level']

        # Should move toward mean (1M)
        assert final > initial
        assert initial < 1000000.0 < final or abs(final - 1000000.0) < abs(initial - 1000000.0)

    def test_reverts_toward_mean_from_above(self):
        """Test that liquidity above mean trends downward."""
        config = LiquiditySDEConfig(
            initial_liquidity="1500000 USD",  # Above mean
            mean_liquidity="1000000 USD",
            mean_reversion_rate="0.5 / day",
            volatility=0.01  # Low volatility
        )
        adapter = LiquidityAdapter(config, seed=42)

        initial = adapter.get_state()['liquidity_level']

        for _ in range(100):
            adapter.step(dt=0.1)

        final = adapter.get_state()['liquidity_level']

        # Should move toward mean
        assert final < initial
        assert final > 1000000.0


class TestStochasticBehavior:
    """Tests for stochastic properties."""

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different paths."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.3
        )

        adapter1 = LiquidityAdapter(config, seed=42)
        adapter2 = LiquidityAdapter(config, seed=123)

        liquidity1 = [adapter1.step(dt=0.1) for _ in range(50)]
        liquidity2 = [adapter2.step(dt=0.1) for _ in range(50)]

        # Paths should diverge - check that they're not identical
        differences = [abs(l1 - l2) for l1, l2 in zip(liquidity1, liquidity2)]
        max_diff = max(differences)
        # With different seeds and high volatility, max difference should be substantial
        assert max_diff > 1000.0  # Should differ by at least 1000 USD at some point

    def test_volatility_affects_variance(self):
        """Test that higher volatility increases variance."""
        config_low = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.05  # Low
        )

        config_high = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.5  # High
        )

        # Run multiple paths for low volatility
        low_vol_paths = []
        for seed in range(10):
            adapter = LiquidityAdapter(config_low, seed=seed)
            path = [adapter.step(dt=0.1) for _ in range(50)]
            low_vol_paths.append(path)

        # Run multiple paths for high volatility
        high_vol_paths = []
        for seed in range(10):
            adapter = LiquidityAdapter(config_high, seed=seed)
            path = [adapter.step(dt=0.1) for _ in range(50)]
            high_vol_paths.append(path)

        # Compare variances
        low_variance = np.var([path[-1] for path in low_vol_paths])
        high_variance = np.var([path[-1] for path in high_vol_paths])

        assert high_variance > low_variance


class TestLiquidityShocks:
    """Tests for external liquidity shocks."""

    def test_apply_positive_shock(self):
        """Test adding liquidity via shock."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.0  # No stochasticity
        )
        adapter = LiquidityAdapter(config)

        initial = adapter.get_state()['liquidity_level']

        # Add 100k
        adapter.apply_shock(amount=100000.0)

        after_shock = adapter.get_state()['liquidity_level']
        assert abs(after_shock - (initial + 100000.0)) < 1.0

    def test_apply_negative_shock(self):
        """Test removing liquidity via shock."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.0
        )
        adapter = LiquidityAdapter(config)

        initial = adapter.get_state()['liquidity_level']

        # Remove 100k
        adapter.apply_shock(amount=-100000.0)

        after_shock = adapter.get_state()['liquidity_level']
        assert abs(after_shock - (initial - 100000.0)) < 1.0

    def test_shock_tracks_cumulative_provision(self):
        """Test that positive shocks are tracked."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.0
        )
        adapter = LiquidityAdapter(config)

        adapter.apply_shock(amount=50000.0)
        adapter.apply_shock(amount=30000.0)

        state = adapter.get_state()
        assert state['cumulative_provision'] >= 80000.0

    def test_shock_tracks_cumulative_removal(self):
        """Test that negative shocks are tracked."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.0
        )
        adapter = LiquidityAdapter(config)

        adapter.apply_shock(amount=-50000.0)
        adapter.apply_shock(amount=-30000.0)

        state = adapter.get_state()
        assert state['cumulative_removal'] >= 80000.0


class TestBounds:
    """Tests for liquidity bounds."""

    def test_min_liquidity_bound(self):
        """Test that liquidity doesn't go below minimum."""
        config = LiquiditySDEConfig(
            initial_liquidity="100000 USD",
            mean_liquidity="100000 USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.5,
            min_liquidity="50000 USD"
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Run many steps
        for _ in range(1000):
            liquidity = adapter.step(dt=0.1)
            assert liquidity >= 50000.0 - 1.0  # Allow tiny numerical error

    def test_max_liquidity_bound(self):
        """Test that liquidity doesn't exceed maximum."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.5,
            max_liquidity="2M USD"
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Run many steps
        for _ in range(1000):
            liquidity = adapter.step(dt=0.1)
            assert liquidity <= 2000000.0 + 1.0  # Allow tiny numerical error

    def test_shock_respects_bounds(self):
        """Test that shocks respect bounds."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.0,
            min_liquidity="500000 USD",
            max_liquidity="1500000 USD"
        )
        adapter = LiquidityAdapter(config)

        # Try to remove too much
        adapter.apply_shock(amount=-1000000.0)
        state = adapter.get_state()
        assert state['liquidity_level'] >= 500000.0

        # Reset and try to add too much
        adapter.reset()
        adapter.apply_shock(amount=1000000.0)
        state = adapter.get_state()
        assert state['liquidity_level'] <= 1500000.0


class TestDiagnostics:
    """Tests for step_with_diagnostics."""

    def test_diagnostics_returns_dict(self):
        """Test that diagnostics returns a dictionary."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        liquidity, diag = adapter.step_with_diagnostics(dt=0.1)

        assert isinstance(diag, dict)
        assert 'drift' in diag
        assert 'diffusion' in diag
        assert 'jump' in diag

    def test_diagnostics_tracks_components(self):
        """Test that diagnostics correctly tracks components."""
        config = LiquiditySDEConfig(
            initial_liquidity="500000 USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.2 / day",
            volatility=0.0  # No diffusion
        )
        adapter = LiquidityAdapter(config, seed=42)

        liquidity, diag = adapter.step_with_diagnostics(dt=1.0)

        # With no volatility, drift should dominate
        assert diag['drift'] != 0.0
        assert abs(diag['diffusion']) < 0.01  # Should be very small

    def test_diagnostics_matches_step(self):
        """Test that liquidity from diagnostics matches step."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Get liquidity from regular step
        liq1 = adapter.step(dt=0.1)

        # Reset and get from diagnostics
        adapter.reset(seed=42)
        liq2, diag = adapter.step_with_diagnostics(dt=0.1)

        assert abs(liq1 - liq2) < 1e-6


class TestReset:
    """Tests for reset functionality."""

    def test_reset_restores_initial_liquidity(self):
        """Test that reset restores initial liquidity."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="500000 USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Run some steps
        for _ in range(50):
            adapter.step(dt=0.1)

        # Reset
        adapter.reset()

        state = adapter.get_state()
        assert abs(state['liquidity_level'] - 1000000.0) < 1.0

    def test_reset_clears_time(self):
        """Test that reset clears time."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config)

        adapter.step(dt=5.0)
        assert adapter.get_state()['time'] > 0

        adapter.reset()
        assert adapter.get_state()['time'] == 0.0

    def test_reset_clears_cumulative_tracking(self):
        """Test that reset clears cumulative provision/removal."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.3
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Run steps to accumulate changes
        for _ in range(100):
            adapter.step(dt=0.1)

        # Reset
        adapter.reset()

        state = adapter.get_state()
        assert state['cumulative_provision'] == 0.0
        assert state['cumulative_removal'] == 0.0
        assert state['jump_count'] == 0

    def test_reset_with_new_seed(self):
        """Test reset with different seed."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        liq1 = adapter.step(dt=0.1)

        adapter.reset(seed=123)
        liq2 = adapter.step(dt=0.1)

        # Different seeds should produce different results
        assert abs(liq1 - liq2) > 0.01


class TestGetState:
    """Tests for get_state functionality."""

    def test_get_state_returns_dict(self):
        """Test that get_state returns a dictionary."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config)

        state = adapter.get_state()
        assert isinstance(state, dict)

    def test_get_state_has_required_keys(self):
        """Test that get_state has all required keys."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config)

        state = adapter.get_state()
        assert 'liquidity_level' in state
        assert 'time' in state
        assert 'cumulative_provision' in state
        assert 'cumulative_removal' in state
        assert 'jump_count' in state

    def test_get_state_returns_python_types(self):
        """Test that get_state values are Python types."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config)

        state = adapter.get_state()
        assert isinstance(state['liquidity_level'], float)
        assert isinstance(state['time'], float)
        assert isinstance(state['cumulative_provision'], float)
        assert isinstance(state['cumulative_removal'], float)
        assert isinstance(state['jump_count'], int)


class TestJAXCompatibility:
    """Tests for JAX compatibility (direct runtime access)."""

    def test_runtime_state_direct_access(self):
        """Test that runtime and state can be accessed directly."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config)

        # Should be able to access runtime and state
        runtime = adapter.runtime
        state = adapter.state

        assert runtime is not None
        assert state is not None

    def test_direct_kernel_usage(self):
        """Test that adapter state works with direct kernel calls."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Use kernel directly
        key = jax.random.PRNGKey(42)
        dt = jnp.array(0.1)

        new_state, liquidity = update_liquidity(
            adapter.runtime,
            adapter.state,
            key,
            dt
        )

        assert liquidity > 0
        assert new_state is not None


class TestRegressionVsKernel:
    """Regression tests against direct kernel usage."""

    def test_adapter_matches_kernel_output(self):
        """Test that adapter produces same results as direct kernel."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Adapter approach
        liq_adapter = adapter.step(dt=0.1)

        # Direct kernel approach
        adapter.reset(seed=42)
        key, subkey = jax.random.split(adapter.key)
        dt = jnp.array(0.1)
        new_state, liq_kernel = update_liquidity(
            adapter.runtime,
            adapter.state,
            subkey,
            dt
        )

        # Should match
        assert abs(liq_adapter - float(liq_kernel)) < 1e-6


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_dt(self):
        """Test that zero dt doesn't cause errors."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config)

        initial = adapter.get_state()['liquidity_level']
        liquidity = adapter.step(dt=0.0)

        # Should barely change
        assert abs(liquidity - initial) < 1000.0

    def test_very_small_dt(self):
        """Test with very small time step."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )
        adapter = LiquidityAdapter(config)

        liquidity = adapter.step(dt=1e-9)
        # Should still be close to initial
        assert 999000.0 < liquidity < 1001000.0

    def test_large_dt_doesnt_crash(self):
        """Test that large time steps don't cause errors."""
        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.1
        )
        adapter = LiquidityAdapter(config, seed=42)

        # Should handle moderate time steps without crashing
        for _ in range(10):
            liquidity = adapter.step(dt=5.0)
            # Just check it's a reasonable value (not NaN, inf, or negative)
            assert 0 < liquidity < 10000000.0  # Within reasonable bounds
            assert not np.isnan(liquidity)
            assert not np.isinf(liquidity)

    def test_zero_volatility_is_deterministic(self):
        """Test that zero volatility produces deterministic results."""
        config = LiquiditySDEConfig(
            initial_liquidity="500000 USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.2 / day",
            volatility=0.0  # Deterministic
        )

        adapter1 = LiquidityAdapter(config, seed=42)
        adapter2 = LiquidityAdapter(config, seed=999)

        # Different seeds but zero volatility = same results
        liq1 = adapter1.step(dt=1.0)
        liq2 = adapter2.step(dt=1.0)

        assert abs(liq1 - liq2) < 1e-6
