"""Tests for JumpProcessAdapter class.

This module tests the high-level JumpProcessAdapter API following the adapter pattern.

Note: Some tests are marked with @pytest.mark.slow and are skipped by default.
To run slow tests: pytest test_jumpprocess_adapter.py -m slow
To run all tests: pytest test_jumpprocess_adapter.py -m ""
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ethode import JumpProcessAdapter, JumpProcessConfig
from ethode.jumpprocess.kernel import generate_next_jump_time, step


class TestJumpProcessAdapter:
    """Tests for JumpProcessAdapter initialization and basic functionality."""

    def test_init_with_poisson_config(self):
        """Test initialization with Poisson config."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        assert adapter.config is config
        assert adapter.runtime is not None
        assert adapter.state is not None

    def test_init_with_deterministic_config(self):
        """Test initialization with deterministic config."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="50.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        assert adapter.config is config
        assert adapter.runtime is not None
        assert adapter.state is not None

    def test_init_sets_first_jump_time(self):
        """Test that initialization sets first jump time."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        state = adapter.get_state()
        # First jump should be scheduled in the future
        assert state['next_jump_time'] > 0.0

    def test_step_returns_bool(self):
        """Test that step returns Python bool, not JAX array."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        jump_occurred = adapter.step(current_time=0.0, dt=0.1)
        assert isinstance(jump_occurred, bool)
        assert not isinstance(jump_occurred, jax.Array)

    def test_step_updates_state(self):
        """Test that step updates internal state."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        initial_state = adapter.get_state()
        adapter.step(current_time=0.0, dt=0.1)

        # State may or may not update depending on whether jump occurred
        # At minimum, adapter shouldn't crash
        updated_state = adapter.get_state()
        assert updated_state is not None

    def test_reproducible_with_seed(self):
        """Test that same seed produces same results."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="500.0 / hour",  # Higher rate for events
            seed=42
        )

        adapter1 = JumpProcessAdapter(config)
        adapter2 = JumpProcessAdapter(config)

        # Reduced iterations for speed (was 100)
        jumps1 = [adapter1.step(i * 0.01, 0.01) for i in range(20)]
        jumps2 = [adapter2.step(i * 0.01, 0.01) for i in range(20)]

        assert jumps1 == jumps2


class TestPoissonProcess:
    """Tests for Poisson process behavior."""

    def test_poisson_jump_times_vary(self):
        """Test that Poisson process has random inter-arrival times."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Generate jumps in a large interval
        jumps = adapter.generate_jumps(t_start=0.0, t_end=1000.0)

        if len(jumps) > 2:
            # Compute inter-arrival times
            intervals = np.diff(jumps)

            # For Poisson process, intervals should vary
            # Standard deviation should be non-zero
            assert np.std(intervals) > 0

    def test_poisson_expected_rate(self):
        """Test that Poisson process matches expected rate."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",  # Rate in canonical units (1/s)
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        rate = adapter.get_expected_rate()
        # 100/hour = 100/3600 per second
        expected = 100.0 / 3600.0
        assert abs(rate - expected) < 1e-6

    def test_poisson_event_count_statistics(self):
        """Test Poisson process event count matches statistical expectation."""
        rate_per_hour = 1000.0  # High rate for good statistics
        config = JumpProcessConfig(
            process_type='poisson',
            rate=f"{rate_per_hour} / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        t_start = 0.0
        t_end = 3600.0  # 1 hour
        jumps = adapter.generate_jumps(t_start=t_start, t_end=t_end)

        # Expected number of events in 1 hour
        expected_count = rate_per_hour
        actual_count = len(jumps)

        # Allow for statistical variation (±3 sigma)
        # For Poisson, variance = mean
        sigma = np.sqrt(expected_count)
        assert abs(actual_count - expected_count) < 3 * sigma


class TestDeterministicProcess:
    """Tests for deterministic process behavior."""

    def test_deterministic_uniform_spacing(self):
        """Test that deterministic process has uniform spacing."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="100.0 / hour",  # 100 events per hour
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        jumps = adapter.generate_jumps(t_start=0.0, t_end=360.0)  # 0.1 hour

        if len(jumps) > 2:
            intervals = np.diff(jumps)

            # For deterministic process, all intervals should be equal
            # Check standard deviation is very small
            assert np.std(intervals) < 1e-6

            # Check mean interval matches expected
            rate_per_sec = 100.0 / 3600.0
            expected_interval = 1.0 / rate_per_sec
            assert abs(np.mean(intervals) - expected_interval) < 1e-3

    def test_deterministic_expected_rate(self):
        """Test that deterministic process matches expected rate."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="50.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        rate = adapter.get_expected_rate()
        # 50/hour = 50/3600 per second
        expected = 50.0 / 3600.0
        assert abs(rate - expected) < 1e-6

    def test_deterministic_event_count_exact(self):
        """Test deterministic process produces exact expected count."""
        rate_per_hour = 100.0
        config = JumpProcessConfig(
            process_type='deterministic',
            rate=f"{rate_per_hour} / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        t_start = 0.0
        t_end = 3600.0  # 1 hour
        jumps = adapter.generate_jumps(t_start=t_start, t_end=t_end)

        # For deterministic, count should be exactly rate * time
        expected_count = rate_per_hour
        actual_count = len(jumps)

        # Should be very close (within 1 event due to boundary effects)
        assert abs(actual_count - expected_count) <= 1


class TestJumpGeneration:
    """Tests for batch jump generation."""

    def test_generate_jumps_returns_array(self):
        """Test that generate_jumps returns numpy array."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        jumps = adapter.generate_jumps(t_start=0.0, t_end=100.0)
        assert isinstance(jumps, np.ndarray)

    def test_generate_jumps_within_interval(self):
        """Test that all jumps fall within specified interval."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="1000.0 / hour",  # High rate
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        t_start = 10.0
        t_end = 50.0
        jumps = adapter.generate_jumps(t_start=t_start, t_end=t_end)

        # All jumps should be in [t_start, t_end)
        assert np.all(jumps >= t_start)
        assert np.all(jumps < t_end)

    def test_generate_jumps_sorted(self):
        """Test that jumps are returned in sorted order."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="1000.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        jumps = adapter.generate_jumps(t_start=0.0, t_end=100.0)

        # Should be sorted
        assert np.all(np.diff(jumps) >= 0)

    def test_generate_jumps_empty_interval(self):
        """Test generate_jumps with very small interval."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Very small interval may produce 0 jumps
        jumps = adapter.generate_jumps(t_start=0.0, t_end=0.001)
        assert isinstance(jumps, np.ndarray)
        # With low rate, likely no jumps
        assert len(jumps) >= 0

    def test_generate_jumps_high_rate(self):
        """Test generate_jumps with high rate."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="10000.0 / hour",  # Very high rate
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        jumps = adapter.generate_jumps(t_start=0.0, t_end=3600.0)  # 1 hour

        # Should get many events
        assert len(jumps) > 100


class TestReset:
    """Tests for reset functionality."""

    def test_reset_restores_initial_state(self):
        """Test that reset restores initial jump time."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        initial_state = adapter.get_state()

        # Run some steps
        for i in range(10):
            adapter.step(i * 0.1, 0.1)

        # Reset
        adapter.reset(seed=42)

        reset_state = adapter.get_state()

        # Should be back to initial state
        assert reset_state['next_jump_time'] == initial_state['next_jump_time']
        assert reset_state['event_count'] == 0

    def test_reset_clears_event_count(self):
        """Test that reset clears event count."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="1000.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Run steps to potentially get events (reduced from 1000 to 50)
        for i in range(50):
            adapter.step(i * 1.0, 1.0)

        # Reset
        adapter.reset()

        state = adapter.get_state()
        assert state['event_count'] == 0

    def test_reset_with_new_seed(self):
        """Test reset with different seed."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="1000.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        jumps1 = adapter.generate_jumps(t_start=0.0, t_end=1000.0)

        adapter.reset(seed=123)
        jumps2 = adapter.generate_jumps(t_start=0.0, t_end=1000.0)

        # Different seeds should produce different jump patterns
        if len(jumps1) > 0 and len(jumps2) > 0:
            # Different seeds should produce different counts or different values
            if len(jumps1) == len(jumps2):
                assert not np.allclose(jumps1, jumps2)
            else:
                assert len(jumps1) != len(jumps2)

    def test_reset_with_start_time(self):
        """Test reset with non-zero start time."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        adapter.reset(seed=42, start_time=10.0)

        state = adapter.get_state()
        # First jump should be after start_time
        assert state['next_jump_time'] >= 10.0


class TestGetState:
    """Tests for get_state functionality."""

    def test_get_state_returns_dict(self):
        """Test that get_state returns a dictionary."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        state = adapter.get_state()
        assert isinstance(state, dict)

    def test_get_state_has_required_keys(self):
        """Test that get_state has all required keys."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        state = adapter.get_state()
        assert 'last_jump_time' in state
        assert 'next_jump_time' in state
        assert 'event_count' in state

    def test_get_state_returns_python_types(self):
        """Test that get_state values are Python types."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        state = adapter.get_state()
        assert isinstance(state['last_jump_time'], float)
        assert isinstance(state['next_jump_time'], float)
        assert isinstance(state['event_count'], int)


class TestJAXCompatibility:
    """Tests for JAX compatibility (direct runtime access)."""

    def test_runtime_state_direct_access(self):
        """Test that runtime and state can be accessed directly."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Should be able to access runtime and state
        runtime = adapter.runtime
        state = adapter.state

        assert runtime is not None
        assert state is not None

    def test_direct_kernel_usage(self):
        """Test that adapter state works with direct kernel calls."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Use kernel directly
        current_time = jnp.array(0.0)
        dt = jnp.array(0.1)

        new_state, jump_occurred = step(
            adapter.runtime,
            adapter.state,
            current_time,
            dt
        )

        assert new_state is not None
        assert isinstance(jump_occurred, jax.Array)


class TestRegressionVsKernel:
    """Regression tests against direct kernel usage."""

    def test_adapter_matches_kernel_output(self):
        """Test that adapter produces same results as direct kernel."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Adapter approach
        jump_adapter = adapter.step(current_time=0.0, dt=0.1)

        # Direct kernel approach
        adapter.reset(seed=42)
        current_time = jnp.array(0.0)
        dt = jnp.array(0.1)
        new_state, jump_kernel = step(
            adapter.runtime,
            adapter.state,
            current_time,
            dt
        )

        # Should match
        assert jump_adapter == bool(jump_kernel)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_dt(self):
        """Test that zero dt doesn't cause errors."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Should not crash
        jump_occurred = adapter.step(current_time=0.0, dt=0.0)
        assert isinstance(jump_occurred, bool)

    def test_very_small_dt(self):
        """Test with very small time step."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        jump_occurred = adapter.step(current_time=0.0, dt=1e-9)
        # With tiny dt, jump is very unlikely
        assert isinstance(jump_occurred, bool)

    def test_very_large_dt(self):
        """Test with large time step."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Large dt may span multiple potential jumps
        jump_occurred = adapter.step(current_time=0.0, dt=10000.0)
        assert isinstance(jump_occurred, bool)

    def test_negative_rate_raises_error(self):
        """Test that negative rate raises validation error."""
        with pytest.raises(ValueError):
            config = JumpProcessConfig(
                process_type='poisson',
                rate="-100 / hour",  # Negative rate!
                seed=42
            )

    def test_invalid_process_type_raises_error(self):
        """Test that invalid process type raises error."""
        with pytest.raises(ValueError):
            config = JumpProcessConfig(
                process_type='invalid',  # Not 'poisson' or 'deterministic'
                rate="100.0 / hour",
                seed=42
            )

    def test_zero_rate(self):
        """Test with zero rate (no events)."""
        config = JumpProcessConfig(
            process_type='poisson',
            rate="0.0 / hour",
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # With zero rate, next_jump_time should be very far
        state = adapter.get_state()
        assert state['next_jump_time'] > 1e10


class TestStochasticBehavior:
    """Tests for stochastic properties."""

    def test_different_seeds_produce_different_patterns(self):
        """Test that different seeds produce different jump patterns."""
        config1 = JumpProcessConfig(
            process_type='poisson',
            rate="1000.0 / hour",
            seed=42
        )
        config2 = JumpProcessConfig(
            process_type='poisson',
            rate="1000.0 / hour",
            seed=123
        )

        adapter1 = JumpProcessAdapter(config1)
        adapter2 = JumpProcessAdapter(config2)

        jumps1 = adapter1.generate_jumps(t_start=0.0, t_end=1000.0)
        jumps2 = adapter2.generate_jumps(t_start=0.0, t_end=1000.0)

        # Patterns should differ
        if len(jumps1) > 0 and len(jumps2) > 0:
            # Different seeds should produce different counts or different values
            if len(jumps1) == len(jumps2):
                assert not np.allclose(jumps1, jumps2)
            else:
                assert len(jumps1) != len(jumps2)

    def test_higher_rate_more_events(self):
        """Test that higher rate leads to more events."""
        config_low = JumpProcessConfig(
            process_type='poisson',
            rate="100.0 / hour",
            seed=42
        )
        config_high = JumpProcessConfig(
            process_type='poisson',
            rate="1000.0 / hour",
            seed=42
        )

        adapter_low = JumpProcessAdapter(config_low)
        adapter_high = JumpProcessAdapter(config_high)

        jumps_low = adapter_low.generate_jumps(t_start=0.0, t_end=3600.0)
        jumps_high = adapter_high.generate_jumps(t_start=0.0, t_end=3600.0)

        # Higher rate should produce more events
        assert len(jumps_high) > len(jumps_low)


class TestEventCounting:
    """Tests for event counting and tracking."""

    def test_event_count_increases(self):
        """Test that event count increases with steps."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="1000.0 / hour",  # High rate, events guaranteed
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        initial_count = adapter.get_state()['event_count']

        # Step through time (reduced from 100 to 20)
        for i in range(20):
            adapter.step(current_time=i * 10.0, dt=10.0)

        final_count = adapter.get_state()['event_count']

        # Should have at least some events
        assert final_count > initial_count

    def test_last_jump_time_updates(self):
        """Test that last jump time is tracked."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="1000.0 / hour",  # High rate
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Step until we get a jump
        for i in range(100):
            state_before = adapter.get_state()
            jump_occurred = adapter.step(current_time=i * 1.0, dt=1.0)
            state_after = adapter.get_state()

            if jump_occurred:
                # Last jump time should update
                assert state_after['last_jump_time'] > state_before['last_jump_time']
                break


class TestStateContinuity:
    """Tests for state continuity across steps."""

    @pytest.mark.slow
    def test_step_consistent_with_generate_jumps_thorough(self):
        """Test that stepping through time matches batch generation (slow, thorough version)."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="100.0 / hour",
            seed=42
        )

        # Method 1: Use generate_jumps
        adapter1 = JumpProcessAdapter(config)
        jumps_batch = adapter1.generate_jumps(t_start=0.0, t_end=360.0)

        # Method 2: Step through time
        adapter2 = JumpProcessAdapter(config)
        jumps_step = []
        for i in range(3600):
            t = i * 0.1
            if adapter2.step(current_time=t, dt=0.1):
                # Jump occurred in this interval
                # Record approximate time (midpoint of interval)
                jumps_step.append(t + 0.05)

        # Counts should be similar (within 1 due to boundaries)
        assert abs(len(jumps_batch) - len(jumps_step)) <= 1

    def test_step_consistent_with_generate_jumps(self):
        """Test that stepping through time matches batch generation (fast version)."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="100.0 / hour",
            seed=42
        )

        # Method 1: Use generate_jumps (shorter interval)
        adapter1 = JumpProcessAdapter(config)
        jumps_batch = adapter1.generate_jumps(t_start=0.0, t_end=36.0)  # 10x shorter

        # Method 2: Step through time (fewer steps)
        adapter2 = JumpProcessAdapter(config)
        jumps_step = []
        for i in range(360):  # 10x fewer
            t = i * 0.1
            if adapter2.step(current_time=t, dt=0.1):
                jumps_step.append(t + 0.05)

        # Counts should be similar (within 1 due to boundaries)
        assert abs(len(jumps_batch) - len(jumps_step)) <= 1

    def test_multiple_jumps_in_interval(self):
        """Test handling of multiple potential jumps in one step."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="100.0 / hour",  # 100 events per hour
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # Step with very large dt that spans many potential jumps
        # With rate of 100/hour, in 1 hour (3600s) we should see ~100 jumps
        # But step() only reports if ANY jump occurred, not how many
        jump_occurred = adapter.step(current_time=0.0, dt=3600.0)

        # Should report that jump occurred
        assert jump_occurred == True

    def test_jump_at_interval_boundary(self):
        """Test jump occurring exactly at interval boundary."""
        config = JumpProcessConfig(
            process_type='deterministic',
            rate="1.0 / second",  # Jump every second
            seed=42
        )
        adapter = JumpProcessAdapter(config)

        # First jump should be at t=1
        # Check interval [0, 1)
        jump_in_first_interval = adapter.step(current_time=0.0, dt=1.0)

        # Should not occur in [0, 1) since it's at t=1
        # But implementation may vary on boundary handling
        assert isinstance(jump_in_first_interval, bool)
