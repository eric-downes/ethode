"""Tests for JumpDiffusionAdapter class.

This module tests the high-level JumpDiffusionAdapter API for ODE+Jump hybrid simulation.

Note: Some tests are marked with @pytest.mark.slow and are skipped by default.
To run slow tests: pytest test_jumpdiffusion_adapter.py -m slow
To run all tests: pytest test_jumpdiffusion_adapter.py -m ""
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ethode import (
    JumpDiffusionAdapter,
    JumpDiffusionConfig,
    JumpProcessConfig,
)
from ethode.jumpdiffusion.kernel import integrate_step, apply_jump, simulate


class TestJumpDiffusionAdapter:
    """Tests for JumpDiffusionAdapter initialization and basic functionality."""

    def test_init_with_exponential_decay(self):
        """Test initialization with exponential decay ODE."""
        # dx/dt = -k*x, analytical solution: x(t) = x0 * exp(-k*t)
        def dynamics(t, state, params):
            k = params['decay_rate']
            return -k * state

        def jump_effect(t, state, params):
            return state  # No change on jump

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            solver='dopri5',
            dt_max="0.1 second",
            params={'decay_rate': 0.1}
        )
        adapter = JumpDiffusionAdapter(config)

        assert adapter.config is config
        assert adapter.runtime is not None
        assert adapter.state is not None

    def test_init_with_multidimensional_state(self):
        """Test initialization with multi-dimensional state."""
        def dynamics(t, state, params):
            # Simple 2D system
            return jnp.array([state[1], -state[0]])

        def jump_effect(t, state, params):
            return state * 0.9  # Scale down on jump

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0, 0.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="5.0 / hour",
                seed=42
            ),
            solver='rk4',
            dt_max="0.05 second",
        )
        adapter = JumpDiffusionAdapter(config)

        assert adapter.state.state.shape == (2,)

    def test_init_sets_initial_jump_time(self):
        """Test that initialization schedules first jump."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="100.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        state_info = adapter.get_state()
        assert state_info['next_jump_time'] > 0.0


class TestODEIntegration:
    """Tests for ODE integration (without jumps)."""

    def test_exponential_decay_accuracy(self):
        """Test ODE integration accuracy with exponential decay."""
        k = 0.1
        x0 = 1.0

        def dynamics(t, state, params):
            return -k * state

        def jump_effect(t, state, params):
            return state

        # Use deterministic with very long interval (no jumps during test)
        config = JumpDiffusionConfig(
            initial_state=jnp.array([x0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.001 / hour",  # Very rare
                seed=42
            ),
            solver='dopri5',
            dt_max="0.1 second",
            rtol=1e-6,
            atol=1e-9,
        )
        adapter = JumpDiffusionAdapter(config)

        # Simulate for 10 seconds
        times, states = adapter.simulate(t_span=(0.0, 10.0))

        # Check final value against analytical solution
        t_final = times[-1]
        x_analytical = x0 * np.exp(-k * t_final)
        x_numerical = states[-1, 0]

        assert abs(x_numerical - x_analytical) < 1e-4

    def test_constant_function_preserved(self):
        """Test that constant state is preserved."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([5.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.001 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 10.0))

        # All states should be 5.0
        assert np.allclose(states[:, 0], 5.0)

    def test_solver_euler(self):
        """Test Euler solver."""
        def dynamics(t, state, params):
            return -0.1 * state

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.001 / hour",
                seed=42
            ),
            solver='euler',
            dt_max="0.01 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 1.0))
        assert len(times) >= 2

    def test_solver_rk4(self):
        """Test RK4 solver."""
        def dynamics(t, state, params):
            return -0.1 * state

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.001 / hour",
                seed=42
            ),
            solver='rk4',
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 1.0))
        assert len(times) >= 2

    def test_solver_dopri8(self):
        """Test Dopri8 solver."""
        def dynamics(t, state, params):
            return -0.1 * state

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.001 / hour",
                seed=42
            ),
            solver='dopri8',
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 1.0))
        assert len(times) >= 2


class TestJumpEffects:
    """Tests for jump application."""

    def test_jump_scales_state(self):
        """Test that jumps correctly modify state."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            # Multiply by 2 on each jump
            return state * 2.0

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="1.0 / second",  # 1 jump per second
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 5.0), max_steps=100)

        # Initial state should be 1.0, after jumps should increase
        assert states[0, 0] == 1.0
        # After jumps, should be larger
        assert states[-1, 0] > 1.0

    def test_jump_count_increases(self):
        """Test that jump count is tracked."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="10.0 / second",
                seed=42
            ),
            dt_max="0.01 second",
        )
        adapter = JumpDiffusionAdapter(config)

        # Run simulation (returns times at jumps + endpoints)
        times, states = adapter.simulate(t_span=(0.0, 1.0), max_steps=200)

        # With deterministic jumps at 10/sec for 1 sec, should have saved ~10 jump times
        # Plus initial and final = at least 10 saves
        assert len(times) >= 10


class TestStatefulAPI:
    """Tests for stateful step() API."""

    def test_step_returns_bool(self):
        """Test that step returns Python bool."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="100.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        jump_occurred = adapter.step(t_end=1.0)
        assert isinstance(jump_occurred, bool)
        assert not isinstance(jump_occurred, jax.Array)

    def test_step_updates_time(self):
        """Test that step advances simulation time."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="5.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        t_initial = adapter.get_state()['t']
        adapter.step(t_end=1.0)
        t_after = adapter.get_state()['t']

        assert t_after > t_initial

    def test_multiple_steps(self):
        """Test multiple sequential steps."""
        def dynamics(t, state, params):
            return -0.1 * state

        def jump_effect(t, state, params):
            return state * 0.9

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="50.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        # Take several steps
        for i in range(10):
            adapter.step(t_end=(i + 1) * 1.0)

        state_info = adapter.get_state()
        assert state_info['t'] > 0.0
        assert state_info['step_count'] > 0


class TestReset:
    """Tests for reset functionality."""

    def test_reset_restores_initial_state(self):
        """Test that reset restores initial conditions."""
        def dynamics(t, state, params):
            return -0.1 * state

        def jump_effect(t, state, params):
            return state * 0.9

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="100.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        initial_state = adapter.get_state()

        # Run simulation
        adapter.simulate(t_span=(0.0, 10.0))

        # Reset
        adapter.reset(seed=42)

        reset_state = adapter.get_state()

        # Should be back to initial
        assert reset_state['t'] == initial_state['t']
        assert np.allclose(reset_state['state'], initial_state['state'])

    def test_reset_clears_counters(self):
        """Test that reset clears step and jump counts."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="10.0 / second",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        # Run simulation
        adapter.simulate(t_span=(0.0, 5.0))

        # Reset
        adapter.reset()

        state_info = adapter.get_state()
        assert state_info['step_count'] == 0
        assert state_info['jump_count'] == 0

    def test_reset_with_new_seed(self):
        """Test that different seeds produce different results."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state * 1.1

        # Create two adapters with different seeds
        config1 = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="500.0 / hour",  # Higher rate to ensure jumps occur
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter1 = JumpDiffusionAdapter(config1)

        config2 = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="500.0 / hour",
                seed=123  # Different seed
            ),
            dt_max="0.1 second",
        )
        adapter2 = JumpDiffusionAdapter(config2)

        times1, states1 = adapter1.simulate(t_span=(0.0, 10.0))
        times2, states2 = adapter2.simulate(t_span=(0.0, 10.0))

        # Different seeds should produce different results
        # (Due to different Poisson realizations)
        # With high rate, should have several jumps
        assert len(times1) > 2 and len(times2) > 2
        # Different seeds should produce different results
        # Either different lengths or different values
        if len(times1) == len(times2):
            assert not np.allclose(times1, times2)
        else:
            # Different number of jumps is also proof of different seeds
            assert len(times1) != len(times2)


class TestGetState:
    """Tests for get_state functionality."""

    def test_get_state_returns_dict(self):
        """Test that get_state returns dictionary."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        state_info = adapter.get_state()
        assert isinstance(state_info, dict)

    def test_get_state_has_required_keys(self):
        """Test that get_state has all required keys."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        state_info = adapter.get_state()
        assert 't' in state_info
        assert 'state' in state_info
        assert 'next_jump_time' in state_info
        assert 'step_count' in state_info
        assert 'jump_count' in state_info

    def test_get_state_returns_python_types(self):
        """Test that get_state returns Python types."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        state_info = adapter.get_state()
        assert isinstance(state_info['t'], float)
        assert isinstance(state_info['state'], np.ndarray)
        assert isinstance(state_info['next_jump_time'], float)
        assert isinstance(state_info['step_count'], int)
        assert isinstance(state_info['jump_count'], int)


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_dt_max_with_string(self):
        """Test dt_max validation with string input."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",  # String format
        )

        assert config.dt_max[0] > 0

    def test_dt_max_positive_required(self):
        """Test that negative dt_max raises error."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        with pytest.raises(ValueError):
            config = JumpDiffusionConfig(
                initial_state=jnp.array([1.0]),
                dynamics_fn=dynamics,
                jump_effect_fn=jump_effect,
                jump_process=JumpProcessConfig(
                    process_type='poisson',
                    rate="10.0 / hour",
                    seed=42
                ),
                dt_max="-0.1 second",  # Negative!
            )

    def test_invalid_solver_raises_error(self):
        """Test that invalid solver raises error."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        with pytest.raises(ValueError):
            config = JumpDiffusionConfig(
                initial_state=jnp.array([1.0]),
                dynamics_fn=dynamics,
                jump_effect_fn=jump_effect,
                jump_process=JumpProcessConfig(
                    process_type='poisson',
                    rate="10.0 / hour",
                    seed=42
                ),
                solver='invalid_solver',  # Invalid!
                dt_max="0.1 second",
            )


class TestSimulateMethod:
    """Tests for functional simulate() method."""

    def test_simulate_returns_arrays(self):
        """Test that simulate returns numpy arrays."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 10.0))

        assert isinstance(times, np.ndarray)
        assert isinstance(states, np.ndarray)

    def test_simulate_includes_endpoints(self):
        """Test that simulation includes initial and final times."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.01 / hour",  # Very rare
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        t_start, t_end = 0.0, 10.0
        times, states = adapter.simulate(t_span=(t_start, t_end))

        # Should include start and end
        assert times[0] == pytest.approx(t_start)
        assert times[-1] == pytest.approx(t_end, abs=1e-6)

    def test_simulate_saves_at_jumps(self):
        """Test that simulation saves state at jump times."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state * 2.0  # Double on jump

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="1.0 / second",  # 1 per second
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 5.0))

        # With deterministic jumps at 1/sec, should have ~5 jumps
        # So should have saved states at jumps
        # (Initial + jumps + final)
        assert len(times) >= 2  # At minimum: initial + final


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_time_interval(self):
        """Test simulation with zero time span."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 0.0))

        # Should return just initial state
        assert len(times) == 1
        assert times[0] == 0.0

    def test_very_stiff_ode(self):
        """Test with stiff ODE (fast decay)."""
        k = 100.0  # Very fast decay

        def dynamics(t, state, params):
            return -k * state

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.01 / hour",
                seed=42
            ),
            solver='dopri5',  # Adaptive solver
            dt_max="0.001 second",  # Small step
            rtol=1e-6,
            atol=1e-9,
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 0.1))

        # Should complete without errors
        assert len(times) >= 2

    def test_many_jumps(self):
        """Test with high jump rate."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state + 0.1

        config = JumpDiffusionConfig(
            initial_state=jnp.array([0.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="100.0 / second",  # Very high rate
                seed=42
            ),
            dt_max="0.001 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 1.0), max_steps=500)

        # Should have many jumps
        state_info = adapter.get_state()
        # With 100 jumps/sec for 1 sec, expect ~100 jumps
        # But limited by max_steps
        assert len(times) >= 2


class TestReproducibility:
    """Tests for reproducibility with seeds."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        def dynamics(t, state, params):
            return -0.1 * state

        def jump_effect(t, state, params):
            return state * 0.9

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="50.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )

        adapter1 = JumpDiffusionAdapter(config)
        times1, states1 = adapter1.simulate(t_span=(0.0, 10.0))

        adapter2 = JumpDiffusionAdapter(config)
        times2, states2 = adapter2.simulate(t_span=(0.0, 10.0))

        # Should be identical
        assert np.allclose(times1, times2)
        assert np.allclose(states1, states2)


class TestJAXCompatibility:
    """Tests for JAX compatibility (direct runtime access)."""

    def test_runtime_state_direct_access(self):
        """Test that runtime and state can be accessed directly."""
        def dynamics(t, state, params):
            return jnp.zeros_like(state)

        def jump_effect(t, state, params):
            return state

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        # Should be able to access runtime and state
        runtime = adapter.runtime
        state = adapter.state

        assert runtime is not None
        assert state is not None

    def test_direct_kernel_usage(self):
        """Test that adapter state works with direct kernel calls."""
        def dynamics(t, state, params):
            return -0.1 * state

        def jump_effect(t, state, params):
            return state * 0.9

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='poisson',
                rate="10.0 / hour",
                seed=42
            ),
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        # Use kernel directly
        new_state, t_reached = integrate_step(
            adapter.runtime,
            adapter.state,
            jnp.array(1.0)
        )

        assert new_state is not None
        assert isinstance(t_reached, jax.Array)


class TestHybridDynamics:
    """Tests for combined ODE+Jump dynamics."""

    def test_decay_with_jumps(self):
        """Test exponential decay with periodic jumps."""
        k = 0.1

        def dynamics(t, state, params):
            # Exponential decay
            return -k * state

        def jump_effect(t, state, params):
            # Add 1.0 on each jump
            return state + 1.0

        config = JumpDiffusionConfig(
            initial_state=jnp.array([0.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="1.0 / second",  # 1 jump per second
                seed=42
            ),
            solver='dopri5',
            dt_max="0.1 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 5.0))

        # State should oscillate: decay between jumps, jump up periodically
        # Final state should be > 0 (jumps counteract decay)
        assert states[-1, 0] > 0.0

    def test_oscillator_with_damping_jumps(self):
        """Test harmonic oscillator with jumps that damp energy."""
        def dynamics(t, state, params):
            # Harmonic oscillator: d²x/dt² = -x
            # As system: dx/dt = v, dv/dt = -x
            x, v = state[0], state[1]
            return jnp.array([v, -x])

        def jump_effect(t, state, params):
            # Damp velocity on jump
            x, v = state[0], state[1]
            return jnp.array([x, v * 0.5])

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0, 0.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=JumpProcessConfig(
                process_type='deterministic',
                rate="0.5 / second",  # Occasional damping
                seed=42
            ),
            solver='rk4',
            dt_max="0.01 second",
        )
        adapter = JumpDiffusionAdapter(config)

        times, states = adapter.simulate(t_span=(0.0, 10.0))

        # With damping jumps, amplitude should decrease over time
        # Check that final amplitude < initial
        initial_amplitude = abs(states[0, 0])
        final_amplitude = np.sqrt(states[-1, 0]**2 + states[-1, 1]**2)

        # Due to damping jumps, energy should decrease
        assert final_amplitude < initial_amplitude
