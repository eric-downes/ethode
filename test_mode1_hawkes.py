"""Tests for Mode 1 (Pre-generated Hawkes) jump-diffusion integration.

This module tests the complete integration of Hawkes processes with ODE dynamics
using Mode 1 (pre-generated event times via jax.lax.scan).
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ethode.jumpdiffusion.config import JumpDiffusionConfig
from ethode.hawkes.config import HawkesConfig
from ethode.jumpdiffusion.kernel import simulate


class TestMode1BasicIntegration:
    """Test basic Mode 1 functionality."""

    def test_mode1_initialization(self):
        """Test Mode 1 config creates correct runtime."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: -0.1 * y,
            jump_effect_fn=lambda t, y, p: y + 0.1,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.3,
                excitation_decay="5 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=100,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)

        assert runtime.scheduler.mode == 1
        assert runtime.scheduler.hawkes is not None
        assert runtime.scheduler.scheduled is None
        assert float(runtime.scheduler.hawkes_max_events) == 100

    def test_mode1_auto_hawkes_dt(self):
        """Test that hawkes_dt is auto-computed if not provided."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.5,
                excitation_decay="2 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=100,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        # Should auto-compute hawkes_dt
        assert config.hawkes_dt is not None
        dt_value, dt_units = config.hawkes_dt
        assert dt_value > 0
        assert dt_units == "second"

    def test_mode1_explicit_hawkes_dt(self):
        """Test explicit hawkes_dt override."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.5,
                excitation_decay="2 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_dt="0.01 second",
            hawkes_max_events=100,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        dt_value, dt_spec = config.hawkes_dt
        assert dt_value == 0.01
        assert dt_spec.symbol == "second"


class TestMode1EventGeneration:
    """Test Mode 1 event generation and statistics."""

    def test_mode1_generates_events(self):
        """Test that Mode 1 generates Hawkes events."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y + 0.1,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.3,
                excitation_decay="5 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=500,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)
        times, states = simulate(runtime, jnp.array([1.0]), (0.0, 3600.0), max_steps=5000)

        # Filter valid times
        valid_mask = jnp.isfinite(times)
        valid_times = times[valid_mask]

        # Should have initial + jumps + final
        num_jumps = len(valid_times) - 2  # Exclude initial and final
        assert num_jumps > 0, "Should generate at least some jumps"

    def test_mode1_event_rate(self):
        """Test that Mode 1 produces correct stationary event rate."""
        base_rate = 100.0  # events/hour
        alpha = 0.5
        expected_stationary = base_rate / (1 - alpha)  # 200 events/hour

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate=f"{base_rate} / hour",
                excitation_strength=alpha,
                excitation_decay="2 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=1000,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)
        times, states = simulate(runtime, jnp.array([1.0]), (0.0, 3600.0), max_steps=10000)

        # Count jumps
        valid_mask = jnp.isfinite(times)
        valid_times = times[valid_mask]
        num_jumps = len([t for t in valid_times if 0 < t < 3600.0])

        observed_rate = num_jumps / 1.0  # events per hour

        # Allow 30% tolerance for stochastic process
        assert abs(observed_rate - expected_stationary) / expected_stationary < 0.3

    def test_mode1_clustering_behavior(self):
        """Test that Mode 1 produces event clustering."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.6,  # Strong clustering
                excitation_decay="1 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=500,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)
        times, states = simulate(runtime, jnp.array([1.0]), (0.0, 1800.0), max_steps=5000)

        # Get jump times
        valid_mask = jnp.isfinite(times)
        valid_times = times[valid_mask]
        jump_times = [float(t) for t in valid_times if 0 < t < 1800.0]

        if len(jump_times) > 10:
            iets = np.diff(jump_times)

            # For clustered process, median << mean (right-skewed distribution)
            median_iet = np.median(iets)
            mean_iet = np.mean(iets)

            # Clustering signature: median significantly less than mean
            assert median_iet < mean_iet, "Clustering should produce median < mean"

    def test_mode1_reproducibility(self):
        """Test that same seed produces same results."""
        def run_simulation(seed):
            config = JumpDiffusionConfig(
                initial_state=jnp.array([1.0]),
                dynamics_fn=lambda t, y, p: -0.1 * y,
                jump_effect_fn=lambda t, y, p: y + 0.1,
                jump_process=HawkesConfig(
                    jump_rate="100 / hour",
                    excitation_strength=0.3,
                    excitation_decay="5 minute",
                    seed=seed
                ),
                hawkes_mode='pregen',
                hawkes_max_events=200,
                solver='euler',
                dt_max="0.1 second",
                params={}
            )
            runtime = config.to_runtime(check_units=True)
            return simulate(runtime, jnp.array([1.0]), (0.0, 600.0), max_steps=1000)

        times1, states1 = run_simulation(42)
        times2, states2 = run_simulation(42)

        # Should be identical
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(states1, states2)


class TestMode1ODEIntegration:
    """Test Mode 1 with various ODE dynamics."""

    def test_mode1_exponential_decay(self):
        """Test Mode 1 with exponential decay ODE."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([10.0]),
            dynamics_fn=lambda t, y, p: -p['k'] * y,
            jump_effect_fn=lambda t, y, p: y + p['jump_size'],
            jump_process=HawkesConfig(
                jump_rate="50 / hour",
                excitation_strength=0.2,
                excitation_decay="10 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=100,
            solver='dopri5',
            dt_max="0.1 second",
            params={'k': 0.1, 'jump_size': 1.0}
        )

        runtime = config.to_runtime(check_units=True)
        times, states = simulate(runtime, jnp.array([10.0]), (0.0, 600.0), max_steps=1000)

        # State should decay between jumps and increase at jumps
        valid_mask = jnp.isfinite(times)
        valid_states = states[valid_mask]

        assert len(valid_states) > 2  # At least initial, some jumps, final
        assert valid_states[0, 0] == 10.0  # Initial condition

    def test_mode1_multidimensional_state(self):
        """Test Mode 1 with multidimensional ODE state."""
        # Harmonic oscillator: x'' = -k*x
        def dynamics(t, state, params):
            x, v = state
            return jnp.array([v, -params['k'] * x])

        def jump_effect(t, state, params):
            x, v = state
            return jnp.array([x, v * 0.9])  # Damped collision

        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0, 0.0]),
            dynamics_fn=dynamics,
            jump_effect_fn=jump_effect,
            jump_process=HawkesConfig(
                jump_rate="20 / hour",
                excitation_strength=0.3,
                excitation_decay="5 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=50,
            solver='dopri5',
            dt_max="0.01 second",
            params={'k': 1.0}
        )

        runtime = config.to_runtime(check_units=True)
        times, states = simulate(runtime, jnp.array([1.0, 0.0]), (0.0, 600.0), max_steps=1000)

        valid_mask = jnp.isfinite(times)
        valid_states = states[valid_mask]

        assert valid_states.shape[1] == 2  # 2D state
        np.testing.assert_array_equal(valid_states[0], jnp.array([1.0, 0.0]))


class TestMode1EdgeCases:
    """Test Mode 1 edge cases and error handling."""

    def test_mode1_high_excitation(self):
        """Test Mode 1 with high excitation strength."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.8,  # High but stable
                excitation_decay="1 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=2000,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)
        # Should not crash
        times, states = simulate(runtime, jnp.array([1.0]), (0.0, 600.0), max_steps=10000)

        valid_mask = jnp.isfinite(times)
        assert jnp.sum(valid_mask) > 0

    def test_mode1_unstable_hawkes_rejected(self):
        """Test that unstable Hawkes (alpha >= 1) is rejected."""
        with pytest.raises(ValueError, match="Excitation strength must be < 1"):
            HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=1.0,  # Unstable
                excitation_decay="5 minute",
                seed=42
            )

    def test_mode1_buffer_overflow_protection(self):
        """Test that buffer overflow is handled gracefully."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate="1000 / hour",  # Very high rate
                excitation_strength=0.5,
                excitation_decay="1 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=100,  # Small buffer
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)

        # Should truncate at buffer size, not crash
        times, states = simulate(runtime, jnp.array([1.0]), (0.0, 3600.0), max_steps=10000)

        valid_mask = jnp.isfinite(times)
        assert jnp.sum(valid_mask) > 0


class TestMode1Solvers:
    """Test Mode 1 with different ODE solvers."""

    @pytest.mark.parametrize("solver", ['euler', 'rk4', 'dopri5', 'dopri8'])
    def test_mode1_solvers(self, solver):
        """Test that all solvers work with Mode 1."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: -0.1 * y,
            jump_effect_fn=lambda t, y, p: y + 0.1,
            jump_process=HawkesConfig(
                jump_rate="50 / hour",
                excitation_strength=0.3,
                excitation_decay="5 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=100,
            solver=solver,
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)
        times, states = simulate(runtime, jnp.array([1.0]), (0.0, 600.0), max_steps=1000)

        valid_mask = jnp.isfinite(times)
        assert jnp.sum(valid_mask) > 2  # At least initial + some events


class TestMode1Units:
    """Test Mode 1 unit handling."""

    def test_mode1_unit_validation(self):
        """Test that units are properly validated."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.3,
                excitation_decay="300 second",  # 5 minutes in seconds
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=100,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)

        # Decay should be converted to canonical units (seconds)
        decay_seconds = float(runtime.scheduler.hawkes.excitation_decay.value)
        assert abs(decay_seconds - 300.0) < 1e-6

    def test_mode1_different_time_units(self):
        """Test Mode 1 with different time unit specifications."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: jnp.zeros_like(y),
            jump_effect_fn=lambda t, y, p: y,
            jump_process=HawkesConfig(
                jump_rate="100 / hour",
                excitation_strength=0.3,
                excitation_decay="5 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_dt="10 millisecond",  # Different unit
            hawkes_max_events=100,
            solver='euler',
            dt_max="100 millisecond",
            params={}
        )

        # Should convert to canonical units (seconds)
        dt_value, dt_spec = config.hawkes_dt
        # Note: hawkes_dt is stored in the units provided, not canonical
        # It gets converted to canonical during to_runtime()
        assert abs(dt_value - 0.01) < 1e-9  # 10ms = 0.01s


class TestMode1Performance:
    """Test Mode 1 performance characteristics."""

    def test_mode1_jit_compilation(self):
        """Test that Mode 1 simulation can be JIT compiled."""
        config = JumpDiffusionConfig(
            initial_state=jnp.array([1.0]),
            dynamics_fn=lambda t, y, p: -0.1 * y,
            jump_effect_fn=lambda t, y, p: y + 0.1,
            jump_process=HawkesConfig(
                jump_rate="50 / hour",
                excitation_strength=0.3,
                excitation_decay="5 minute",
                seed=42
            ),
            hawkes_mode='pregen',
            hawkes_max_events=100,
            solver='euler',
            dt_max="0.1 second",
            params={}
        )

        runtime = config.to_runtime(check_units=True)

        # JIT compile
        simulate_jit = jax.jit(
            lambda: simulate(runtime, jnp.array([1.0]), (0.0, 600.0), max_steps=1000)
        )

        # Should not crash
        times, states = simulate_jit()
        assert jnp.sum(jnp.isfinite(times)) > 0
