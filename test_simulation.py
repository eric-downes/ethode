"""Tests for high-level Simulation facade.

This module tests the Simulation class and pure functional interfaces
for orchestrating multiple subsystems.
"""

import pytest
import jax
import jax.numpy as jnp

from ethode.simulation import Simulation, simulate_controller_step
from ethode.adapters import ControllerAdapter
from ethode.controller.config import ControllerConfig
from ethode.runtime import ControllerState


class TestSimulation:
    """Tests for Simulation class."""

    def test_init_with_controller(self):
        """Test initialization with ControllerAdapter."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.001 USD", "0.003 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        assert sim.controller is adapter
        assert sim.fee is None
        assert sim.liquidity is None

    def test_init_requires_controller_adapter(self):
        """Test that initialization requires a ControllerAdapter."""
        with pytest.raises(TypeError, match="controller must be a ControllerAdapter"):
            sim = Simulation(controller="not an adapter")

    def test_step_basic(self):
        """Test basic step functionality."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        output = sim.step(error=1.0, dt=0.1)

        assert isinstance(output, float)
        assert output != 0.0  # Should have non-zero response

    def test_step_updates_controller_state(self):
        """Test that step updates controller state."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        initial_integral = float(sim.controller.state.integral)
        sim.step(error=1.0, dt=0.1)

        # State should have changed
        assert float(sim.controller.state.integral) != initial_integral

    def test_step_matches_adapter_directly(self):
        """Test that Simulation.step() matches ControllerAdapter.step()."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )

        # Create two separate adapters with same config
        adapter1 = ControllerAdapter(config)
        adapter2 = ControllerAdapter(config)
        sim = Simulation(controller=adapter1)

        # Run same sequence
        errors = [1.0, 0.5, 0.2, 0.0, -0.1]
        dt = 0.1

        sim_outputs = []
        adapter_outputs = []

        for error in errors:
            sim_outputs.append(sim.step(error, dt))
            adapter_outputs.append(adapter2.step(error, dt))

        # Should produce identical outputs
        for sim_out, adapter_out in zip(sim_outputs, adapter_outputs):
            assert abs(sim_out - adapter_out) < 1e-10

    def test_reset(self):
        """Test reset functionality."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        # Run a step to accumulate state
        sim.step(error=1.0, dt=0.1)
        assert float(sim.controller.state.integral) != 0.0

        # Reset should clear state
        sim.reset()
        assert float(sim.controller.state.integral) == 0.0

    def test_get_state(self):
        """Test get_state returns all subsystem states."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        sim.step(error=1.0, dt=0.1)
        state = sim.get_state()

        assert isinstance(state, dict)
        assert 'controller' in state
        assert isinstance(state['controller'], dict)
        assert 'integral' in state['controller']

    def test_runtime_state_direct_access(self):
        """Test that runtime and state can be accessed for JAX."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        # Should be able to access runtime and state
        runtime = sim.controller.runtime
        state = sim.controller.state

        assert runtime is not None
        assert state is not None


class TestSimulateControllerStep:
    """Tests for pure functional simulate_controller_step."""

    def test_basic_functionality(self):
        """Test basic functionality of simulate_controller_step."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)

        runtime = adapter.runtime
        state = adapter.state
        error = jnp.array(1.0)
        dt = jnp.array(0.1)

        new_state, output = simulate_controller_step(runtime, state, error, dt)

        assert isinstance(new_state, ControllerState)
        assert isinstance(output, jax.Array)
        assert float(output) != 0.0

    def test_matches_controller_step(self):
        """Test that simulate_controller_step matches controller_step."""
        from ethode.controller.kernel import controller_step

        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)

        runtime = adapter.runtime
        state = adapter.state
        error = jnp.array(1.0)
        dt = jnp.array(0.1)

        # Both should produce identical results
        state1, output1 = simulate_controller_step(runtime, state, error, dt)
        state2, output2 = controller_step(runtime, state, error, dt)

        assert float(output1) == float(output2)
        assert float(state1.integral) == float(state2.integral)

    def test_with_jax_scan(self):
        """Test simulate_controller_step with jax.lax.scan."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)

        runtime = adapter.runtime
        initial_state = adapter.state

        # Define scan function
        def step_fn(state, inputs):
            error, dt = inputs
            return simulate_controller_step(runtime, state, error, dt)

        # Run scan
        errors = jnp.array([1.0, 0.5, 0.2, 0.0, -0.1])
        dts = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])

        final_state, outputs = jax.lax.scan(step_fn, initial_state, (errors, dts))

        # Verify results
        assert outputs.shape == (5,)
        assert isinstance(final_state, ControllerState)
        assert float(final_state.integral) != 0.0

    def test_scan_matches_sequential_steps(self):
        """Test that scan produces same results as sequential steps."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )

        # Sequential execution with adapter
        adapter = ControllerAdapter(config)
        seq_outputs = []
        errors = [1.0, 0.5, 0.2, 0.0, -0.1]
        dt = 0.1
        for error in errors:
            seq_outputs.append(adapter.step(error, dt))

        # Scan execution
        adapter2 = ControllerAdapter(config)
        runtime = adapter2.runtime
        initial_state = adapter2.state

        def step_fn(state, inputs):
            error, dt = inputs
            return simulate_controller_step(runtime, state, error, dt)

        errors_jax = jnp.array(errors)
        dts_jax = jnp.array([dt] * len(errors))

        final_state, scan_outputs = jax.lax.scan(step_fn, initial_state, (errors_jax, dts_jax))

        # Compare outputs
        for seq, scan in zip(seq_outputs, scan_outputs):
            assert abs(seq - float(scan)) < 1e-6


class TestSimulationIntegration:
    """Integration tests for Simulation."""

    def test_complete_simulation_workflow(self):
        """Test complete workflow: create, step, reset, step again."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        # First sequence
        outputs1 = []
        for error in [1.0, 0.5, 0.2]:
            outputs1.append(sim.step(error, 0.1))

        # Reset
        sim.reset()

        # Second sequence (should match first)
        outputs2 = []
        for error in [1.0, 0.5, 0.2]:
            outputs2.append(sim.step(error, 0.1))

        # Should be identical
        for out1, out2 in zip(outputs1, outputs2):
            assert abs(out1 - out2) < 1e-10


class TestSimulationScan:
    """Tests for Simulation.scan() method."""

    def test_scan_basic_functionality(self):
        """Test basic scan functionality."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        errors = jnp.array([1.0, 0.5, 0.2, 0.0, -0.1])
        dts = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])

        outputs, final_state = sim.scan(errors, dts)

        # Check output shape
        assert outputs.shape == (5,)
        assert isinstance(final_state, ControllerState)

    def test_scan_matches_sequential_steps(self):
        """Test that scan produces same results as sequential step() calls."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )

        # Sequential execution
        adapter1 = ControllerAdapter(config)
        sim1 = Simulation(controller=adapter1)
        errors_list = [1.0, 0.5, 0.2, 0.0, -0.1]
        dt = 0.1
        seq_outputs = []
        for error in errors_list:
            seq_outputs.append(sim1.step(error, dt))

        # Scan execution
        adapter2 = ControllerAdapter(config)
        sim2 = Simulation(controller=adapter2)
        errors_jax = jnp.array(errors_list)
        dts_jax = jnp.array([dt] * len(errors_list))
        scan_outputs, final_state = sim2.scan(errors_jax, dts_jax)

        # Compare outputs
        for seq, scan in zip(seq_outputs, scan_outputs):
            assert abs(seq - float(scan)) < 1e-6

    def test_scan_updates_internal_state(self):
        """Test that scan updates internal controller state."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        # Initial state should be zero
        assert float(sim.controller.state.integral) == 0.0

        errors = jnp.array([1.0, 0.5, 0.2])
        dts = jnp.array([0.1, 0.1, 0.1])

        outputs, final_state = sim.scan(errors, dts)

        # Internal state should be updated
        assert float(sim.controller.state.integral) != 0.0
        assert float(sim.controller.state.integral) == float(final_state.integral)

    def test_scan_state_continuity(self):
        """Test that multiple scan calls maintain state continuity."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        # First batch
        errors1 = jnp.array([1.0, 0.5])
        dts1 = jnp.array([0.1, 0.1])
        outputs1, state1 = sim.scan(errors1, dts1)

        # Second batch (should start from state1)
        errors2 = jnp.array([0.2, 0.0])
        dts2 = jnp.array([0.1, 0.1])
        outputs2, state2 = sim.scan(errors2, dts2)

        # Compare with single continuous batch
        adapter2 = ControllerAdapter(config)
        sim2 = Simulation(controller=adapter2)
        errors_all = jnp.array([1.0, 0.5, 0.2, 0.0])
        dts_all = jnp.array([0.1, 0.1, 0.1, 0.1])
        outputs_all, state_all = sim2.scan(errors_all, dts_all)

        # Final states should match
        assert abs(float(state2.integral) - float(state_all.integral)) < 1e-6
        assert abs(float(state2.last_error) - float(state_all.last_error)) < 1e-6

    def test_scan_with_jit(self):
        """Test scan performance with JIT compilation."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        # Create JIT-compiled version
        @jax.jit
        def jitted_scan_step(runtime, state, errors, dts):
            def step_fn(s, inputs):
                error, dt = inputs
                from ethode.controller.kernel import controller_step
                return controller_step(runtime, s, error, dt)
            return jax.lax.scan(step_fn, state, (errors, dts))

        errors = jnp.array([1.0, 0.5, 0.2, 0.0, -0.1])
        dts = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])

        # Run jitted version
        jit_final_state, jit_outputs = jitted_scan_step(
            sim.controller.runtime,
            sim.controller.state,
            errors,
            dts
        )

        # Run regular scan
        adapter2 = ControllerAdapter(config)
        sim2 = Simulation(controller=adapter2)
        scan_outputs, scan_final_state = sim2.scan(errors, dts)

        # Results should match
        for jit_out, scan_out in zip(jit_outputs, scan_outputs):
            assert abs(float(jit_out) - float(scan_out)) < 1e-6

    def test_scan_after_reset(self):
        """Test that scan works correctly after reset."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        errors = jnp.array([1.0, 0.5, 0.2])
        dts = jnp.array([0.1, 0.1, 0.1])

        # First run
        outputs1, state1 = sim.scan(errors, dts)

        # Reset
        sim.reset()

        # Second run (should match first)
        outputs2, state2 = sim.scan(errors, dts)

        # Outputs should be identical
        for out1, out2 in zip(outputs1, outputs2):
            assert abs(float(out1) - float(out2)) < 1e-10

    def test_scan_with_varying_dt(self):
        """Test scan with non-uniform time steps."""
        config = ControllerConfig(
            kp="0.2 / day",
            ki="0.02 / day**2",
            kd=0.0,
            tau="7 day",
            noise_band=("0.0 USD", "1e9 USD")
        )
        adapter = ControllerAdapter(config)
        sim = Simulation(controller=adapter)

        errors = jnp.array([1.0, 0.5, 0.2, 0.0])
        dts = jnp.array([0.1, 0.05, 0.2, 0.15])  # Varying time steps

        outputs, final_state = sim.scan(errors, dts)

        # Should work without errors
        assert outputs.shape == (4,)
        assert isinstance(final_state, ControllerState)
