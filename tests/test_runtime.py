"""Tests for runtime Penzai structures."""

import pytest
import dataclasses
import jax
import jax.numpy as jnp
import penzai as pz
from penzai.core import struct

from ethode.units import UnitSpec
from ethode.runtime import (
    QuantityNode,
    ControllerRuntime,
    ControllerState,
    SimulationOutput,
    MarketRuntime,
    SystemRuntime,
    tree_info,
)


class TestQuantityNode:
    """Test QuantityNode struct."""

    def test_creation(self):
        """Test basic QuantityNode creation."""
        spec = UnitSpec(dimension="time", symbol="s", to_canonical=1.0)
        node = QuantityNode(
            value=jnp.array(5.0),
            units=spec
        )
        assert node.value == 5.0
        assert node.units.dimension == "time"

    def test_from_float(self):
        """Test creating QuantityNode from float."""
        spec = UnitSpec(dimension="length", symbol="m")
        node = QuantityNode.from_float(3.14, spec)
        assert node.to_float() == pytest.approx(3.14)
        assert node.value.dtype == jnp.float32

    def test_pytree_behavior(self):
        """Test that QuantityNode works as a pytree."""
        spec = UnitSpec(dimension="time", symbol="s")
        node = QuantityNode.from_float(10.0, spec)

        # Flatten and unflatten
        leaves, treedef = jax.tree_util.tree_flatten(node)
        assert len(leaves) == 1  # Only value is a leaf
        assert leaves[0] == 10.0

        # Reconstruct
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed.value == node.value
        assert reconstructed.units == node.units  # Metadata preserved

    def test_jax_transformations(self):
        """Test QuantityNode with JAX transformations."""
        spec = UnitSpec(dimension="price", symbol="USD")
        node = QuantityNode.from_float(100.0, spec)

        # JIT compilation
        @jax.jit
        def double_value(q: QuantityNode) -> QuantityNode:
            return dataclasses.replace(q, value=q.value * 2)

        doubled = double_value(node)
        assert doubled.to_float() == 200.0
        assert doubled.units == spec  # Units preserved

        # Gradient
        def square(q: QuantityNode) -> jax.Array:
            return q.value ** 2

        grad_fn = jax.grad(square)
        grad = grad_fn(node)
        assert grad.value == 200.0  # d/dx(x^2) = 2x at x=100


class TestControllerRuntime:
    """Test ControllerRuntime struct."""

    def test_creation(self):
        """Test creating ControllerRuntime."""
        time_spec = UnitSpec(dimension="time", symbol="s")
        rate_spec = UnitSpec(dimension="1/time", symbol="1/s")
        price_spec = UnitSpec(dimension="price", symbol="USD")

        runtime = ControllerRuntime(
            kp=QuantityNode.from_float(0.2, rate_spec),
            ki=QuantityNode.from_float(0.02, rate_spec),
            kd=QuantityNode.from_float(0.0, time_spec),
            tau=QuantityNode.from_float(86400.0, time_spec),
            noise_band_low=QuantityNode.from_float(0.001, price_spec),
            noise_band_high=QuantityNode.from_float(0.003, price_spec),
        )

        assert runtime.get_kp() == 0.2
        assert runtime.get_ki() == 0.02
        assert runtime.get_tau() == 86400.0

        low, high = runtime.get_noise_band()
        assert low == 0.001
        assert high == 0.003

    def test_optional_fields(self):
        """Test optional fields in ControllerRuntime."""
        spec = UnitSpec(dimension="dimensionless", symbol="")
        runtime = ControllerRuntime(
            kp=QuantityNode.from_float(1.0, spec),
            ki=QuantityNode.from_float(0.0, spec),
            kd=QuantityNode.from_float(0.0, spec),
            tau=QuantityNode.from_float(1.0, spec),
            noise_band_low=QuantityNode.from_float(0.0, spec),
            noise_band_high=QuantityNode.from_float(1.0, spec),
            output_min=QuantityNode.from_float(-10.0, spec),
            output_max=QuantityNode.from_float(10.0, spec),
        )

        assert runtime.output_min is not None
        assert runtime.output_min.to_float() == -10.0
        assert runtime.output_max.to_float() == 10.0
        assert runtime.rate_limit is None

    def test_pytree_structure(self):
        """Test ControllerRuntime as pytree."""
        spec = UnitSpec(dimension="dimensionless", symbol="")
        runtime = ControllerRuntime(
            kp=QuantityNode.from_float(1.0, spec),
            ki=QuantityNode.from_float(0.5, spec),
            kd=QuantityNode.from_float(0.1, spec),
            tau=QuantityNode.from_float(10.0, spec),
            noise_band_low=QuantityNode.from_float(0.0, spec),
            noise_band_high=QuantityNode.from_float(1.0, spec),
        )

        # Check tree structure
        leaves, treedef = jax.tree_util.tree_flatten(runtime)
        # Should have 6 leaves (the values from each QuantityNode)
        assert len(leaves) == 6

        # Tree map example
        def scale_values(x):
            return x * 2

        scaled = jax.tree.map(scale_values, runtime)
        assert scaled.kp.value == 2.0
        assert scaled.ki.value == 1.0
        # Units should be preserved
        assert scaled.kp.units == spec


class TestControllerState:
    """Test ControllerState struct."""

    def test_zero_initialization(self):
        """Test zero initialization of state."""
        state = ControllerState.zero()
        assert state.integral == 0.0
        assert state.last_error == 0.0
        assert state.last_output == 0.0
        assert state.time == 0.0

    def test_reset_integral(self):
        """Test resetting integral."""
        state = ControllerState(
            integral=jnp.array(100.0),
            last_error=jnp.array(5.0),
            last_output=jnp.array(10.0),
            time=jnp.array(50.0)
        )

        reset_state = state.reset_integral()
        assert reset_state.integral == 0.0
        assert reset_state.last_error == 5.0  # Unchanged
        assert reset_state.last_output == 10.0  # Unchanged
        assert reset_state.time == 50.0  # Unchanged

    def test_jit_compatibility(self):
        """Test that state works with JIT."""
        @jax.jit
        def update_state(state: ControllerState, error: jax.Array) -> ControllerState:
            new_integral = state.integral + error
            return dataclasses.replace(state,
                integral=new_integral,
                last_error=error
            )

        state = ControllerState.zero()
        updated = update_state(state, jnp.array(2.5))
        assert updated.integral == 2.5
        assert updated.last_error == 2.5


class TestSimulationOutput:
    """Test SimulationOutput struct."""

    def test_creation_with_metrics(self):
        """Test creating output with metrics."""
        state = ControllerState.zero()
        output = SimulationOutput(
            control=jnp.array(0.5),
            state=state,
            error=jnp.array(0.1),
            metrics={"proportional": 0.2, "integral": 0.3}
        )

        assert output.control == 0.5
        assert output.error == 0.1
        assert output.metrics["proportional"] == 0.2

    def test_pytree_with_metrics(self):
        """Test that metrics (non-pytree) are handled correctly."""
        state = ControllerState.zero()
        output = SimulationOutput(
            control=jnp.array(1.0),
            state=state,
            error=jnp.array(0.0),
            metrics={"test": "value"}
        )

        leaves, treedef = jax.tree_util.tree_flatten(output)
        # Metrics should not be in leaves (it's metadata)
        # Should have: control + state fields (4) + error = 6 leaves
        assert len(leaves) == 6


class TestSystemRuntime:
    """Test SystemRuntime struct."""

    def test_complete_system(self):
        """Test creating complete system runtime."""
        spec = UnitSpec(dimension="dimensionless", symbol="")
        price_spec = UnitSpec(dimension="price", symbol="USD")

        controller = ControllerRuntime(
            kp=QuantityNode.from_float(1.0, spec),
            ki=QuantityNode.from_float(0.0, spec),
            kd=QuantityNode.from_float(0.0, spec),
            tau=QuantityNode.from_float(1.0, spec),
            noise_band_low=QuantityNode.from_float(0.0, price_spec),
            noise_band_high=QuantityNode.from_float(1.0, price_spec),
        )

        market = MarketRuntime(
            liquidity=QuantityNode.from_float(1e6, price_spec),
            volatility=QuantityNode.from_float(0.02, spec),
            base_fee=QuantityNode.from_float(0.005, spec),
        )

        system = SystemRuntime(
            controller=controller,
            market=market
        )

        assert system.controller.kp.value == 1.0
        assert system.market.liquidity.value == 1e6

    def test_tree_info_utility(self):
        """Test the tree_info utility function."""
        spec = UnitSpec(dimension="dimensionless", symbol="")
        node = QuantityNode.from_float(5.0, spec)

        info = tree_info(node)
        assert "1 leaves" in info
        assert "Structure" in info


class TestJAXIntegration:
    """Test integration with JAX features."""

    def test_vmap_over_parameters(self):
        """Test vmapping over different parameter values."""
        spec = UnitSpec(dimension="dimensionless", symbol="")

        def compute_output(kp_value: float) -> float:
            node = QuantityNode.from_float(kp_value, spec)
            return node.value * 2

        vmapped = jax.vmap(compute_output)
        kp_values = jnp.array([0.1, 0.2, 0.3, 0.4])
        outputs = vmapped(kp_values)

        assert outputs.shape == (4,)
        assert jnp.allclose(outputs, kp_values * 2)

    def test_nested_structs_with_jit(self):
        """Test nested struct compilation."""
        spec = UnitSpec(dimension="dimensionless", symbol="")

        @jax.jit
        def process_system(sys: SystemRuntime) -> jax.Array:
            if sys.market is not None:
                return sys.controller.kp.value + sys.market.volatility.value
            return sys.controller.kp.value

        controller = ControllerRuntime(
            kp=QuantityNode.from_float(1.0, spec),
            ki=QuantityNode.from_float(0.0, spec),
            kd=QuantityNode.from_float(0.0, spec),
            tau=QuantityNode.from_float(1.0, spec),
            noise_band_low=QuantityNode.from_float(0.0, spec),
            noise_band_high=QuantityNode.from_float(1.0, spec),
        )

        market = MarketRuntime(
            liquidity=QuantityNode.from_float(1e6, spec),
            volatility=QuantityNode.from_float(0.5, spec),
            base_fee=QuantityNode.from_float(0.01, spec),
        )

        system = SystemRuntime(controller=controller, market=market)
        result = process_system(system)
        assert result == 1.5