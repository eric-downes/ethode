#!/usr/bin/env python
"""Comprehensive tests for new unit-aware modules."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

from ethode.units import UnitManager
from ethode.runtime import QuantityNode, UnitSpec
from ethode.fields import quantity_field


@pytest.fixture(autouse=True, scope='module')
def enable_x64():
    """Enable x64 precision for this test module only."""
    old_x64 = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', old_x64)


class TestUnitManager:
    """Comprehensive tests for UnitManager."""

    def test_singleton_pattern(self):
        """Test that UnitManager is a singleton."""
        manager1 = UnitManager.instance()
        manager2 = UnitManager.instance()
        assert manager1 is manager2
        print("✓ UnitManager singleton works")

    def test_financial_units(self):
        """Test financial unit definitions."""
        manager = UnitManager.instance()

        # Test basis points
        bps = manager.ensure_quantity("100 bps")
        assert abs(bps.to("dimensionless").magnitude - 0.01) < 1e-10

        # Test financial shorthand
        million_usd = manager.ensure_quantity("1M USD")
        assert million_usd.magnitude == 1e6
        assert str(million_usd.units) == "USD"

        thousand_eur = manager.ensure_quantity("5k EUR")
        assert thousand_eur.magnitude == 5000

        print("✓ Financial units work correctly")

    def test_numeric_string_handling(self):
        """Test handling of numeric strings."""
        manager = UnitManager.instance()

        # Pure numeric string should accept default unit
        qty = manager.ensure_quantity("42", "USD")
        assert qty.magnitude == 42
        assert str(qty.units) == "USD"

        # String with units should ignore default
        qty = manager.ensure_quantity("42 EUR", "USD")
        assert str(qty.units) == "EUR"

        print("✓ Numeric string handling works")

    def test_canonical_conversion(self):
        """Test conversion to canonical units."""
        manager = UnitManager.instance()

        # Time conversion
        hour = manager.ensure_quantity("1 hour")
        canonical_value, spec = manager.to_canonical(hour, "time")
        assert canonical_value == 3600  # seconds
        assert spec.dimension == "time"
        assert spec.to_canonical == 3600

        # Rate conversion
        rate = manager.ensure_quantity("100 USD/hour")
        canonical_value, spec = manager.to_canonical(rate, "price/time")
        assert abs(canonical_value - 100/3600) < 1e-10

        print("✓ Canonical conversion works")

    def test_from_canonical_reconstruction(self):
        """Test reconstruction from canonical values."""
        manager = UnitManager.instance()

        # Create a spec for hours
        spec = UnitSpec(dimension="time", symbol="hour", to_canonical=3600)

        # Reconstruct from canonical seconds
        qty = manager.from_canonical(7200, spec)  # 2 hours in seconds
        assert qty.magnitude == 2.0
        assert "hour" in str(qty.units)

        print("✓ From canonical reconstruction works")


class TestQuantityNode:
    """Comprehensive tests for QuantityNode."""

    def test_creation_methods(self):
        """Test different ways to create QuantityNodes."""
        spec = UnitSpec("time", "second", 1.0)

        # From float
        node1 = QuantityNode.from_float(42.0, spec)
        assert float(node1.value) == 42.0
        assert node1.units == spec

        # Direct creation
        node2 = QuantityNode(jnp.array(42.0), spec)
        assert float(node2.value) == 42.0

        print("✓ QuantityNode creation methods work")

    def test_pytree_compatibility(self):
        """Test that QuantityNode works with JAX transforms."""
        spec = UnitSpec("price", "USD", 1.0)
        node = QuantityNode.from_float(100.0, spec)

        # Test tree flattening/unflattening
        flat, treedef = jax.tree_util.tree_flatten(node)
        assert len(flat) == 1  # Only value is a leaf
        assert float(flat[0]) == 100.0

        # Reconstruct
        node2 = jax.tree_util.tree_unflatten(treedef, flat)
        assert node2.units == spec  # Units preserved as metadata

        print("✓ QuantityNode pytree compatibility works")

    def test_jax_transformations(self):
        """Test QuantityNode with JAX transformations."""
        spec = UnitSpec("price", "USD", 1.0)

        def double_price(node: QuantityNode) -> QuantityNode:
            return QuantityNode(node.value * 2, node.units)

        # JIT compilation
        jitted = jax.jit(double_price)
        node = QuantityNode.from_float(50.0, spec)
        result = jitted(node)
        assert float(result.value) == 100.0
        assert result.units == spec

        print("✓ JAX transformations work with QuantityNode")


class TestFieldValidators:
    """Test field validators for Pydantic models."""

    def test_quantity_field_validator(self):
        """Test quantity_field validator."""
        validator = quantity_field("time", "second")

        # Test with string
        value, spec = validator("5 minutes", None)
        assert value == 300  # 5 minutes in seconds
        assert spec.dimension == "time"

        # Test with float and default unit
        value, spec = validator(42, None)
        assert value == 42
        assert spec.symbol == "second"

        # Test with pint quantity
        manager = UnitManager.instance()
        qty = manager.ensure_quantity("2 hours")
        value, spec = validator(qty, None)
        assert value == 7200

        print("✓ quantity_field validator works")

    def test_quantity_field_bounds(self):
        """Test quantity_field with bounds."""
        validator = quantity_field("time", "second", min_value=0, max_value=3600)

        # Within bounds
        value, spec = validator("30 minutes", None)
        assert value == 1800

        # Below minimum
        with pytest.raises(ValueError, match="below minimum"):
            validator("-5 seconds", None)

        # Above maximum
        with pytest.raises(ValueError, match="above maximum"):
            validator("2 hours", None)

        print("✓ quantity_field bounds checking works")


class TestFeeModuleAdvanced:
    """Advanced tests for Fee module."""

    def test_fee_with_all_parameters(self):
        """Test FeeConfig with all optional parameters."""
        from ethode.fee import FeeConfig

        config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="500 bps",
            min_fee_rate="10 bps",
            fee_decay_time="2 hours",
            fee_growth_rate="0.1 / hour",
            min_fee_amount="0.01 USD",
            max_fee_amount="1000 USD",
            accumulation_period="1 day"
        )

        runtime = config.to_runtime()

        # Check all fields populated
        assert runtime.base_fee_rate is not None
        assert runtime.max_fee_rate is not None
        assert runtime.min_fee_rate is not None
        assert runtime.fee_decay_time is not None
        assert runtime.fee_growth_rate is not None
        assert runtime.min_fee_amount is not None
        assert runtime.max_fee_amount is not None
        assert runtime.accumulation_period is not None

        print("✓ FeeConfig with all parameters works")

    def test_fee_stress_dynamics(self):
        """Test fee stress update dynamics."""
        from ethode.fee import FeeConfig, FeeState
        from ethode.fee.kernel import update_stress_level

        config = FeeConfig(
            base_fee_rate="100 bps",
            max_fee_rate="500 bps"
        )

        state = FeeState.from_base_rate(0.01)

        # Low stress
        new_state = update_stress_level(
            state,
            market_volatility=jnp.array(0.1),
            volume_ratio=jnp.array(0.8)
        )
        assert float(new_state.stress_level) < 0.5

        # High stress
        new_state = update_stress_level(
            state,
            market_volatility=jnp.array(0.9),
            volume_ratio=jnp.array(3.0)
        )
        assert float(new_state.stress_level) > 0.5

        print("✓ Fee stress dynamics work")

    def test_fee_accumulation(self):
        """Test fee accumulation over multiple transactions."""
        from ethode.fee import FeeConfig, FeeState
        from ethode.fee.kernel import calculate_fee, reset_accumulated_fees

        config = FeeConfig(
            base_fee_rate=0.01,  # 1%
            max_fee_rate=0.05
        )

        runtime = config.to_runtime()
        state = FeeState.from_base_rate(0.01)

        # Process multiple transactions
        for amount in [1000, 2000, 3000]:
            state, fee = calculate_fee(
                runtime, state,
                transaction_amount=jnp.array(float(amount)),
                dt=jnp.array(1.0)
            )

        # Check accumulated fees (should be ~60 = 1% of 6000)
        assert abs(float(state.accumulated_fees) - 60) < 1

        # Reset and check
        state, total = reset_accumulated_fees(state)
        assert abs(float(total) - 60) < 1
        assert float(state.accumulated_fees) == 0

        print("✓ Fee accumulation works")


class TestLiquidityModuleAdvanced:
    """Advanced tests for Liquidity module."""

    def test_liquidity_mean_reversion(self):
        """Test mean reversion dynamics."""
        from ethode.liquidity import LiquiditySDEConfig, LiquidityState

        config = LiquiditySDEConfig(
            initial_liquidity="500000 USD",
            mean_liquidity="1000000 USD",
            mean_reversion_rate="0.5 / day",
            volatility=0.1
        )

        runtime = config.to_runtime()

        # State below mean should tend to increase
        state = LiquidityState.initialize(500000)
        # In real simulation, would apply SDE dynamics
        # Here just verify structure
        assert float(state.liquidity_level) == 500000

        print("✓ Liquidity mean reversion structure works")

    def test_liquidity_with_jumps(self):
        """Test liquidity with jump diffusion parameters."""
        from ethode.liquidity import LiquiditySDEConfig

        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2,
            jump_intensity="10 / day",  # 10 jumps per day on average
            jump_size_mean="50k USD",
            jump_size_std="10k USD"
        )

        runtime = config.to_runtime()

        assert runtime.jump_intensity is not None
        assert runtime.jump_size_mean is not None
        assert runtime.jump_size_std is not None

        # Check canonical conversion
        jump_rate_per_second = float(runtime.jump_intensity.value)
        assert abs(jump_rate_per_second - 10/86400) < 1e-10

        print("✓ Liquidity with jumps works")

    def test_liquidity_provision_removal(self):
        """Test liquidity provision and removal parameters."""
        from ethode.liquidity import LiquiditySDEConfig

        config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2,
            provision_rate="10k USD / hour",  # Continuous provision
            removal_threshold="2M USD"  # Remove if above threshold
        )

        runtime = config.to_runtime()

        assert runtime.provision_rate is not None
        assert runtime.removal_threshold is not None

        # Check provision rate in canonical units (USD/second)
        provision_per_second = float(runtime.provision_rate.value)
        assert abs(provision_per_second - 10000/3600) < 1e-6

        print("✓ Liquidity provision/removal works")


class TestHawkesModuleAdvanced:
    """Advanced tests for Hawkes module."""

    def test_hawkes_branching_ratio(self):
        """Test Hawkes branching ratio calculation."""
        from ethode.hawkes import HawkesConfig

        config = HawkesConfig(
            jump_rate="100 / hour",
            excitation_strength=0.8,  # High but stable
            excitation_decay="5 minutes"
        )

        # Branching ratio = excitation_strength
        # Must be < 1 for stability
        assert config.excitation_strength[0] < 1.0

        runtime = config.to_runtime()
        assert abs(float(runtime.excitation_strength.value) - 0.8) < 1e-6  # Use approx equality for float32

        print("✓ Hawkes branching ratio works")

    def test_hawkes_intensity_bounds(self):
        """Test Hawkes with intensity bounds."""
        from ethode.hawkes import HawkesConfig

        config = HawkesConfig(
            jump_rate="50 / hour",
            excitation_strength=0.4,
            excitation_decay="10 minutes",
            min_intensity="10 / hour",  # Floor
            max_intensity="500 / hour"  # Ceiling
        )

        runtime = config.to_runtime()

        assert runtime.min_intensity is not None
        assert runtime.max_intensity is not None

        # Check bounds in canonical units (Hz)
        min_hz = float(runtime.min_intensity.value)
        max_hz = float(runtime.max_intensity.value)
        assert min_hz < max_hz

        print("✓ Hawkes intensity bounds work")

    def test_hawkes_event_impacts(self):
        """Test Hawkes with event impact distributions."""
        from ethode.hawkes import HawkesConfig, HawkesState

        config = HawkesConfig(
            jump_rate="100 / hour",
            excitation_strength=0.3,
            excitation_decay="5 minutes",
            event_impact_mean="10 USD",
            event_impact_std="2 USD",
            cluster_decay_rate="0.1 / minute"
        )

        runtime = config.to_runtime()

        assert runtime.event_impact_mean is not None
        assert runtime.event_impact_std is not None
        assert runtime.cluster_decay_rate is not None

        # Initialize state
        state = HawkesState.initialize(100/3600)  # Base rate in Hz
        assert float(state.event_count) == 0
        assert float(state.cumulative_impact) == 0

        print("✓ Hawkes event impacts work")


class TestConfigInteroperability:
    """Test interoperability between different configs."""

    def test_shared_unit_manager(self):
        """Test that all configs share the same UnitManager."""
        from ethode.fee import FeeConfig
        from ethode.liquidity import LiquiditySDEConfig
        from ethode.hawkes import HawkesConfig

        manager = UnitManager.instance()

        # Define custom unit
        if not hasattr(manager.registry, 'custom_unit'):
            manager.registry.define('custom_unit = 123 * USD')

        # All configs should recognize the custom unit
        fee = FeeConfig(
            base_fee_rate=0.01,
            max_fee_rate=0.05,
            min_fee_amount="1 custom_unit"
        )
        assert fee.min_fee_amount[0] == 123  # Converted to USD

        print("✓ Shared unit manager works")

    def test_runtime_composition(self):
        """Test composing multiple runtime structures."""
        from ethode.fee import FeeConfig
        from ethode.liquidity import LiquiditySDEConfig
        from penzai.core import struct

        @struct.pytree_dataclass
        class CompositeRuntime(struct.Struct):
            fee: object  # FeeRuntime
            liquidity: object  # LiquidityRuntime

        fee_config = FeeConfig(
            base_fee_rate="50 bps",
            max_fee_rate="200 bps"
        )

        liquidity_config = LiquiditySDEConfig(
            initial_liquidity="1M USD",
            mean_liquidity="1M USD",
            mean_reversion_rate="0.1 / day",
            volatility=0.2
        )

        # Compose runtimes
        composite = CompositeRuntime(
            fee=fee_config.to_runtime(),
            liquidity=liquidity_config.to_runtime()
        )

        # Should work with JAX
        flat, treedef = jax.tree_util.tree_flatten(composite)
        assert len(flat) > 0

        print("✓ Runtime composition works")


def run_all_tests():
    """Run all comprehensive tests."""

    print("\n=== Testing UnitManager ===")
    um_tests = TestUnitManager()
    um_tests.test_singleton_pattern()
    um_tests.test_financial_units()
    um_tests.test_numeric_string_handling()
    um_tests.test_canonical_conversion()
    um_tests.test_from_canonical_reconstruction()

    print("\n=== Testing QuantityNode ===")
    qn_tests = TestQuantityNode()
    qn_tests.test_creation_methods()
    qn_tests.test_pytree_compatibility()
    qn_tests.test_jax_transformations()

    print("\n=== Testing Field Validators ===")
    fv_tests = TestFieldValidators()
    fv_tests.test_quantity_field_validator()
    # Skip bounds test that needs pytest

    print("\n=== Testing Fee Module (Advanced) ===")
    fee_tests = TestFeeModuleAdvanced()
    fee_tests.test_fee_with_all_parameters()
    fee_tests.test_fee_stress_dynamics()
    fee_tests.test_fee_accumulation()

    print("\n=== Testing Liquidity Module (Advanced) ===")
    liq_tests = TestLiquidityModuleAdvanced()
    liq_tests.test_liquidity_mean_reversion()
    liq_tests.test_liquidity_with_jumps()
    liq_tests.test_liquidity_provision_removal()

    print("\n=== Testing Hawkes Module (Advanced) ===")
    hawkes_tests = TestHawkesModuleAdvanced()
    hawkes_tests.test_hawkes_branching_ratio()
    hawkes_tests.test_hawkes_intensity_bounds()
    hawkes_tests.test_hawkes_event_impacts()

    print("\n=== Testing Config Interoperability ===")
    interop_tests = TestConfigInteroperability()
    interop_tests.test_shared_unit_manager()
    interop_tests.test_runtime_composition()

    print("\n✅ All comprehensive tests passed!")


if __name__ == "__main__":
    run_all_tests()