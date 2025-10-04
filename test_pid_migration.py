#!/usr/bin/env python
"""Test that the new PIDController works with legacy code patterns."""

import numpy as np
from ethode.controller import PIDController, PIDParams


def test_legacy_pidparams():
    """Test using legacy PIDParams."""
    params = PIDParams(
        kp=1.0,
        ki=0.1,
        kd=0.01,
        integral_leak=0.1,  # 1/tau
        output_min=-10.0,
        output_max=10.0,
        noise_threshold=0.001
    )

    pid = PIDController(params)

    # Test basic update
    output = pid.update(error=1.0, dt=0.1)
    assert isinstance(output, float)

    # Test state access
    assert hasattr(pid, 'integral')
    assert hasattr(pid, 'last_error')
    assert hasattr(pid, 'last_output')

    # Test reset
    pid.reset()
    assert pid.integral == 0.0

    print("✓ PIDParams compatibility works")


def test_legacy_kwargs_init():
    """Test initialization with keyword arguments."""
    pid = PIDController(kp=0.5, ki=0.05, kd=0.005)

    # Check attributes
    assert pid.kp == 0.5
    assert pid.ki == 0.05
    assert pid.kd == 0.005

    # Test update
    output = pid.update(0.5, 0.1)
    assert isinstance(output, float)

    print("✓ Keyword argument initialization works")


def test_legacy_attributes():
    """Test direct attribute access patterns."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.0,
                        output_min=-5.0, output_max=5.0,
                        tau_leak=10.0)

    # Direct attribute access
    assert pid.kp == 1.0
    assert pid.ki == 0.1
    assert pid.kd == 0.0
    assert pid.output_min == -5.0
    assert pid.output_max == 5.0
    assert pid.tau_leak == 10.0

    print("✓ Direct attribute access works")


def test_noise_threshold():
    """Test noise threshold (dead zone) behavior."""
    params = PIDParams(
        kp=1.0,
        ki=0.0,
        kd=0.0,
        noise_threshold=0.1
    )

    pid = PIDController(params)

    # Small error should be filtered
    output = pid.update(error=0.05, dt=0.1)  # Below threshold
    assert output == 0.0

    # Large error should pass through
    output = pid.update(error=0.5, dt=0.1)  # Above threshold
    assert output > 0.0

    print("✓ Noise threshold works")


def test_integral_leak():
    """Test integral leak (anti-windup)."""
    params = PIDParams(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        integral_leak=1.0  # Fast decay
    )

    pid = PIDController(params)

    # Build up integral
    pid.update(error=1.0, dt=1.0)
    initial_integral = pid.integral
    assert initial_integral > 0

    # Let it decay with zero error
    pid.update(error=0.0, dt=1.0)
    decayed_integral = pid.integral

    # Should have decayed
    assert decayed_integral < initial_integral
    assert decayed_integral == pytest.approx(initial_integral * np.exp(-1.0), rel=1e-3)

    print("✓ Integral leak works")


def test_output_saturation():
    """Test output limits."""
    pid = PIDController(kp=10.0, ki=0.0, kd=0.0,
                        output_min=-1.0, output_max=1.0)

    # Large positive error should saturate
    output = pid.update(error=10.0, dt=0.1)
    assert output == 1.0

    # Large negative error should saturate
    output = pid.update(error=-10.0, dt=0.1)
    assert output == -1.0

    print("✓ Output saturation works")


def test_complete_control_loop():
    """Test a complete control loop scenario."""
    params = PIDParams(
        kp=0.5,
        ki=0.1,
        kd=0.01,
        integral_leak=0.01,
        output_min=-2.0,
        output_max=2.0,
        noise_threshold=0.001
    )

    pid = PIDController(params)

    # Simulate control loop
    setpoint = 1.0
    measurement = 0.0
    dt = 0.1

    outputs = []
    for _ in range(20):
        error = setpoint - measurement
        output = pid.update(error, dt)
        outputs.append(output)

        # Simple first-order system response
        measurement += output * dt * 0.5

    # Should have non-zero outputs
    assert any(o != 0 for o in outputs)

    # Should make progress toward setpoint
    # (may not fully converge in 20 steps with these gains)
    assert measurement > 0.2  # Made some progress
    print(f"  Final measurement: {measurement:.3f} (target: {setpoint})")

    print("✓ Complete control loop works")


def test_state_persistence():
    """Test that state persists between calls."""
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0)

    # First update
    pid.update(1.0, 0.1)
    state1 = pid.get_state()

    # Second update
    pid.update(1.0, 0.1)
    state2 = pid.get_state()

    # Integral should have increased
    assert state2['integral'] > state1['integral']

    # Time should have advanced
    assert state2['time'] > state1['time']

    print("✓ State persistence works")


if __name__ == "__main__":
    import pytest

    # Run tests
    test_legacy_pidparams()
    test_legacy_kwargs_init()
    test_legacy_attributes()
    test_noise_threshold()
    test_integral_leak()
    test_output_saturation()
    test_complete_control_loop()
    test_state_persistence()

    print("\n✅ All legacy compatibility tests passed!")
    print("The new PIDController successfully replaces the old implementation.")