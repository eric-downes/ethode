#!/usr/bin/env python
"""Demo comparing old and new PID controller APIs."""

import numpy as np
import jax.numpy as jnp
from ethode.controller import (
    # Legacy API
    PIDController, PIDParams,
    # New API
    ControllerConfig, controller_step, ControllerState
)


def demo_legacy_api():
    """Demonstrate the legacy PID API (backward compatible)."""
    print("=" * 60)
    print("LEGACY API (Backward Compatible)")
    print("=" * 60)

    # Create controller with old API
    params = PIDParams(
        kp=1.0,
        ki=0.1,
        kd=0.01,
        integral_leak=0.01,  # Decay rate
        output_min=-10.0,
        output_max=10.0,
        noise_threshold=0.001
    )
    pid = PIDController(params)

    print(f"Created PIDController with:")
    print(f"  kp={pid.kp}, ki={pid.ki}, kd={pid.kd}")
    print(f"  output limits: [{pid.output_min}, {pid.output_max}]")

    # Run control loop
    print("\nRunning control loop:")
    errors = [0.5, 0.4, 0.3, 0.2, 0.1]
    dt = 0.1

    for i, error in enumerate(errors):
        output = pid.update(error, dt)
        print(f"  Step {i+1}: error={error:.2f} → output={output:.3f}")
        print(f"           integral={pid.integral:.3f}, last_error={pid.last_error:.3f}")

    # Reset
    pid.reset()
    print("\nController reset - integral:", pid.integral)


def demo_new_api():
    """Demonstrate the new unit-aware API."""
    print("\n" + "=" * 60)
    print("NEW UNIT-AWARE API")
    print("=" * 60)

    # Create config with units
    config = ControllerConfig(
        kp="1.0 / second",       # Can use units!
        ki="0.1 / second",
        kd="0.01 second",
        tau="100 second",        # Time constant (1/leak_rate)
        noise_band=("0.001", "10.0"),  # Dimensionless in this case
        output_min=-10.0,
        output_max=10.0
    )

    print("Created ControllerConfig with units:")
    print(config.summary(format="text"))

    # Convert to runtime
    runtime = config.to_runtime()
    state = ControllerState.zero()

    # Run control loop
    print("\nRunning control loop (JAX-based):")
    errors = [0.5, 0.4, 0.3, 0.2, 0.1]
    dt = 0.1

    for i, error in enumerate(errors):
        state, output = controller_step(
            runtime,
            state,
            jnp.array(error),
            jnp.array(dt)
        )
        print(f"  Step {i+1}: error={error:.2f} → output={float(output):.3f}")
        print(f"           integral={float(state.integral):.3f}, "
              f"last_error={float(state.last_error):.3f}")


def demo_advanced_features():
    """Demonstrate advanced features of the new API."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES (New API Only)")
    print("=" * 60)

    # Create config with real-world units
    config = ControllerConfig(
        kp="0.2 / day",
        ki="0.02 / day",  # Simplified unit
        kd="0 hour",
        tau="1 week",
        noise_band=("1 milliUSD", "3 milliUSD"),  # Financial units
        output_min="-100 USD",
        output_max="100 USD"
    )

    print("Real-world units example:")
    print(config.summary(format="text"))

    # Vectorized execution with vmap
    print("\nVectorized control (parallel execution):")
    runtime = config.to_runtime()
    state = ControllerState.zero()

    # Multiple errors to process in parallel
    errors = jnp.array([0.001, 0.002, 0.003, 0.004, 0.005])

    import jax

    # Define vectorized step
    def run_step(error):
        return controller_step(runtime, state, error, jnp.array(3600.0))[1]

    # Run in parallel!
    vmapped_step = jax.vmap(run_step)
    outputs = vmapped_step(errors)

    print(f"  Input errors: {errors}")
    print(f"  Outputs (parallel): {outputs}")

    # JIT compilation for speed
    print("\nJIT-compiled execution:")
    jitted_step = jax.jit(lambda e: controller_step(runtime, state, e, jnp.array(3600.0)))

    import time
    # First call includes compilation
    start = time.perf_counter()
    _, output1 = jitted_step(jnp.array(0.002))
    first_time = time.perf_counter() - start

    # Subsequent calls are much faster
    start = time.perf_counter()
    _, output2 = jitted_step(jnp.array(0.003))
    second_time = time.perf_counter() - start

    print(f"  First call (with compilation): {first_time*1000:.3f} ms")
    print(f"  Second call (compiled): {second_time*1000:.3f} ms")
    print(f"  Speedup: {first_time/second_time:.1f}x")


def main():
    """Run all demonstrations."""
    print("\n🎛️  PID CONTROLLER API COMPARISON\n")

    demo_legacy_api()
    demo_new_api()
    demo_advanced_features()

    print("\n" + "=" * 60)
    print("✅ Both APIs work seamlessly!")
    print("   - Legacy API: Unchanged, backward compatible")
    print("   - New API: Unit-aware, JAX-optimized, vectorizable")
    print("=" * 60)


if __name__ == "__main__":
    main()