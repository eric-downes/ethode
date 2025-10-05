#!/usr/bin/env python
"""Demo script showing ControllerConfig usage with user-friendly units."""

import jax.numpy as jnp
from ethode.controller import ControllerConfig, controller_step
from ethode.runtime import ControllerState


def main():
    """Demonstrate the controller configuration and execution."""

    # 1. Create configuration with user-friendly units
    print("Creating controller configuration with friendly units...")
    config = ControllerConfig(
        kp="0.2 / day",                    # Proportional gain: 0.2 per day
        ki="0.2 / day / 7 day",            # Integral gain: 0.2 per day per week
        kd="0.0 hour",                     # No derivative action
        tau="1 week",                      # Integral leak time constant
        noise_band=("1 milliUSD", "3 milliUSD"),  # Error filtering band
        output_min="-100 USD",             # Minimum output
        output_max="100 USD",              # Maximum output
    )

    # 2. Display configuration summary
    print("\n" + config.summary(format="text"))

    # 3. Convert to runtime structure for JAX
    print("\nConverting to runtime structure...")
    runtime = config.to_runtime(dtype=jnp.float32)
    print(f"  kp value (in Hz): {runtime.kp.value:.2e}")
    print(f"  tau value (in seconds): {runtime.tau.value:.0f}")

    # 4. Initialize controller state
    state = ControllerState.zero(dtype=jnp.float32)

    # 5. Run control loop with some example errors
    print("\nRunning control loop...")
    errors = [0.005, 0.003, 0.002, 0.001, 0.0005]  # USD errors
    dt = 3600.0  # 1 hour time step

    for i, error in enumerate(errors):
        state, output = controller_step(
            runtime,
            state,
            jnp.array(error, dtype=jnp.float32),
            jnp.array(dt, dtype=jnp.float32)
        )
        print(f"  Step {i+1}: error={error:.4f} USD, output={output:.4f} USD")

    # 6. Convert back to user-friendly units for display
    print("\nConverting runtime back to friendly units...")
    output_config = ControllerConfig.from_runtime(runtime)
    print(f"  kp: {output_config.kp}")
    print(f"  tau: {output_config.tau}")
    print(f"  noise_band: ({output_config.noise_band_low}, {output_config.noise_band_high})")

    # 7. Show markdown summary (useful for notebooks)
    print("\nMarkdown summary:")
    print(config.summary(format="markdown"))


if __name__ == "__main__":
    main()