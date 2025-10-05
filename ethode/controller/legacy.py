"""Legacy PIDController wrapper for backward compatibility.

This module provides a PIDController class that mimics the old API
while using the new unit-aware implementation internally.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import jax.numpy as jnp

from .config import ControllerConfig
from .kernel import controller_step, controller_step_with_diagnostics
from ..runtime import ControllerState, UnitSpec


@dataclass
class PIDParams:
    """Legacy PID controller parameters (backward compatibility).

    Maps to the new ControllerConfig internally.
    """
    kp: float = 1.0                # Proportional gain
    ki: float = 0.1                # Integral gain
    kd: float = 0.0                # Derivative gain
    integral_leak: float = 0.0     # Integral decay rate (0 = no leak)
    output_min: float = -np.inf    # Output bounds
    output_max: float = np.inf
    noise_threshold: float = 0.0   # Dead zone threshold

    def to_config(self) -> ControllerConfig:
        """Convert to new ControllerConfig."""
        # Convert integral_leak to tau (time constant)
        # integral_leak is a rate, tau is 1/rate
        tau = 1.0 / self.integral_leak if self.integral_leak > 0 else 1e10

        # Handle infinite bounds
        output_min = None if np.isinf(self.output_min) else self.output_min
        output_max = None if np.isinf(self.output_max) else self.output_max

        return ControllerConfig(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            tau=tau,
            noise_band=(self.noise_threshold, 1e10),  # Large upper bound
            output_min=output_min,
            output_max=output_max,
        )


class PIDController:
    """Legacy PID controller wrapper for backward compatibility.

    This class provides the old PIDController API while using the new
    JAX-based implementation internally. It maintains stateful behavior
    for compatibility with existing code.
    """

    def __init__(self, params: Optional[PIDParams] = None, **kwargs):
        """Initialize PID controller with parameters.

        Can be called with:
        - PIDParams object: PIDController(params)
        - Individual parameters: PIDController(kp=1.0, ki=0.1, kd=0.01)
        - Mix of both (kwargs override params)
        """
        # Handle different initialization patterns
        if params is None:
            params = PIDParams()

        # Apply any keyword overrides
        for key, value in kwargs.items():
            if hasattr(params, key):
                setattr(params, key, value)
            elif key in ('tau_leak', 'error_filter'):
                # Handle extended API fields
                if key == 'tau_leak' and value is not None:
                    params.integral_leak = 1.0 / value if value > 0 else 0.0
            elif key == 'rate_limit':
                # Store for later use
                self.rate_limit = value
            else:
                # Store as attribute for compatibility
                setattr(self, key, value)

        self.p = params
        self.params = params  # Alias for compatibility

        # Create config and runtime
        self.config = params.to_config()

        # Add rate limit if specified
        if hasattr(self, 'rate_limit') and self.rate_limit is not None:
            # Properly construct the rate_limit with UnitSpec using unit conversion
            from ..units import UnitManager
            import pint
            manager = UnitManager.instance()

            # Handle different input types for rate_limit
            if isinstance(self.rate_limit, pint.Quantity):
                # Already a pint Quantity, use directly
                qty = self.rate_limit
            elif isinstance(self.rate_limit, str):
                # String that might contain units
                qty = manager.ensure_quantity(self.rate_limit, "USD/second")
            else:
                # Numeric value - assume USD/second
                qty = manager.ensure_quantity(f"{self.rate_limit} USD/second", "USD/second")

            # Convert to canonical form for price/time dimension
            rate_value, rate_spec = manager.to_canonical(qty, "price/time")
            self.config.rate_limit = (rate_value, rate_spec)

            # Rebuild runtime with updated config
            self.runtime = self.config.to_runtime()
        else:
            self.runtime = self.config.to_runtime()

        # Initialize state
        self.state = ControllerState.zero()

        # Legacy state access (for backward compatibility)
        self.integral = 0.0
        self.last_error = None
        self.last_output = 0.0

        # Error filter (if provided)
        self.error_filter = kwargs.get('error_filter', None)

        # Direct attribute access for gains (backward compatibility)
        self.kp = params.kp
        self.ki = params.ki
        self.kd = params.kd

        # Bounds and limits
        self.output_min = params.output_min if not np.isinf(params.output_min) else None
        self.output_max = params.output_max if not np.isinf(params.output_max) else None
        self.tau_leak = 1.0 / params.integral_leak if params.integral_leak > 0 else None
        self.noise_threshold = params.noise_threshold

    def update(self, error: float, dt: float) -> float:
        """Update controller and return output.

        Args:
            error: Current error (setpoint - measurement)
            dt: Time step

        Returns:
            Controller output (float)
        """
        # Apply error filter if provided (legacy support)
        if self.error_filter is not None:
            error = self.error_filter(error)

        # Convert to JAX arrays
        error_jax = jnp.array(float(error))
        dt_jax = jnp.array(float(dt))

        # Run controller step
        self.state, output = controller_step(
            self.runtime,
            self.state,
            error_jax,
            dt_jax
        )

        # Update legacy state variables for backward compatibility
        self.integral = float(self.state.integral)
        self.last_error = float(self.state.last_error)
        self.last_output = float(self.state.last_output)

        # Return as Python float
        return float(output)

    def reset(self):
        """Reset controller state."""
        self.state = ControllerState.zero()
        self.integral = 0.0
        self.last_error = None
        self.last_output = 0.0

    def get_state(self) -> dict:
        """Get current controller state as dictionary."""
        return {
            'integral': self.integral,
            'last_error': self.last_error,
            'last_output': self.last_output,
            'time': float(self.state.time)
        }

    def set_state(self, integral: float = 0.0, last_error: Optional[float] = None,
                  last_output: float = 0.0):
        """Set controller state (for testing or initialization)."""
        self.state = ControllerState(
            integral=jnp.array(integral),
            last_error=jnp.array(last_error if last_error is not None else 0.0),
            last_output=jnp.array(last_output),
            time=self.state.time
        )
        self.integral = integral
        self.last_error = last_error
        self.last_output = last_output

    def update_with_diagnostics(self, error: float, dt: float) -> tuple[float, dict]:
        """Update controller and return output with diagnostics.

        Args:
            error: Current error
            dt: Time step

        Returns:
            Tuple of (output, diagnostics_dict)
        """
        # Apply error filter if provided
        if self.error_filter is not None:
            error = self.error_filter(error)

        # Convert to JAX arrays
        error_jax = jnp.array(float(error))
        dt_jax = jnp.array(float(dt))

        # Run controller step with diagnostics
        self.state, output, diagnostics = controller_step_with_diagnostics(
            self.runtime,
            self.state,
            error_jax,
            dt_jax
        )

        # Update legacy state
        self.integral = float(self.state.integral)
        self.last_error = float(self.state.last_error)
        self.last_output = float(self.state.last_output)

        # Convert diagnostics to Python floats
        diag_dict = {k: float(v) for k, v in diagnostics.items()}

        return float(output), diag_dict


def noise_barrier(error: float, low: float, high: float) -> float:
    """Legacy noise barrier function for backward compatibility.

    Args:
        error: Error signal
        low: Lower threshold
        high: Upper threshold

    Returns:
        Filtered error
    """
    abs_error = abs(error)

    if abs_error < low:
        return 0.0
    elif abs_error > high:
        return error
    else:
        # Linear interpolation in band
        factor = (abs_error - low) / (high - low)
        return error * factor