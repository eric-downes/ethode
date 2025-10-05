"""Legacy PIDController wrapper for backward compatibility.

This module provides a PIDController class that mimics the old API
while using the new unit-aware implementation internally.

.. deprecated:: 2.0
   The PIDParams class and legacy PIDController will be removed in v3.0.
   Please migrate to ethode.controller.PIDController with ControllerConfig.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import warnings
import numpy as np
import jax.numpy as jnp

from .config import ControllerConfig
from .kernel import controller_step, controller_step_with_diagnostics
from ..runtime import ControllerState, UnitSpec


@dataclass
class PIDParams:
    """Legacy PID controller parameters (backward compatibility).

    .. deprecated:: 2.0
       PIDParams is deprecated. Use ControllerConfig instead.

    Maps to the new ControllerConfig internally.
    """
    kp: float = 1.0                # Proportional gain
    ki: float = 0.1                # Integral gain
    kd: float = 0.0                # Derivative gain
    integral_leak: float = 0.0     # Integral decay rate (0 = no leak)
    output_min: float = -np.inf    # Output bounds
    output_max: float = np.inf
    noise_threshold: float = 0.0   # Dead zone threshold

    def __post_init__(self):
        warnings.warn(
            "PIDParams is deprecated and will be removed in v3.0. "
            "Please use ethode.controller.ControllerConfig instead. "
            "Example migration:\n"
            "  Old: params = PIDParams(kp=1.0, ki=0.1, kd=0.01)\n"
            "  New: config = ControllerConfig(kp=1.0, ki=0.1, kd=0.01)",
            DeprecationWarning,
            stacklevel=3  # Stack: user -> __init__ -> __post_init__
        )

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
        - ControllerConfig object: PIDController(config) (new API)
        - Individual parameters: PIDController(kp=1.0, ki=0.1, kd=0.01)
        - Mix of both (kwargs override params)
        """
        from ..controller import ControllerConfig

        # Type handling - distinguish between new and legacy API
        if isinstance(params, ControllerConfig):
            # New API: no deprecation warning
            self.config = params
            self.params = None  # No PIDParams for new API
            self.p = None

            # Warn only if kwargs are trying to override the config
            if kwargs and not (len(kwargs) == 1 and 'rate_limit' in kwargs):
                warnings.warn(
                    "Passing kwargs to override a ControllerConfig is deprecated. "
                    "Modify the config directly before passing it.",
                    DeprecationWarning,
                    stacklevel=2
                )

        elif isinstance(params, PIDParams) or params is None:
            # Legacy API: show deprecation warning
            warnings.warn(
                "Legacy PIDController is deprecated and will be removed in v3.0. "
                "Please use ethode.controller.PIDController (main module) instead. "
                "The new PIDController has improved unit handling and JAX integration.",
                DeprecationWarning,
                stacklevel=2
            )

            # Handle different initialization patterns
            if params is None:
                params = PIDParams()

            # Apply any keyword overrides to params before creating config
            for key, value in kwargs.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                elif key in ('tau_leak', 'error_filter'):
                    # Handle extended API fields
                    if key == 'tau_leak' and value is not None:
                        params.integral_leak = 1.0 / value if value > 0 else 0.0
                elif key == 'rate_limit':
                    # Store for later processing
                    pass  # Will be handled below
                else:
                    # Store as attribute for compatibility
                    setattr(self, key, value)

            self.p = params
            self.params = params  # Alias for compatibility

            # Create config and runtime
            self.config = params.to_config()

        else:
            raise TypeError(
                f"PIDController expects PIDParams, ControllerConfig, or None, "
                f"got {type(params).__name__}"
            )

        # Process rate_limit for both pathways (if specified in kwargs)
        rate_limit = kwargs.get('rate_limit')
        if rate_limit is not None:
            # Properly construct the rate_limit with UnitSpec using unit conversion
            from ..units import UnitManager
            import pint
            manager = UnitManager.instance()

            # Handle different input types for rate_limit
            if isinstance(rate_limit, pint.Quantity):
                # Already a pint Quantity, use directly
                qty = rate_limit
            elif isinstance(rate_limit, str):
                # Check if string is just a number (like "5" or "3.5")
                try:
                    numeric_value = float(rate_limit)
                    # Treat numeric strings same as floats - apply default unit
                    qty = manager.ensure_quantity(f"{numeric_value} USD/second", "USD/second")
                except ValueError:
                    # String has units (like "5 USD/hour"), parse as-is
                    qty = manager.ensure_quantity(rate_limit, "USD/second")
            else:
                # Numeric value - assume USD/second
                qty = manager.ensure_quantity(f"{rate_limit} USD/second", "USD/second")

            # Convert to canonical form for price/time dimension
            rate_value, rate_spec = manager.to_canonical(qty, "price/time")
            self.config.rate_limit = (rate_value, rate_spec)

            # Store original value for compatibility
            self.rate_limit = rate_limit

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
        if self.params is not None:
            self.kp = self.params.kp
            self.ki = self.params.ki
            self.kd = self.params.kd
        else:
            # Extract from config for new API
            self.kp = self.config.kp[0] if self.config.kp else 0.0
            self.ki = self.config.ki[0] if self.config.ki else 0.0
            self.kd = self.config.kd[0] if self.config.kd else 0.0

        # Bounds and limits
        if self.params is not None:
            self.output_min = self.params.output_min if not np.isinf(self.params.output_min) else None
            self.output_max = self.params.output_max if not np.isinf(self.params.output_max) else None
        else:
            # Extract from config for new API
            self.output_min = self.config.output_min[0] if self.config.output_min else None
            self.output_max = self.config.output_max[0] if self.config.output_max else None
        if self.params is not None:
            self.tau_leak = 1.0 / self.params.integral_leak if self.params.integral_leak > 0 else None
            self.noise_threshold = self.params.noise_threshold
        else:
            # Extract from config for new API
            self.tau_leak = self.config.tau[0] if self.config.tau else None
            self.noise_threshold = self.config.noise_band[0] if self.config.noise_band else 0.0

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

        # Check if integral was manually set (backward compatibility)
        if abs(self.integral - float(self.state.integral)) > 1e-10:
            # Integral was manually changed, update the JAX state
            self.state = ControllerState(
                integral=jnp.array(self.integral),
                last_error=self.state.last_error,
                last_output=self.state.last_output,
                time=self.state.time
            )

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