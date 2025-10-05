"""High-level adapter classes for stateful simulation workflows.

This module provides adapter classes that wrap the low-level JAX runtime
structures with stateful, user-friendly APIs. Adapters handle unit validation,
state management, and provide convenient interfaces for both interactive
(notebook) and batch (JAX scan) usage.

The adapter pattern is designed to be reusable across all subsystems
(controller, fee, liquidity, etc.).
"""

from __future__ import annotations
from typing import Union, Tuple, Optional, Any
import warnings

import jax
import jax.numpy as jnp
import pint

from .controller.config import ControllerConfig
from .controller.legacy import PIDParams
from .controller.kernel import controller_step, controller_step_with_diagnostics
from .runtime import ControllerRuntime, ControllerState
from .units import UnitManager


__all__ = [
    'ControllerAdapter',
]


class ControllerAdapter:
    """High-level adapter for PID controller with stateful API.

    This is the primary high-level API for controller usage. It accepts
    either ControllerConfig (new API) or PIDParams (legacy), validates
    units by default, and provides stateful methods for easy usage.

    For JAX power users, direct access to .runtime and .state is provided
    for use with jax.lax.scan, jax.jit, etc.

    Example:
        >>> config = ControllerConfig(kp=1.0, ki=0.1, kd=0.01, tau=100.0,
        ...                           noise_band=(0.01, 1e9))
        >>> adapter = ControllerAdapter(config)
        >>> output = adapter.step(error=0.5, dt=0.1)

        # JAX power users can access runtime directly:
        >>> runtime = adapter.runtime
        >>> state = adapter.state
        >>> new_state, output = controller_step(runtime, state, error, dt)

    Args:
        params: Either ControllerConfig (recommended) or PIDParams (legacy)
        check_units: Whether to validate dimensional consistency (default: True)

    Attributes:
        config: The ControllerConfig used to build the runtime
        runtime: JAX-ready ControllerRuntime structure
        state: Current ControllerState
    """

    def __init__(
        self,
        params: Union[ControllerConfig, PIDParams],
        *,
        check_units: bool = True
    ):
        """Initialize the controller adapter.

        Args:
            params: Controller configuration (ControllerConfig or PIDParams)
            check_units: Validate dimensional consistency (default: True)

        Raises:
            ValueError: If unit validation fails
        """
        # Handle both new and legacy params
        if isinstance(params, PIDParams):
            warnings.warn(
                "Using PIDParams with ControllerAdapter is deprecated. "
                "Please migrate to ControllerConfig for the full feature set. "
                "Example:\n"
                "  config = ControllerConfig(kp=1.0, ki=0.1, kd=0.01, "
                "tau=100.0, noise_band=(0.01, 1e9))",
                DeprecationWarning,
                stacklevel=2
            )
            self.config = params.to_config()
        else:
            self.config = params

        # Build runtime (without re-validating inside to_runtime)
        self.runtime = self.config.to_runtime(check_units=False)

        # Always validate units by default using EXISTING validation functions
        # Do NOT re-implement validation logic here
        if check_units:
            from .validation import validate_controller_dimensions
            manager = UnitManager.instance()
            dimensions = getattr(self.config, 'dimensions', None)

            # Call existing validation - this checks dimensional consistency
            try:
                validate_controller_dimensions(
                    self.runtime,
                    manager=manager,
                    dimensions=dimensions
                )
            except (ValueError, pint.DimensionalityError) as e:
                raise ValueError(f"Unit validation failed: {e}")

        # Initialize state
        self.state = ControllerState.zero()

    def step(self, error: float, dt: float) -> float:
        """Execute one control step (stateful).

        Updates internal state and returns control output.

        Args:
            error: Current error (setpoint - measurement)
            dt: Time step size

        Returns:
            Control output value
        """
        error_jax = jnp.array(float(error))
        dt_jax = jnp.array(float(dt))

        self.state, output = controller_step(
            self.runtime,
            self.state,
            error_jax,
            dt_jax
        )

        return float(output)

    def step_with_diagnostics(
        self,
        error: float,
        dt: float
    ) -> Tuple[float, dict]:
        """Execute control step with diagnostic information.

        Like step() but also returns intermediate values for debugging.

        Args:
            error: Current error
            dt: Time step size

        Returns:
            Tuple of (control_output, diagnostics_dict)
        """
        error_jax = jnp.array(float(error))
        dt_jax = jnp.array(float(dt))

        self.state, output, diagnostics = controller_step_with_diagnostics(
            self.runtime,
            self.state,
            error_jax,
            dt_jax
        )

        # Convert diagnostics to Python floats
        diag_dict = {k: float(v) for k, v in diagnostics.items()}

        return float(output), diag_dict

    def step_with_units(
        self,
        error_qty: pint.Quantity,
        dt_qty: pint.Quantity
    ) -> pint.Quantity:
        """Execute control step with pint quantities (for debugging).

        This calls the existing controller_step_units() validation function
        to check dimensional consistency at runtime. Useful for debugging
        unit issues.

        Args:
            error_qty: Error as pint Quantity
            dt_qty: Time step as pint Quantity

        Returns:
            Control output as pint Quantity
        """
        # Call EXISTING controller_step_units() from validation.py
        # Do NOT re-implement the pint logic here
        from .validation import controller_step_units

        state_dict, output = controller_step_units(
            self.runtime, self.state, error_qty, dt_qty
        )

        # Update internal state from returned dict
        self.state = ControllerState(
            integral=jnp.array(float(state_dict['integral'].magnitude)),
            last_error=jnp.array(float(state_dict['last_error'].magnitude)),
            last_output=jnp.array(float(state_dict['last_output'].magnitude)),
            time=jnp.array(float(state_dict['time'].magnitude))
        )

        return output

    def reset(self):
        """Reset controller state to zero."""
        self.state = ControllerState.zero()

    def get_state(self) -> dict:
        """Get current state as dictionary of Python floats.

        Returns:
            Dictionary with state values
        """
        return {
            'integral': float(self.state.integral),
            'last_error': float(self.state.last_error),
            'last_output': float(self.state.last_output),
            'time': float(self.state.time),
        }
