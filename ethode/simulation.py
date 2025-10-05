"""High-level simulation facade for multi-subsystem orchestration.

This module provides a Simulation class that orchestrates multiple subsystems
(controller, fee, liquidity, etc.) with both stateful and functional interfaces.
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .adapters import ControllerAdapter
from .runtime import ControllerRuntime, ControllerState
from .controller.kernel import controller_step


__all__ = [
    'Simulation',
    'simulate_controller_step',
]


class Simulation:
    """High-level simulation facade for orchestrating multiple subsystems.

    This class provides a convenient interface for running simulations with
    multiple subsystems (controller, fee, liquidity, etc.). It supports both:

    - Stateful API (for notebook/interactive use): `.step()`, `.reset()`
    - Functional API (for JAX transformations): direct access to `.controller.runtime` and `.controller.state`

    Currently supports controller subsystem, with fee/liquidity integration planned.

    Example (stateful):
        >>> from ethode import ControllerAdapter, ControllerConfig, Simulation
        >>> config = ControllerConfig(kp="0.2/day", ki="0.02/day**2", kd=0.0,
        ...                           tau="7 day", noise_band=("0.001 USD", "0.003 USD"))
        >>> adapter = ControllerAdapter(config)
        >>> sim = Simulation(controller=adapter)
        >>> output = sim.step(error=1.0, dt=0.1)
        >>> sim.reset()

    Example (functional):
        >>> runtime = sim.controller.runtime
        >>> state = sim.controller.state
        >>> new_state, output = controller_step(runtime, state, jnp.array(1.0), jnp.array(0.1))
    """

    def __init__(
        self,
        *,
        controller: ControllerAdapter,
        fee=None,
        liquidity=None
    ):
        """Initialize simulation with subsystems.

        Args:
            controller: ControllerAdapter instance (required)
            fee: FeeAdapter instance (reserved for future use)
            liquidity: LiquidityAdapter instance (reserved for future use)
        """
        self.controller = controller
        self.fee = fee  # Reserved for future FeeAdapter
        self.liquidity = liquidity  # Reserved for future LiquidityAdapter

        # Validate subsystems
        if not isinstance(controller, ControllerAdapter):
            raise TypeError(
                f"controller must be a ControllerAdapter, got {type(controller).__name__}"
            )

    def step(self, error: float, dt: float) -> float:
        """Execute one simulation step (stateful).

        Orchestrates all subsystems in the correct order. Updates internal
        state and returns control output.

        Args:
            error: Current error (setpoint - measurement)
            dt: Time step size

        Returns:
            Control output value
        """
        # Controller step
        control = self.controller.step(error, dt)

        # TODO: Integrate fee/liquidity subsystems when ready
        # if self.fee:
        #     fee_output = self.fee.step(...)
        # if self.liquidity:
        #     liquidity_output = self.liquidity.step(...)

        return control

    def reset(self):
        """Reset all subsystem states."""
        self.controller.reset()

        # TODO: Reset fee/liquidity when ready
        # if self.fee:
        #     self.fee.reset()
        # if self.liquidity:
        #     self.liquidity.reset()

    def get_state(self) -> dict:
        """Get current state of all subsystems.

        Returns:
            Dictionary with subsystem states
        """
        state = {
            'controller': self.controller.get_state()
        }

        # TODO: Add fee/liquidity states when ready
        # if self.fee:
        #     state['fee'] = self.fee.get_state()
        # if self.liquidity:
        #     state['liquidity'] = self.liquidity.get_state()

        return state

    def scan(
        self,
        errors: jax.Array,
        dts: jax.Array
    ) -> Tuple[jax.Array, ControllerState]:
        """Convenience wrapper for jax.lax.scan over a sequence of steps.

        This method efficiently processes a batch of error/dt pairs using JAX's
        scan operation. It automatically updates the internal controller state
        to the final state after processing all steps.

        Args:
            errors: Array of error values [n_steps]
            dts: Array of time steps [n_steps]

        Returns:
            Tuple of (outputs, final_state) where:
                - outputs: Array of control outputs [n_steps]
                - final_state: Final controller state after all steps

        Example:
            >>> errors = jnp.array([1.0, 0.5, 0.2, 0.0, -0.1])
            >>> dts = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
            >>> outputs, final_state = sim.scan(errors, dts)
            >>> # Internal state is now updated to final_state
        """
        # Define step function for scan
        def step_fn(state, inputs):
            error, dt = inputs
            new_state, output = controller_step(
                self.controller.runtime, state, error, dt
            )
            return new_state, output

        # Run scan starting from current state
        initial_state = self.controller.state
        final_state, outputs = jax.lax.scan(
            step_fn, initial_state, (errors, dts)
        )

        # Update internal controller state
        self.controller.state = final_state

        return outputs, final_state


def simulate_controller_step(
    runtime: ControllerRuntime,
    state: ControllerState,
    error: jax.Array,
    dt: jax.Array
) -> Tuple[ControllerState, jax.Array]:
    """Pure functional controller step for JAX transformations.

    This is a convenience wrapper around controller_step that provides
    a clear interface for JAX power users who want to use jax.lax.scan,
    jax.jit, etc.

    Args:
        runtime: Controller runtime parameters
        state: Current controller state
        error: Error signal (JAX array)
        dt: Time step (JAX array)

    Returns:
        Tuple of (new_state, control_output)

    Example:
        >>> # Use with jax.lax.scan for batch processing
        >>> def step_fn(state, inputs):
        ...     error, dt = inputs
        ...     return simulate_controller_step(runtime, state, error, dt)
        >>>
        >>> errors = jnp.array([1.0, 0.5, 0.2])
        >>> dts = jnp.array([0.1, 0.1, 0.1])
        >>> final_state, outputs = jax.lax.scan(step_fn, initial_state, (errors, dts))
    """
    return controller_step(runtime, state, error, dt)
