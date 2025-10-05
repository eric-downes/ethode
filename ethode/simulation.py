"""High-level simulation facade for multi-subsystem orchestration.

This module provides a Simulation class that orchestrates multiple subsystems
(controller, fee, liquidity, etc.) with both stateful and functional interfaces.
"""

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .adapters import ControllerAdapter, FeeAdapter, LiquidityAdapter, HawkesAdapter
from .runtime import ControllerRuntime, ControllerState
from .controller.kernel import controller_step


__all__ = [
    'Simulation',
    'simulate_controller_step',
]


class Simulation:
    """High-level simulation facade for orchestrating multiple subsystems.

    This class provides a convenient interface for running simulations with
    multiple subsystems (controller, fee, liquidity, hawkes). It supports both:

    - Stateful API (for notebook/interactive use): `.step()`, `.reset()`
    - Functional API (for JAX transformations): direct access to adapter runtime/state

    All subsystems are optional. Provide only the subsystems you need.

    Example (controller only):
        >>> from ethode import ControllerAdapter, ControllerConfig, Simulation
        >>> config = ControllerConfig(kp="0.2/day", ki="0.02/day**2", kd=0.0,
        ...                           tau="7 day", noise_band=("0.001 USD", "0.003 USD"))
        >>> adapter = ControllerAdapter(config)
        >>> sim = Simulation(controller=adapter)
        >>> inputs = {'error': 1.0}
        >>> outputs = sim.step(inputs, dt=0.1)

    Example (multi-subsystem):
        >>> sim = Simulation(
        ...     controller=ControllerAdapter(controller_config),
        ...     fee=FeeAdapter(fee_config),
        ...     liquidity=LiquidityAdapter(liquidity_config)
        ... )
        >>> inputs = {'error': 1.0, 'market_volatility': 0.5}
        >>> outputs = sim.step(inputs, dt=0.1)
        >>> # outputs = {'control': 0.02, 'fee': 0.001, 'liquidity': 1000000.0}
    """

    def __init__(
        self,
        *,
        controller: Optional[ControllerAdapter] = None,
        fee: Optional[FeeAdapter] = None,
        liquidity: Optional[LiquidityAdapter] = None,
        hawkes: Optional[HawkesAdapter] = None
    ):
        """Initialize simulation with subsystems.

        Args:
            controller: ControllerAdapter instance (optional)
            fee: FeeAdapter instance (optional)
            liquidity: LiquidityAdapter instance (optional)
            hawkes: HawkesAdapter instance (optional)
        """
        self.controller = controller
        self.fee = fee
        self.liquidity = liquidity
        self.hawkes = hawkes

        # Validate subsystems
        if controller is not None and not isinstance(controller, ControllerAdapter):
            raise TypeError(
                f"controller must be a ControllerAdapter, got {type(controller).__name__}"
            )
        if fee is not None and not isinstance(fee, FeeAdapter):
            raise TypeError(
                f"fee must be a FeeAdapter, got {type(fee).__name__}"
            )
        if liquidity is not None and not isinstance(liquidity, LiquidityAdapter):
            raise TypeError(
                f"liquidity must be a LiquidityAdapter, got {type(liquidity).__name__}"
            )
        if hawkes is not None and not isinstance(hawkes, HawkesAdapter):
            raise TypeError(
                f"hawkes must be a HawkesAdapter, got {type(hawkes).__name__}"
            )

    def step(self, inputs: dict, dt: float) -> dict:
        """Execute one simulation step (stateful).

        Orchestrates all subsystems in the correct order. Updates internal
        states and returns outputs from all active subsystems.

        Execution order:
        1. Controller - computes control signal from error
        2. Fee - calculates transaction fees (may use control output)
        3. Liquidity - updates liquidity level (independent)
        4. Hawkes - generates events (independent)

        Args:
            inputs: Dictionary with subsystem inputs:
                - 'error': float - For controller (required if controller exists)
                - 'market_volatility': float - For fee stress (optional)
                - 'volume_ratio': float - For fee stress (optional)
            dt: Time step size

        Returns:
            Dictionary with subsystem outputs:
                - 'control': float - Controller output (if controller exists)
                - 'fee': float - Fee amount (if fee exists)
                - 'liquidity': float - Liquidity level (if liquidity exists)
                - 'event_occurred': bool - Event flag (if hawkes exists)

        Example:
            >>> inputs = {'error': 1.0, 'market_volatility': 0.5}
            >>> outputs = sim.step(inputs, dt=0.1)
            >>> print(outputs['control'])
        """
        outputs = {}

        # 1. Controller subsystem
        if self.controller is not None:
            if 'error' not in inputs:
                raise ValueError("'error' required in inputs when controller is active")
            control = self.controller.step(inputs['error'], dt)
            outputs['control'] = control

        # 2. Fee subsystem
        if self.fee is not None:
            # Update stress based on market conditions if provided
            if 'market_volatility' in inputs:
                self.fee.update_stress(
                    volatility=inputs['market_volatility'],
                    volume_ratio=inputs.get('volume_ratio', 1.0)
                )

            # Calculate fee for transaction amount
            # Use controller output if available, otherwise use explicit transaction amount
            if 'transaction_amount' in inputs:
                transaction_amount = inputs['transaction_amount']
            elif 'control' in outputs:
                transaction_amount = abs(outputs['control'])
            else:
                transaction_amount = 0.0

            fee = self.fee.step(transaction_amount, dt)
            outputs['fee'] = fee

        # 3. Liquidity subsystem (independent stochastic update)
        if self.liquidity is not None:
            liquidity = self.liquidity.step(dt)
            outputs['liquidity'] = liquidity

        # 4. Hawkes subsystem (independent event generation)
        if self.hawkes is not None:
            event_occurred = self.hawkes.step(dt)
            outputs['event_occurred'] = event_occurred

        return outputs

    def reset(self):
        """Reset all subsystem states.

        Resets each active subsystem to its initial state. Stochastic subsystems
        (liquidity, hawkes) maintain their current random seeds.
        """
        if self.controller is not None:
            self.controller.reset()
        if self.fee is not None:
            self.fee.reset()
        if self.liquidity is not None:
            self.liquidity.reset()
        if self.hawkes is not None:
            self.hawkes.reset()

    def get_state(self) -> dict:
        """Get current state of all subsystems.

        Returns:
            Dictionary with subsystem states. Only includes states for
            active subsystems.

        Example:
            >>> state = sim.get_state()
            >>> print(state['controller'])
            {'error_integral': 0.0, 'last_error': 0.0, ...}
        """
        state = {}

        if self.controller is not None:
            state['controller'] = self.controller.get_state()
        if self.fee is not None:
            state['fee'] = self.fee.get_state()
        if self.liquidity is not None:
            state['liquidity'] = self.liquidity.get_state()
        if self.hawkes is not None:
            state['hawkes'] = self.hawkes.get_state()

        return state

    def scan(
        self,
        errors: jax.Array,
        dts: jax.Array
    ) -> Tuple[jax.Array, ControllerState]:
        """Convenience wrapper for jax.lax.scan over a sequence of controller steps.

        NOTE: This method currently supports controller subsystem only. For
        multi-subsystem batch processing, use direct JAX transformations on
        the adapter runtime/state structures.

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

        Raises:
            ValueError: If controller subsystem is not active

        Example:
            >>> errors = jnp.array([1.0, 0.5, 0.2, 0.0, -0.1])
            >>> dts = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
            >>> outputs, final_state = sim.scan(errors, dts)
            >>> # Internal state is now updated to final_state
        """
        if self.controller is None:
            raise ValueError("scan() requires controller subsystem to be active")

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
