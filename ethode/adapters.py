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

import numpy as np
import jax
import jax.numpy as jnp
import pint

from .controller.config import ControllerConfig
from .controller.legacy import PIDParams
from .controller.kernel import controller_step, controller_step_with_diagnostics
from .runtime import ControllerRuntime, ControllerState
from .units import UnitManager
from .fee.config import FeeConfig
from .fee.kernel import calculate_fee, calculate_fee_with_diagnostics, update_stress_level
from .fee.runtime import FeeRuntime, FeeState
from .liquidity.config import LiquiditySDEConfig
from .liquidity.kernel import update_liquidity, update_liquidity_with_diagnostics, apply_liquidity_shock
from .liquidity.runtime import LiquidityRuntime, LiquidityState
from .hawkes.config import HawkesConfig
from .hawkes.kernel import generate_event, generate_event_with_diagnostics, apply_external_shock
from .hawkes.runtime import HawkesRuntime, HawkesState


__all__ = [
    'ControllerAdapter',
    'FeeAdapter',
    'LiquidityAdapter',
    'HawkesAdapter',
    'JumpProcessAdapter',
    'JumpDiffusionAdapter',
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


class FeeAdapter:
    """High-level adapter for fee calculations with stateful API.

    Provides stateful API with automatic unit validation and convenient
    access to JAX runtime structures for fee calculations with stress-based
    adjustments.

    For JAX power users, direct access to .runtime and .state is provided
    for use with jax.lax.scan, jax.jit, etc.

    Example:
        >>> config = FeeConfig(
        ...     base_fee_rate="50 bps",
        ...     max_fee_rate="200 bps",
        ...     fee_decay_time="1 day"
        ... )
        >>> adapter = FeeAdapter(config)
        >>> fee = adapter.step(transaction_amount=100.0, dt=0.1)
        >>> adapter.update_stress(volatility=0.5, volume_ratio=1.2)

        # JAX power users can access runtime directly:
        >>> runtime = adapter.runtime
        >>> state = adapter.state
        >>> new_state, fee = calculate_fee(runtime, state, amount, dt)

    Args:
        config: FeeConfig instance
        check_units: Whether to validate units (default: True)

    Attributes:
        config: The FeeConfig used to build the runtime
        runtime: JAX-ready FeeRuntime structure
        state: Current FeeState
    """

    def __init__(
        self,
        config: FeeConfig,
        *,
        check_units: bool = True
    ):
        """Initialize fee adapter.

        Args:
            config: FeeConfig instance
            check_units: Whether to validate units (default: True)

        Raises:
            ValueError: If unit validation fails
        """
        self.config = config

        # Build runtime
        self.runtime = config.to_runtime()

        # Validate units if requested
        if check_units:
            # Fee dimension validation if needed
            # For now, we trust the config validators
            pass

        # Initialize state with base fee rate
        base_rate = float(self.runtime.base_fee_rate.value)
        self.state = FeeState.from_base_rate(base_rate)

    def step(self, transaction_amount: float, dt: float) -> float:
        """Calculate fee for a transaction (stateful).

        Updates internal state and returns fee amount.

        Args:
            transaction_amount: Transaction amount in USD
            dt: Time step since last update

        Returns:
            Fee amount (float)
        """
        # Convert inputs to JAX arrays
        amount_jax = jnp.array(float(transaction_amount), dtype=jnp.float32)
        dt_jax = jnp.array(float(dt), dtype=jnp.float32)

        # Call pure kernel
        self.state, fee = calculate_fee(
            self.runtime,
            self.state,
            amount_jax,
            dt_jax
        )

        return float(fee)

    def step_with_diagnostics(
        self,
        transaction_amount: float,
        dt: float
    ) -> Tuple[float, dict]:
        """Calculate fee with diagnostic information.

        Like step() but also returns intermediate values for debugging.

        Args:
            transaction_amount: Transaction amount in USD
            dt: Time step since last update

        Returns:
            Tuple of (fee_amount, diagnostics_dict)
        """
        amount_jax = jnp.array(float(transaction_amount), dtype=jnp.float32)
        dt_jax = jnp.array(float(dt), dtype=jnp.float32)

        self.state, fee, diagnostics = calculate_fee_with_diagnostics(
            self.runtime,
            self.state,
            amount_jax,
            dt_jax
        )

        return float(fee), diagnostics

    def update_stress(self, volatility: float, volume_ratio: float):
        """Update market stress level.

        Updates the internal stress level based on market volatility
        and trading volume. Higher stress increases fees.

        Args:
            volatility: Market volatility indicator [0, 1]
            volume_ratio: Volume relative to normal [0, inf)
        """
        vol_jax = jnp.array(float(volatility), dtype=jnp.float32)
        ratio_jax = jnp.array(float(volume_ratio), dtype=jnp.float32)

        self.state = update_stress_level(self.state, vol_jax, ratio_jax)

    def reset(self):
        """Reset fee state to initial values.

        Resets to base fee rate with zero accumulated fees.
        """
        base_rate = float(self.runtime.base_fee_rate.value)
        self.state = FeeState.from_base_rate(base_rate)

    def get_state(self) -> dict:
        """Get current state as dictionary of Python floats.

        Returns:
            Dictionary with state values including:
            - current_fee_rate: Current effective fee rate
            - accumulated_fees: Total fees accumulated
            - last_update_time: Time of last fee update
            - stress_level: Current market stress level [0, 1]
        """
        return {
            'current_fee_rate': float(self.state.current_fee_rate),
            'accumulated_fees': float(self.state.accumulated_fees),
            'last_update_time': float(self.state.last_update_time),
            'stress_level': float(self.state.stress_level),
        }


class LiquidityAdapter:
    """High-level adapter for stochastic liquidity dynamics with stateful API.

    Provides stateful API for liquidity modeling using stochastic differential
    equations (SDEs). Implements Ornstein-Uhlenbeck mean reversion with
    optional jump processes.

    For JAX power users, direct access to .runtime and .state is provided
    for use with jax.lax.scan, jax.jit, etc.

    Example:
        >>> config = LiquiditySDEConfig(
        ...     initial_liquidity="1M USD",
        ...     mean_liquidity="1M USD",
        ...     mean_reversion_rate="0.1 / day",
        ...     volatility=0.2
        ... )
        >>> adapter = LiquidityAdapter(config, seed=42)
        >>> liquidity = adapter.step(dt=0.1)
        >>> adapter.apply_shock(amount=100000.0)  # Add 100k liquidity

        # JAX power users can access runtime directly:
        >>> runtime = adapter.runtime
        >>> state = adapter.state
        >>> new_state, liq = update_liquidity(runtime, state, key, dt)

    Args:
        config: LiquiditySDEConfig instance
        check_units: Whether to validate units (default: True)
        seed: Random seed for stochastic updates (default: 0)

    Attributes:
        config: The LiquiditySDEConfig used to build the runtime
        runtime: JAX-ready LiquidityRuntime structure
        state: Current LiquidityState
        key: Current PRNG key
    """

    def __init__(
        self,
        config: LiquiditySDEConfig,
        *,
        check_units: bool = True,
        seed: int = 0
    ):
        """Initialize liquidity adapter.

        Args:
            config: LiquiditySDEConfig instance
            check_units: Whether to validate units (default: True)
            seed: Random seed for stochastic updates

        Raises:
            ValueError: If unit validation fails
        """
        self.config = config

        # Build runtime
        self.runtime = config.to_runtime()

        # Validate units if requested
        if check_units:
            # Liquidity dimension validation if needed
            # For now, we trust the config validators
            pass

        # Initialize state with initial liquidity
        initial_liquidity = float(self.runtime.initial_liquidity.value)
        self.state = LiquidityState.initialize(initial_liquidity)

        # Initialize PRNG key for stochastic updates
        self.key = jax.random.PRNGKey(seed)

    def step(self, dt: float) -> float:
        """Update liquidity level stochastically (stateful).

        Updates internal state using stochastic differential equation
        and returns current liquidity level.

        Args:
            dt: Time step size

        Returns:
            Current liquidity level (float)
        """
        # Split key for this step
        self.key, subkey = jax.random.split(self.key)

        dt_jax = jnp.array(float(dt), dtype=jnp.float32)

        # Update state
        self.state, liquidity = update_liquidity(
            self.runtime,
            self.state,
            subkey,
            dt_jax
        )

        return float(liquidity)

    def step_with_diagnostics(
        self,
        dt: float
    ) -> Tuple[float, dict]:
        """Update liquidity with diagnostic information.

        Like step() but also returns intermediate values for debugging.

        Args:
            dt: Time step size

        Returns:
            Tuple of (liquidity_level, diagnostics_dict)
        """
        # Split key for this step
        self.key, subkey = jax.random.split(self.key)

        dt_jax = jnp.array(float(dt), dtype=jnp.float32)

        self.state, liquidity, diagnostics = update_liquidity_with_diagnostics(
            self.runtime,
            self.state,
            subkey,
            dt_jax
        )

        return float(liquidity), diagnostics

    def apply_shock(self, amount: float):
        """Apply external liquidity shock (provision or removal).

        Updates liquidity immediately without advancing time. Use this
        to model discrete events like large LP entries/exits.

        Args:
            amount: Liquidity change (positive = add, negative = remove)
        """
        shock_jax = jnp.array(float(amount), dtype=jnp.float32)
        self.state = apply_liquidity_shock(self.state, shock_jax, self.runtime)

    def reset(self, seed: Optional[int] = None):
        """Reset liquidity state to initial values.

        Args:
            seed: New random seed (optional, keeps current if None)
        """
        initial_liquidity = float(self.runtime.initial_liquidity.value)
        self.state = LiquidityState.initialize(initial_liquidity)

        if seed is not None:
            self.key = jax.random.PRNGKey(seed)

    def get_state(self) -> dict:
        """Get current state as dictionary of Python floats.

        Returns:
            Dictionary with state values including:
            - liquidity_level: Current liquidity in USD
            - time: Current simulation time
            - cumulative_provision: Total liquidity added
            - cumulative_removal: Total liquidity removed
            - jump_count: Number of jumps occurred
        """
        return {
            'liquidity_level': float(self.state.liquidity_level),
            'time': float(self.state.time),
            'cumulative_provision': float(self.state.cumulative_provision),
            'cumulative_removal': float(self.state.cumulative_removal),
            'jump_count': int(self.state.jump_count),
        }


class HawkesAdapter:
    """High-level adapter for Hawkes self-exciting processes with stateful API.

    Provides stateful API for modeling event arrivals where each event
    increases the probability of future events. Implements self-exciting
    Hawkes processes with exponential decay.

    For JAX power users, direct access to .runtime and .state is provided
    for use with jax.lax.scan, jax.jit, etc.

    Example:
        >>> config = HawkesConfig(
        ...     jump_rate="100 / hour",
        ...     excitation_strength=0.3,
        ...     excitation_decay="5 minutes"
        ... )
        >>> adapter = HawkesAdapter(config, seed=42)
        >>> event_occurred = adapter.step(dt=0.1)
        >>> adapter.apply_shock(intensity_boost=500.0)  # News shock

        # JAX power users can access runtime directly:
        >>> runtime = adapter.runtime
        >>> state = adapter.state
        >>> new_state, event, impact = generate_event(runtime, state, key, dt)

    Args:
        config: HawkesConfig instance
        check_units: Whether to validate units (default: True)
        seed: Random seed for stochastic updates (default: 0)

    Attributes:
        config: The HawkesConfig used to build the runtime
        runtime: JAX-ready HawkesRuntime structure
        state: Current HawkesState
        key: Current PRNG key
    """

    def __init__(
        self,
        config: HawkesConfig,
        *,
        check_units: bool = True,
        seed: int = 0
    ):
        """Initialize Hawkes adapter.

        Args:
            config: HawkesConfig instance
            check_units: Whether to validate units (default: True)
            seed: Random seed for stochastic updates

        Raises:
            ValueError: If unit validation fails or process is unstable
        """
        self.config = config

        # Build runtime
        self.runtime = config.to_runtime()

        # Validate units if requested
        if check_units:
            # Hawkes dimension validation if needed
            # For now, we trust the config validators
            pass

        # Initialize state with base rate
        base_rate = float(self.runtime.jump_rate.value)
        self.state = HawkesState.initialize(base_rate)

        # Initialize PRNG key for stochastic updates
        self.key = jax.random.PRNGKey(seed)

    def step(self, dt: float) -> bool:
        """Simulate one time step (stateful).

        Updates internal state and checks if an event occurred.

        Args:
            dt: Time step size (should be small for accurate simulation,
                typically << 1/intensity)

        Returns:
            True if an event occurred, False otherwise
        """
        # Split key for this step
        self.key, subkey = jax.random.split(self.key)

        dt_jax = jnp.array(float(dt), dtype=jnp.float32)

        # Generate potential event
        self.state, event_occurred, _ = generate_event(
            self.runtime,
            self.state,
            subkey,
            dt_jax
        )

        return bool(event_occurred)

    def step_with_diagnostics(
        self,
        dt: float
    ) -> Tuple[bool, dict]:
        """Simulate one time step with diagnostic information.

        Like step() but also returns intermediate values for debugging.

        Args:
            dt: Time step size

        Returns:
            Tuple of (event_occurred, diagnostics_dict)
        """
        # Split key for this step
        self.key, subkey = jax.random.split(self.key)

        dt_jax = jnp.array(float(dt), dtype=jnp.float32)

        self.state, event_occurred, impact, diagnostics = generate_event_with_diagnostics(
            self.runtime,
            self.state,
            subkey,
            dt_jax
        )

        return bool(event_occurred), diagnostics

    def apply_shock(self, intensity_boost: float):
        """Apply external intensity shock.

        Models external events that temporarily increase event intensity,
        such as news announcements or market disruptions.

        Args:
            intensity_boost: Amount to add to current intensity
        """
        boost_jax = jnp.array(float(intensity_boost), dtype=jnp.float32)
        self.state = apply_external_shock(self.state, boost_jax, self.runtime)

    def get_intensity(self) -> float:
        """Get current process intensity.

        Returns:
            Current intensity (events/time)
        """
        return float(self.state.current_intensity)

    def get_branching_ratio(self) -> float:
        """Get the branching ratio (stability metric).

        Returns:
            Branching ratio. < 1 for stable, >= 1 for unstable
        """
        from .hawkes.kernel import get_branching_ratio
        return get_branching_ratio(self.runtime)

    def get_stationary_intensity(self) -> float:
        """Get long-term average intensity (if stable).

        Returns:
            Stationary intensity

        Raises:
            ValueError: If process is unstable
        """
        from .hawkes.kernel import get_stationary_intensity
        return get_stationary_intensity(self.runtime)

    def reset(self, seed: Optional[int] = None):
        """Reset Hawkes state to initial values.

        Args:
            seed: New random seed (optional, keeps current if None)
        """
        base_rate = float(self.runtime.jump_rate.value)
        self.state = HawkesState.initialize(base_rate)

        if seed is not None:
            self.key = jax.random.PRNGKey(seed)

    def get_state(self) -> dict:
        """Get current state as dictionary of Python types.

        Returns:
            Dictionary with state values including:
            - current_intensity: Current process intensity
            - time: Current simulation time
            - event_count: Total number of events
            - last_event_time: Time of most recent event
            - cumulative_impact: Sum of all event impacts
        """
        return {
            'current_intensity': float(self.state.current_intensity),
            'time': float(self.state.time),
            'event_count': int(self.state.event_count),
            'last_event_time': float(self.state.last_event_time),
            'cumulative_impact': float(self.state.cumulative_impact),
        }


class JumpProcessAdapter:
    """High-level adapter for jump processes with stateful API.

    This is the primary high-level API for jump process usage.

    Example:
        >>> config = JumpProcessConfig(
        ...     process_type='poisson',
        ...     rate="100 / day",
        ...     seed=42
        ... )
        >>> adapter = JumpProcessAdapter(config)
        >>>
        >>> # Sequential usage
        >>> for t in range(100):
        ...     jump_occurred = adapter.step(t * 0.1, dt=0.1)
        ...     if jump_occurred:
        ...         # Handle event
        ...         pass
        >>>
        >>> # Batch generation
        >>> jump_times = adapter.generate_jumps(0.0, 10.0)

    Args:
        config: JumpProcessConfig instance
        check_units: Whether to validate dimensional consistency (default: True)

    Attributes:
        config: The JumpProcessConfig used
        runtime: JAX-ready runtime structure
        state: Current JumpProcessState
    """

    def __init__(
        self,
        config: 'JumpProcessConfig',
        *,
        check_units: bool = True
    ):
        from .jumpprocess.config import JumpProcessConfig
        from .jumpprocess.runtime import JumpProcessState
        from .jumpprocess.kernel import generate_next_jump_time

        self.config = config
        self.runtime = config.to_runtime(check_units=check_units)
        self._current_seed = config.seed if config.seed is not None else 0
        self.state = JumpProcessState.zero(
            seed=self._current_seed,
            start_time=0.0
        )

        # Initialize first jump time from t=0
        self.state, _ = generate_next_jump_time(
            self.runtime, self.state, jnp.array(0.0)
        )

    def step(self, current_time: float, dt: float) -> bool:
        """
        Step forward in time, check if jump occurred.

        Updates internal state.

        Args:
            current_time: Current time
            dt: Time step

        Returns:
            True if jump occurred in [current_time, current_time + dt)
        """
        from .jumpprocess.kernel import step

        self.state, occurred = step(
            self.runtime,
            self.state,
            jnp.array(float(current_time)),
            jnp.array(float(dt))
        )
        return bool(occurred)

    def generate_jumps(self, t_start: float, t_end: float) -> np.ndarray:
        """
        Generate all jump times in interval [t_start, t_end).

        Args:
            t_start: Start time
            t_end: End time

        Returns:
            Numpy array of jump times
        """
        from .jumpprocess.kernel import generate_jumps_in_interval

        jumps = generate_jumps_in_interval(
            self.runtime,
            t_start,
            t_end,
            seed=self._current_seed
        )
        return np.array(jumps)

    def reset(self, seed: Optional[int] = None, start_time: float = 0.0):
        """Reset state to initial conditions.

        Args:
            seed: Random seed (uses config seed if None)
            start_time: Time to start from (default 0.0)
        """
        from .jumpprocess.runtime import JumpProcessState
        from .jumpprocess.kernel import generate_next_jump_time

        seed = seed if seed is not None else self.config.seed
        seed = seed if seed is not None else 0
        self._current_seed = seed
        self.state = JumpProcessState.zero(seed, start_time)

        # Generate first jump
        self.state, _ = generate_next_jump_time(
            self.runtime, self.state, jnp.array(start_time)
        )

    def get_expected_rate(self) -> float:
        """Get expected event rate (events per unit time)."""
        return float(self.runtime.rate.value)

    def get_state(self) -> dict:
        """Get current state as dictionary."""
        return {
            'last_jump_time': float(self.state.last_jump_time),
            'next_jump_time': float(self.state.next_jump_time),
            'event_count': int(self.state.event_count),
        }


class JumpDiffusionAdapter:
    """High-level adapter for ODE+Jump simulations with stateful API.

    This is the primary API for hybrid continuous/discrete simulations.
    Follows the ethode adapter pattern with stateful and functional interfaces.

    Example:
        >>> # Define dynamics
        >>> def my_dynamics(t, state, params):
        ...     x, v = state
        ...     return jnp.array([v, -params['k'] * x])  # Harmonic oscillator
        >>>
        >>> def my_jump(t, state, params):
        ...     x, v = state
        ...     return jnp.array([x, -v * 0.9])  # Damped collision
        >>>
        >>> # Configure
        >>> config = JumpDiffusionConfig(
        ...     initial_state=jnp.array([1.0, 0.0]),
        ...     dynamics_fn=my_dynamics,
        ...     jump_effect_fn=my_jump,
        ...     jump_process=JumpProcessConfig(
        ...         process_type='poisson',
        ...         rate="10 / second",
        ...     ),
        ...     solver='dopri5',
        ...     dt_max="0.01 second",
        ...     params={'k': 1.0},
        ... )
        >>>
        >>> # Stateful API
        >>> adapter = JumpDiffusionAdapter(config)
        >>> times, states = adapter.simulate(t_span=(0.0, 10.0))
        >>>
        >>> # Or step-by-step
        >>> adapter.reset()
        >>> for i in range(100):
        ...     jump_occurred = adapter.step(t_end=adapter.state.t + 0.1)
        ...     if jump_occurred:
        ...         print(f"Jump at t={adapter.state.t}")

    Args:
        config: JumpDiffusionConfig instance
        check_units: Whether to validate dimensional consistency

    Attributes:
        config: The configuration used
        runtime: JAX-ready runtime structure
        state: Current simulation state (JumpDiffusionState)
    """

    def __init__(
        self,
        config: 'JumpDiffusionConfig',
        *,
        check_units: bool = True
    ):
        from .jumpdiffusion.config import JumpDiffusionConfig

        self.config = config
        self.runtime = config.to_runtime(check_units=check_units)

        # Initialize state immediately for stateful API
        self._initialize_state(t0=0.0)

    def _initialize_state(self, t0: float = 0.0, seed: Optional[int] = None):
        """Initialize simulation state with jump buffer.

        Args:
            t0: Initial time
            seed: Random seed (uses runtime seed if None)
        """
        from .jumpdiffusion.runtime import JumpDiffusionState, ScheduledJumpBuffer
        from .jumpprocess.runtime import JumpProcessState
        from .jumpdiffusion.kernel import _generate_poisson_event

        # Use provided seed or runtime seed
        if seed is None:
            seed = int(self.runtime.scheduler.seed)

        rng_key = jax.random.PRNGKey(seed)
        mode = self.runtime.scheduler.mode
        initial_state = self.config.initial_state

        # Default buffer size for stateful API (can be extended dynamically)
        STATEFUL_BUFFER_SIZE = 10000

        if mode == 0:
            # Mode 0 (Poisson): Lazy generation
            key1, key2 = jax.random.split(rng_key)
            first_event, new_key = _generate_poisson_event(
                self.runtime.scheduler.scheduled,
                jnp.array(t0),
                key1
            )

            event_times = jnp.full(STATEFUL_BUFFER_SIZE, jnp.inf, dtype=initial_state.dtype)
            event_times = event_times.at[0].set(first_event)

            jump_buffer = ScheduledJumpBuffer(
                event_times=event_times,
                count=jnp.array(STATEFUL_BUFFER_SIZE, dtype=jnp.int32),
                next_index=jnp.array(0, dtype=jnp.int32),
                rng_key=new_key,
                cumulative_excitation=jnp.array(0.0, dtype=initial_state.dtype),
                last_update_time=jnp.array(t0, dtype=initial_state.dtype)
            )

        elif mode == 1:
            # Mode 1 (Pre-gen Hawkes): Not supported for stateful API
            # Requires knowing t_span in advance
            raise NotImplementedError(
                "Stateful API (step/reset) not supported for Mode 1 (Pre-generated Hawkes). "
                "Use simulate() method instead, which requires t_span."
            )

        else:  # mode == 2
            # Mode 2 (Online Hawkes): Not yet implemented
            raise NotImplementedError(
                "Stateful API (step/reset) not yet implemented for Mode 2 (Online Hawkes). "
                "Use simulate() method instead."
            )

        # Create initial state
        self.state = JumpDiffusionState.zero(
            initial_state,
            jump_buffer,
            t0=t0
        )

    def simulate(
        self,
        t_span: Tuple[float, float],
        max_steps: int = 100000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full ODE+Jump simulation over time span (functional).

        Saves state at: initial time, each jump time, and final time.
        Does not modify internal state. For stateful stepping, use step().

        Args:
            t_span: (t_start, t_end) simulation interval
            max_steps: Maximum number of steps (safety limit for total saves)

        Returns:
            (times, states) where:
            - times: 1D array of time points at jump times + t_end
            - states: 2D array (n_times, state_dim) with corresponding states

        Note:
            Internal padding is automatically filtered out. You receive only actual trajectory.
        """
        from .jumpdiffusion.kernel import simulate

        times_jax, states_jax = simulate(
            self.runtime,
            self.config.initial_state,
            t_span,
            max_steps=max_steps,
        )

        # Filter out padding (keep times <= t_end)
        mask = times_jax <= t_span[1]
        return np.array(times_jax[mask]), np.array(states_jax[mask])

    def step(
        self,
        t_end: float,
    ) -> bool:
        """
        Take single step: integrate to next jump or t_end (stateful).

        Updates internal self.state.

        Args:
            t_end: Maximum time to integrate to

        Returns:
            jump_occurred: True if a jump occurred during this step
        """
        from .jumpdiffusion.kernel import integrate_step, apply_jump

        new_state, t_reached = integrate_step(
            self.runtime,
            self.state,
            jnp.array(t_end)
        )

        # Check if jump occurred
        idx = new_state.jump_buffer.next_index
        next_jump_time = new_state.jump_buffer.event_times[idx]
        jump_occurred = jnp.logical_and(
            jnp.isclose(new_state.t, next_jump_time, rtol=0, atol=1e-9),
            new_state.t < t_end
        )

        # Apply jump if occurred
        if jump_occurred:
            new_state = apply_jump(self.runtime, new_state)

        self.state = new_state
        return bool(jump_occurred)

    def reset(self, t0: float = 0.0, seed: Optional[int] = None):
        """
        Reset simulation to initial conditions.

        Args:
            t0: Initial time
            seed: New random seed (uses config seed if None)

        Note:
            Stateful stepping (step/reset) currently only supports Mode 0 (Poisson).
            For Hawkes processes, use the functional simulate() method instead.
        """
        self._initialize_state(t0=t0, seed=seed)

    def get_state(self) -> dict:
        """
        Get current simulation state as dictionary.

        Returns:
            Dictionary with:
            - 't': current time
            - 'state': current state vector
            - 'next_jump_time': time of next scheduled jump (from buffer)
            - 'step_count': number of ODE steps taken
            - 'jump_count': number of jumps processed
        """
        if self.state is None:
            raise RuntimeError("State not initialized. Call simulate() or step() first.")

        idx = int(self.state.jump_buffer.next_index)
        next_jump_time = float(self.state.jump_buffer.event_times[idx])

        return {
            't': float(self.state.t),
            'state': np.array(self.state.state),
            'next_jump_time': next_jump_time,
            'step_count': int(self.state.step_count),
            'jump_count': int(self.state.jump_count),
        }

    def set_state(self, state: 'JumpDiffusionState'):
        """
        Set simulation state directly.

        Args:
            state: JumpDiffusionState to use
        """
        self.state = state
