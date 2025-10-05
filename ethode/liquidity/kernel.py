"""Liquidity SDE kernel functions for JAX.

This module provides JIT-compiled functions for stochastic liquidity dynamics.
Implements Ornstein-Uhlenbeck mean reversion with optional jump processes.
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from .runtime import LiquidityRuntime, LiquidityState


def update_liquidity(
    runtime: LiquidityRuntime,
    state: LiquidityState,
    key: jax.Array,
    dt: jax.Array
) -> Tuple[LiquidityState, jax.Array]:
    """Update liquidity level using stochastic differential equation.

    Implements Ornstein-Uhlenbeck process with optional jumps:
        dL = κ(L_mean - L)dt + σ√L dW + dJ

    Where:
        - κ is mean reversion rate
        - L_mean is long-term mean liquidity
        - σ is volatility
        - dW is Brownian motion
        - dJ is jump process (optional)

    Args:
        runtime: Liquidity configuration
        state: Current liquidity state
        key: JAX PRNG key for randomness
        dt: Time step

    Returns:
        Tuple of (new_state, current_liquidity)

    Example:
        >>> import jax
        >>> config = LiquiditySDEConfig(
        ...     initial_liquidity="1M USD",
        ...     mean_liquidity="1M USD",
        ...     mean_reversion_rate="0.1 / day",
        ...     volatility=0.2
        ... )
        >>> runtime = config.to_runtime()
        >>> state = LiquidityState.initialize(1000000.0)
        >>> key = jax.random.PRNGKey(42)
        >>> new_state, liquidity = update_liquidity(runtime, state, key, jnp.array(0.1))
    """
    # Extract parameters
    mean_liquidity = float(runtime.mean_liquidity.value)
    reversion_rate = float(runtime.mean_reversion_rate.value)
    volatility = float(runtime.volatility.value)

    # Mean reversion term: κ(L_mean - L)dt
    drift = reversion_rate * (mean_liquidity - state.liquidity_level) * dt

    # Diffusion term: σ√L dW (volatility scales with sqrt of liquidity)
    key, subkey = jax.random.split(key)
    # Use sqrt(liquidity_level) for realistic volatility scaling
    vol_scale = jnp.sqrt(jnp.maximum(state.liquidity_level, 0.0))
    diffusion = volatility * vol_scale * jnp.sqrt(dt) * jax.random.normal(subkey)

    # Jump term (if configured)
    jump = jnp.array(0.0)
    jump_occurred = False
    if runtime.jump_intensity is not None:
        key, subkey = jax.random.split(key)
        # Poisson process: probability of jump in dt
        jump_prob = float(runtime.jump_intensity.value) * dt
        jump_occurred = jax.random.uniform(subkey) < jump_prob

        if runtime.jump_size_mean is not None:
            key, subkey = jax.random.split(key)
            jump_mean = float(runtime.jump_size_mean.value)
            jump_std = float(runtime.jump_size_std.value) if runtime.jump_size_std else 0.0

            # Jump size drawn from normal distribution
            jump_size = jump_mean + jump_std * jax.random.normal(subkey)
            jump = jnp.where(jump_occurred, jump_size, 0.0)

    # Update liquidity
    new_liquidity = state.liquidity_level + drift + diffusion + jump

    # Apply bounds if configured
    if runtime.min_liquidity is not None:
        new_liquidity = jnp.maximum(new_liquidity, float(runtime.min_liquidity.value))
    if runtime.max_liquidity is not None:
        new_liquidity = jnp.minimum(new_liquidity, float(runtime.max_liquidity.value))

    # Ensure non-negative
    new_liquidity = jnp.maximum(new_liquidity, 0.0)

    # Track provisions and removals
    liquidity_change = new_liquidity - state.liquidity_level
    provision = jnp.maximum(liquidity_change, 0.0)
    removal = jnp.maximum(-liquidity_change, 0.0)

    # Create new state
    new_state = LiquidityState(
        liquidity_level=new_liquidity,
        time=state.time + dt,
        cumulative_provision=state.cumulative_provision + provision,
        cumulative_removal=state.cumulative_removal + removal,
        jump_count=state.jump_count + jnp.where(jump_occurred, 1, 0),
    )

    return new_state, new_liquidity


def update_liquidity_with_diagnostics(
    runtime: LiquidityRuntime,
    state: LiquidityState,
    key: jax.Array,
    dt: jax.Array
) -> Tuple[LiquidityState, jax.Array, dict]:
    """Update liquidity with diagnostic information.

    Args:
        runtime: Liquidity configuration
        state: Current state
        key: PRNG key
        dt: Time step

    Returns:
        Tuple of (new_state, liquidity, diagnostics)

    Diagnostics include:
        - drift: Mean reversion component
        - diffusion: Stochastic component
        - jump: Jump component (if jumps occurred)
        - hit_min_bound: Whether minimum bound was hit
        - hit_max_bound: Whether maximum bound was hit
        - provision: Liquidity added
        - removal: Liquidity removed
    """
    # Store initial values
    initial_liquidity = float(state.liquidity_level)

    # Extract parameters for diagnostics
    mean_liquidity = float(runtime.mean_liquidity.value)
    reversion_rate = float(runtime.mean_reversion_rate.value)
    volatility = float(runtime.volatility.value)

    # Calculate components
    drift = reversion_rate * (mean_liquidity - state.liquidity_level) * dt

    key, subkey = jax.random.split(key)
    vol_scale = jnp.sqrt(jnp.maximum(state.liquidity_level, 0.0))
    noise = jax.random.normal(subkey)
    diffusion = volatility * vol_scale * jnp.sqrt(dt) * noise

    # Jump component
    jump = jnp.array(0.0)
    jump_occurred = False
    if runtime.jump_intensity is not None:
        key, subkey = jax.random.split(key)
        jump_prob = float(runtime.jump_intensity.value) * dt
        jump_occurred = jax.random.uniform(subkey) < jump_prob

        if runtime.jump_size_mean is not None:
            key, subkey = jax.random.split(key)
            jump_mean = float(runtime.jump_size_mean.value)
            jump_std = float(runtime.jump_size_std.value) if runtime.jump_size_std else 0.0
            jump_size = jump_mean + jump_std * jax.random.normal(subkey)
            jump = jnp.where(jump_occurred, jump_size, 0.0)

    # Before bounds
    unbounded_liquidity = state.liquidity_level + drift + diffusion + jump

    # After bounds
    new_liquidity = unbounded_liquidity
    if runtime.min_liquidity is not None:
        new_liquidity = jnp.maximum(new_liquidity, float(runtime.min_liquidity.value))
    if runtime.max_liquidity is not None:
        new_liquidity = jnp.minimum(new_liquidity, float(runtime.max_liquidity.value))
    new_liquidity = jnp.maximum(new_liquidity, 0.0)

    # Check if bounds were hit
    hit_min_bound = False
    hit_max_bound = False
    if runtime.min_liquidity is not None:
        hit_min_bound = unbounded_liquidity < float(runtime.min_liquidity.value)
    if runtime.max_liquidity is not None:
        hit_max_bound = unbounded_liquidity > float(runtime.max_liquidity.value)

    # Track changes
    liquidity_change = new_liquidity - state.liquidity_level
    provision = jnp.maximum(liquidity_change, 0.0)
    removal = jnp.maximum(-liquidity_change, 0.0)

    # Create new state
    new_state = LiquidityState(
        liquidity_level=new_liquidity,
        time=state.time + dt,
        cumulative_provision=state.cumulative_provision + provision,
        cumulative_removal=state.cumulative_removal + removal,
        jump_count=state.jump_count + jnp.where(jump_occurred, 1, 0),
    )

    # Build diagnostics
    diagnostics = {
        'drift': float(drift),
        'diffusion': float(diffusion),
        'jump': float(jump),
        'jump_occurred': bool(jump_occurred),
        'hit_min_bound': bool(hit_min_bound),
        'hit_max_bound': bool(hit_max_bound),
        'provision': float(provision),
        'removal': float(removal),
        'unbounded_liquidity': float(unbounded_liquidity),
        'distance_from_mean': float(new_liquidity - mean_liquidity),
    }

    return new_state, new_liquidity, diagnostics


def apply_liquidity_shock(
    state: LiquidityState,
    shock_amount: jax.Array,
    runtime: LiquidityRuntime = None
) -> LiquidityState:
    """Apply external liquidity shock (provision or removal).

    This can be used to model external events like:
    - Large liquidity provider entering/exiting
    - Protocol-level liquidity injections
    - Emergency liquidity removal

    Args:
        state: Current state
        shock_amount: Liquidity change (positive = add, negative = remove)
        runtime: Optional runtime for bounds checking

    Returns:
        Updated state with shock applied

    Example:
        >>> state = LiquidityState.initialize(1000000.0)
        >>> # Add 100k liquidity
        >>> new_state = apply_liquidity_shock(state, jnp.array(100000.0))
        >>> print(new_state.liquidity_level)  # 1100000.0
    """
    new_liquidity = state.liquidity_level + shock_amount

    # Apply bounds if runtime provided
    if runtime is not None:
        if runtime.min_liquidity is not None:
            new_liquidity = jnp.maximum(new_liquidity, float(runtime.min_liquidity.value))
        if runtime.max_liquidity is not None:
            new_liquidity = jnp.minimum(new_liquidity, float(runtime.max_liquidity.value))

    # Ensure non-negative
    new_liquidity = jnp.maximum(new_liquidity, 0.0)

    # Track provision/removal
    provision = jnp.maximum(shock_amount, 0.0)
    removal = jnp.maximum(-shock_amount, 0.0)

    return LiquidityState(
        liquidity_level=new_liquidity,
        time=state.time,  # Time doesn't advance for shocks
        cumulative_provision=state.cumulative_provision + provision,
        cumulative_removal=state.cumulative_removal + removal,
        jump_count=state.jump_count,
    )


def deterministic_update(
    runtime: LiquidityRuntime,
    state: LiquidityState,
    dt: jax.Array
) -> Tuple[LiquidityState, jax.Array]:
    """Update liquidity using only deterministic mean reversion (no noise).

    Useful for:
    - Testing and validation
    - Deterministic simulations
    - Understanding baseline dynamics

    Args:
        runtime: Liquidity configuration
        state: Current state
        dt: Time step

    Returns:
        Tuple of (new_state, liquidity)
    """
    mean_liquidity = float(runtime.mean_liquidity.value)
    reversion_rate = float(runtime.mean_reversion_rate.value)

    # Only drift, no diffusion or jumps
    drift = reversion_rate * (mean_liquidity - state.liquidity_level) * dt
    new_liquidity = state.liquidity_level + drift

    # Apply bounds
    if runtime.min_liquidity is not None:
        new_liquidity = jnp.maximum(new_liquidity, float(runtime.min_liquidity.value))
    if runtime.max_liquidity is not None:
        new_liquidity = jnp.minimum(new_liquidity, float(runtime.max_liquidity.value))
    new_liquidity = jnp.maximum(new_liquidity, 0.0)

    # Track changes
    liquidity_change = new_liquidity - state.liquidity_level
    provision = jnp.maximum(liquidity_change, 0.0)
    removal = jnp.maximum(-liquidity_change, 0.0)

    new_state = LiquidityState(
        liquidity_level=new_liquidity,
        time=state.time + dt,
        cumulative_provision=state.cumulative_provision + provision,
        cumulative_removal=state.cumulative_removal + removal,
        jump_count=state.jump_count,
    )

    return new_state, new_liquidity
