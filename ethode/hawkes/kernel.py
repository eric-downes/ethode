"""Hawkes process kernel functions for JAX.

This module provides JIT-compiled functions for self-exciting Hawkes processes.
Implements event generation with exponentially decaying excitation.
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from .runtime import HawkesRuntime, HawkesState


def update_intensity(
    runtime: HawkesRuntime,
    state: HawkesState,
    dt: jax.Array
) -> Tuple[HawkesState, jax.Array]:
    """Update Hawkes intensity with time decay.

    Updates the current intensity by decaying past excitation:
        λ(t + dt) = λ₀ + (λ(t) - λ₀) * exp(-dt / τ)

    Args:
        runtime: Hawkes configuration
        state: Current state
        dt: Time step

    Returns:
        Tuple of (new_state, current_intensity)

    Example:
        >>> config = HawkesConfig(
        ...     jump_rate="100 / hour",
        ...     excitation_strength=0.3,
        ...     excitation_decay="5 minutes"
        ... )
        >>> runtime = config.to_runtime()
        >>> state = HawkesState.initialize(100.0 / 3600.0)
        >>> new_state, intensity = update_intensity(runtime, state, jnp.array(1.0))
    """
    base_rate = float(runtime.jump_rate.value)
    decay_time = float(runtime.excitation_decay.value)

    # Exponential decay toward base rate
    # λ(t+dt) = λ₀ + (λ(t) - λ₀) * exp(-dt/τ)
    decay_factor = jnp.exp(-dt / decay_time)
    new_intensity = base_rate + (state.current_intensity - base_rate) * decay_factor

    # Apply bounds if configured
    if runtime.min_intensity is not None:
        new_intensity = jnp.maximum(new_intensity, float(runtime.min_intensity.value))
    if runtime.max_intensity is not None:
        new_intensity = jnp.minimum(new_intensity, float(runtime.max_intensity.value))

    # Update state
    new_state = HawkesState(
        current_intensity=new_intensity,
        time=state.time + dt,
        event_count=state.event_count,
        last_event_time=state.last_event_time,
        cumulative_impact=state.cumulative_impact,
    )

    return new_state, new_intensity


def generate_event(
    runtime: HawkesRuntime,
    state: HawkesState,
    key: jax.Array,
    dt: jax.Array
) -> Tuple[HawkesState, bool, jax.Array]:
    """Check if an event occurs and update state accordingly.

    Uses thinning algorithm: event occurs with probability λ(t) * dt
    for small dt.

    Args:
        runtime: Hawkes configuration
        state: Current state
        key: PRNG key
        dt: Time step (should be small for accurate simulation)

    Returns:
        Tuple of (new_state, event_occurred, event_impact)
    """
    # First decay intensity
    state, current_intensity = update_intensity(runtime, state, dt)

    # Check if event occurs (Poisson process with rate λ(t))
    # For small dt: P(event) ≈ λ(t) * dt
    event_prob = current_intensity * dt
    key, subkey = jax.random.split(key)
    event_occurred = jax.random.uniform(subkey) < event_prob

    # Calculate event impact if it occurred
    event_impact = jnp.array(0.0)
    if runtime.event_impact_mean is not None:
        key, subkey = jax.random.split(key)
        impact_mean = float(runtime.event_impact_mean.value)
        impact_std = float(runtime.event_impact_std.value) if runtime.event_impact_std else 0.0
        event_impact = jnp.where(
            event_occurred,
            impact_mean + impact_std * jax.random.normal(subkey),
            0.0
        )

    # If event occurred, increase intensity (self-excitation)
    excitation = float(runtime.excitation_strength.value)
    base_rate = float(runtime.jump_rate.value)

    # Add excitation: λ -> λ + α * λ₀
    new_intensity = jnp.where(
        event_occurred,
        state.current_intensity + excitation * base_rate,
        state.current_intensity
    )

    # Apply bounds
    if runtime.max_intensity is not None:
        new_intensity = jnp.minimum(new_intensity, float(runtime.max_intensity.value))

    # Update state
    new_state = HawkesState(
        current_intensity=new_intensity,
        time=state.time,
        event_count=state.event_count + jnp.where(event_occurred, 1, 0),
        last_event_time=jnp.where(event_occurred, state.time, state.last_event_time),
        cumulative_impact=state.cumulative_impact + event_impact,
    )

    return new_state, bool(event_occurred), event_impact


def generate_event_with_diagnostics(
    runtime: HawkesRuntime,
    state: HawkesState,
    key: jax.Array,
    dt: jax.Array
) -> Tuple[HawkesState, bool, jax.Array, dict]:
    """Generate event with diagnostic information.

    Args:
        runtime: Hawkes configuration
        state: Current state
        key: PRNG key
        dt: Time step

    Returns:
        Tuple of (new_state, event_occurred, event_impact, diagnostics)

    Diagnostics include:
        - intensity_before_decay: Intensity before time decay
        - intensity_after_decay: Intensity after decay
        - event_probability: Probability of event in this timestep
        - excitation_added: Amount of excitation added (if event occurred)
        - intensity_after_event: Final intensity
    """
    intensity_before = float(state.current_intensity)

    # Decay intensity
    state_decayed, intensity_after_decay = update_intensity(runtime, state, dt)

    # Event probability
    event_prob = float(intensity_after_decay * dt)

    # Check for event
    key, subkey = jax.random.split(key)
    event_occurred = jax.random.uniform(subkey) < event_prob

    # Calculate impact
    event_impact = jnp.array(0.0)
    if runtime.event_impact_mean is not None:
        key, subkey = jax.random.split(key)
        impact_mean = float(runtime.event_impact_mean.value)
        impact_std = float(runtime.event_impact_std.value) if runtime.event_impact_std else 0.0
        event_impact = jnp.where(
            event_occurred,
            impact_mean + impact_std * jax.random.normal(subkey),
            0.0
        )

    # Self-excitation
    excitation = float(runtime.excitation_strength.value)
    base_rate = float(runtime.jump_rate.value)
    excitation_added = jnp.where(event_occurred, excitation * base_rate, 0.0)

    new_intensity = state_decayed.current_intensity + excitation_added

    # Apply bounds
    if runtime.max_intensity is not None:
        new_intensity = jnp.minimum(new_intensity, float(runtime.max_intensity.value))

    # Update state
    new_state = HawkesState(
        current_intensity=new_intensity,
        time=state_decayed.time,
        event_count=state_decayed.event_count + jnp.where(event_occurred, 1, 0),
        last_event_time=jnp.where(event_occurred, state_decayed.time, state_decayed.last_event_time),
        cumulative_impact=state_decayed.cumulative_impact + event_impact,
    )

    diagnostics = {
        'intensity_before_decay': float(intensity_before),
        'intensity_after_decay': float(intensity_after_decay),
        'event_probability': float(event_prob),
        'event_occurred': bool(event_occurred),
        'excitation_added': float(excitation_added),
        'intensity_after_event': float(new_intensity),
        'event_impact': float(event_impact),
        'time_since_last_event': float(state.time - state.last_event_time),
    }

    return new_state, bool(event_occurred), event_impact, diagnostics


def step(
    runtime: HawkesRuntime,
    state: HawkesState,
    key: jax.Array,
    dt: jax.Array
) -> Tuple[HawkesState, jax.Array]:
    """Single step of Hawkes process (decay + potential event).

    This is the main stepping function that combines intensity decay
    and event generation.

    Args:
        runtime: Hawkes configuration
        state: Current state
        key: PRNG key
        dt: Time step

    Returns:
        Tuple of (new_state, current_intensity)
    """
    new_state, event_occurred, event_impact = generate_event(runtime, state, key, dt)
    return new_state, new_state.current_intensity


def get_branching_ratio(runtime: HawkesRuntime) -> float:
    """Calculate the branching ratio of the Hawkes process.

    The branching ratio is the average number of events triggered by
    a single event. For stability, this must be < 1.

    Args:
        runtime: Hawkes configuration

    Returns:
        Branching ratio (dimensionless)

    Note:
        For a simple Hawkes process with exponential decay:
        branching_ratio = α (excitation_strength)

        The process is:
        - Subcritical (stable) if α < 1
        - Critical if α = 1
        - Supercritical (explosive) if α > 1
    """
    return float(runtime.excitation_strength.value)


def get_stationary_intensity(runtime: HawkesRuntime) -> float:
    """Calculate stationary (equilibrium) intensity.

    For a stable Hawkes process (α < 1), the long-term average intensity is:
        λ_∞ = λ₀ / (1 - α)

    Args:
        runtime: Hawkes configuration

    Returns:
        Stationary intensity (events/time)

    Raises:
        ValueError: If process is unstable (α >= 1)
    """
    alpha = float(runtime.excitation_strength.value)
    if alpha >= 1.0:
        raise ValueError(
            f"Process is unstable (α={alpha} >= 1). "
            "Stationary intensity is undefined."
        )

    base_rate = float(runtime.jump_rate.value)
    return base_rate / (1.0 - alpha)


def apply_external_shock(
    state: HawkesState,
    intensity_boost: jax.Array,
    runtime: HawkesRuntime = None
) -> HawkesState:
    """Apply external intensity shock.

    Models external events that temporarily increase process intensity,
    such as news announcements, large trades, or regulatory changes.

    Args:
        state: Current state
        intensity_boost: Amount to add to current intensity
        runtime: Optional runtime for bounds checking

    Returns:
        Updated state with boosted intensity

    Example:
        >>> # Simulate news shock
        >>> state = HawkesState.initialize(100.0)
        >>> shocked_state = apply_external_shock(state, jnp.array(500.0))
    """
    new_intensity = state.current_intensity + intensity_boost

    # Apply bounds if runtime provided
    if runtime is not None:
        if runtime.min_intensity is not None:
            new_intensity = jnp.maximum(new_intensity, float(runtime.min_intensity.value))
        if runtime.max_intensity is not None:
            new_intensity = jnp.minimum(new_intensity, float(runtime.max_intensity.value))

    return HawkesState(
        current_intensity=new_intensity,
        time=state.time,
        event_count=state.event_count,
        last_event_time=state.last_event_time,
        cumulative_impact=state.cumulative_impact,
    )
