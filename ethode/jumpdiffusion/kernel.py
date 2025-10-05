"""Jump diffusion kernel functions for JAX.

This module provides JIT-compiled functions for ODE+Jump hybrid simulation.
"""

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Any

from .runtime import JumpDiffusionRuntime, JumpDiffusionState, ScheduledJumpBuffer
from ..jumpprocess import JumpProcessState
from ..jumpprocess.kernel import generate_next_jump_time
from ..hawkes import HawkesRuntime


def integrate_step(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
    t_end: jax.Array,
) -> Tuple[JumpDiffusionState, jax.Array]:
    """Integrate ODE from current time until next jump or t_end.

    Args:
        runtime: Runtime configuration
        state: Current simulation state
        t_end: Maximum time to integrate to

    Returns:
        (updated_state, time_reached)
        - updated_state: State after integration
        - time_reached: Actual time reached (min of next_jump_time, t_end)
    """
    # Determine integration end time (read from buffer - same for all modes)
    idx = state.jump_buffer.next_index
    next_jump_time = state.jump_buffer.event_times[idx]
    t_target = jnp.minimum(next_jump_time, t_end)

    # Integrate ODE from state.t to t_target
    # Use runtime.solver_type to select solver
    state_new = _ode_integrate(
        runtime.dynamics_fn,
        state.state,
        state.t,
        t_target,
        runtime.dt_max.value,
        runtime.rtol,
        runtime.atol,
        runtime.params,
        runtime.solver_type,
    )

    # Update state using dataclasses.replace for Penzai structs
    updated_state = dataclasses.replace(
        state,
        t=t_target,
        state=state_new,
        step_count=state.step_count + 1,
    )

    return updated_state, t_target


def _generate_poisson_event(
    runtime,
    current_t: jax.Array,
    rng_key: jax.Array
) -> Tuple[float, jax.Array]:
    """Generate next Poisson event time."""
    # Create temporary state (only for interface compatibility)
    temp_state = JumpProcessState(
        last_jump_time=current_t,
        next_jump_time=current_t,  # Will be overwritten
        rng_key=rng_key,
        event_count=jnp.array(0, dtype=jnp.int32)
    )

    new_state, next_time = generate_next_jump_time(runtime, temp_state, current_t)
    return next_time, new_state.rng_key


def _default_exponential_decay(E: Any, dt: jax.Array, hawkes: HawkesRuntime) -> Any:
    """Default excitation decay: E(t+dt) = E(t) * exp(-beta * dt).

    Args:
        E: Current cumulative excitation (can be pytree)
        dt: Time since last event
        hawkes: HawkesRuntime containing excitation_decay parameter

    Returns:
        Decayed excitation value
    """
    beta = hawkes.excitation_decay.value
    # Handle both scalar and pytree E
    if isinstance(E, dict):
        # Pytree case (e.g., power-law with elapsed time)
        return {k: (v * jnp.exp(-beta * dt) if k == "value" else v) for k, v in E.items()}
    else:
        # Scalar case
        return E * jnp.exp(-beta * dt)


def _default_unit_jump(E: Any, hawkes: HawkesRuntime) -> Any:
    """Default excitation jump: E(t+) = E(t-) + 1.

    Args:
        E: Cumulative excitation before event (can be pytree)
        hawkes: HawkesRuntime (unused in default)

    Returns:
        Excitation after event
    """
    if isinstance(E, dict):
        # Pytree case
        return {k: (v + 1.0 if k == "value" else v) for k, v in E.items()}
    else:
        # Scalar case
        return E + 1.0


def _default_linear_intensity(lambda_0: jax.Array, E: Any, hawkes: HawkesRuntime) -> jax.Array:
    """Default intensity: lambda(t) = lambda_0 + alpha * E(t).

    Args:
        lambda_0: State-dependent base rate
        E: Current cumulative excitation (can be pytree)
        hawkes: HawkesRuntime containing excitation_strength parameter

    Returns:
        Current intensity
    """
    alpha = hawkes.excitation_strength.value
    # Extract value from E if it's a pytree
    E_value = E["value"] if isinstance(E, dict) else E
    return lambda_0 + alpha * E_value


def _generate_hawkes_event_online(
    scheduler,
    cumulative_excitation: Any,
    current_t: jax.Array,
    ode_state: jax.Array,
    rng_key: jax.Array,
    max_rejections: int = 1000
) -> Tuple[float, jax.Array, Any]:
    """Generate next Hawkes event using JAX-compatible thinning with bounded loop.

    Uses cumulative excitation accumulator E(t) instead of reconstructing from history.
    Implements Ogata's thinning algorithm with jax.lax.while_loop.

    Args:
        scheduler: JumpSchedulerRuntime with static callable fields
        cumulative_excitation: Current value of E(t) accumulator (can be pytree)
        current_t: Current simulation time
        ode_state: Current ODE state vector (for state-dependent λ₀)
        rng_key: JAX PRNG key
        max_rejections: Safety limit on thinning iterations

    Returns:
        Tuple of:
        - next_event_time: Time of next event (jnp.inf if max_rejections hit)
        - new_rng_key: Updated PRNG key
        - updated_excitation: E(t) value at next_event_time
    """
    # Get user-provided functions or defaults from static fields
    lambda_0_fn = scheduler.lambda_0_fn or (lambda s: scheduler.hawkes.jump_rate.value)
    decay_fn = scheduler.excitation_decay_fn or _default_exponential_decay
    intensity_fn = scheduler.intensity_fn or _default_linear_intensity

    # Compute state-dependent base rate
    lambda_0 = lambda_0_fn(ode_state)

    def cond_fn(carry):
        """Continue while not accepted and iterations < max."""
        _, _, _, accepted, iter_count = carry
        return jnp.logical_and(
            jnp.logical_not(accepted),
            iter_count < max_rejections
        )

    def body_fn(carry):
        """Single thinning iteration: propose candidate, accept/reject."""
        rng, t_candidate, E_current, _, iter_count = carry

        # Current intensity at candidate time
        lambda_current = intensity_fn(lambda_0, E_current, scheduler.hawkes)

        # Sample inter-event time using current intensity as upper bound
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        dt = jax.random.exponential(subkey1) / lambda_current
        t_next = t_candidate + dt

        # Decay excitation to t_next
        E_next = decay_fn(E_current, dt, scheduler.hawkes)

        # Intensity at proposed time
        lambda_next = intensity_fn(lambda_0, E_next, scheduler.hawkes)

        # Accept/reject using thinning (accept if λ(t_next) / λ_current >= U)
        u = jax.random.uniform(subkey2)
        accepted = u * lambda_current <= lambda_next

        return (rng, t_next, E_next, accepted, iter_count + 1)

    # Run bounded thinning loop
    init = (rng_key, current_t, cumulative_excitation, False, 0)
    final_rng, final_t, final_E, final_accepted, final_iter = jax.lax.while_loop(
        cond_fn, body_fn, init
    )

    # Fail-safe: if max_rejections hit, return inf (no more events in this simulation)
    t_event = jnp.where(final_accepted, final_t, jnp.inf)

    return t_event, final_rng, final_E


def apply_jump(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
) -> JumpDiffusionState:
    """Apply jump effect and generate next event (mode-dependent).

    This is the ONLY function that differs by mode:
    - Mode 0: Generate next Poisson event
    - Mode 1: Advance buffer pointer (events pre-generated)
    - Mode 2: Generate next Hawkes event with state-dependent λ₀

    Args:
        runtime: Runtime configuration
        state: State at jump time (before jump)

    Returns:
        State after jump with next event time
    """
    # Apply jump effect to state (same for all modes)
    state_after_jump = runtime.jump_effect_fn(
        state.t,
        state.state,
        runtime.params
    )

    # Generate/retrieve next event time based on mode
    mode = runtime.scheduler.mode
    next_idx = state.jump_buffer.next_index
    buffer_capacity = state.jump_buffer.count

    if mode == 0:
        # Mode 0: Poisson - lazy generation
        next_time, new_key = _generate_poisson_event(
            runtime.scheduler.scheduled,
            state.t,
            state.jump_buffer.rng_key
        )
        new_buffer = dataclasses.replace(
            state.jump_buffer,
            event_times=state.jump_buffer.event_times.at[next_idx + 1].set(next_time),
            next_index=next_idx + 1,
            rng_key=new_key
        )
    elif mode == 1:
        # Mode 1: Pre-generated Hawkes - just advance pointer
        new_buffer = dataclasses.replace(
            state.jump_buffer,
            next_index=next_idx + 1
        )
    else:  # mode == 2
        # Mode 2: Online Hawkes - lazy generation with cumulative excitation

        # Decay excitation from last update to now
        dt_since_last = state.t - state.jump_buffer.last_update_time
        decay_fn = runtime.scheduler.excitation_decay_fn or _default_exponential_decay
        decayed_excitation = decay_fn(
            state.jump_buffer.cumulative_excitation,
            dt_since_last,
            runtime.scheduler.hawkes
        )

        # Add excitation from this event
        jump_fn = runtime.scheduler.excitation_jump_fn or _default_unit_jump
        new_excitation = jump_fn(decayed_excitation, runtime.scheduler.hawkes)

        # Generate next event using cumulative excitation
        next_time, new_key, final_excitation = _generate_hawkes_event_online(
            runtime.scheduler,
            cumulative_excitation=new_excitation,
            current_t=state.t,
            ode_state=state.state,  # ← Access current ODE state!
            rng_key=state.jump_buffer.rng_key,
            max_rejections=int(runtime.scheduler.hawkes_max_events)
        )

        # Apply buffer overflow guard: if full, set next_time to inf (stop generating)
        next_time = jax.lax.cond(
            next_idx + 1 < buffer_capacity,
            lambda: next_time,
            lambda: jnp.inf  # Buffer full - no more events
        )

        # Update buffer with cumulative excitation fields
        new_buffer = dataclasses.replace(
            state.jump_buffer,
            event_times=state.jump_buffer.event_times.at[next_idx + 1].set(next_time),
            next_index=next_idx + 1,
            rng_key=new_key,
            cumulative_excitation=final_excitation,
            last_update_time=state.t
        )

    return dataclasses.replace(
        state,
        state=state_after_jump,
        jump_buffer=new_buffer,
        jump_count=state.jump_count + 1,
    )


def simulate(
    runtime: JumpDiffusionRuntime,
    initial_state: jax.Array,
    t_span: Tuple[float, float],
    max_steps: int = 100000,
) -> Tuple[jax.Array, jax.Array]:
    """Run full ODE+Jump simulation using jax.lax.scan for JIT compatibility.

    Saves state at: initial time, each jump time, and final time.
    Does NOT save at every ODE integration step (see Future Extensions for dense_output).

    Args:
        runtime: Runtime configuration
        initial_state: Initial state vector
        t_span: (t_start, t_end) simulation time span
        max_steps: Maximum number of steps (safety limit for total saves)

    Returns:
        (times, states)
        - times: Array of time points [shape: (n_saves,)] padded with inf
        - states: Array of states [shape: (n_saves,) + state_shape] padded with final state

    Note:
        - Output is padded to max_steps length. Filter by `times < t_end` to get actual trajectory.
        - Current implementation: saves only at jump times and t_end
        - Future: dense_output (save every ODE step) and custom save_at times
    """
    from ..jumpprocess.runtime import JumpProcessState

    t_start, t_end = t_span
    MODE2_BUFFER_SIZE = 100  # Fixed small buffer for Mode 2

    # Tolerance for time comparisons to avoid spurious iterations from diffrax roundoff
    TIME_ATOL = 1e-9

    # Initialize RNG key
    rng_key = jax.random.PRNGKey(int(runtime.scheduler.seed))

    # Initialize jump buffer based on scheduler mode
    mode = runtime.scheduler.mode

    if mode == 0:
        # Mode 0 (Poisson): Lazy - initialize empty buffer with first event
        key1, key2 = jax.random.split(rng_key)
        first_event, new_key = _generate_poisson_event(
            runtime.scheduler.scheduled,
            jnp.array(t_start),
            key1
        )

        event_times = jnp.full(max_steps, jnp.inf, dtype=initial_state.dtype)
        event_times = event_times.at[0].set(first_event)

        jump_buffer = ScheduledJumpBuffer(
            event_times=event_times,
            count=jnp.array(max_steps, dtype=jnp.int32),  # Capacity
            next_index=jnp.array(0, dtype=jnp.int32),
            rng_key=new_key,
            cumulative_excitation=jnp.array(0.0, dtype=initial_state.dtype),  # Not used for mode 0
            last_update_time=jnp.array(t_start, dtype=initial_state.dtype)
        )

    elif mode == 1:
        # Mode 1 (Pre-gen Hawkes): Pre-generate all events upfront
        from ..hawkes.scheduler import generate_schedule

        # Extract float from hawkes_dt to make it static for JIT
        # (scan requires static length, which depends on dt being static)
        hawkes_dt_value = float(runtime.scheduler.hawkes_dt)

        events, _ = generate_schedule(
            runtime.scheduler.hawkes,
            t_span,
            hawkes_dt_value,
            int(runtime.scheduler.hawkes_max_events),
            int(runtime.scheduler.seed),
            dtype=initial_state.dtype  # Match dtype to avoid upcasting
        )

        # Count valid events
        valid_mask = events < t_end
        count = jnp.sum(valid_mask.astype(jnp.int32))

        jump_buffer = ScheduledJumpBuffer(
            event_times=events,
            count=count,
            next_index=jnp.array(0, dtype=jnp.int32),
            rng_key=rng_key,  # Not used for mode 1
            cumulative_excitation=jnp.array(0.0, dtype=initial_state.dtype),  # Not used for mode 1
            last_update_time=jnp.array(t_start, dtype=initial_state.dtype)
        )

    else:  # mode == 2
        # Mode 2 (Online Hawkes): Lazy generation with cumulative excitation accumulator
        key1, key2 = jax.random.split(rng_key)

        # Initialize with zero excitation (no past events)
        initial_excitation = jnp.array(0.0, dtype=initial_state.dtype)

        # Generate first event
        first_event, new_key, first_excitation = _generate_hawkes_event_online(
            runtime.scheduler,
            cumulative_excitation=initial_excitation,
            current_t=jnp.array(t_start),
            ode_state=initial_state,
            rng_key=key1,
            max_rejections=int(runtime.scheduler.hawkes_max_events)
        )

        # Mode 2: Fixed buffer size of 100 (small constant for lazy filling)
        event_times = jnp.full(MODE2_BUFFER_SIZE, jnp.inf, dtype=initial_state.dtype)
        event_times = event_times.at[0].set(first_event)

        jump_buffer = ScheduledJumpBuffer(
            event_times=event_times,
            count=jnp.array(MODE2_BUFFER_SIZE, dtype=jnp.int32),  # Capacity (not max_events!)
            next_index=jnp.array(0, dtype=jnp.int32),
            rng_key=new_key,
            cumulative_excitation=first_excitation,
            last_update_time=jnp.array(t_start, dtype=initial_state.dtype)
        )

    # Initialize simulation state with buffer
    sim_state = JumpDiffusionState.zero(
        initial_state,
        jump_buffer,
        t0=t_start
    )

    # Pre-allocate arrays for results
    # Note: Use full shape + dtype to handle arbitrary state dimensions
    times = jnp.full(max_steps, jnp.inf)
    states = jnp.zeros((max_steps,) + initial_state.shape, dtype=initial_state.dtype)

    # Set initial values
    times = times.at[0].set(t_start)
    states = states.at[0].set(initial_state)

    def scan_fn(carry, _):
        """Single scan step: integrate + possibly jump."""
        sim_state, times, states, idx = carry

        # Stop if we've reached t_end
        def continue_sim(_):
            # Integrate to next jump or t_end
            new_state, t_reached = integrate_step(runtime, sim_state, jnp.array(t_end))

            # Check if we hit a jump
            # Use tolerance for both jump detection and end-time check to avoid roundoff issues
            idx = new_state.jump_buffer.next_index
            next_jump_time = new_state.jump_buffer.event_times[idx]
            jump_occurred = jnp.logical_and(
                jnp.isclose(new_state.t, next_jump_time, rtol=0, atol=TIME_ATOL),
                new_state.t < t_end - TIME_ATOL
            )

            # Apply jump if occurred
            new_state = jax.lax.cond(
                jump_occurred,
                lambda s: apply_jump(runtime, s),
                lambda s: s,
                new_state
            )

            # Save state if we jumped or reached end (with tolerance)
            should_save = jnp.logical_or(jump_occurred, new_state.t >= t_end - TIME_ATOL)

            # Guard against overflow: only save if idx+1 < max_steps
            can_save = idx + 1 < max_steps
            should_save_safe = jnp.logical_and(should_save, can_save)

            new_idx = jax.lax.cond(
                should_save_safe,
                lambda _: idx + 1,
                lambda _: idx,
                None
            )

            new_times = jax.lax.cond(
                should_save_safe,
                lambda _: times.at[new_idx].set(new_state.t),
                lambda _: times,
                None
            )

            new_states = jax.lax.cond(
                should_save_safe,
                lambda _: states.at[new_idx].set(new_state.state),
                lambda _: states,
                None
            )

            return new_state, new_times, new_states, new_idx

        def stop_sim(_):
            # Already at t_end, don't modify anything
            return sim_state, times, states, idx

        # Only continue if current time < t_end (with tolerance to avoid spurious iterations)
        new_state, new_times, new_states, new_idx = jax.lax.cond(
            sim_state.t < t_end - TIME_ATOL,
            continue_sim,
            stop_sim,
            None
        )

        return (new_state, new_times, new_states, new_idx), None

    # Run scan
    (final_state, final_times, final_states, final_idx), _ = jax.lax.scan(
        scan_fn,
        (sim_state, times, states, jnp.array(0, dtype=jnp.int32)),
        None,
        length=max_steps
    )

    # Post-simulation check: Mode 2 buffer overflow detection
    if mode == 2:
        final_idx_buffer = int(final_state.jump_buffer.next_index)
        buffer_capacity = int(final_state.jump_buffer.count)

        if final_idx_buffer >= buffer_capacity - 1:
            raise RuntimeError(
                f"Mode 2 buffer overflow: Simulation exhausted {buffer_capacity}-element buffer. "
                f"This indicates extremely rapid jumps or pathological simulation. "
                f"Possible causes: (1) Intensity too high, (2) Stuck in loop, (3) Bug in kernel. "
                f"Consider: Reducing excitation_strength, checking lambda_0_fn, or using Mode 1."
            )

    return final_times, final_states


def _ode_integrate(
    dynamics_fn: Callable,
    y0: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    dt_max: float,
    rtol: float,
    atol: float,
    params: Any,
    solver_type: int,
) -> jax.Array:
    """Internal ODE integration using diffrax.

    Args:
        dynamics_fn: Right-hand side function
        y0: Initial state
        t0: Start time
        t1: End time
        dt_max: Maximum step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        params: User parameters
        solver_type: Solver identifier (0=euler, 1=rk4, 2=dopri5, 3=dopri8)

    Returns:
        Final state at t1
    """
    import diffrax

    # Map solver type int to diffrax solver
    # 0=euler, 1=rk4 (Tsit5), 2=dopri5, 3=dopri8
    solvers = [
        diffrax.Euler(),
        diffrax.Tsit5(),  # Using Tsit5 as RK4 alternative (5th order)
        diffrax.Dopri5(),
        diffrax.Dopri8(),
    ]
    solver = solvers[solver_type]

    # Define ODE term
    def vector_field(t, y, args):
        return dynamics_fn(t, y, params)

    term = diffrax.ODETerm(vector_field)

    # Choose appropriate step size controller
    # Euler doesn't have error estimates, so use constant step size
    if solver_type == 0:  # Euler
        stepsize_controller = diffrax.ConstantStepSize()
    else:  # Adaptive methods (Tsit5, Dopri5, Dopri8)
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # Solve
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt_max,
        y0=y0,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t1=True),
    )

    return solution.ys[-1]
