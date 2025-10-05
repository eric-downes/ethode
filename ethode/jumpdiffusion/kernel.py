"""Jump diffusion kernel functions for JAX.

This module provides JIT-compiled functions for ODE+Jump hybrid simulation.
"""

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Any

from .runtime import JumpDiffusionRuntime, JumpDiffusionState
from ..jumpprocess.kernel import generate_next_jump_time


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
    # Determine integration end time
    next_jump_time = state.jump_state.next_jump_time
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


def apply_jump(
    runtime: JumpDiffusionRuntime,
    state: JumpDiffusionState,
) -> JumpDiffusionState:
    """Apply jump effect and generate next jump time.

    Args:
        runtime: Runtime configuration
        state: State at jump time (before jump)

    Returns:
        State after jump with new next_jump_time
    """
    # Apply jump effect to state
    state_after_jump = runtime.jump_effect_fn(
        state.t,
        state.state,
        runtime.params
    )

    # Generate next jump time
    jump_state_new, _ = generate_next_jump_time(
        runtime.jump_runtime,
        state.jump_state,
        state.t
    )

    # Update state using dataclasses.replace for Penzai structs
    return dataclasses.replace(
        state,
        state=state_after_jump,
        jump_state=jump_state_new,
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

    # Tolerance for time comparisons to avoid spurious iterations from diffrax roundoff
    TIME_ATOL = 1e-9

    # Initialize jump process state
    jump_state_init = JumpProcessState.zero(
        seed=runtime.jump_runtime.seed,
        start_time=t_start
    )
    jump_state_init, _ = generate_next_jump_time(
        runtime.jump_runtime,
        jump_state_init,
        jnp.array(t_start)
    )

    # Initialize simulation state
    sim_state = JumpDiffusionState.zero(
        initial_state,
        jump_state_init,
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
            jump_occurred = jnp.logical_and(
                jnp.isclose(new_state.t, new_state.jump_state.next_jump_time, rtol=0, atol=TIME_ATOL),
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
