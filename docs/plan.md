# Ethode Improvement Plan

Needs revision; we have decided to rebase loops on JAX+penzai so we
can handle units without the awkward workarounds previously used.

## P9 Blockers (fix before rd-sim sweeps)
- Replace `AutoDefault`/`DEFAULTS` (ethode.py:35-83) with explicit `__post_init__` defaults to avoid shared mutable state (`list`, `dict`) and to support type hints without `isinstance(type_, t)` hacks.
- Rework `FinDiffSim.sim` (ethode.py:119-133): `nt` must be an `int`, the update should call `self.func(t, state, params)`, and we need quantity-safe stepping (current call `self.func(data[t-1])` crashes for any subclass).
- Harden magnitude utilities: centralise `mag()`/`wmag()` to handle `pint` + numpy arrays without rebuilding sequences elementwise each call (current recursion is O(n²) for tuples of arrays).
- Provide a vetted PID implementation (reusable by rd-sim) with configurable deadzone, saturation, rate limits, and explicit state reset hooks (current copy in `stochastic_extensions.py` is an ad-hoc duplicate).
- Standardise solve_ivp wrappers: expose tolerances, method selection, and event hooks; current `ODESim` hardcodes `Radau` and does not propagate failure diagnostics needed for automated sweeps.

## Near-term Enhancements
- Define a light-weight simulation base interface: `Sim.run(config, observers)` returning structured artefacts (df, diagnostics) so downstream code stops mutating `self.df` in-place.
- Add an observation/feature registry to `_add_outputs` allowing declarative witness definitions (avoid `f.__code__.co_varnames` heuristics).
- Package unit handling: expose a single `create_unit_registry()` helper and require sims to receive a registry instead of relying on globals.
- Introduce reusable stochastic process primitives (Hawkes, Poisson, compound jumps) with vectorised sampling and deterministic seeding.
- Fill unit tests for all utilities (AutoDefault successor, PID, jump processes) and wire into CI.

## Longer-term
- Provide config schemas (pydantic/dataclasses) for sims to guarantee typed parameters.
- Ship plotting/report helpers (matplotlib/seaborn wrappers) behind optional dependency guard.
- Investigate migrating to JAX/numba for hot loops once semantics stabilise.
