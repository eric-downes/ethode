"""Hawkes process module with unit-aware configuration.

This module provides configuration and runtime structures for
self-exciting Hawkes processes used in modeling clustered events.
"""

from .config import HawkesConfig, HawkesConfigOutput
from .runtime import HawkesRuntime, HawkesState
from .kernel import (
    update_intensity,
    generate_event,
    generate_event_with_diagnostics,
    step,
    get_branching_ratio,
    get_stationary_intensity,
    apply_external_shock,
)

__all__ = [
    'HawkesConfig',
    'HawkesConfigOutput',
    'HawkesRuntime',
    'HawkesState',
    'update_intensity',
    'generate_event',
    'generate_event_with_diagnostics',
    'step',
    'get_branching_ratio',
    'get_stationary_intensity',
    'apply_external_shock',
]