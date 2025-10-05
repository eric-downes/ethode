"""Hawkes process module with unit-aware configuration.

This module provides configuration and runtime structures for
self-exciting Hawkes processes used in modeling clustered events.
"""

from .config import HawkesConfig, HawkesConfigOutput
from .runtime import HawkesRuntime, HawkesState

__all__ = [
    'HawkesConfig',
    'HawkesConfigOutput',
    'HawkesRuntime',
    'HawkesState',
]