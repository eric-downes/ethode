"""Jump process module with unit-aware configuration.

This module provides configuration and runtime structures for
Poisson and deterministic jump processes.
"""

from .config import JumpProcessConfig, JumpProcessConfigOutput
from .runtime import JumpProcessRuntime, JumpProcessState
from .kernel import (
    generate_next_jump_time,
    check_jump_occurred,
    step,
    generate_jumps_in_interval,
)

__all__ = [
    'JumpProcessConfig',
    'JumpProcessConfigOutput',
    'JumpProcessRuntime',
    'JumpProcessState',
    'generate_next_jump_time',
    'check_jump_occurred',
    'step',
    'generate_jumps_in_interval',
]
