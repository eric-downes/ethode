"""Jump diffusion processes for hybrid ODE+Jump simulation.

This module provides configuration, runtime structures, and kernel functions
for simulating systems with both continuous dynamics (ODEs) and discrete jumps.
"""

from .config import JumpDiffusionConfig, JumpDiffusionConfigOutput
from .runtime import JumpDiffusionRuntime, JumpDiffusionState
from .kernel import integrate_step, apply_jump, simulate

__all__ = [
    'JumpDiffusionConfig',
    'JumpDiffusionConfigOutput',
    'JumpDiffusionRuntime',
    'JumpDiffusionState',
    'integrate_step',
    'apply_jump',
    'simulate',
]
