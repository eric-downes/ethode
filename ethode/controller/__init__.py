"""Controller module for PID control with unit awareness."""

from .config import ControllerConfig, ControllerConfigOutput
from .kernel import controller_step, controller_step_with_diagnostics
from .legacy import PIDController, PIDParams, noise_barrier
from ..runtime import ControllerState, ControllerRuntime

__all__ = [
    # New API
    'ControllerConfig',
    'ControllerConfigOutput',
    'ControllerState',
    'ControllerRuntime',
    'controller_step',
    'controller_step_with_diagnostics',
    # Legacy compatibility
    'PIDController',
    'PIDParams',
    'noise_barrier',
]