"""Controller module for PID control with unit awareness."""

from .config import ControllerConfig, ControllerConfigOutput
from .dimensions import (
    ControllerDimensions,
    DIMENSIONLESS,
    FINANCIAL,
    TEMPERATURE,
    VELOCITY,
    POSITION
)
from .kernel import controller_step, controller_step_with_diagnostics
from .legacy import PIDController, PIDParams, noise_barrier
from ..runtime import ControllerState, ControllerRuntime
from ..adapters import ControllerAdapter

__all__ = [
    # New API (core tier)
    'ControllerConfig',
    'ControllerConfigOutput',
    'ControllerDimensions',
    'ControllerState',
    'ControllerRuntime',
    'controller_step',
    'controller_step_with_diagnostics',
    # High-level adapter (recommended for day-to-day use)
    'ControllerAdapter',
    # Dimension schemas
    'DIMENSIONLESS',
    'FINANCIAL',
    'TEMPERATURE',
    'VELOCITY',
    'POSITION',
    # Legacy compatibility (deprecated)
    'PIDController',
    'PIDParams',
    'noise_barrier',
]