"""Ethode: Unit-aware dynamical systems simulation framework."""

from .units import UnitManager, UnitSpec, QuantityInput
from .fields import quantity_field, tuple_quantity_field
from .runtime import (
    QuantityNode,
    ControllerRuntime,
    ControllerState,
    SimulationOutput,
    MarketRuntime,
    SystemRuntime,
)
from .controller import ControllerConfig, ControllerConfigOutput, controller_step

__all__ = [
    # Units
    'UnitManager',
    'UnitSpec',
    'QuantityInput',
    # Fields
    'quantity_field',
    'tuple_quantity_field',
    # Runtime structures
    'QuantityNode',
    'ControllerRuntime',
    'ControllerState',
    'SimulationOutput',
    'MarketRuntime',
    'SystemRuntime',
    # Controller
    'ControllerConfig',
    'ControllerConfigOutput',
    'controller_step',
]