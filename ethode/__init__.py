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
from .controller import (
    ControllerConfig,
    ControllerConfigOutput,
    ControllerDimensions,
    controller_step,
    DIMENSIONLESS,
    FINANCIAL,
    TEMPERATURE,
    VELOCITY,
    POSITION,
)
from .fee import (
    FeeConfig,
    FeeConfigOutput,
    FeeRuntime,
    FeeState,
    calculate_fee,
)
from .liquidity import (
    LiquiditySDEConfig,
    LiquiditySDEConfigOutput,
    LiquidityRuntime,
    LiquidityState,
    update_liquidity,
)
from .hawkes import (
    HawkesConfig,
    HawkesConfigOutput,
    HawkesRuntime,
    HawkesState,
    generate_event,
)
from .jumpprocess import (
    JumpProcessConfig,
    JumpProcessConfigOutput,
    JumpProcessRuntime,
    JumpProcessState,
    generate_jumps_in_interval,
)
from .adapters import (
    ControllerAdapter,
    FeeAdapter,
    LiquidityAdapter,
    HawkesAdapter,
    JumpProcessAdapter,
)
from .simulation import (
    Simulation,
    simulate_controller_step,
)

# Lazy import for legacy API to avoid deprecation warnings on every import
# Users who need legacy API must explicitly import from ethode.legacy
import sys as _sys
import importlib as _importlib

def __getattr__(name):
    """Lazy loading for legacy attributes to avoid deprecation warnings."""
    # List of legacy attributes
    legacy_attrs = {
        'U', 'ETH', 'Yr', 'One', 'mag', 'wmag', 'output',
        'dataclass', 'field', 'AutoDefault', 'Params', 'Sim',
        'FinDiffParams', 'FinDiffSim', 'ODESim'
    }

    if name in legacy_attrs:
        # Lazy load the legacy module only when accessed
        if 'ethode.legacy' not in _sys.modules:
            _importlib.import_module('ethode.legacy')
        return getattr(_sys.modules['ethode.legacy'], name)

    raise AttributeError(f"module 'ethode' has no attribute '{name}'")

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
    # Controller (core tier)
    'ControllerConfig',
    'ControllerConfigOutput',
    'ControllerDimensions',
    'controller_step',
    # Fee (core tier)
    'FeeConfig',
    'FeeConfigOutput',
    'FeeRuntime',
    'FeeState',
    'calculate_fee',
    # Liquidity (core tier)
    'LiquiditySDEConfig',
    'LiquiditySDEConfigOutput',
    'LiquidityRuntime',
    'LiquidityState',
    'update_liquidity',
    # Hawkes (core tier)
    'HawkesConfig',
    'HawkesConfigOutput',
    'HawkesRuntime',
    'HawkesState',
    'generate_event',
    # JumpProcess (core tier)
    'JumpProcessConfig',
    'JumpProcessConfigOutput',
    'JumpProcessRuntime',
    'JumpProcessState',
    'generate_jumps_in_interval',
    # Adapters (high-level tier)
    'ControllerAdapter',
    'FeeAdapter',
    'LiquidityAdapter',
    'HawkesAdapter',
    'JumpProcessAdapter',
    # Simulation facade
    'Simulation',
    'simulate_controller_step',
    # Dimension schemas
    'DIMENSIONLESS',
    'FINANCIAL',
    'TEMPERATURE',
    'VELOCITY',
    'POSITION',
    # Legacy (deprecated) - available via lazy loading
    'U', 'ETH', 'Yr', 'One', 'mag', 'wmag', 'output',
    'dataclass', 'field', 'AutoDefault', 'Params', 'Sim',
    'FinDiffParams', 'FinDiffSim', 'ODESim',
]