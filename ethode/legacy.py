"""Legacy imports from the old ethode.py file.

This module provides backward compatibility for code that expects
imports from the old ethode.py file.
"""

import warnings

# Import everything from the old ethode.py file
import sys
import importlib.util
from pathlib import Path

# Load the old ethode.py file (one directory up from this file)
ethode_py_path = Path(__file__).parent.parent / "ethode.py"
spec = importlib.util.spec_from_file_location("ethode_legacy", str(ethode_py_path))
if spec and spec.loader:
    ethode_legacy = importlib.util.module_from_spec(spec)
    sys.modules['ethode_legacy'] = ethode_legacy
    try:
        spec.loader.exec_module(ethode_legacy)

        # Export commonly used items
        U = ethode_legacy.U
        ETH = ethode_legacy.ETH
        Yr = ethode_legacy.Yr
        One = ethode_legacy.One
        mag = ethode_legacy.mag
        wmag = ethode_legacy.wmag
        output = ethode_legacy.output
        dataclass = ethode_legacy.dataclass
        field = ethode_legacy.field
        AutoDefault = ethode_legacy.AutoDefault
        Params = ethode_legacy.Params
        Sim = ethode_legacy.Sim
        FinDiffParams = ethode_legacy.FinDiffParams
        FinDiffSim = ethode_legacy.FinDiffSim
        ODESim = ethode_legacy.ODESim

    except Exception as e:
        warnings.warn(f"Could not load legacy ethode module: {e}", ImportWarning)
        # Define fallbacks
        U = None
        ETH = None
        Yr = None
        One = None
        mag = None
        wmag = None
        output = None
        dataclass = None
        field = None
        AutoDefault = None
        Params = None
        Sim = None
        FinDiffParams = None
        FinDiffSim = None
        ODESim = None

__all__ = [
    'U', 'ETH', 'Yr', 'One', 'mag', 'wmag', 'output',
    'dataclass', 'field', 'AutoDefault', 'Params', 'Sim',
    'FinDiffParams', 'FinDiffSim', 'ODESim'
]