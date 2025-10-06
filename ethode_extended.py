"""
Extended ethode framework with flexible unit loading and other improvements
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence, Optional, Dict, Any
from collections.abc import Iterator

from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pint

# Import the new PIDController for backward compatibility
from ethode.controller import PIDController

Num = np.float64|float|int
Nums = tuple[Num, ...]
Q = pint.Quantity
Qs = tuple[Q, ...]

# Create unit registry without auto-loading
def create_unit_registry(auto_load_eth_units: bool = True) -> pint.UnitRegistry:
    """Create a unit registry with optional ETH units loading"""
    registry = pint.UnitRegistry()
    if auto_load_eth_units:
        try:
            registry.load_definitions('eth_units.txt')
        except FileNotFoundError:
            pass  # Gracefully handle missing file
    return registry

# Default registry for backward compatibility
U = create_unit_registry(auto_load_eth_units=True)

# Common units - only define if successfully loaded
One = 1 * U.dimensionless
Yr = 1 * U.years if hasattr(U, 'years') else None
ETH = 1 * U.ETH if hasattr(U, 'ETH') else None
ETH_Data = tuple[tuple[str, Q],...]

DEFAULTS = {
    bool: False,
    str: '',
    list: [],
    dict: {},
    None: None,
    int|float|np.float64: 0,
    pint.Quantity: 1e-3,
}

# Add ETH-specific defaults only if ETH units loaded
if ETH is not None:
    DEFAULTS.update({
        ETH: 1e6,
        ETH/Yr: 1e3,
        1/Yr: 1e-3,
        One: 1e-1,
    })

# Functional utilities (unchanged)
def mag(x: Num|Nums|Q|Qs) -> Num|Nums:
    """Extract magnitude from quantities, preserving structure"""
    if isinstance(x, pint.Quantity):
        return x.magnitude
    elif isinstance(x, Sequence):
        con = type(x)
        return con(mag(e) for e in x)
    else: 
        return x

def wmag(f: Callable) -> Callable:
    """Decorator to optionally extract magnitudes from function output"""
    def wf(*args, _nomag: bool = False, **kwargs):
        out = f(*args, **kwargs)
        return out if _nomag else mag(out)
    return wf

def output(depends: list[str] = None):
    """
    Decorator to mark output functions with optional dependency specification
    
    Args:
        depends: List of variable names this output depends on
    """
    def decorator(f: Callable) -> Callable:
        f.is_output = True
        f.output_depends = depends
        return f
    
    # Handle both @output and @output(depends=[...])
    if callable(depends):
        # Called as @output without parentheses
        func = depends
        func.is_output = True
        func.output_depends = None
        return func
    else:
        # Called as @output(depends=[...])
        return decorator

# Base classes with improvements
@dataclass
class AutoDefault:
    """Automatically assign defaults based on type annotations"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, type_ in cls.__annotations__.items():
            if hasattr(cls, name): 
                continue
            if type_ not in (types := DEFAULTS.keys()):
                for t in types:
                    if isinstance(type_, t):
                        type_ = t
                        break
            setattr(cls, name, field(
                    default = DEFAULTS.get(type_, None)))

@dataclass
class Params(AutoDefault):
    """Base parameter class"""
    init_conds: tuple[tuple[str, Any], ...]
    tspan: tuple[Any, ...]
    
@dataclass
class Sim(AutoDefault):
    """Base simulation class with improved output handling"""
    params: Params
    df: pd.DataFrame = None
    
    def sim(self) -> None: 
        pass
    
    def test(self, tol: float = 1e-12) -> bool: 
        pass
    
    @staticmethod
    def func(t: Any, v: tuple[Any,...], p: Params) -> tuple[Any,...]: 
        pass
    
    def _graph(self) -> None: 
        pass
    
    def _output_fcns(self) -> Iterator[Callable]:
        """Find output functions on both Sim and Params"""
        # Check Sim instance
        for name in dir(self):
            if callable(f := getattr(self, name)) and \
               getattr(f, 'is_output', False):
                yield f
                
        # Also check params
        if hasattr(self, 'params'):
            for name in dir(self.params):
                if callable(f := getattr(self.params, name)) and \
                   getattr(f, 'is_output', False):
                    yield f
                    
    def _add_outputs(self) -> None:
        """Add output function results as columns to dataframe"""
        df = self.df
        colset = set(df.columns)
        
        for f in self._output_fcns():
            # Determine function name and handle collisions
            fname = f.__name__
            if fname in df.columns:
                # Add suffix to avoid collision
                obj_name = 'params' if hasattr(self.params, fname) else 'sim'
                fname = f"{fname}_{obj_name}"
                
            # Determine dependencies
            if hasattr(f, 'output_depends') and f.output_depends:
                # Use specified dependencies
                deps = f.output_depends
            else:
                # Infer from function signature
                deps = list(colset.intersection(f.__code__.co_varnames))
                
            if deps and all(d in colset for d in deps):
                # Get the object that has this method (self or self.params)
                obj = self if hasattr(self, f.__name__) else self.params
                # Apply function row-wise
                df[fname] = df[deps].apply(
                    lambda row: getattr(obj, f.__name__)(**row.to_dict()), axis=1)

# PID Controller now imported from new module (see imports at top)

# TWAP (Time-Weighted Average Price) calculator
@dataclass
class TWAP:
    """
    Time-Weighted Average Price calculator
    
    Maintains a sliding window of price observations
    """
    window: float  # Window size in time units
    
    # Internal state
    observations: list[tuple[float, float]] = field(default_factory=list, init=False)
    
    def update(self, time: float, value: float):
        """Add new observation and clean old ones"""
        self.observations.append((time, value))
        
        # Remove observations outside window
        cutoff_time = time - self.window
        self.observations = [(t, v) for t, v in self.observations 
                           if t >= cutoff_time]
    
    @property
    def value(self) -> float:
        """Calculate current TWAP"""
        if not self.observations:
            return 0.0
            
        if len(self.observations) == 1:
            return self.observations[0][1]
            
        # Calculate time-weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i in range(1, len(self.observations)):
            t0, v0 = self.observations[i-1]
            t1, v1 = self.observations[i]
            dt = t1 - t0
            avg_value = (v0 + v1) / 2  # Trapezoidal integration
            
            weighted_sum += avg_value * dt
            total_weight += dt
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0

# Enhanced ODE simulation classes
@dataclass
class FinDiffParams(Params):
    dt: Any = 1
    
@dataclass
class FinDiffSim(Sim):
    params: FinDiffParams
    def sim(self, graph: tuple[str,...] = None) -> None:
        p = self.params
        names, values = zip(*p.init_conds)
        nt = int(np.ceil((mag(p.tspan[1]) - mag(p.tspan[0]))/mag(p.dt)))
        data = np.ndarray(shape = (nt, len(p.init_conds)),
                          dtype = np.float64)
        data[0] = mag(values)
        for t in range(1, nt):
            data[t] = data[t-1] + mag(p.dt) * mag(self.func(data[t-1]))
        self.df = pd.DataFrame(data, columns = names)
        self.df['t'] = np.arange(mag(p.tspan[0]), mag(p.tspan[1]), mag(p.dt))
        self._add_outputs()
        self._graph()

@dataclass
class ODESim(Sim):
    out: OdeResult = None
    def sim(self, graph: tuple[str,...] = None) -> None:
        p = self.params
        names, values = zip(*p.init_conds)
        out = solve_ivp(fun = wmag(self.func),
                        t_span = mag(p.tspan),
                        y0 = mag(values),
                        method = 'Radau',
                        args = (p,))
        df = pd.DataFrame(out.y.T, columns = names)
        df['t'] = out.t
        self.out = out
        self.df = df
        self._add_outputs()
        self._graph()

# Base class for controlled ODE systems
@dataclass  
class ControlledODESim(ODESim):
    """
    Base class for ODE systems with controllers
    
    Manages controller state and updates
    """
    controllers: Dict[str, PIDController] = field(default_factory=dict)
    
    def get_errors(self, t: float, state: Dict[str, float], p: Params) -> Dict[str, float]:
        """
        Calculate errors for each controller
        
        Should be overridden by subclasses
        
        Returns:
            Dict mapping controller name to error value
        """
        return {}
        
    def sim(self, graph: tuple[str,...] = None) -> None:
        """Enhanced sim that manages controller state"""
        # Reset all controllers
        for controller in self.controllers.values():
            controller.reset()
            
        # Run base simulation
        super().sim(graph)

# Enhanced equilibrium testing
def test_equilibrium(sim: Sim, 
                    tolerances: Dict[str, float] = None,
                    default_tol: float = 1e-6) -> bool:
    """
    Test equilibrium with per-variable tolerances
    
    Args:
        sim: Simulation instance (must have been run)
        tolerances: Dict mapping variable names to tolerances
        default_tol: Default tolerance for unlisted variables
        
    Returns:
        True if all variables within tolerance at end of simulation
    """
    if sim.df is None:
        return False
        
    tolerances = tolerances or {}
    final_row = sim.df.iloc[-1]
    
    # Check each variable
    for col in sim.df.columns:
        if col == 't':
            continue
            
        tol = tolerances.get(col, default_tol)
        # Assume equilibrium target is stored in params or is 0
        target = getattr(sim.params, f"{col}_target", 0.0)
        
        if abs(final_row[col] - target) > tol:
            return False
            
    return True

def test_convergence(sim: Sim,
                    variable: str,
                    target: float,
                    tolerance: float,
                    after_time: float = None) -> bool:
    """
    Test that a variable converges to target within tolerance
    
    Args:
        sim: Simulation instance
        variable: Variable name to check
        target: Target value
        tolerance: Acceptable deviation
        after_time: Only check after this time (optional)
        
    Returns:
        True if converged
    """
    if sim.df is None or variable not in sim.df.columns:
        return False
        
    df = sim.df
    if after_time is not None:
        df = df[df.t >= after_time]
        
    if len(df) == 0:
        return False
        
    # Check if all values after time are within tolerance
    return (abs(df[variable] - target) <= tolerance).all()

if __name__ == "__main__":
    print("Ethode extended framework loaded successfully")