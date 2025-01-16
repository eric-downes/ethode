from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence

from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pint

Num = np.float64|float|int
Nums = tuple[Num, ...]
Q = pint.Quantity
Qs = tuple[Q, ...]

U = pint.UnitRegistry()
U.load_definitions('eth_units.txt')
One = 1 * U.dimensionless
Yr = 1 * U.years
ETH = 1 * U.ETH
ETH_Data = tuple[tuple[str, Q],...]

DEFAULTS = {
    bool: False,
    str: '',
    list: [],
    dict: {},
    None: None,
    int|float|np.float64: 0,
    ETH: 1e6,
    ETH/Yr: 1e3,
    1/Yr: 1e-3,
    One: 1e-1,
    pint.Quantity: 1e-3,
}

# functional

def mag(x: Num|Nums|Q|Qs) -> Num|Nums:
    if isinstance(x, pint.Quantity):
        return x.magnitude
    elif isinstance(x, Sequence):
        con = type(x)
        return con(mag(e) for e in x)
    else: return x

def wmag(f:Callable) -> Callable:
    def wf(*args, _nomag:bool = False, **kwargs):
        out = f(*args, **kwargs)
        return out if _nomag else mag(out)
    return wf

def output(f:Callable) -> Callable:
    f.is_output = True
    return f

# classy

@dataclass
class AutoDefault:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, type_ in cls.__annotations__.items():
            if hasattr(cls, name): continue
            if type_ not in (types := DEFAULTS.keys()):
                for t in types:
                    if isinstance(type_, t):
                        type_ = t
                        break
            setattr(cls, name, field(
                    default = DEFAULTS.get(type_, None)))

@dataclass
class Params(AutoDefault):
    init_conds: tuple[tuple[str,Q], ...]
    tspan: tuple[Q, ...]
    
@dataclass
class Sim(AutoDefault):
    params: Params
    def _output_fcns(self) -> Iterator[Callable]:
        for name in dir(self):
            if callable(f := getattr(self, name)) and \
               getattr(f, 'is_output', False):
                yield f
    def sim(self) -> tuple[pd.DataFrame, OdeResult]: pass
    def test(self, tol:float = 1e-12) -> bool: pass
    @staticmethod
    def func(t:Q, v:tuple[Q,...], p:Params) -> tuple[Q,...]: pass

@dataclass
class ODESim(Sim):
    def sim(self, graph:tuple[str,...] = None
            ) -> tuple[pd.DataFrame, Any]:
        p = self.params
        names, values = zip(*p.init_conds)
        out = solve_ivp(fun = wmag(self.func),
                        t_span = mag(p.tspan),
                        y0 = mag(values),
                        method = 'Radau',
                        args = (p,))
        df = pd.DataFrame(out.y.T, columns = names)
        df['t'] = out.t
        colset = set(df.columns)
        for f in self._output_fcns():
            while (fname := f.__name__) in df.columns:
                fname += '_'
            df[fname] = df[list(colset.intersection(
                f.__code__.co_varnames))].apply(
                    lambda row: f(**row), axis=1)
        self.df = df
        self.out = out
        return None
