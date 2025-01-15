from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence

from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_ivp
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
ETH_Data = tuple[tuple[str, ETH],...]


DEFAULTS = {
    bool: False,
    str: '',
    list: [],
    dict: {},
    None: None,
    int|float|np.float64: 0,
    pint.Quantity: .1,
}

# functional

def mag(x: Num|Nums|Q|Qs) -> Num|Nums:
    if isinstance(x, pint.Quantity):
        return x.magnitude
    elif isinstance(x, Sequence):
        con = type(x)
        return con(mag(e) for e in x)
    else: return x

def wmag(f:Callable|Q) -> Callable:
    def wf(*args, _nomag:bool = True, **kwargs):
        out = f(*args, **kwargs)
        return out if _nomag else mag(out)
    return wf

# classy

@dataclass # thanks Claude
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
    def sim(self) -> list[tuple[Q, ...]]: pass
    def test(self, tol:float = 1e-12) -> bool: pass
    @staticmethod
    def func(t:Q, v:tuple[Q,...], p:Params) -> bool: pass

@dataclass
class ODESim(Sim):
    def sim(self) -> tuple[pd.DataFrame, Any]:
        p = self.params
        names, values = zip(*p.init_conds)
        out = solve_ivp(fun = wmag(self.func),
                        t_span = mag(p.tspan),
                        y0 = mag(values),
                        method = 'Radau',
                        args = (self.params,))
        df = pd.DataFrame(out.y.T, columns = names)
        df['t'] = out.t
        return df, out
