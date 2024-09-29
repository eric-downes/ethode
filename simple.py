from __future__ import annotations
from typing import TypeVar, Annotated, Generic
from dataclasses import dataclass

from scipy.integrate import odeint
import numpy as np
import pint

Q = TypeVar('Q')
Num = float|int

Units = pint.UnitRegistry()
Units.load_definitions('eth_units.txt')
one = 1 * Units.dimensionless

# GENERIC

@dataclass
class Params: pass

@dataclass
class Sim:
    ic: tuple[Q, ...]
    tinfo: tuple[Num, Num, Num]
    params: Params
    def sim(self) -> list[tuple[Q, ...]]: pass
    @staticmethod
    def test(v:tuple[Q,...], t:Num, tol:float) -> bool: pass
    @staticmethod
    def func(v:tuple[Q,...], t:Num, p:Params) -> bool: pass
    
@dataclass
class ODESim(Sim):
    def sim(self) -> list[tuple[Q, ...]]:
        return odeint(func = self.func,
                      y0 = self.ic,
                      t = np.arange(*self.tinfo),
                      args = (self.params,))

        
