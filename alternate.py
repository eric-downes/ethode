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

# GENERIC

@dataclass
class Params: pass

@dataclass
class Sim: pass
    ic: tuple[Q, ...]
    tinfo: tuple[Num, Num, Num]
    params: Params
    def sim(self) -> list[tuple[Q, ...]]: pass
    @staticmethod
    def test(v:tuple[Q,...], t:Num, tol:float) -> bool: pass
    @staticmethod
    def func(v:tuple[Q,...], t:Num, p:Params) -> bool: pass
    
class ODESim(Sim):
    def sim(self) -> list[tuple[Q, ...]]:
        return odeint(func = self.func,
                      y0 = self.ic,
                      t = np.arange(*self.tinfo),
                      args = self.params)

#############################
# CONTENT BELOW
#############################

# simple ESC model

class SimpleParams(Params):
    r: float # staking reinvestment fraction in (0,1)
    f: Units.EPB # tx fees rate per 1 ETH
    y: Units.EPB # issuance rate per 
    e: Units.ETH = 1 * Units.ETH
    
class SimpleESU(ODESim):
    @staticmethod
    def test(v:tuple[Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             tol:float = 1e-12) -> None:
        E,S,U = v
        troo = all(E > 0) & all(S > 0) & all(U > 0)
        troo &= all(abs(E - S - U) < tol)
        return troo
    @staticmethod
    def func(v:tuple[Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             p:Params):
        E,S,U = v        
        dE = p.y * np.sqrt(p.e / S) * S
        fees = p.f * (U / p.e) ** 2
        profit = dE + fees
        dS = p.r * profit
        dU = (1 - p.r) * profit - fees
        return dE, dS, dU
        
# complex ESC Model template

class ComplexESU(SimpleESU):
    @staticmethod
    def func(v:tuple[Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             p:Params):
        E,S,U = v
        r = p.renvst()
        f = p.fees(U)
        dE = p.issuance(S) * S
        profit = dE + f
        dS = r * profit
        dU = (1 - r) * profit - f
        return dE, dS, dU

# .. w/ quadratic trading

class QuadTradingParams(Params):
    def issuance(self, s: Units.ETH) -> Units.ETH:
        return self.y * np.sqrt(self.e / s)
    def fees(self, u: Units.ETH) -> Units.ETH:
        return self.f * (u / self.e) ** 2
    def renvst(self) -> Units.EPB:
        return self.r

# ... w/ linear trading

class LinTradingParams(QuadTradingParams):
    def fees(self, u: Units.ETH) -> Units.ETH:
        return self.f * (u / self.e)


'''
SimFunc = Callable[[tuple[Q, ...], Sequence[Num], Params],
                   tuple[Q, ...]]

@dataclass
class Params: pass

@dataclass
class ODESim(Sim):
    func: SimFunc
    ic: tuple[Q, ...]
    tinfo: tuple[Num, ...]
    params: Params
    def sim(self) -> list[tuple[Q, ...]]:
        return odeint(func = self.func,
                      y0 = self.ic,
                      t = np.arange(*self.tinfo),
                      args = params)
    def update(self, vin:tuple[Q, ...], tin: tuple[Num]): pass

class MixedSim:
    mngr: SimMngr
    def mixed_sim(self) -> list[tuple[Num,...], tuple[Num,...]]:
        with self.mngr() as m:
            while m.simulating():
                con, dis = m.next_step()
                m.update( con.sim() )
                m.update( dis.sim() )
        return m.vs, m.ts

class SimMngr:
    disfcn: SimFcn
    confcn: SimFcn
    vs: list[tuple[Q, ...]]
    ts: tuple[Num, ...]
    def simulating(self) -> bool: pass
    def next_step(self) -> tuple[Sim, Sim]: pass
    def update(self) -> None: pass
    

'''
