from __future__ import annotations
from typing import TypeVar, Annotated, Generic
from dataclasses import dataclass
from functools import partial

from pydantic_pint import PydanticPintQuantity as PPQuantity
from pydantic import BaseModel, Field, model_validator
from scipy.integrate import odeint
import numpy as np
import brainpy # SDE
import pint


T = TypeVar('T')
Num = float|int
Vars = tuple[T,T,T,T,T,T]

Unit = TypeVar('Unit', bound = pint.Unit)
Quantity = pint.Quantity
U = pint.UnitRegistry()
U.load_definitions('eth_units.txt')

EPB = Annotated[Quantity, PPQuantity("EPB", ureg = U)]
Blk = Annotated[Quantity, PPQuantity("Block", ureg = U)]
ETH = Annotated[Quantity, PPQuantity("ETH", ureg = U)]

@dataclass
class SimulationManager:
    t_start: Num
    t_stop: Num
    dt: Num
    init_conds: Vars[Num]

class IterationManager: pass

class ESAItMngr(IterationManager):
    def __init__(self, params: dict[str, Num]):
        self._uq: list[T] = [0]
        self._sq: list[T] = [0]
        self.params = params
        self.variables = {v:i for i,v in enumerate(('W','S','C','Qs','Qu','B'))}
        self.dvars = {}
        for vstr in self.variables:
            dv = 'd' + vstr
            self.dvars[dv] = {}
            setattr(self, dv, self.dvars[dv])
        
    def burn(self, x:Num) -> Num:
        return self.params['burn'] * x        

    def init(self, *args) -> None: pass
    def fees_off(self, *args) -> None: pass

    def issuance(self, v:Vars[float], t:float) -> float:
        return np.sqrt(v[self.variables['S']])
    def pop_unstaking_q(self, *args) -> float:
        return self._uq.pop() if self._uq else 0
    def pop_staking_q(self, *args) -> float:
        return self._sq.pop() if self._sq else 0
    def fees_on(self, *args) -> None: pass
    
    def withdraw(self, v:Vars[float], t:float) -> tuple[float, float]:
        W = v[self.variables['W']]
        self.dB['wdraw'] = (dB := self.burn(W))
        return -W, W - dB
    def reinvest(self, v:Vars[float], t:float) -> tuple[float, float]:
        wdraw = abs(self.dC['wdraw'])
        negdC = self.params['rnvst'] * wdraw
        self.dB['rnvst'] = (dB := self.burn(negdC))
        return negdC - dB, -negdC
    def push_staking_q(self, v:Vars[float], t:float) -> tuple[float, float]:
        return 0, 0
    def push_unstaking_q(self, v:Vars[float], t:float) -> tuple[float, float]:
        return 0, 0

    def trading_volume_fees(self, v:Vars[float], t:float) -> tuple[float, float]:
        C_idx = self.variables['C']
        self.dB['vol'] = (dB := self.burn(fees := (v[C_idx] / 100) ** 2))
        return fees - dB, -fees

    def sum_deltas(self) -> Vars[T]:
        s = []
        for d in self.dvars.values():
            s += [sum(d.values(), start = 0)]
        return tuple(s)



class ETHModel:
    def __init__(self,
                 sim: SimulationManager,
                 it: IterationManager):
        self.it = it
        self.sim = sim
        
    def odesim(self):
        sim = self.sim
        out = odeint(func = self.iterate,
                     y0 = self.sim.init_conds,
                     t = np.arange(sim.t_start, sim.t_stop, sim.dt),
                     args = (self.it,))
        return out
    
    @staticmethod
    def iterate(v:Vars[float], t:float, it:tuple[IterationManager]):

        W, S, C, Qs, Qu, B = v
        
        # setup this iteration
        it.init(v, t)

        # no tx fees
        it.fees_off()
        it.dW['issuance'] = it.issuance(v, t)
        it.dC['pop_unstake'] = it.pop_unstaking_q(v, t)
        it.dS['pop_stake'] = it.pop_staking_q(v, t)

        # tx fees; start collecting fees
        # all following calculations will generate fees implicitly
        # some of ea flow is burnt (p.it.dB++) according to p.fcn
        # and remainder of ea flow goes to p.it.dW awaiting withdrawal
        it.fees_on(v, t)
        
        # withdraw staking rewards into circulating ETH
        it.dW['wdraw'], it.dC['wdraw'] = it.withdraw(v, t)
        # add staking reinvestment to staking queue
        it.dQs['reinvest'], it.dC['reinvest'] = it.reinvest(v, t)
        # FIFO staking queue
        it.dC['push_stake'], it.dQs['push_stake'] = it.push_staking_q(v, t)
        # FIFO unstaking queue
        it.dS['push_unstake'], it.dQu['push_unstake'] = it.push_unstaking_q(v, t)

        # trading volume
        it.dW['vol'], it.dC['vol'] = it.trading_volume_fees(v, t)

        # calculate the totals and return
        return it.sum_deltas()

    
    
if __name__ == '__main__':
    sim = SimulationManager(
        t_start = 0,
        t_stop = 10,
        dt = 1e-3,
        init_conds = tuple(x for x in (0,.3,.7,0,0,0)),
    )
    it = ESAItMngr(params = {'rnvst': .6, 'burn': .02})
    esa = ETHModel(sim = sim, it = it)
    out = esa.odesim()
