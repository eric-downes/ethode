from __future__ import annotations
from functools import partial
from typing import TypeVar

import brainpy # SDE
import numpy as np
from scipy.integrate import odeint
from pydantic import BaseModel, Field, model_validator

T = TypeVar('T')
Num = float|int
Vars = tuple[T,T,T,T,T]


class IterationManager[T](BaseModel):
    # simplest, when everything is a number instead of a distribution
    dW: dict[str,T] = {}
    dS: dict[str,T] = {}
    dC: dict[str,T] = {}
    dQs: dict[str,T] = {}
    dQu: dict[str,T] = {}
    dB: dict[str,T] = {}
    def init(self, v:Vars[T], t:Num) -> None: pass
    def sum_deltas(self) -> Vars[T]: pass


class SimulationManager[T](BaseModel):
    time_range: tuple[Num, Num]
    dt: Num
    init_conds: Vars[T]


class FunctionManager[T](BaseModel):
    def fees_off(self) -> None: pass
    def fees_on(self, v:Vars[T], t:Num) -> None: pass
    def issuance(self, v:Vars[T], t:Num) -> T: pass
    def pop_unstake(self, v:Vars[T], t:Num) -> T: pass
    def pop_stake(self, v:Vars[T], t:Num) -> T: pass
    def withdraw(self, v:Vars[T], t:Num) -> tuple[T,T]: pass
    def reinvest(self, v:Vars[T], t:Num) -> tuple[T,T]: pass
    def push_stake(self, v:Vars[T], t:Num) -> tuple[T,T]: pass
    def trading_volume_fees(self, v:Vars[T], t:Num) -> tuple[T,T]: pass

    
class ETHModel[T](BaseModel):
    sim: SimulationManager
    it: IterationManager
    fcn: FunctionManager
    @staticmethod
    def iterate(self, v:Vars[T], t:float):

        # setup this iteration
        self.it.init(v,t)

        # no tx fees
        self.fcn.fees_off()
        self.it.dW['issuance'] = self.fcn.issuance(v,t)
        self.it.dC['pop_unstake'] = self.fcn.exit_unstaking_queue(v,t)
        self.it.dS['pop_stake'] = self.fcn.exit_staking_queue(v,t)

        # tx fees; start collecting fees
        # all following calculations will generate fees implicitly
        # some of ea flow is burnt (p.it.dB++) according to p.fcn
        # and remainder of ea flow goes to p.it.dW awaiting withdrawal
        self.fcn.fees_on(v,t)
        
        # withdraw staking rewards into circulating ETH
        self.it.dW['wdraw'], self.it.dC['wdraw'] = self.fcn.withdraw(v,t)
        # add staking reinvestment to staking queue
        self.it.dQs['reinvest'], self.it.dC['reinvest'] = self.fcn.reinvest(v,t)
        # FIFO staking queue
        self.it.dC['push_stake'], self.it.dQs['push_stake'] = self.fcn.push_stake(v,t)

        # trading volume
        self.it.dW['vol'], self.it.dC['vol'] = self.fcn.trading_volume_fees(v,t)

        # calculate the totals and return
        return self.it.sum_deltas()
