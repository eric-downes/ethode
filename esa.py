from pydantic_pint import PydanticPintQuantity as PPQuantity
from typing import Annotated, Generic
import pint

from generic import *

Unit = TypeVar('Unit', bound = pint.Unit)
Quantity = pint.Quantity
U = pint.UnitRegistry()
U.load_definitions('eth_units.txt')

EPB = Annotated[Quantity, PPQuantity("EPB", ureg = U)]
Blk = Annotated[Quantity, PPQuantity("Block", ureg = U)]
ETH = Annotated[Quantity, PPQuantity("ETH", ureg = U)]

class ESAItMngr(BaseModel):
    _uq: list[T] = [0]
    _sq: list[T] = [0]
    @property
    def variables(self) -> tuple[str,...]:
        return ('W','S','C','Qs','Qu','B')
    @property
    def params(self) -> dict[str, Num]:
        return {'rnvst': .6, 'burn': .02})
    @model_validate
    def _val_(self) -> ESAItMngr
        for vstr in self.variables:
            setattr(self, 'd' + vstr, {})
        return self
    def sum_deltas(self) -> Vars[T]:
        s = []
        for d in self.variables:
            s.append(sum(getattr(self, d).values(), start = 0))
        return tuple(s)
    def _burn(self, x:Num) -> Num:
        return self.params['burn'] * x
    @staticmethod
    def issuance(staked:ETH) -> EPB:
        return math.sqrt(staked)
    def fees_off(self) -> None: pass
    def fees_on(self, *args) -> None: pass
    def push_stake(self, *args) -> tuple[EPB, EPB]:
        return 0, 0
    def pop_unstake(self, *args) -> EPB:
        return self._uq.pop()
    def pop_stake(self, *args) -> EPB:
        return self._sq.pop()
    def withdraw(self, W:ETH) -> tuple[EPB, EPB]:
        self.dB['burn'] += (dB := self._burn(W))
        return -W, W - dB
    def reinvest(self, wdraw:ETH) -> tuple[EPB, EPB]:
        negdC = self.params['rnvst'] * wdraw
        self.dB['rnvst'] = (dB := self._burn(negdC))
        return negdC - dB, -negdC
    def trading_volume_fees(self, v, t) -> tuple[EPB, EPB]:
        self.it.dB = (dB := self._burn(fees := (v[C_idx] / 100) ** 2))
        return fees - dB, -fees


class ETHModel[T](BaseModel):
    params: dict[str, float]
    sim: SimulationManager
    it: IterationManager
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
        self.it.dQs['reinvest'], self.it.dC['reinvest'] \
            = self.fcn.reinvest(,t)
        # FIFO staking queue
        self.it.dC['push_stake'], self.it.dQs['push_stake'] \
            = self.fcn.push_stake(v, t)

        # trading volume
        self.it.dW['vol'], self.it.dC['vol'] \
            = self.fcn.trading_volume_fees(v,t)

        # calculate the totals and return
        return self.it.sum_deltas()

    
class ODETHModel(ETHModel):
    def odesim(self):
        out = [self.sim.init_conds]
        for t in range(*self.sim.time_range, self.sim.dt):
            out.append( scipy.integrate.odeint(
                self.iterate, out[-1], [t, t + self.sim.dt], self))
        return out
    
    
if __name__ == '__main__':
    esa_sim_mngr = SimulationManager(
        block_minmax = (0 * U.EPB, 1000 * U.EPB),
        dt = 1e-3,
        init_conds = tuple(x * U.ETH for x in (0,.3,.7,0,0,0)),
    )
    it = ODEIterMngr()
    params = 
    esa = ODETHModel[EPB](
        sim = esa_sim_mngr, 
        it = it,
        fcn = ESAFcnMngr(
            
            it = it))
    out = esa.odesim()
