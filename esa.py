import pint

from generic import *


Unit = TypeVar('Unit', bound = pint.Unit)
U = pint.UnitRegistry()
U.load_definitions('eth_units.txt')

EPB = U.ETH / U.block


class ODEIterMngr(IterationManager[EPB]):
    def sum_deltas(self) -> Vars[EPB]:
        s = []
        for d in self.model_fields:
            s.append(sum(getattr(self, d).values(), start = 0))
        return tuple(s)

    
class ODESimMngr(SimulationManager[EPB]):
    time_range: range = (0, 1000)
    dt: float = 1e-3
    init_conds: Vars[EPB] = tuple(x * U.ETH for x in (0, .3, .7, 0, 0, 0))

    
class ESAFcnMngr(BaseModel):
    params: dict[str, float]
    it: IterationManager
    _uq: list[float] = [0]
    _sq: list[float] = [0]
    def _burn(self, x:Num) -> Num:
        return self.params['burn'] * x
    def fees_off(self) -> None: pass
    def fees_on(self, *args) -> None: pass
    def issuance(self, v:Vars[EPB], t:float) -> EPB:
        return math.sqrt(v[S_idx])
    def pop_unstake(self, *args) -> EPB:
        return self._uq.pop()
    def pop_stake(self, *args) -> EPB:
        return self._sq.pop()
    def withdraw(self, v, t) -> tuple[EPB, EPB]:
        dW = v[W_idx]
        self.it.dB['burn'] += (dB := self._burn(dW))
        return -dW, dW - dB
    def reinvest(self, *args) -> tuple[EPB, EPB]:
        negdC = self.params['rnvst'] * self.it.dC['wdraw']
        self.it.dB['rnvst'] = (dB := self._burn(negdC))
        return negdC - dB, -negdC
    def push_stake(self, *args) -> tuple[EPB, EPB]:
        return 0, 0
    def trading_volume_fees(self, v, t) -> tuple[EPB, EPB]:
        self.it.dB = (dB := self._burn(fees := (v[C_idx] / 100) ** 2))
        return fees - dB, -fees

    
class ODEModel[EPB](ETHModel):
    def odesim(self):
        out = [self.sim.init_conds]
        for t in range(*self.sim.time_range, self.sim.dt):
            out.append( scipy.integrate.odeint(
                self.iterate, out[-1], [t, t + self.sim.dt], self))
        return out
    

if __name__ == '__main__':
    it = ODEIterMngr()
    params = {'rnvst': .6, 'burn': .02}    
    esa = ETHModel[EPB](ODESimMngr(), it, ESAFcnMngr(params, it))
    out = esa.odesim()
