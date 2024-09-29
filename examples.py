from simple import *

#############################
# CONTENT BELOW
#############################

# simple ESC model

ETHx3 = (Units['ETH'], Units['ETH'], Units['ETH'])

@dataclass
class SimpleParams(Params):
    r: Units.dimensionless # staking reinvestment fraction in (0,1)
    f: Units.EPB # tx fees apy
    y: Units.EPB # issuance apy
    e: Units.ETH = 1 * Units.ETH
    
@dataclass
class SimpleESU(ODESim):
    @staticmethod
    def test(v:tuple[*ETHx3],
             t:Units.Block,
             tol:float = 1e-12) -> bool:
        E,S,U = v
        troo = all(E > 0) & all(S > 0) & all(U > 0)
        troo &= all(abs(E - S - U) < tol)
        return troo
    @staticmethod
    @Units.wraps(ETHx3, (ETHx3, Units['block'], Units.dimensionless))
    def func(v:tuple[Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             p:Params) -> tuple[Units.ETH, Units.ETH, Units.ETH]:
        E,S,U = v
        dE = p.y * np.sqrt(p.e / S) * S
        fees = p.f * (U / p.e) ** 2
        profit = dE + fees
        dS = p.r * profit
        dU = (1 - p.r) * profit - fees
        return dE, dS, dU
        
# complex ESC Model template

@dataclass
class ComplexESU(SimpleESU):
    @staticmethod
    def func(v:tuple[Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             p:Params) -> tuple[Units.ETH, Units.ETH, Units.ETH]:
        # load variables into a dict
        d = {k:v for k,v in zip(('E','S','U'), v)}
        d['t'] = t
        # compute functions
        r = p.renvst(**d)
        f = p.fees(**d)
        dE = p.issuance(**d) * S
        profit = dE + f
        dS = r * profit
        dU = (1 - r) * profit - f
        return dE, dS, dU

# .. w/ quadratic trading

@dataclass
class QuadTradingParams(Params):
    def issuance(self, S: Units.ETH, **kwargs) -> Units.ETH:
        return self.y * np.sqrt(self.e / S)
    def fees(self, U: Units.ETH, **kwargs) -> Units.ETH:
        return self.f * (U / self.e) ** 2
    def renvst(self, **kwargs) -> Units.EPB:
        return self.r

# ... w/ linear trading

@dataclass
class LinTradingParams(QuadTradingParams):
    def fees(self, U: Units.ETH, **kwargs) -> Units.ETH:
        return self.f * (U / self.e)

@dataclass
class KineticTradingParams(QuadTradingParams):
    k: Units.ETH = 2 ** 10 * Units.ETH
    def fees(self, U: Units.ETH, **kwargs) -> Units.ETH:
        u = U / self.e
        k = self.k / self.e
        return self.f * u ** 2 /(u + k)

### ESCB

@dataclass
class ESCB(ODESim):
    @staticmethod
    def test(v:tuple[Units.ETH, Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             tol:float = 1e-12) -> bool:
        E,S,C,B = v
        troo = all(E > 0) & all(S > 0) & all(C > 0) & all(B > 0)
        troo &= all(abs(E - S - C - B) < tol)
        return troo
    @staticmethod
    def func(v:tuple[Units.ETH, Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             p:Params) -> tuple[Units.ETH, Units.ETH, Units.ETH, Units.ETH]:
        # load variables into a dict
        d = {k:v for k,v in zip(('E','S','C','B'), v)}
        d['t'] = t
        # compute functions
        r = p.renvst(**d)
        f_tot, f_burned = p.fees(**d)
        f_reward = f_tot - f_burned
        # compute derivatives
        dE = p.issuance(**d) * S
        profit = dE + f_reward
        dS = r * profit
        dC = (1 - r) * profit - f_tot
        dB = f_burned
        return dE, dS, dC, dB
    
@dataclass
class ESCBParams(Params):
    r: Units.dimensionless # staking reinvestment fraction in (0,1)
    f: Units.PB # tx fees apy
    y: Units.PB # issuance apy
    b: Units.dimensionless # burn
    c: Units.EPB = 0 * Units.EPB # cost
    e: Units.ETH = 1 * Units.ETH # sets scale for dimless calc
    k: Units.ETH = 1 * Units.ETH # michaelis-menton-like parameter
    def issuance(self, S: Units.ETH, **kwargs) -> Units.PB:
        return self.y * np.sqrt(self.e / S)
    def renvst(self, **kwargs) -> Units.dimensionless:
        i = self.issuance(**kwargs)
        return self.r * (one - (self.c / i).to('')) 
    def fees(self, C: Units.ETH, **kwargs) -> tuple[Units.EPB, Units.EPB]:
        fees = self.f * C * C / (C + self.k)
        return fees, self.burned(fees, **kwargs)
    def burned(self, fees: Units.EPB, **kwargs) -> Units.EPB:
        return self.b * fees


'''
class QESCB(ESCB):
    @staticmethod
    def test(v:tuple[Units.ETH, Units.ETH, Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             tol:float = 1e-12) -> bool:
        E,S,C,B,Q = v
        troo = all(E > 0) & all(S > 0) & all(C > 0) & all(B > 0)
        troo &= all(abs(E - S - C - B - Q) < tol)
        return troo
    @staticmethod
    def func(v:tuple[Units.ETH, Units.ETH, Units.ETH, Units.ETH, Units.ETH],
             t:Units.Block,
             p:Params):
        # load variables into a dict
        d = {k:v for k,v in zip(('E','S','C','B', 'Q'), v)}
        d['t'] = t
        # compute functions
        r = p.renvst(**d)
        f_tot, f_burned = p.fees(**d)
        f_reward = f_tot - f_burned
        dQs_net, dQu_net = p.queue(**d)
        # compute derivatives
        dE = p.issuance(**d) * S
        profit = dE + f_reward
        dS = r * profit + dQs_net
        dC = (1 - r) * profit - f_tot + dQu_net
        dB = f_burned
        return dE, dS, dC, dB

    
class QESCBKinTradParams(Params):
    r: float # staking reinvestment fraction in (0,1)
    f: Units.EPB # tx fees apy
    y: Units.EPB # issuance apy
    b: float # burn
    c: Units.EPB # cost
    e: Units.ETH = 1 * Units.ETH # sets scale for dimless calc
    k: Units.ETH = 2 ** 20 * Units.ETH # michaelis-menton-like parameter
    qt: Unit.Block = 1000
    def issuance(self, S: Units.ETH, **kwargs) -> Units.EPB:
        return self.y * np.sqrt(self.e / S)
    def renvst(self, **kwargs) -> Units.EPB:
        i = self.issuance(**kwargs)
        return self.r * (1 - self.c / i) 
    def fees(self, C: Units.ETH, **kwargs) -> tuple[Units.EPB, Units.EPB]:
        u = C / self.e
        k = self.k / self.e
        fees = self.f * u ** 2 /(u + k)
        return fees, self.burned(fees, **kwargs)
    def burned(self, fees: Units.EPB, **kwargs) -> Units.EPB:
        return self.b * fees

    


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
