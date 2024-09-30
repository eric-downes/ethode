from simple import *

#############################
# CONTENT BELOW
#############################

# simple ESC model

PB = Units['PB']
ETH = Units['ETH']
PE = 1 / ETH
EPB = Units['EPB']
Dless = Units['']
Block = Units['Block']
Price = Units['PriceETHUSD']

def ETHx(n:int) -> tuple[ETH,...]:
    return (ETH, ) * n

@dataclass
class SimpleParams(Params):
    r: float # staking reinvestment fraction in (0,1)
    f: EPB # tx fees apy
    y: EPB # issuance apy
    e: ETH = 1 * ETH
    
@dataclass
class SimpleESU(ODESim):
    @staticmethod
    def test(v:ETHx(3), t:Block, tol:float = 1e-12) -> bool:
        E,S,U = v
        troo = all(E > 0) & all(S > 0) & all(U > 0)
        troo &= all(abs(E - S - U) < tol)
        return troo
    @staticmethod
    @Units.wraps(ETHx(3), (ETHx(3), Block, Dless))
    def func(v:EWTHx(3), t:Block, p:Params) -> ETHx(3):
        E,S,U = v
        dE = p.y * np.sqrt(p.e / S) * S
        fees = p.f * (U / p.e)
        profit = dE + fees
        dS = p.r * profit
        dU = (1 - p.r) * profit - fees
        return dE, dS, dU
        
# complex ESC Model template

@dataclass
class ComplexESU(SimpleESU):
    @staticmethod
    @Units.wraps(ETHx(3), (ETHx(3), Block, Dless))
    def func(v:ETHx(3), t:Block, p:Params) -> ETHx(3):
        # load variables into a dict
        d = {k:v for k,v in zip(('E','S','U'), v)}
        E,S,U = v
        d['t'] = t
        # compute functions
        r = p.renvst(**d)
        f = p.fees(**d)
        dE = p.yield_curve(**d) * S
        profit = dE + f
        dS = r * profit
        dU = (1 - r) * profit - f
        return dE, dS, dU

# .. w/ quadratic trading

@dataclass
class LinearTradingParams(Params):
    def yield_curve(self, S: ETH, **kwargs) -> ETH:
        return self.y * np.sqrt(self.e / S)
    def fees(self, U: ETH, **kwargs) -> ETH:
        return self.f * (U / self.e)
    def renvst(self, **kwargs) -> EPB:
        return self.r

# ... w/ linear trading

@dataclass
class KineticTradingParams(LinearTradingParams):
    k: ETH = 2 ** 10 * ETH
    def fees(self, U: ETH, **kwargs) -> ETH:
        u = U / self.e
        k = self.k / self.e
        return self.f * u ** 2 /(u + k)

# change yield curve

@dataclass
class AndersCurveParams(LinearTradingParams):
    k: PE = 1 / ETH
    def yield_curve(self, S: ETH, **kwargs) -> ETH:
        return self.y * np.sqrt(self.e / S / (1 + self.k * S))

    
### ESCB

InflData = (Price, ) + ETHx(5)

@dataclass
class InflSim(ODESim):
    @staticmethod
    @Units.wraps(InflData, (InflData, Block, Dless))
    def func(v:InflData, t:Block, p:Params) -> ETHx(4):
        # load variables into a dict
        d = {k:v for k,v in zip(('P','E','S','L','C','B'), v)}
        P,E,S,L,C,B = v
        d['t'] = t
        # compute functions
        tot_fees, solo_pfees, lsp_pfees = p.fees(**d)
        burned_fees = tot_fees - solo_pfees - lsp_pfees
        # compute derivatives
        dP = (p.dlog_utility(**d) - p.dlog_supply(**d)) * P
        dE = (y := p.yield_curve(**d)) * (S + L)
        dS = y * S + solo_pfees - (K := p.usd_cost(**d) / P)
        dL = (r := p.lsp_renvst(**d)) * (lsp_rev := y * L + lsp_pfees)
        dC = K + (1 - r) * lsp_rev - tx_fees
        dB = burned_fees
        return dP, dE, dS, dL, dC, dB
    
@dataclass
class ESCBParams(Params):
    def fees(self, **kwargs) -> ETHx(3):
        tot_fees, solo_pfees, lsp_pfees = p.fees(**d)
        burned_fees = tot_fees - solo_pfees - lsp_pfees
        # compute derivatives
        dP = (p.dlog_utility(**d) - p.dlog_supply(**d)) * P
        dE = (y := p.yield_curve(**d)) * (S + L)
        dS = y * S + solo_pfees - (K := p.usd_cost(**d) / P)
        dL = (r := p.lsp_renvst(**d)) * (lsp_rev := y * L + lsp_pfees)
        dC = K + (1 - r) * lsp_rev - tx_fees
        dB = burned_fees
        return dP, dE, dS, dL, dC, dB

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
