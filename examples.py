from ethode import *

'''
You can define params separately, in case you want to use the same
parameters in different models.
'''

@dataclass
class ConstParams(Params):
    y1: 1/Yr = 166.3 #1/Yr is a unit (pint quantity type)
    b: One = 5e-1  # One is "dimensionaless" having no units
    f: 1/Yr = 8e-3 # use units in annotation but *always assign pure int|float*
    j: 1/Yr = 1e-5 
    r: One = .65
    qs: 1/Yr = 1e-4
    qu: 1/Yr = 1e-4
    s1: ETH = 1   # pedantic so sqrt is dimless
    def yld(self, S:ETH, **kwargs) -> 1/Yr:
        return self.y1 * np.sqrt(self.s1 / S)

'''
Things are intended to be as modular as possible.
Here is (my understanding of) Anders Elowsson's fav yield curve,
the new `AndersParams` inherits the old `ConstParams`
https://ethresear.ch/t/properties-of-issuance-level-consensus-incentives-and-variability-across-potential-reward-curves/18448
'''

@dataclass
class AndersParams(ConstParams):
    y1: 1/Yr = 166.3 * 100 / 64
    k: 1/ETH = 2**(-11)
    def yld(self, S:ETH, **kwargs) -> ETH:
        return self.y1 * np.sqrt(self.s1 / S / (1 + self.k * S))
    
'''
Beware!  Params always have a default set of (possibly terrible)
initial conditions.  You probably want to update your base params with
different `init_conds` and `tspan` as below.

Right now `solve_ivp` picks time points to plot, which might not be
enough to make pretty plots.
'''
    
@dataclass
class SUConstParams(ConstParams):
    init_conds: ETH_Data = (('S', 120e6 * .3), ('U', 120e6 * .7))
    tspan: tuple[Yr, Yr] = (0, 100)
@dataclass
class SUConstSim(ODESim):
    # std pattern to get around initialization by py interp
    # we want a separate copy of Params() for each sim instance
    params: Params = field(default_factory = SUConstParams)
    @staticmethod # always need this: scipy wants a specific signature
    def func(t:Yr, v:tuple[ETH, ETH], p:Params) -> tuple[ETH/Yr, ETH/Yr]:
        # self.params is passed as "p"
        S, U = v
        dS = (p.r * (y := p.yld(S)) - p.j - p.qu) * S + \
            ((rf := p.r * p.f) * (1 - p.b) + p.qs) * U
        dU = ((1 - p.r) * y + p.qu) * S - \
            (rf + (1 - p.r) * p.b * p.f + p.qs) * U
        return dS, dU
su = SUConstSim()
su.sim() # su.df contains the data, su.out contains metadata 

"""
`@output` literally just adds an attribute to the method so we know to
include it as a column in the dataframe `su.df`

A practice you may want to adopt if you have many dynamic vars is to
define e.g. `def sfrac(self, S:ETH, U:ETH, **kwargs) -> One:` instead;
then you can unpack vars inside `func` using a dictionary `val_dict`
and calculate sfrac using `p.sfrac(**val_dict)
"""

@dataclass
class SUaConstParams(SUConstParams):
    @output
    def sfrac(self, S:ETH, U:ETH) -> One:
        return S / (S + U)
    @output
    def alpha(self, S:ETH, U:ETH) -> 1/Yr:
        s = self.sfrac(S,U)
        return self.yld(S) * s - self.b * self.f * (1 - s) - self.j * s
@dataclass
class SUaConstSim(SUConstSim):
    params: Params = field(default_factory = SUaConstParams)
su_a = SUaConstSim()
su_a.sim()

"""
you will inherit the other variables, but can always define new ones

MegaBurn is maybe a misleading name, the burn isn't that high here.
It's post mega-burn, which is something made up to get high inflation.
"""

@dataclass
class MegaBurnParams(SUaConstParams):
    init_conds: ETH_Data = (('S', 1.2e6 * .4), ('U', 1.2e6 * .6))
    tspan: tuple[Yr, Yr] = (0, 200)
    b: One   = 1e-3
    qs: 1/Yr = 2e-1
@dataclass
class MegaBurnSim(SUaConstSim):
    params: Params = field(default_factory = MegaBurnParams)
zomg = MegaBurnSim()
zomg.sim()

"""
you can define custom tests
to sanity check your model and atb least catch stupid mistakes
hopefully more to come on this... would like to automate
unit sanity-checking without screwing up scipy internals
"""

@dataclass
class BurnlessParams(Params):
    r: One = .5 # staking reinvestment fraction in (0,1)
    f: 1/Yr = .002 # tx fees apy
    y: 1/Yr = 166.3 # issuance apy
    e: ETH = 1
    init_conds: (('E', 100), ('S', 30), ('U', 70))
    
@dataclass
class Burnless(ODESim):
    params: Params = field(default_factory = BurnlessParams)
    def test(self, tol:float = 1e-12) -> bool:
        self.sim()
        df = self.df
        E,S,U = df.E, df.S, df.U
        troo = (E > 0).all() & (S > 0).all() & (U > 0).all()
        troo &= (abs(E - S - U) < tol).all()
        return troo
    @staticmethod
    def func(t:Yr, v:tuple[ETH,ETH,ETH], p:Params) -> ETH_Data:
        E,S,U = v
        dE = p.y * np.sqrt(p.e / S) * S
        fees = p.f * (U / p.e)
        profit = dE + fees
        dS = p.r * profit
        dU = (1 - p.r) * profit - fees
        return dE, dS, dU
        
"""
A sketch of a Price Inflation model w/ price and LSTs distinguished from solo
"""
USD = U.USD
Price = U.USD / ETH
InflData = tuple[*(ETH,)*5, Price]
DInfl = tuple[*(ETH/Yr,)*5, Price/Yr]
EPY = ETH/Yr

@dataclass
class InflParams(Params):
    def tot_fees_mev(self, **kwargs) -> EPY: pass
    def burned_fees(self, tot_fees:EPY, **kwargs) -> EPY: pass
    def split_post_burn(self, post_burn_fees: EPY, **kwargs) -> (EPY,)*2: pass
    def dlog_utility(self, **kwargs) -> 1/Yr: pass
    def dlog_supply(self, **kwargs) -> 1/Yr: pass
    def yield_curve(self, **kwargs) -> 1/Yr: pass
    def usd_val_cost(self, **kwargs) -> USD: pass
    def lsp_rnvst(self, **kwargs) -> One: pass
    def lsp_pfees(self, **kwargs) -> EPY: pass

@dataclass
class InflSim(ODESim):
    params: Params = field(default_factory = InflParams)    
    @staticmethod
    def func(t:Yr, v:InflData, p:Params) -> DInfl:
        # load variables into a dict
        d = {k:v for k,v in zip(('P','E','S','L','U','O'), v)}
        P,E,S,L,U,O = v
        d['t'] = t
        # compute fees
        tx_fees = p.tot_fees_mev(**d)
        burned_fees = p.burned_fees(tx_fees, **d)
        post_burn_fees = tot_fees - burned_fees
        solo_pfees, lsp_pfees = p.split_post_burn(post_burn_fees, **d)
        # compute derivatives
        dP = (p.dlog_utility(**d) - p.dlog_supply(**d)) * P
        dE = (y := p.yield_curve(**d)) * (S + L)
        dS = y * S + solo_pfees - (K := p.usd_val_cost(**d))
        dL = (r := p.lsp_rnvst(**d)) * (lsp_rev := y * L + lsp_pfees)
        dU = K + (1 - r) * lsp_rev - tx_fees
        dO = burned_fees
        return dP, dE, dS, dL, dU, dO
