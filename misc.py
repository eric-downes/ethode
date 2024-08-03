from matplotlib import pyplot as plt
import numpy as np

'''
solving the toy model:
T = Total ETH
S = Staked ETH
U = Unstabled ETH

dT/dt = y(S)T = dS/dt + dU/dt

say unstaked ETH is fixed at a fraction of staked ETH; 

demand is met; dU/dt = 0 leads to

dt = dS /[ y(S)(S + U) ]

Using y(S) = k / sqrt(S) and scaling s = S/U

k/sqrt(U) * dt = sqrt(s)/(s + 1) * ds

The definite integral [0,t):

kt/(2 sqrt(U)) = [sqrt(s) - atan( sqrt(s) )]{s(0), s(t)}
(C + k't) = sqrt(s(t)) - atan( sqrt(s(t)) )

Now we must invert this; the inversion can be done visually and
we inspect the result to confirm.  (Probably can use Lagrange Inversion as well?)

'''

x = np.arange(-7,7,.05)
y = x - np.arctan(x)
yy = x + np.arctan(x)

plt.plot(x,y,'b-', y,x,'r-', x, yy,'go')

from pint import UnitRegistry
import numpy as np

U = UnitRegistry()
U.load_definitions('eth_units.txt')

NBLOCKS = 1_000

np.random.seed()

gas_scenarios = { # how much gas used per block
    'target': np.ones(NBLOCKS) * U.gas_target,
    'linear': np.linspace(U.gas_target, U.gas_max, num = NBLOCKS),
    'normal': U.gas_std * np.random.randn(NBLOCKS) + U.gas_target
    }
    
def fees

