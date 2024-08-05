'''
use
brainpy -- SDE
scipy.integrate.odeint -- ODE
pydantic -- parameters & fcns

'''

from functools import partial

import numpy as np
from pydantic import BaseModel, Field

def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

from scipy.integrate import odeint
sol = odeint(pend, y0, t, args=(b, c))


def model(variables: tuple[tuple|float], time: float,
          params: dict[str, Callable | float]) -> tuple:



def model(variables: tuple[tuple|float], time: float,
          params: dict[str, Callable | float]) -> tuple:
