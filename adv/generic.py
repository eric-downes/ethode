from __future__ import annotations
from functools import partial
from typing import TypeVar

import brainpy # SDE
import numpy as np
from scipy.integrate import odeint
from pydantic import BaseModel, Field, model_validator

T = TypeVar('T')
Num = float|int
Vars = tuple[T,T,T,T,T,T]

class SimulationManager[T](BaseModel):
    t_minmax: tuple[Num, Num]
    dt: Num
    init_conds: Vars[T]
