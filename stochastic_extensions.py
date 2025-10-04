"""
General-purpose stochastic simulation extensions for ethode
Includes jump processes, Hawkes processes, and PID controllers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import pint

# Import base classes - these are still in the old ethode.py file
# We need to be careful about the module/package naming conflict
import importlib.util
spec = importlib.util.spec_from_file_location("ethode_base", "./ethode.py")
ethode_base = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ethode_base)
    Sim = ethode_base.Sim
    Params = ethode_base.Params
    U = ethode_base.U
    Yr = ethode_base.Yr
    One = ethode_base.One
    mag = ethode_base.mag
    wmag = ethode_base.wmag
except:
    # Fallback - import directly if the above fails
    from ethode import Sim, Params, U, Yr, One, mag, wmag

# Import PID controller from new ethode package
from ethode.controller import PIDController, PIDParams

# Type aliases
JumpEvent = Tuple[float, Callable]  # (time, effect_function)

@dataclass
class StochasticParams(Params):
    """Parameters for stochastic simulations"""
    seed: int = None

@dataclass
class JumpProcessParams(StochasticParams):
    """Parameters for jump-diffusion processes"""
    jump_rate: float = 100.0  # Events per time unit
    jump_process_type: str = 'poisson'  # 'poisson', 'hawkes', 'deterministic'

@dataclass
class HawkesParams(JumpProcessParams):
    """Parameters for Hawkes (self-exciting) processes"""
    jump_process_type: str = 'hawkes'
    excitation_strength: float = 0.3  # How much each event increases intensity
    excitation_decay: float = 1.0     # Decay timescale for excitation

class JumpProcess:
    """Base class for jump processes"""
    
    def __init__(self, params: JumpProcessParams, rng: Optional[np.random.Generator] = None):
        self.p = params
        self.rng = rng or np.random.default_rng(params.seed)
        
    def generate_jumps(self, t_start: float, t_end: float) -> List[float]:
        """Generate jump times in interval [t_start, t_end)"""
        if self.p.jump_process_type == 'poisson':
            return self._generate_poisson_jumps(t_start, t_end)
        elif self.p.jump_process_type == 'deterministic':
            return self._generate_deterministic_jumps(t_start, t_end)
        else:
            raise ValueError(f"Unknown jump process type: {self.p.jump_process_type}")
            
    def _generate_poisson_jumps(self, t_start: float, t_end: float) -> List[float]:
        """Generate Poisson process jump times"""
        if self.p.jump_rate <= 0:
            return []
        dt = t_end - t_start
        n_jumps = self.rng.poisson(self.p.jump_rate * dt)
        jump_times = t_start + self.rng.uniform(0, dt, n_jumps)
        return sorted(jump_times.tolist())
    
    def _generate_deterministic_jumps(self, t_start: float, t_end: float) -> List[float]:
        """Generate regularly spaced jumps"""
        dt = 1.0 / self.p.jump_rate
        jumps = []
        t = t_start + dt
        while t < t_end:
            jumps.append(t)
            t += dt
        return jumps

class HawkesProcess(JumpProcess):
    """Hawkes self-exciting point process"""
    
    def __init__(self, params: HawkesParams, rng: Optional[np.random.Generator] = None):
        super().__init__(params, rng)
        self.history: List[float] = []
        
    def intensity(self, t: float) -> float:
        """Calculate intensity at time t given history"""
        lambda_t = self.p.jump_rate
        
        # Add excitation from past events
        for t_i in self.history:
            if t > t_i:
                lambda_t += (self.p.excitation_strength * 
                           np.exp(-(t - t_i) / self.p.excitation_decay))
                
        return lambda_t
    
    def generate_jumps(self, t_start: float, t_end: float) -> List[float]:
        """Generate Hawkes process using thinning algorithm"""
        jumps = []
        t = t_start
        
        # Handle zero rate case
        if self.p.jump_rate <= 0:
            return jumps
            
        # Upper bound for thinning
        lambda_max = self.p.jump_rate * (1 + 10 * self.p.excitation_strength)
        
        while t < t_end:
            # Generate candidate
            dt = self.rng.exponential(1 / lambda_max)
            t += dt
            
            if t >= t_end:
                break
                
            # Accept/reject
            if self.rng.uniform() < self.intensity(t) / lambda_max:
                jumps.append(t)
                self.history.append(t)
                
        return jumps

@dataclass
class JumpDiffusionSim(Sim):
    """Simulation with both continuous dynamics and jump events"""
    params: JumpProcessParams
    jump_process: Optional[JumpProcess] = None
    
    def __post_init__(self):
        if self.jump_process is None:
            if isinstance(self.params, HawkesParams):
                self.jump_process = HawkesProcess(self.params)
            else:
                self.jump_process = JumpProcess(self.params)
    
    def jump_effect(self, t: float, state: np.ndarray) -> np.ndarray:
        """Effect of a jump on state - override in subclass"""
        return state
    
    def sim(self, callback: Optional[Callable] = None) -> None:
        """Simulate with jumps and optional callback at each event"""
        p = self.params
        
        # Generate jump times
        jump_times = self.jump_process.generate_jumps(
            mag(p.tspan[0]), mag(p.tspan[1])
        )
        
        # Merge continuous integration points with jumps
        all_times = sorted(set([mag(p.tspan[0])] + jump_times + [mag(p.tspan[1])]))
        
        # Initial conditions
        names, values = zip(*p.init_conds)
        state = np.array(mag(values))
        
        # Storage
        times = [all_times[0]]
        states = [state.copy()]
        
        # Integrate between jumps
        for i in range(len(all_times) - 1):
            t_start = all_times[i]
            t_end = all_times[i + 1]
            
            if t_end > t_start:
                # Continuous evolution
                sol = solve_ivp(
                    fun=lambda t, y: mag(self.func(t * U.years, 
                                                  tuple(y * U.dimensionless), p)),
                    t_span=[t_start, t_end],
                    y0=state,
                    method='RK45',
                    dense_output=True
                )
                
                # Update state
                state = sol.y[:, -1]
                
                # Store intermediate points
                t_dense = np.linspace(t_start, t_end, 10)[1:-1]
                for t in t_dense:
                    times.append(t)
                    states.append(sol.sol(t))
            
            # Apply jump if this is a jump time
            if t_end in jump_times:
                state = self.jump_effect(t_end, state)
                if callback:
                    callback(t_end, state, 'jump')
                    
            times.append(t_end)
            states.append(state.copy())
            
        # Create dataframe
        self.df = pd.DataFrame(np.array(states), columns=names)
        self.df['t'] = times
        self._add_outputs()
        self._graph()

# Utility functions for common patterns

def noise_barrier(error: float, threshold1: float, threshold2: float) -> float:
    """
    Apply noise barrier scaling (dead zone with linear ramp)
    
    Args:
        error: Input error signal
        threshold1: Below this, output is 0
        threshold2: Above this, output equals input
        
    Returns:
        Scaled error signal
    """
    abs_error = abs(error)
    if abs_error < threshold1:
        return 0.0
    elif abs_error < threshold2:
        scale = (abs_error - threshold1) / (threshold2 - threshold1)
        return scale * error if error >= 0 else -scale * abs(error)
    else:
        return error

def rate_limit(value: float, previous: float, max_change: float, dt: float) -> float:
    """Apply rate limiting to a signal"""
    max_delta = max_change * dt
    delta = value - previous
    if abs(delta) > max_delta:
        return previous + max_delta * np.sign(delta)
    return value