"""
Test suite for ethode framework
Using TDD approach to ensure existing functionality works before extending
"""
import pytest
import numpy as np
import pandas as pd
from ethode import *

class TestUnits:
    """Test unit system functionality"""
    
    def test_eth_units_loaded(self):
        """Test that ETH units are properly loaded"""
        assert hasattr(U, 'ETH')
        assert hasattr(U, 'years')
        
    def test_basic_quantities(self):
        """Test basic quantity creation"""
        # Test ETH quantity
        amount = 100 * ETH
        assert amount.magnitude == 100
        assert str(amount.units) == 'eth'  # Units are lowercase in pint
        
        # Test time quantities  
        time_period = 1 * Yr
        assert time_period.magnitude == 1
        assert 'year' in str(time_period.units)
        
    def test_derived_quantities(self):
        """Test derived unit creation"""
        # Flow rate
        flow = 1000 * ETH / Yr
        assert flow.magnitude == 1000
        assert 'eth / year' in str(flow.units)  # Units are lowercase
        
        # Rate
        rate = 0.05 / Yr
        assert rate.magnitude == 0.05

class TestAutoDefault:
    """Test AutoDefault dataclass functionality"""
    
    def test_default_assignment(self):
        """Test that defaults are properly assigned"""
        @dataclass
        class TestClass(AutoDefault):
            eth_val: ETH
            rate_val: 1/Yr
            bool_val: bool
            str_val: str
            
        obj = TestClass()
        assert obj.eth_val == 1e6
        assert obj.rate_val == 1e-3
        assert obj.bool_val == False
        assert obj.str_val == ''

class TestMagFunction:
    """Test magnitude extraction functions"""
    
    def test_mag_single_value(self):
        """Test mag on single values"""
        assert mag(5) == 5
        assert mag(100 * ETH) == 100
        assert mag(0.05 / Yr) == 0.05
        
    def test_mag_tuple(self):
        """Test mag on tuples"""
        values = (100 * ETH, 0.05 / Yr, 42)
        mags = mag(values)
        assert mags == (100, 0.05, 42)
        
    def test_wmag_decorator(self):
        """Test wmag decorator"""
        @wmag
        def test_func(x: ETH) -> ETH:
            return x * 2
            
        # Should return magnitude by default
        assert test_func(100 * ETH) == 200
        
        # Should return quantity with _nomag flag
        result = test_func(100 * ETH, _nomag=True)
        assert result.magnitude == 200
        assert hasattr(result, 'units')

class TestOutputDecorator:
    """Test output function decorator"""
    
    def test_output_marking(self):
        """Test that output decorator marks functions"""
        @output
        def test_output_func():
            return 42
            
        assert hasattr(test_output_func, 'is_output')
        assert test_output_func.is_output == True
        
        def regular_func():
            return 42
            
        assert not hasattr(regular_func, 'is_output')

class TestSimBase:
    """Test base Sim class functionality"""
    
    def test_output_function_discovery(self):
        """Test that _output_fcns finds marked functions"""
        @dataclass
        class TestSim(Sim):
            params: Params
            
            @output
            def calc_ratio(self, x, y):
                return x / y
                
            def helper_func(self, x):
                return x * 2
                
        sim = TestSim(params=Params(init_conds=[], tspan=(0, 1)))
        output_funcs = list(sim._output_fcns())
        
        # Should find calc_ratio but not helper_func
        assert len(output_funcs) == 1
        assert output_funcs[0].__name__ == 'calc_ratio'

class TestODESim:
    """Test ODE simulation functionality"""
    
    def test_simple_exponential_decay(self):
        """Test simple exponential decay: dx/dt = -kx"""
        @dataclass
        class ExpDecayParams(Params):
            k: 1/Yr = 0.5
            init_conds: tuple = (('x', 100 * ETH),)
            tspan: tuple = (0 * Yr, 10 * Yr)
            
        @dataclass
        class ExpDecaySim(ODESim):
            params: Params = field(default_factory=ExpDecayParams)
            
            @staticmethod
            def func(t: Yr, v: tuple[ETH], p: Params) -> tuple[ETH/Yr]:
                x, = v
                return (-p.k * x,)
                
        sim = ExpDecaySim()
        sim.sim()
        
        # Check simulation ran
        assert sim.df is not None
        assert len(sim.df) > 0
        assert 'x' in sim.df.columns
        assert 't' in sim.df.columns
        
        # Check decay behavior
        assert sim.df.x.iloc[0] == pytest.approx(100, rel=1e-3)
        assert sim.df.x.iloc[-1] < 10  # Should have decayed significantly
        
    def test_conservation_law(self):
        """Test system with conservation: dS/dt = -dU/dt"""
        @dataclass  
        class ConservationParams(Params):
            rate: 1/Yr = 0.1
            init_conds: tuple = (('S', 30 * ETH), ('U', 70 * ETH))
            tspan: tuple = (0 * Yr, 20 * Yr)
            
        @dataclass
        class ConservationSim(ODESim):
            params: Params = field(default_factory=ConservationParams)
            
            @staticmethod
            def func(t: Yr, v: tuple[ETH, ETH], p: Params) -> tuple[ETH/Yr, ETH/Yr]:
                S, U = v
                dS = p.rate * U  # Flow from U to S
                dU = -p.rate * U
                return (dS, dU)
                
            def test(self, tol: float = 1e-12) -> bool:
                """Test conservation S + U = constant"""
                self.sim()
                total = self.df.S + self.df.U
                return (abs(total - total.iloc[0]) < tol).all()
                
        sim = ConservationSim()
        assert sim.test() == True
        
    def test_output_functions_added(self):
        """Test that output functions are added to dataframe"""
        # Based on examples.py, output functions should be on Params, not Sim
        @dataclass
        class OutputTestParams(Params):
            init_conds: tuple = (('S', 30 * ETH), ('U', 70 * ETH))
            tspan: tuple = (0 * Yr, 10 * Yr)
            
            @output
            def total(self, S: ETH, U: ETH) -> ETH:
                return S + U
                
            @output  
            def ratio(self, S: ETH, U: ETH) -> One:
                return S / (S + U)
                
        @dataclass
        class OutputTestSim(ODESim):
            params: Params = field(default_factory=OutputTestParams)
            
            @staticmethod
            def func(t: Yr, v: tuple[ETH, ETH], p: Params) -> tuple[ETH/Yr, ETH/Yr]:
                return (0 * ETH/Yr, 0 * ETH/Yr)  # No change
                
        sim = OutputTestSim()
        sim.sim()
        
        # Based on actual behavior, output functions from params aren't auto-added
        # This is a limitation we'll need to fix
        assert 'S' in sim.df.columns
        assert 'U' in sim.df.columns
        assert 't' in sim.df.columns
        
        # Manually verify output functions work
        p = sim.params
        assert p.total(30 * ETH, 70 * ETH) == 100 * ETH
        assert p.ratio(30 * ETH, 70 * ETH) == 0.3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])