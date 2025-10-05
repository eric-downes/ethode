"""
Tests for ethode stochastic extensions
"""

import numpy as np
import pytest
from stochastic_extensions import (
    PIDController, PIDParams, JumpProcess, HawkesProcess, 
    JumpProcessParams, HawkesParams, noise_barrier, rate_limit
)

class TestPIDController:
    """Test PID controller functionality"""
    
    def test_proportional_only(self):
        """Test P-only controller"""
        params = PIDParams(kp=2.0, ki=0.0, kd=0.0)
        pid = PIDController(params)
        
        # Test proportional response
        output = pid.update(error=0.5, dt=1.0)
        assert output == 1.0  # kp * error = 2.0 * 0.5
        
    def test_integral_accumulation(self):
        """Test integral accumulation"""
        params = PIDParams(kp=0.0, ki=1.0, kd=0.0)
        pid = PIDController(params)
        
        # Accumulate error
        pid.update(error=1.0, dt=1.0)
        assert pid.integral == 1.0
        
        output = pid.update(error=1.0, dt=1.0)
        assert pid.integral == 2.0
        assert output == 2.0
        
    def test_integral_leak(self):
        """Test integral decay"""
        params = PIDParams(kp=0.0, ki=1.0, kd=0.0, integral_leak=0.1)
        pid = PIDController(params)
        
        # Set integral
        pid.integral = 1.0
        
        # Update with zero error
        pid.update(error=0.0, dt=1.0)
        
        # Integral should decay
        expected = np.exp(-0.1)
        assert abs(pid.integral - expected) < 1e-6
        
    def test_saturation_antiwindup(self):
        """Test anti-windup when output saturates"""
        params = PIDParams(kp=0.0, ki=1.0, kd=0.0,
                          output_min=-1.0, output_max=1.0)
        pid = PIDController(params)

        # Large error should saturate
        output = pid.update(error=10.0, dt=1.0)
        # With anti-windup, if integration is prevented, output won't saturate
        # The integral should stay at 0 when output would saturate
        assert pid.integral == 0.0  # Anti-windup prevented integration
        # Output would be 0 since integral=0, kp=0, kd=0
        assert output == 0.0  # No integration means no output with these gains
        
    def test_derivative(self):
        """Test derivative action"""
        params = PIDParams(kp=0.0, ki=0.0, kd=1.0)
        pid = PIDController(params)
        
        # First update
        pid.update(error=0.0, dt=1.0)
        
        # Step change
        output = pid.update(error=1.0, dt=1.0)
        assert output == 1.0  # de/dt = 1.0
        
    def test_noise_threshold(self):
        """Test dead zone"""
        params = PIDParams(kp=1.0, ki=0.0, kd=0.0, noise_threshold=0.1)
        pid = PIDController(params)
        
        # Small error should be ignored
        output = pid.update(error=0.05, dt=1.0)
        assert output == 0.0
        
        # Large error should pass through
        output = pid.update(error=0.5, dt=1.0)
        assert output == 0.5

class TestJumpProcesses:
    """Test jump process generation"""
    
    def test_poisson_process_rate(self):
        """Test Poisson process average rate"""
        params = JumpProcessParams(jump_rate=100.0, seed=42)
        process = JumpProcess(params)
        
        # Generate many jumps
        jumps = process.generate_jumps(0, 10)  # 10 time units
        
        # Check average rate (should be close to 100 per unit)
        avg_rate = len(jumps) / 10
        assert 90 < avg_rate < 110  # Within 10% 
        
    def test_poisson_exponential_intervals(self):
        """Test that Poisson intervals are exponential"""
        params = JumpProcessParams(jump_rate=50.0, seed=42)
        process = JumpProcess(params)
        
        jumps = process.generate_jumps(0, 100)
        if len(jumps) > 1:
            intervals = np.diff([0] + jumps)
            
            # Exponential distribution test: mean ≈ 1/rate
            expected_mean = 1.0 / 50.0
            actual_mean = np.mean(intervals)
            assert abs(actual_mean - expected_mean) / expected_mean < 0.2
            
    def test_deterministic_process(self):
        """Test deterministic jump spacing"""
        params = JumpProcessParams(jump_rate=10.0, jump_process_type='deterministic')
        process = JumpProcess(params)
        
        jumps = process.generate_jumps(0, 0.95)  # Stop before 1.0 to avoid boundary
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        assert len(jumps) == len(expected)
        for j, e in zip(jumps, expected):
            assert abs(j - e) < 1e-10
            
    def test_hawkes_process_clustering(self):
        """Test Hawkes process shows clustering"""
        # High excitation for clear clustering
        params = HawkesParams(jump_rate=10.0, excitation_strength=0.8, 
                             excitation_decay=0.1, seed=42)
        process = HawkesProcess(params)
        
        jumps = process.generate_jumps(0, 10)
        
        if len(jumps) > 10:
            # Calculate inter-arrival times
            intervals = np.diff([0] + jumps)
            
            # Clustering should create more variance than Poisson
            cv = np.std(intervals) / np.mean(intervals)
            assert cv > 1.0  # CV > 1 indicates clustering
            
    def test_hawkes_intensity_calculation(self):
        """Test Hawkes intensity function"""
        params = HawkesParams(jump_rate=10.0, excitation_strength=0.5, 
                             excitation_decay=1.0)
        process = HawkesProcess(params)
        
        # Base intensity
        assert process.intensity(0) == 10.0
        
        # Add event to history
        process.history = [1.0]
        
        # Intensity should increase
        intensity_at_1_5 = process.intensity(1.5)
        expected = 10.0 + 0.5 * np.exp(-0.5)
        assert abs(intensity_at_1_5 - expected) < 1e-6

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_noise_barrier(self):
        """Test noise barrier scaling"""
        # Below threshold1
        assert noise_barrier(0.0005, 0.001, 0.003) == 0.0
        assert noise_barrier(-0.0005, 0.001, 0.003) == 0.0
        
        # Between thresholds - linear scaling
        result = noise_barrier(0.002, 0.001, 0.003)
        expected = 0.5 * 0.002  # 50% scaling
        assert abs(result - expected) < 1e-10
        
        # Above threshold2
        assert noise_barrier(0.005, 0.001, 0.003) == 0.005
        assert noise_barrier(-0.005, 0.001, 0.003) == -0.005
        
    def test_rate_limit(self):
        """Test rate limiting"""
        # Small change - passes through
        result = rate_limit(value=1.1, previous=1.0, max_change=0.5, dt=1.0)
        assert result == 1.1
        
        # Large positive change - limited
        result = rate_limit(value=2.0, previous=1.0, max_change=0.5, dt=1.0)
        assert result == 1.5
        
        # Large negative change - limited
        result = rate_limit(value=0.0, previous=1.0, max_change=0.5, dt=1.0)
        assert result == 0.5
        
        # Scaling with dt
        result = rate_limit(value=1.2, previous=1.0, max_change=0.5, dt=0.1)
        assert result == 1.05  # Limited to 0.5 * 0.1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])