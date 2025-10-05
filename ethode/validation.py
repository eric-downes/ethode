"""Unit validation system for catching dimension mismatches early.

This module provides pint-based validation that runs before JAX compilation
to catch unit errors during development while keeping runtime performance.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import pint
from dataclasses import dataclass

from .units import UnitManager
from .runtime import ControllerRuntime, ControllerState


def controller_step_units(
    runtime: ControllerRuntime,
    state: ControllerState,
    error: pint.Quantity,
    dt: pint.Quantity,
    manager: Optional[UnitManager] = None
) -> Tuple[Dict[str, pint.Quantity], pint.Quantity]:
    """Pint-based validation of controller step logic.

    This function mirrors the logic of controller_step but operates on pint
    quantities to catch unit mismatches. It's meant to be run once during
    config validation, not during actual simulation.

    Args:
        runtime: Controller runtime with QuantityNodes
        state: Current controller state
        error: Error signal as pint Quantity
        dt: Time step as pint Quantity
        manager: UnitManager instance (uses singleton if None)

    Returns:
        Tuple of (state_dict, control_output) with pint quantities

    Raises:
        pint.DimensionalityError: If units don't match properly
    """
    if manager is None:
        manager = UnitManager.instance()

    # Convert runtime values to pint quantities
    kp = runtime.kp.to_quantity(manager)
    ki = runtime.ki.to_quantity(manager)
    kd = runtime.kd.to_quantity(manager)
    tau = runtime.tau.to_quantity(manager)
    noise_low = runtime.noise_band_low.to_quantity(manager)
    noise_high = runtime.noise_band_high.to_quantity(manager)

    # Convert state to quantities with appropriate units
    # The integral accumulates error*time, so its dimension depends on error's dimension
    integral_units = str(error.units) + " * second"
    integral = manager.registry.Quantity(float(state.integral), integral_units)
    last_error = manager.registry.Quantity(float(state.last_error), str(error.units))

    # Noise band logic (simplified for validation)
    abs_error = abs(error)
    if abs_error < noise_low:
        filtered_error = manager.registry.Quantity(0.0, error.units)
    elif abs_error > noise_high:
        filtered_error = error
    else:
        # Linear interpolation in band
        band_factor = (abs_error - noise_low) / (noise_high - noise_low)
        filtered_error = error * float(band_factor.magnitude)

    # Integral leak
    decay_ratio = dt / tau
    # Convert to dimensionless for exponential
    decay_factor = float(decay_ratio.to('dimensionless').magnitude)
    # Apply exponential decay
    import numpy as np
    leaked_integral = integral * np.exp(-decay_factor)

    # Update integral
    new_integral = leaked_integral + filtered_error * dt

    # Derivative
    if float(dt.magnitude) > 0:
        derivative = (filtered_error - last_error) / dt
    else:
        derivative = manager.registry.Quantity(0.0, 'USD/second')

    # PID terms - these operations will raise DimensionalityError if units mismatch
    p_term = kp * filtered_error
    i_term = ki * new_integral
    d_term = kd * derivative

    # Try to sum the terms - this will fail if they don't have compatible dimensions
    # But first, check if any terms need to be made dimensionless
    try:
        # Try direct addition first
        raw_output = p_term + i_term + d_term
    except pint.DimensionalityError:
        # If direct addition fails, try to find a common dimension
        # This can happen when gains are specified in non-standard ways

        # Get the dimensions of each term
        p_dim = p_term.dimensionality
        i_dim = i_term.dimensionality
        d_dim = d_term.dimensionality

        # Find the most likely output dimension
        # We need an actual unit, not just a dimensionality
        output_unit = None
        if p_term.magnitude != 0:
            output_unit = p_term.units
        elif i_term.magnitude != 0:
            output_unit = i_term.units
        elif d_term.magnitude != 0:
            output_unit = d_term.units
        else:
            # All terms are zero, use error units
            output_unit = error.units

        # Try to convert all terms to the output unit
        # This helps when units are specified in non-standard but compatible ways
        try:
            p_converted = p_term.to(output_unit) if p_term.magnitude != 0 else p_term
            i_converted = i_term.to(output_unit) if i_term.magnitude != 0 else i_term
            d_converted = d_term.to(output_unit) if d_term.magnitude != 0 else d_term
            raw_output = p_converted + i_converted + d_converted
        except pint.DimensionalityError:
            # If conversion also fails, report the incompatibility
            raise pint.DimensionalityError(
                p_term.units, i_term.units,
                extra_msg=f"PID terms have incompatible units: P={p_term.units}, I={i_term.units}, D={d_term.units}"
            )

    # Apply limits if specified
    if runtime.output_min is not None:
        output_min = runtime.output_min.to_quantity(manager)
        # This will fail if output and limit have incompatible units
        raw_output = max(raw_output, output_min)

    if runtime.output_max is not None:
        output_max = runtime.output_max.to_quantity(manager)
        raw_output = min(raw_output, output_max)

    # Rate limiting
    if runtime.rate_limit is not None:
        rate_limit = runtime.rate_limit.to_quantity(manager)
        max_change = rate_limit * dt
        # Simplified rate limiting for validation
        final_output = raw_output
    else:
        final_output = raw_output

    # Return updated state as dict and output
    state_dict = {
        'integral': new_integral,
        'last_error': filtered_error,
        'last_output': final_output,
        'time': manager.registry.Quantity(float(state.time), 'second') + dt
    }

    return state_dict, final_output


def validate_controller_dimensions(
    runtime: ControllerRuntime,
    manager: Optional[UnitManager] = None,
    dimensions: Optional['ControllerDimensions'] = None
) -> Dict[str, str]:
    """Validate that controller dimensions are self-consistent.

    Runs a dry-run with pint quantities to check dimensional consistency.

    Args:
        runtime: ControllerRuntime to validate
        manager: UnitManager instance
        dimensions: Optional dimension schema (uses default if not provided)

    Returns:
        Dict mapping term names to their dimensions

    Raises:
        pint.DimensionalityError: If dimensions don't match
        ValueError: If validation fails
    """
    if manager is None:
        manager = UnitManager.instance()

    # Use provided dimensions or create default from runtime
    if dimensions is None:
        from ethode.controller.dimensions import FINANCIAL
        dimensions = FINANCIAL

    # Map dimension names to pint units
    dimension_to_units = {
        "price": "USD",
        "temperature": "kelvin",
        "length": "meter",
        "force": "newton",
        "pressure": "pascal",
        "flow_rate": "liter/second",
        "dimensionless": "dimensionless",
        "time": "second",
        "price/time": "USD/second",
        "length/time": "meter/second",
    }

    # Get appropriate units for error from schema
    # If not in our map, try to use it directly as a unit string
    error_units = dimension_to_units.get(dimensions.error_dim, dimensions.error_dim)

    # Create test quantities
    test_error = manager.registry.Quantity(1.0, error_units)
    test_dt = manager.registry.Quantity(0.1, 'second')

    # Create zero state
    test_state = ControllerState.zero()

    # Run validation step
    try:
        state_dict, output = controller_step_units(
            runtime, test_state, test_error, test_dt, manager
        )

        # Extract dimensions
        dimensions = {
            'error': str(test_error.dimensionality),
            'dt': str(test_dt.dimensionality),
            'output': str(output.dimensionality),
            'integral': str(state_dict['integral'].dimensionality),
            'kp': str(runtime.kp.to_quantity(manager).dimensionality),
            'ki': str(runtime.ki.to_quantity(manager).dimensionality),
            'kd': str(runtime.kd.to_quantity(manager).dimensionality),
            'tau': str(runtime.tau.to_quantity(manager).dimensionality),
        }

        # Calculate what each PID term produces
        kp = runtime.kp.to_quantity(manager)
        ki = runtime.ki.to_quantity(manager)
        kd = runtime.kd.to_quantity(manager)

        # P term: kp * error
        p_term = kp * test_error

        # I term: ki * (error * time)
        test_integral = test_error * test_dt
        i_term = ki * test_integral

        # D term: kd * (error / time)
        test_derivative = test_error / test_dt
        d_term = kd * test_derivative

        # Check if terms have compatible dimensions
        # They might not be exactly the same but should be convertible
        output_dim = None
        incompatible = []

        # Find a common dimension
        for term, name in [(p_term, 'P'), (i_term, 'I'), (d_term, 'D')]:
            if term.magnitude == 0:
                continue  # Skip zero terms
            if output_dim is None:
                output_dim = term.dimensionality
            elif term.dimensionality != output_dim:
                try:
                    # Try to convert to output dimension
                    _ = term.to(output_dim)
                except pint.DimensionalityError:
                    incompatible.append(f"{name}: {term.dimensionality}")

        if incompatible and output_dim:
            # Check if it's just a case of one term being zero
            non_zero_terms = [
                (p_term, 'P'), (i_term, 'I'), (d_term, 'D')
            ]
            non_zero_terms = [(t, n) for t, n in non_zero_terms if t.magnitude != 0]

            if len(non_zero_terms) == 1:
                # Only one non-zero term, that's OK
                pass
            else:
                # Multiple non-zero terms with incompatible dimensions
                raise ValueError(
                    f"PID terms have incompatible dimensions:\n"
                    f"  P term: {p_term.dimensionality}\n"
                    f"  I term: {i_term.dimensionality}\n"
                    f"  D term: {d_term.dimensionality}\n"
                    f"  Incompatible: {incompatible}"
                )

        return dimensions

    except pint.DimensionalityError as e:
        raise ValueError(f"Controller dimension validation failed: {e}")


@dataclass
class ValidationReport:
    """Report from unit validation."""
    success: bool
    dimensions: Dict[str, str]
    warnings: list[str]
    errors: list[str]

    def __str__(self) -> str:
        """Format as readable report."""
        lines = ["=== Unit Validation Report ==="]
        lines.append(f"Status: {'PASS' if self.success else 'FAIL'}")

        if self.dimensions:
            lines.append("\nDimensions:")
            for key, dim in self.dimensions.items():
                lines.append(f"  {key}: {dim}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")

        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ❌ {e}")

        return "\n".join(lines)


def validate_config_units(config: Any, verbose: bool = False) -> ValidationReport:
    """Validate units in a config object.

    Args:
        config: Config object with to_runtime() method
        verbose: If True, print validation details

    Returns:
        ValidationReport with results
    """
    report = ValidationReport(
        success=True,
        dimensions={},
        warnings=[],
        errors=[]
    )

    try:
        # Build runtime to check conversions work
        runtime = config.to_runtime(check_units=False)  # Don't recurse

        # If it's a controller config, do deeper validation
        if hasattr(config, 'kp'):  # ControllerConfig
            try:
                # Pass dimensions schema if available
                config_dimensions = getattr(config, 'dimensions', None)
                dimensions = validate_controller_dimensions(runtime, dimensions=config_dimensions)
                report.dimensions = dimensions

                # Only warn about dimensionless if not explicitly using dimensionless schema
                if config_dimensions and config_dimensions.error_dim == "dimensionless" and config_dimensions.output_dim == "dimensionless":
                    # Dimensionless schema is intentional, no warning
                    pass
                elif 'dimensionless' in str(dimensions.get('kp', '')):
                    report.warnings.append(
                        "kp appears dimensionless - consider using frequency units like '1/hour' or specify a dimensions schema"
                    )

            except ValueError as e:
                report.success = False
                report.errors.append(str(e))
            except Exception as e:
                # Catch any other exception
                report.success = False
                report.errors.append(f"Validation error: {type(e).__name__}: {e}")

    except Exception as e:
        report.success = False
        import traceback
        error_msg = f"Failed to build runtime: {type(e).__name__}: {str(e)}"
        if verbose:
            error_msg += f"\n{traceback.format_exc()}"
        report.errors.append(error_msg)

    if verbose:
        print(report)

    return report