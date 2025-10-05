"""Controller dimension schema for unit validation.

This module provides a schema system that allows users to declare the dimensions
of their controller signals, enabling automatic derivation of correct gain dimensions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ControllerDimensions:
    """Schema for controller signal dimensions.

    This class defines the dimensions of controller signals, allowing the system
    to automatically derive the correct dimensions for gains and validate consistency.

    Attributes:
        error_dim: Dimension of the error signal (e.g., "price", "temperature", "dimensionless")
        output_dim: Dimension of the controller output (e.g., "price/time", "dimensionless")
        time_dim: Dimension of time (default: "time")

    Examples:
        # Financial controller: USD error → USD/time output
        ControllerDimensions(error_dim="price", output_dim="price/time")

        # Temperature controller: kelvin error → dimensionless output (heater %)
        ControllerDimensions(error_dim="temperature", output_dim="dimensionless")

        # Dimensionless controller: dimensionless error → dimensionless output
        ControllerDimensions(error_dim="dimensionless", output_dim="dimensionless")
    """

    error_dim: str = "price"           # Default: financial error (USD)
    output_dim: str = "price/time"     # Default: financial rate output (USD/time)
    time_dim: str = "time"             # Default: standard time dimension

    def expected_gain_dimension(self, gain_type: str) -> str:
        """Calculate the expected dimension for a gain parameter.

        Based on the dimensional analysis of PID control:
        - P term: kp * error = output → kp has dimension output/error
        - I term: ki * (error * time) = output → ki has dimension output/(error * time)
        - D term: kd * (error / time) = output → kd has dimension (output * time)/error

        Args:
            gain_type: Type of gain ("kp", "ki", or "kd")

        Returns:
            Expected dimension string for the gain

        Raises:
            ValueError: If gain_type is not recognized
        """
        if gain_type == "kp":
            # Proportional: output/error
            if self.output_dim == "dimensionless" and self.error_dim == "dimensionless":
                return "dimensionless"
            elif self.output_dim == self.error_dim:
                return "dimensionless"
            elif self.error_dim == "dimensionless":
                return self.output_dim
            elif self.output_dim == "dimensionless":
                return f"1/{self.error_dim}"
            # Simplify common cases
            elif self.error_dim == "price" and self.output_dim == "price/time":
                return "1/time"
            elif self.error_dim == "temperature" and self.output_dim == "dimensionless":
                return "1/temperature"
            elif self.error_dim == "length" and self.output_dim == "length/time":
                return "1/time"
            elif self.error_dim == "length" and self.output_dim == "force":
                return "force/length"
            else:
                # General case: output/error
                return f"({self.output_dim})/({self.error_dim})"

        elif gain_type == "ki":
            # Integral: output/(error * time)
            if self.output_dim == "dimensionless" and self.error_dim == "dimensionless":
                return "1/time"
            elif self.error_dim == "price" and self.output_dim == "price/time":
                # Special case: output is already error/time
                return "1/time^2"
            elif self.error_dim == "dimensionless":
                if self.output_dim == "price/time":
                    return "price/time^2"
                else:
                    return f"({self.output_dim})/time"
            elif self.output_dim == "dimensionless":
                if self.error_dim == "temperature":
                    return "1/(temperature*time)"
                else:
                    return f"1/({self.error_dim}*time)"
            elif self.error_dim == "length" and self.output_dim == "length/time":
                return "1/time^2"
            elif self.error_dim == "length" and self.output_dim == "force":
                return "force/(length*time)"
            else:
                # General case
                return f"({self.output_dim})/({self.error_dim}*time)"

        elif gain_type == "kd":
            # Derivative: (output * time)/error
            if self.output_dim == "dimensionless" and self.error_dim == "dimensionless":
                return "time"
            elif self.error_dim == "price" and self.output_dim == "price/time":
                # Special case: output is error/time
                return "dimensionless"
            elif self.error_dim == "dimensionless":
                return f"({self.output_dim})*time"
            elif self.output_dim == "dimensionless":
                if self.error_dim == "temperature":
                    return "time/temperature"
                else:
                    return f"time/{self.error_dim}"
            elif self.error_dim == "length" and self.output_dim == "length/time":
                return "dimensionless"
            elif self.error_dim == "length" and self.output_dim == "force":
                return "(force*time)/length"
            else:
                # General case
                return f"({self.output_dim}*time)/({self.error_dim})"

        else:
            raise ValueError(f"Unknown gain type: {gain_type}")

    def expected_dimensions(self) -> Dict[str, str]:
        """Get expected dimensions for all parameters.

        Returns:
            Dictionary mapping parameter names to expected dimensions
        """
        return {
            "kp": self.expected_gain_dimension("kp"),
            "ki": self.expected_gain_dimension("ki"),
            "kd": self.expected_gain_dimension("kd"),
            "tau": self.time_dim,
            "noise_band": self.error_dim,
            "output": self.output_dim,
            "output_min": self.output_dim,
            "output_max": self.output_dim,
            "rate_limit": f"{self.output_dim}/{self.time_dim}",
        }

    @classmethod
    def from_string(cls, spec: str) -> ControllerDimensions:
        """Create from a string specification.

        Args:
            spec: String like "temperature->dimensionless" or "price->price/time"

        Returns:
            ControllerDimensions instance
        """
        if "->" in spec:
            error_dim, output_dim = spec.split("->")
            return cls(error_dim=error_dim.strip(), output_dim=output_dim.strip())
        else:
            # Assume it's just the error dimension with default output
            return cls(error_dim=spec.strip())

    def __str__(self) -> str:
        """String representation."""
        return f"{self.error_dim} → {self.output_dim}"

    def validate_gain_dimension(self, gain_type: str, actual_dim: str) -> bool:
        """Check if an actual dimension matches the expected for a gain.

        Args:
            gain_type: Type of gain ("kp", "ki", or "kd")
            actual_dim: The actual dimension string from the config

        Returns:
            True if dimensions match or are compatible
        """
        expected = self.expected_gain_dimension(gain_type)

        # Direct match
        if actual_dim == expected:
            return True

        # Check common equivalences
        equivalences = {
            "1/time": ["frequency", "1/second", "Hz"],
            "1/time^2": ["1/second^2", "Hz^2", "1/second/second"],
            "dimensionless": ["", "1"],
            "time": ["second", "seconds"],
        }

        for canonical, variants in equivalences.items():
            if expected == canonical and actual_dim in variants:
                return True
            if actual_dim == canonical and expected in variants:
                return True

        return False


# Common dimension schemas for convenience
DIMENSIONLESS = ControllerDimensions(
    error_dim="dimensionless",
    output_dim="dimensionless"
)

FINANCIAL = ControllerDimensions(
    error_dim="price",
    output_dim="price/time"
)

TEMPERATURE = ControllerDimensions(
    error_dim="temperature",
    output_dim="dimensionless"
)

VELOCITY = ControllerDimensions(
    error_dim="length",
    output_dim="length/time"
)

POSITION = ControllerDimensions(
    error_dim="length",
    output_dim="force"
)