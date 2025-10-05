"""Unit management system for ethode using pint and Penzai.

This module provides the foundation for unit-aware computations with:
- UnitManager: Singleton registry management and unit conversions
- UnitSpec: Metadata for units that survives JAX transformations
- Conversion utilities between pint quantities and canonical floats
"""

from __future__ import annotations

import pint
from pint import UndefinedUnitError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Any, ClassVar
import jax.numpy as jnp
import penzai as pz

# Type alias for inputs that can be converted to quantities
QuantityInput = Union[str, float, int, pint.Quantity]


@dataclass(frozen=True)
class UnitSpec:
    """Immutable metadata for units that can be attached to JAX arrays.

    Attributes:
        dimension: Physical dimension string (e.g., "time", "length", "price")
        symbol: Unit symbol string (e.g., "s", "m", "USD")
        to_canonical: Factor to convert from this unit to canonical
    """
    dimension: str
    symbol: str
    to_canonical: float = 1.0

    def __hash__(self):
        return hash((self.dimension, self.symbol, self.to_canonical))


class UnitManager:
    """Manages unit registry and conversions for the ethode system.

    Provides:
    - Singleton pint.UnitRegistry access
    - Canonical unit definitions per dimension
    - Conversion utilities to/from canonical floats
    - Custom unit loading from definition files
    """

    _instance: ClassVar[Optional[UnitManager]] = None

    def __init__(self, registry: Optional[pint.UnitRegistry] = None):
        """Initialize with optional custom registry.

        Args:
            registry: Custom pint registry. If None, creates default.
        """
        self.registry = registry or pint.UnitRegistry()

        # Common aliases for convenience (must come before canonical_units)
        self._setup_aliases()

        # Define canonical units for each dimension
        # These are the internal units we convert everything to
        self.canonical_units = {
            "time": self.registry.second,
            "frequency": self.registry.Hz,  # 1/time
            "1/time": self.registry.Hz,  # Alternative name for frequency
            "length": self.registry.meter,
            "mass": self.registry.kilogram,
            "temperature": self.registry.kelvin,
            "current": self.registry.ampere,
            "substance": self.registry.mole,
            "luminosity": self.registry.candela,
            "price": self.registry.USD,  # Financial dimension
            "dimensionless": self.registry.dimensionless,
            # Mixed dimensions
            "price/time": self.registry.USD / self.registry.second,
            "volume/time": self.registry.dimensionless / self.registry.second,  # tokens/second
            "price*time": self.registry.USD * self.registry.second,
        }

        # Load custom definitions if available
        self.load_custom_units()

    @classmethod
    def instance(cls) -> UnitManager:
        """Get or create the singleton instance.

        Returns:
            The global UnitManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _setup_aliases(self) -> None:
        """Set up common unit aliases."""
        # Financial units (must be defined first)
        try:
            # Check if USD exists
            _ = self.registry.USD
        except (AttributeError, pint.UndefinedUnitError):
            # Define basic currency units
            self.registry.define('USD = [currency]')
            self.registry.define('EUR = 1.1 * USD')  # Approximate
            self.registry.define('GBP = 1.3 * USD')  # Approximate

        # Time aliases
        if not hasattr(self.registry, 'day'):
            self.registry.define('day = 24 * hour')
        if not hasattr(self.registry, 'week'):
            self.registry.define('week = 7 * day')
        if not hasattr(self.registry, 'year'):
            self.registry.define('year = 365.25 * day')

        # Financial percentages - basis points
        try:
            # Define percent if not already defined
            if not hasattr(self.registry, 'percent'):
                self.registry.define('percent = 0.01 = %')

            # Define basis point (1/100th of a percent)
            if not hasattr(self.registry, 'basis_point'):
                self.registry.define('basis_point = 0.0001 = bps = bp')
        except (pint.DefinitionSyntaxError, pint.RedefinitionError):
            # May already be defined
            pass

        # Large number prefixes for financial units
        try:
            if not hasattr(self.registry, 'million'):
                self.registry.define('million = 1e6')
            if not hasattr(self.registry, 'billion'):
                self.registry.define('billion = 1e9')
            if not hasattr(self.registry, 'thousand'):
                self.registry.define('thousand = 1e3 = k')
        except (pint.DefinitionSyntaxError, pint.RedefinitionError):
            # May already be defined
            pass

    def load_custom_units(self, paths: Optional[list[Path]] = None) -> None:
        """Load custom unit definitions from files.

        Args:
            paths: List of paths to unit definition files.
                   If None, looks for eth_units.txt and btc_units.txt
                   in the same directory as this module.

        Raises:
            FileNotFoundError: If a specified path doesn't exist
            Exception: If unit definitions cannot be loaded
        """
        if paths is None:
            # Look for unit files next to this module
            module_dir = Path(__file__).parent
            default_paths = [
                module_dir / 'eth_units.txt',
                module_dir / 'btc_units.txt',
            ]
            # Only try to load files that exist
            paths = [p for p in default_paths if p.exists()]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Unit definition file not found: {path}")

            self.registry.load_definitions(str(path))

    def load_aliases(self, path: Path) -> None:
        """Load additional unit aliases from a file.

        Args:
            path: Path to alias definition file

        Raises:
            FileNotFoundError: If path doesn't exist
            Exception: If definitions cannot be loaded
        """
        if not path.exists():
            raise FileNotFoundError(f"Alias definition file not found: {path}")

        self.registry.load_definitions(str(path))

    def ensure_quantity(
        self,
        value: QuantityInput,
        default_unit: Optional[str] = None
    ) -> pint.Quantity:
        """Convert input to a pint Quantity.

        Args:
            value: String, number, or Quantity to convert
            default_unit: Unit to use if value is dimensionless

        Returns:
            pint.Quantity object

        Raises:
            ValueError: If string cannot be parsed as quantity
        """
        if isinstance(value, pint.Quantity):
            return value
        elif isinstance(value, str):
            # Handle financial shorthand
            # Replace M/k/K with proper multipliers for financial contexts
            if 'USD' in value or '$' in value or 'EUR' in value or 'GBP' in value:
                import re
                # Handle patterns like "500k USD" -> "500000 USD"
                value = re.sub(r'(\d+(?:\.\d+)?)M\s*(USD|EUR|GBP|\$)', r'\g<1>e6 \2', value)
                value = re.sub(r'(\d+(?:\.\d+)?)k\s*(USD|EUR|GBP|\$)', r'\g<1>e3 \2', value)
                value = re.sub(r'(\d+(?:\.\d+)?)K\s*(USD|EUR|GBP|\$)', r'\g<1>e3 \2', value)
                # Alternative: multiply the number directly
                # This handles patterns without space: "500kUSD"
                value = re.sub(r'(\d+(?:\.\d+)?)M(USD|EUR|GBP)', r'\g<1>e6 \2', value)
                value = re.sub(r'(\d+(?:\.\d+)?)k(USD|EUR|GBP)', r'\g<1>e3 \2', value)

            # Parse string as quantity
            try:
                q = self.registry(value)
                # If parsing returns just a number, wrap it as dimensionless
                if not isinstance(q, pint.Quantity):
                    q = self.registry.Quantity(q, default_unit or 'dimensionless')
                return q
            except Exception as e:
                raise ValueError(f"Cannot parse '{value}' as quantity: {e}")
        else:
            # Numeric value - apply default unit if provided
            if default_unit:
                return self.registry.Quantity(value, default_unit)
            else:
                return self.registry.Quantity(value, 'dimensionless')

    def to_canonical(
        self,
        quantity: pint.Quantity,
        dimension: str
    ) -> tuple[float, UnitSpec]:
        """Convert quantity to canonical units for dimension.

        Args:
            quantity: pint Quantity to convert
            dimension: Target dimension name

        Returns:
            Tuple of (canonical_value, unit_spec)

        Raises:
            ValueError: If quantity dimension doesn't match target
        """
        if dimension not in self.canonical_units:
            # For unknown dimensions, use the quantity as-is
            return (
                quantity.magnitude,
                UnitSpec(
                    dimension=dimension,
                    symbol=str(quantity.units),
                    to_canonical=1.0
                )
            )

        canonical_unit = self.canonical_units[dimension]

        # Convert to canonical units
        try:
            canonical_quantity = quantity.to(canonical_unit)
        except pint.DimensionalityError as e:
            raise ValueError(
                f"Cannot convert {quantity} to dimension '{dimension}': {e}"
            )

        # Calculate conversion factor from units (not magnitudes)
        # This avoids division by zero and works correctly for all values
        # e.g., for hours to seconds: 1 hour = 3600 seconds, so factor = 3600
        original_units = quantity.units
        one_original = self.registry.Quantity(1.0, original_units)
        one_in_canonical = one_original.to(canonical_unit)
        conversion_factor = float(one_in_canonical.magnitude)

        return (
            canonical_quantity.magnitude,
            UnitSpec(
                dimension=dimension,
                symbol=str(original_units),
                to_canonical=conversion_factor
            )
        )

    def from_canonical(
        self,
        value: float,
        spec: UnitSpec
    ) -> pint.Quantity:
        """Reconstruct pint Quantity from canonical value and spec.

        Args:
            value: Canonical float value
            spec: UnitSpec with dimension and symbol info

        Returns:
            pint.Quantity in original units
        """
        # to_canonical tells us: original * to_canonical = canonical
        # So: original = canonical / to_canonical
        original_value = value / spec.to_canonical if spec.to_canonical != 0 else value

        # Create quantity with original units
        return self.registry.Quantity(original_value, spec.symbol)

    def get_canonical_unit(self, dimension: str) -> pint.Unit:
        """Get the canonical unit for a dimension.

        Args:
            dimension: Dimension name

        Returns:
            Canonical pint.Unit for dimension
        """
        if dimension not in self.canonical_units:
            return self.registry.dimensionless
        return self.canonical_units[dimension]

    def infer_dimension(self, quantity: pint.Quantity) -> str:
        """Infer dimension name from quantity.

        Args:
            quantity: pint Quantity to analyze

        Returns:
            Best matching dimension name
        """
        # Check dimensionality against known canonical units
        for dim_name, canonical_unit in self.canonical_units.items():
            try:
                # If conversion succeeds, this is the right dimension
                _ = quantity.to(canonical_unit)
                return dim_name
            except pint.DimensionalityError:
                continue

        # Unknown dimension - use string representation
        return str(quantity.dimensionality)