"""Hawkes process configuration with unit-aware Pydantic models.

This module provides configuration for self-exciting Hawkes processes,
which model clustered event arrivals where each event increases the
probability of future events.
"""

from __future__ import annotations
from typing import Tuple, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
import pint

from ..units import UnitManager
from ..fields import quantity_field
from ..runtime import UnitSpec, QuantityNode
from .runtime import HawkesRuntime, HawkesState


class HawkesConfig(BaseModel):
    """Configuration for Hawkes self-exciting point process.

    The Hawkes process intensity is:
        λ(t) = λ₀ + Σ α * exp(-(t - tᵢ) / τ)

    Where:
        - λ₀ is the base intensity (jump_rate)
        - α is the excitation strength
        - τ is the decay time constant
        - tᵢ are past event times

    Example:
        >>> config = HawkesConfig(
        ...     jump_rate="100 / hour",
        ...     excitation_strength=0.3,
        ...     excitation_decay="5 minutes",
        ...     max_intensity="1000 / hour"
        ... )
    """

    # Base intensity
    jump_rate: Tuple[float, UnitSpec] = Field(
        description="Base event rate (events/time)"
    )

    # Excitation parameters
    excitation_strength: Tuple[float, UnitSpec] = Field(
        description="How much each event increases intensity (dimensionless, < 1 for stability)"
    )

    excitation_decay: Tuple[float, UnitSpec] = Field(
        description="Time constant for excitation decay"
    )

    # Bounds
    max_intensity: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Maximum allowed intensity"
    )

    min_intensity: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Minimum intensity floor"
    )

    # Event impact
    event_impact_mean: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Mean impact of each event (e.g., price change)"
    )

    event_impact_std: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Standard deviation of event impact"
    )

    # Clustering parameters
    cluster_decay_rate: Optional[Tuple[float, UnitSpec]] = Field(
        default=None,
        description="Rate of cluster dissipation"
    )

    # Random seed for reproducibility
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for event generation (optional)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Validators
    _validate_jump_rate = field_validator("jump_rate", mode="before")(
        quantity_field("1/time", "1/second")
    )

    _validate_excitation = field_validator("excitation_strength", mode="before")(
        quantity_field("dimensionless", "dimensionless")
    )

    _validate_decay = field_validator("excitation_decay", mode="before")(
        quantity_field("time", "second")
    )

    @field_validator("max_intensity", mode="before")
    def _validate_max_intensity(cls, v):
        if v is None:
            return None
        return quantity_field("1/time", "1/second")(v, None)

    @field_validator("min_intensity", mode="before")
    def _validate_min_intensity(cls, v):
        if v is None:
            return None
        return quantity_field("1/time", "1/second")(v, None)

    @field_validator("event_impact_mean", mode="before")
    def _validate_impact_mean(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("event_impact_std", mode="before")
    def _validate_impact_std(cls, v):
        if v is None:
            return None
        return quantity_field("price", "USD")(v, None)

    @field_validator("cluster_decay_rate", mode="before")
    def _validate_cluster_decay(cls, v):
        if v is None:
            return None
        return quantity_field("1/time", "1/second")(v, None)

    @field_validator("excitation_strength", mode="after")
    def _validate_stability(cls, value: Tuple[float, UnitSpec]) -> Tuple[float, UnitSpec]:
        """Ensure excitation strength < 1 for process stability."""
        if value[0] >= 1.0:
            raise ValueError(
                f"Excitation strength must be < 1 for stability, got {value[0]}. "
                "Values >= 1 can lead to explosive event rates."
            )
        if value[0] < 0:
            raise ValueError(f"Excitation strength must be >= 0, got {value[0]}")

        return value

    @field_validator("jump_rate", "excitation_decay", mode="after")
    def _validate_positive(cls, value: Tuple[float, UnitSpec]) -> Tuple[float, UnitSpec]:
        """Ensure rates and times are positive."""
        if value[0] <= 0:
            raise ValueError(f"Value must be positive, got {value[0]}")
        return value

    def to_runtime(self, manager: Optional[UnitManager] = None) -> HawkesRuntime:
        """Convert to runtime structure for JAX.

        Args:
            manager: Optional UnitManager instance

        Returns:
            HawkesRuntime structure with QuantityNodes
        """
        def to_node(value_spec: Optional[Tuple[float, UnitSpec]]) -> Optional[QuantityNode]:
            if value_spec is None:
                return None
            return QuantityNode.from_float(value_spec[0], value_spec[1])

        return HawkesRuntime(
            jump_rate=to_node(self.jump_rate),
            excitation_strength=to_node(self.excitation_strength),
            excitation_decay=to_node(self.excitation_decay),
            max_intensity=to_node(self.max_intensity),
            min_intensity=to_node(self.min_intensity),
            event_impact_mean=to_node(self.event_impact_mean),
            event_impact_std=to_node(self.event_impact_std),
            cluster_decay_rate=to_node(self.cluster_decay_rate),
        )

    @staticmethod
    def from_runtime(runtime: HawkesRuntime, manager: Optional[UnitManager] = None) -> HawkesConfigOutput:
        """Create output config from runtime structure.

        Args:
            runtime: HawkesRuntime to convert
            manager: Optional UnitManager instance

        Returns:
            HawkesConfigOutput with pint quantities
        """
        if manager is None:
            manager = UnitManager.instance()

        def to_quantity(node: Optional[QuantityNode]) -> Optional[pint.Quantity]:
            if node is None:
                return None
            return manager.from_canonical(float(node.value), node.units)

        return HawkesConfigOutput(
            jump_rate=to_quantity(runtime.jump_rate),
            excitation_strength=to_quantity(runtime.excitation_strength),
            excitation_decay=to_quantity(runtime.excitation_decay),
            max_intensity=to_quantity(runtime.max_intensity),
            min_intensity=to_quantity(runtime.min_intensity),
            event_impact_mean=to_quantity(runtime.event_impact_mean),
            event_impact_std=to_quantity(runtime.event_impact_std),
            cluster_decay_rate=to_quantity(runtime.cluster_decay_rate),
        )

    def summary(self, format: str = "markdown") -> str:
        """Generate summary of Hawkes configuration.

        Args:
            format: Output format ('markdown', 'text', or 'dict')

        Returns:
            Formatted summary string
        """
        if format == "dict":
            return str(self.model_dump())

        manager = UnitManager.instance()
        lines = []

        if format == "markdown":
            lines.append("# Hawkes Process Configuration\n")
            lines.append("| Parameter | Value | Units |")
            lines.append("|-----------|--------|-------|")

            def format_row(name: str, value: Optional[Tuple[float, UnitSpec]]) -> str:
                if value is None:
                    return f"| {name} | - | - |"
                qty = manager.from_canonical(value[0], value[1])
                return f"| {name} | {qty.magnitude:.4g} | {qty.units} |"

        else:  # text format
            lines.append("Hawkes Process Configuration")
            lines.append("-" * 40)

            def format_row(name: str, value: Optional[Tuple[float, UnitSpec]]) -> str:
                if value is None:
                    return f"  {name}: -"
                qty = manager.from_canonical(value[0], value[1])
                return f"  {name}: {qty.magnitude:.4g} {qty.units}"

        lines.append(format_row("Jump rate", self.jump_rate))
        lines.append(format_row("Excitation strength", self.excitation_strength))
        lines.append(format_row("Excitation decay", self.excitation_decay))

        if self.max_intensity:
            lines.append(format_row("Max intensity", self.max_intensity))
        if self.event_impact_mean:
            lines.append(format_row("Event impact mean", self.event_impact_mean))

        # Add stability check
        lines.append("")
        stability = "STABLE" if self.excitation_strength[0] < 1 else "UNSTABLE"
        lines.append(f"Stability: {stability} (α = {self.excitation_strength[0]:.3f})")

        return "\n".join(lines)


class HawkesConfigOutput(BaseModel):
    """Output format for Hawkes configuration with pint quantities."""

    jump_rate: pint.Quantity
    excitation_strength: pint.Quantity
    excitation_decay: pint.Quantity
    max_intensity: Optional[pint.Quantity] = None
    min_intensity: Optional[pint.Quantity] = None
    event_impact_mean: Optional[pint.Quantity] = None
    event_impact_std: Optional[pint.Quantity] = None
    cluster_decay_rate: Optional[pint.Quantity] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)