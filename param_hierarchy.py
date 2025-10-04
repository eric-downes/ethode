"""
Parameter hierarchy system for tracking parameter sources and confidence levels.

This module provides base classes for organizing parameters by their source:
1. Protocol-defined (from team/whitepaper)
2. Empirical (from data/calibration)
3. Arbitrary (placeholder values for testing)
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional, Set
from enum import Enum
import warnings

class ParamSource(Enum):
    """Source and confidence level of parameters"""
    PROTOCOL = "protocol"      # Defined by protocol team/whitepaper
    EMPIRICAL = "empirical"    # Derived from data or calibration
    ARBITRARY = "arbitrary"    # Placeholder/testing values
    DERIVED = "derived"        # Computed from other parameters

@dataclass
class ParamMetadata:
    """Metadata for a single parameter"""
    value: Any
    source: ParamSource
    description: str
    units: Optional[str] = None
    reference: Optional[str] = None  # e.g., "RAI team", "Liquity docs", "calibrated from DEX data"
    confidence: Optional[float] = None  # 0-1 confidence score
    range: Optional[tuple] = None  # (min, max) valid range

@dataclass 
class HierarchicalParams:
    """Base class for hierarchical parameter organization"""
    
    # Metadata tracking
    _metadata: Dict[str, ParamMetadata] = field(default_factory=dict, init=False, repr=False)
    _warn_on_arbitrary: bool = field(default=True, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize metadata tracking"""
        if not hasattr(self, '_metadata'):
            self._metadata = {}
        
        # Check for missing metadata
        param_fields = {f.name for f in fields(self) 
                       if not f.name.startswith('_')}
        missing = param_fields - set(self._metadata.keys())
        
        if missing and self._warn_on_arbitrary:
            warnings.warn(
                f"Parameters without metadata (assumed ARBITRARY): {missing}",
                UserWarning
            )
    
    def set_param(self, name: str, value: Any, source: ParamSource, 
                  description: str = "", **kwargs):
        """Set a parameter with metadata"""
        setattr(self, name, value)
        self._metadata[name] = ParamMetadata(
            value=value,
            source=source,
            description=description,
            **kwargs
        )
    
    def get_metadata(self, name: str) -> Optional[ParamMetadata]:
        """Get metadata for a parameter"""
        return self._metadata.get(name)
    
    def get_params_by_source(self, source: ParamSource) -> Dict[str, Any]:
        """Get all parameters from a specific source"""
        return {
            name: getattr(self, name)
            for name, meta in self._metadata.items()
            if meta.source == source
        }
    
    def validate_params(self) -> Dict[str, str]:
        """Validate parameters against their metadata"""
        issues = {}
        for name, meta in self._metadata.items():
            value = getattr(self, name, None)
            
            # Check range
            if meta.range and value is not None:
                min_val, max_val = meta.range
                if not (min_val <= value <= max_val):
                    issues[name] = f"Value {value} outside range {meta.range}"
            
            # Warn about arbitrary params
            if meta.source == ParamSource.ARBITRARY and self._warn_on_arbitrary:
                issues[name] = f"Using ARBITRARY value: {value}"
        
        return issues
    
    def summary(self) -> str:
        """Generate a summary of parameters by source"""
        lines = ["Parameter Summary:"]
        lines.append("=" * 50)
        
        for source in ParamSource:
            params = self.get_params_by_source(source)
            if params:
                lines.append(f"\n{source.value.upper()} Parameters:")
                lines.append("-" * 30)
                for name, value in params.items():
                    meta = self._metadata[name]
                    unit_str = f" [{meta.units}]" if meta.units else ""
                    ref_str = f" ({meta.reference})" if meta.reference else ""
                    lines.append(f"  {name}: {value}{unit_str}{ref_str}")
                    if meta.description:
                        lines.append(f"    {meta.description}")
        
        # Count by source
        counts = {}
        for meta in self._metadata.values():
            counts[meta.source.value] = counts.get(meta.source.value, 0) + 1
        
        lines.append("\n" + "=" * 50)
        lines.append("Parameter Counts:")
        for source, count in counts.items():
            lines.append(f"  {source}: {count}")
        
        return "\n".join(lines)
    
    def to_dict(self, include_metadata: bool = False) -> Dict:
        """Convert to dictionary, optionally including metadata"""
        if not include_metadata:
            # Return all parameters that have metadata
            return {
                name: getattr(self, name)
                for name in self._metadata.keys()
            }
        else:
            return {
                name: {
                    'value': getattr(self, name),
                    'source': meta.source.value,
                    'description': meta.description,
                    'units': meta.units,
                    'reference': meta.reference,
                    'confidence': meta.confidence,
                    'range': meta.range
                }
                for name, meta in self._metadata.items()
            }


@dataclass
class ProtocolParams(HierarchicalParams):
    """Parameters defined by the protocol team"""
    
    def __post_init__(self):
        """Mark all parameters as protocol-defined by default"""
        # First set metadata for any defined parameters
        for f in fields(self):
            if not f.name.startswith('_') and f.name not in self._metadata:
                value = getattr(self, f.name)
                self._metadata[f.name] = ParamMetadata(
                    value=value,
                    source=ParamSource.PROTOCOL,
                    description=f"Protocol-defined parameter",
                    reference="Protocol specification"
                )
        super().__post_init__()


@dataclass
class EmpiricalParams(HierarchicalParams):
    """Parameters derived from data or calibration"""
    
    def __post_init__(self):
        """Mark parameters as empirical by default"""
        for f in fields(self):
            if not f.name.startswith('_') and f.name not in self._metadata:
                value = getattr(self, f.name)
                self._metadata[f.name] = ParamMetadata(
                    value=value,
                    source=ParamSource.EMPIRICAL,
                    description=f"Empirically-derived parameter",
                    reference="Calibrated from data"
                )
        super().__post_init__()


@dataclass
class ArbitraryParams(HierarchicalParams):
    """Parameters with arbitrary/placeholder values"""
    
    def __post_init__(self):
        """Mark parameters as arbitrary by default"""
        for f in fields(self):
            if not f.name.startswith('_') and f.name not in self._metadata:
                value = getattr(self, f.name)
                self._metadata[f.name] = ParamMetadata(
                    value=value,
                    source=ParamSource.ARBITRARY,
                    description=f"Arbitrary placeholder value",
                    reference="Testing only"
                )
        super().__post_init__()


class CompositeParams(HierarchicalParams):
    """Composite parameters combining multiple sources"""
    
    def __init__(self, protocol: Optional[ProtocolParams] = None,
                 empirical: Optional[EmpiricalParams] = None,
                 arbitrary: Optional[ArbitraryParams] = None,
                 **kwargs):
        """Initialize from component parameter sets"""
        
        # Combine all parameters
        all_params = {}
        self._metadata = {}
        
        # Add in order of precedence
        for param_set, source_override in [
            (arbitrary, None),
            (empirical, None),
            (protocol, None)
        ]:
            if param_set:
                for f in fields(param_set):
                    if not f.name.startswith('_'):
                        value = getattr(param_set, f.name)
                        all_params[f.name] = value
                        
                        # Copy metadata if available
                        if hasattr(param_set, '_metadata'):
                            meta = param_set._metadata.get(f.name)
                            if meta:
                                self._metadata[f.name] = meta
        
        # Set all parameters
        for name, value in all_params.items():
            setattr(self, name, value)
        
        # Add any kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)
            if name not in self._metadata:
                self._metadata[name] = ParamMetadata(
                    value=value,
                    source=ParamSource.ARBITRARY,
                    description="Added via kwargs"
                )
        
        super().__post_init__()
    
    def require_non_arbitrary(self, param_names: Set[str]) -> bool:
        """Check that specified parameters are not arbitrary"""
        issues = []
        for name in param_names:
            meta = self._metadata.get(name)
            if meta and meta.source == ParamSource.ARBITRARY:
                issues.append(f"{name} is ARBITRARY")
        
        if issues:
            warnings.warn(f"Non-arbitrary parameters required: {', '.join(issues)}")
            return False
        return True