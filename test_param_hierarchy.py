"""
Tests for parameter hierarchy system
"""

import pytest
from param_hierarchy import (
    ParamSource, ParamMetadata, HierarchicalParams,
    ProtocolParams, EmpiricalParams, ArbitraryParams,
    CompositeParams
)
from dataclasses import dataclass


class TestParamMetadata:
    """Test parameter metadata"""
    
    def test_metadata_creation(self):
        """Test creating parameter metadata"""
        meta = ParamMetadata(
            value=1.0,
            source=ParamSource.PROTOCOL,
            description="Test parameter",
            units="dimensionless",
            reference="Test source",
            confidence=0.9,
            range=(0, 2)
        )
        
        assert meta.value == 1.0
        assert meta.source == ParamSource.PROTOCOL
        assert meta.description == "Test parameter"
        assert meta.units == "dimensionless"
        assert meta.reference == "Test source"
        assert meta.confidence == 0.9
        assert meta.range == (0, 2)


class TestHierarchicalParams:
    """Test hierarchical parameter base class"""
    
    def test_set_and_get_param(self):
        """Test setting and getting parameters with metadata"""
        params = HierarchicalParams()
        
        params.set_param('test_value', 42, ParamSource.EMPIRICAL,
                        "Test parameter", units="units", confidence=0.8)
        
        assert params.test_value == 42
        meta = params.get_metadata('test_value')
        assert meta is not None
        assert meta.source == ParamSource.EMPIRICAL
        assert meta.confidence == 0.8
    
    def test_get_params_by_source(self):
        """Test filtering parameters by source"""
        params = HierarchicalParams()
        
        params.set_param('protocol_param', 1, ParamSource.PROTOCOL, "Protocol")
        params.set_param('empirical_param', 2, ParamSource.EMPIRICAL, "Empirical")
        params.set_param('arbitrary_param', 3, ParamSource.ARBITRARY, "Arbitrary")
        
        protocol = params.get_params_by_source(ParamSource.PROTOCOL)
        assert len(protocol) == 1
        assert protocol['protocol_param'] == 1
        
        empirical = params.get_params_by_source(ParamSource.EMPIRICAL)
        assert len(empirical) == 1
        assert empirical['empirical_param'] == 2
        
        arbitrary = params.get_params_by_source(ParamSource.ARBITRARY)
        assert len(arbitrary) == 1
        assert arbitrary['arbitrary_param'] == 3
    
    def test_validate_params_range(self):
        """Test parameter validation against ranges"""
        params = HierarchicalParams()
        params._warn_on_arbitrary = False  # Suppress warnings for test
        
        params.set_param('in_range', 5, ParamSource.PROTOCOL, "In range", range=(0, 10))
        params.set_param('out_range', 15, ParamSource.PROTOCOL, "Out of range", range=(0, 10))
        
        issues = params.validate_params()
        assert 'in_range' not in issues
        assert 'out_range' in issues
        assert 'outside range' in issues['out_range']
    
    def test_summary_generation(self):
        """Test parameter summary generation"""
        params = HierarchicalParams()
        
        params.set_param('param1', 1, ParamSource.PROTOCOL, "First parameter")
        params.set_param('param2', 2, ParamSource.EMPIRICAL, "Second parameter")
        
        summary = params.summary()
        assert 'PROTOCOL' in summary
        assert 'EMPIRICAL' in summary
        assert 'param1: 1' in summary
        assert 'param2: 2' in summary
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        params = HierarchicalParams()
        
        params.set_param('param1', 1, ParamSource.PROTOCOL, "Test")
        params.set_param('param2', 2, ParamSource.EMPIRICAL, "Test")
        
        # Without metadata
        d = params.to_dict(include_metadata=False)
        assert d['param1'] == 1
        assert d['param2'] == 2
        
        # With metadata
        d_meta = params.to_dict(include_metadata=True)
        assert d_meta['param1']['value'] == 1
        assert d_meta['param1']['source'] == 'protocol'
        assert d_meta['param2']['value'] == 2
        assert d_meta['param2']['source'] == 'empirical'


class TestSpecializedParams:
    """Test specialized parameter classes"""
    
    def test_protocol_params_default_source(self):
        """Test that ProtocolParams defaults to protocol source"""
        @dataclass
        class TestProtocol(ProtocolParams):
            param1: float = 1.0
            param2: float = 2.0
        
        params = TestProtocol()
        
        meta1 = params.get_metadata('param1')
        assert meta1.source == ParamSource.PROTOCOL
        
        meta2 = params.get_metadata('param2')
        assert meta2.source == ParamSource.PROTOCOL
    
    def test_empirical_params_default_source(self):
        """Test that EmpiricalParams defaults to empirical source"""
        @dataclass
        class TestEmpirical(EmpiricalParams):
            param1: float = 1.0
            param2: float = 2.0
        
        params = TestEmpirical()
        
        meta1 = params.get_metadata('param1')
        assert meta1.source == ParamSource.EMPIRICAL
        
        meta2 = params.get_metadata('param2')
        assert meta2.source == ParamSource.EMPIRICAL
    
    def test_arbitrary_params_default_source(self):
        """Test that ArbitraryParams defaults to arbitrary source"""
        @dataclass
        class TestArbitrary(ArbitraryParams):
            param1: float = 1.0
            param2: float = 2.0
        
        params = TestArbitrary()
        
        meta1 = params.get_metadata('param1')
        assert meta1.source == ParamSource.ARBITRARY
        
        meta2 = params.get_metadata('param2')
        assert meta2.source == ParamSource.ARBITRARY


class TestCompositeParams:
    """Test composite parameter combination"""
    
    def test_composite_combination(self):
        """Test combining multiple parameter sources"""
        @dataclass
        class TestProtocol(ProtocolParams):
            protocol_param: float = 1.0
        
        @dataclass
        class TestEmpirical(EmpiricalParams):
            empirical_param: float = 2.0
        
        @dataclass
        class TestArbitrary(ArbitraryParams):
            arbitrary_param: float = 3.0
        
        protocol = TestProtocol()
        empirical = TestEmpirical()
        arbitrary = TestArbitrary()
        
        composite = CompositeParams(
            protocol=protocol,
            empirical=empirical,
            arbitrary=arbitrary
        )
        
        # Check all parameters are present
        assert composite.protocol_param == 1.0
        assert composite.empirical_param == 2.0
        assert composite.arbitrary_param == 3.0
        
        # Check metadata is preserved
        assert composite.get_metadata('protocol_param').source == ParamSource.PROTOCOL
        assert composite.get_metadata('empirical_param').source == ParamSource.EMPIRICAL
        assert composite.get_metadata('arbitrary_param').source == ParamSource.ARBITRARY
    
    def test_require_non_arbitrary(self):
        """Test checking for non-arbitrary parameters"""
        @dataclass
        class TestProtocol(ProtocolParams):
            good_param: float = 1.0
        
        @dataclass
        class TestArbitrary(ArbitraryParams):
            bad_param: float = 2.0
        
        composite = CompositeParams(
            protocol=TestProtocol(),
            arbitrary=TestArbitrary()
        )
        
        # Should pass for protocol param
        assert composite.require_non_arbitrary({'good_param'}) == True
        
        # Should fail for arbitrary param
        with pytest.warns(UserWarning, match="bad_param is ARBITRARY"):
            assert composite.require_non_arbitrary({'bad_param'}) == False
        
        # Mixed should fail
        with pytest.warns(UserWarning):
            assert composite.require_non_arbitrary({'good_param', 'bad_param'}) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])