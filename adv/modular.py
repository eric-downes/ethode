from future import __annotations__
from generic import *

Num = int|float|complex
T = TypeVar('T')
ManyToOne_OneType = Callable[[T, ...], T]
ManyToMany_OneType = Callable[[T, ...], tuple[T,...]]
Term = FuncOfOneType[Num]
Terms = tuple[Term, ...]

class Node(BaseModel):
    to_internal_type: Callable[[type], Num]
    symbol: Symbol
    model: Model = None

Nodes = tuple[Node, ...]

class Flow(BaseModel):
    sources: Nodes
    targets: Nodes
    internal_function: ManyToMany_OneType[Num]
    model: Model = None
    @model_validator(mode = 'after')
    def _validate_(self) -> Flow:
        a = self.internal_function.__annotations__.copy()
        assert len(ret := a.pop('return')) == len(self.targets)
        return self


class DYDT(BaseModel):
    # we are going to generate these, so should do strict tyoe checking
    # that the equation agrees with the model its a part of
    variable_types: tuple[type,...]
    output_type: type
    param_types: dict[str, type]
    function: Callable
    discrete_time: bool = False
    @model_validator(mode='after')
    def _fcn_sig_check_(self) -> DYDT:
        a = self.function.__anotations__.copy()
        assert a.pop('return') == self.output_type
        assert a.pop('time') == int if self.discrete_time else float
        for (aname, atyp), vtyp in zip(a.items(), variable_types):
            if vtyp:
                assert atyp == vtyp 
            pass
        for n, in a.pop('return'):
            pass            
    def to_term_dict(self) -> dict[Node, Terms]:
        d = {}
        for src in self.sources:
            pass
            
        
'''        
                 
    
    


class ConversionDict(


        
ConversionDict = dict[type, Terms]

class Model(BaseModel):
    def __init__(
            self,
            nodes:Nodes,
            term_dict:dict[Node, Terms]): pass
    def convert_function(
            self,
            fcn: FunctionOfOneType
                
                        

class Edge(BaseModel): pass

class Node(BaseModel):
    node_type: type
    belongs_to: Model = None

Nodes = tuple[Node,...]

class ManyToMany(Edge):
    typed_function: Callable[[Node,...], Nodes]
    internal_type: type
    model: Model
    @model_validator(mode = 'after')
    def _validate_(self) -> Flow:
        a = self.function.__annotations__
        d = self.model.conversion_dict
        for to_type in a.pop('return').__args__:
            assert to_type in d[self.internal_type]
        for from_type in a.values():
            assert self.internal_type in self.model.conv
        
                 
        for typ in a.__args__:
            assert
        
        
        
        


class ManyToMany(Flow):
    inputs: Nodes
    outputs: Nodes
    function: Callable[[Nodes], Nodes]
    @model_validator(mode='after')
    def _check_topology_(self) -> ManyToMany:
        a = self.function.__annotation__
        len(a

class Source: pass
class Sink: pass
class Operad: pass
class CoOperad: pass


def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt


sol = odeint(pend, y0, t, args=(b, c))


def model(variables: tuple[tuple|float], time: float,
          params: dict[str, Callable | float]) -> tuple:

'''
