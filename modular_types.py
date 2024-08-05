'''
models have assumptions
parameters have constraints

'''

class Validatable(BaseModel): pass
class Restriction(BaseModel): pass

class Module(Validatable): pass # can contain other modules...
class Parameter(Validatable): pass

class Model(Module): pass


class Constraint(Restriction): pass

class Assumption(Restriction): pass

    

