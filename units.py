from contextlib import contextmanager
from typing import TypeVar, Optional
import functools

import pint
from pint import UnitRegistry

Unit = TypeVar('Unit', bound = pint.Unit)

U = UnitRegistry()
U.load_definitions('eth_units.txt')

def unitful(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert (d := func.__annotations__.copy()), \
            f'function {func} must be annotated'
        while kwargs:
            k, v = kwargs.popitem()
            assert isinstance(v, t := d.pop(k)), \
                f'input value {k}:{v} fails to be type {t} in kwargs'
        ret_type = d.pop('return')
        for v, (k, t) in zip(args, d.items()):
            assert isinstance(v, t), \
                f'input value {k}:{v} fails to be type {t} in args'
        ret = func(*args, **kwargs)
        assert isinstance(ret, ret_type), \
            f'output value {ret} fails to be type {ret_type}'
        return ret
    return wrapper
    
@unitful
def total_issuance(eth_supply: U.ETH,
                 epochs_per_year: Optional[U.year] = None) -> U.ETH:
    # example function ... need to cleanup
    if not epochs_per_year:
        epochs_per_year = U.ave_epoch.to(U.year)
    return U.base_reward_factor * epochs_per_year * \
        math.sqrt(eth_supply / U.giga_eth) * U.ETH / U.year

