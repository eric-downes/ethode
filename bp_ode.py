from scipy.integrate import odeint
import numpy as np

def model(variables: tuple[tuple|float], time: float,
          params: dict[str, Callable | float]) -> tuple:

    # unpacking
    supply_buy, supply_sell, supply_staked = variables[0 : 3]
    base_gas_fee, price = variables[3 : 5]

    # constants
    frac = params['fraction_buyer_fees']
    eip1559_const = params['eip_1559_constant']
    eip1559_target = params['eip_1559_target_gas']

    # issuance curve
    yld = params['yield_curve'](supply_staked)

    # EXOGENOUS VARS BP does not model
    gasc = params['gas_consumed_this_block'](time) # he treats this as a dep var
    gasc_prev = params['gas_consumed_last_block'](time) 
    gasp = params['gas_price_ETH'](time) # doesn't model this beyond reflexivity
    forecast_inflation = params['expected_inflation_function']
    
    # basic conservation eqs
    supply = supply_buy + supply_sell + supply_staked
    dsupply = supply * yld  -  base_gas_fee * gasc # mass conservation
    fees = gasc * gasp # eq 5 non-log version
    assert fees >= base_gas_fee
    
    # BIG BP ASSUMPTIONS
    # no demand effects on price.... ????
    dprice = - price * dsupply / (1 + dsupply)
    # negA = volume ETH bought by S_buy; bizzare?
    negA = supply / supply_staked * forecast_inflation(dsupply)
        
    # evolution    
    # supply == S_staked + S_sell + S_buy
    dS_buy = negA  -  gasc * fees * frac
    dS_sell = - negA  -  gasc * fees * (1 - frac)
    dS_staked = supply * yld  +  gasc * (fees  -  base_fee)
    dbase_fee = base_fee * (gasc_prev / eip1559_target  -  1) * eip1559_const
    
    dvars = [dS_buy, dS_sell, dS_staked, dbase_fee, dprice]        
    return dvars

def simulate(start_block:float, end_block:float, dt:float = None):
    # time setup
    assert 0 < start_block < end_block
    nblocks = end_block - start_block
    # i.c. and exogenous vars
    out = [get_initial()]
    params = get_params(nblocks)
    # do the compute & return
    for t in time_range:
        out.append( odeint(model, out[-1], [t, t + dt], params) )
    return out

def get_params(nblocks:int, dt:float) -> dict[str, float|Callable]:
    _gas = np.random.randn(nblocks) * U.time
    params = {'fraction_buyer_fees': .5 * U.dimless,
              'eip_1559_constant': 1/8 * U.dimless,
              'eip_1559_gas_target': U.gas_target,
              'yield_curve': lambda staked: 2.6 / math.sqrt(staked) * U.dimless,
              'gas_consumed_this_block': lambda t: _gas[t // dt],
              'gas_consumed_last_block': lambda t: _gas[max(0, t // dt - 1)],
              'gas_price_ETH': lambda x: 600 * U.gwei,
              'expected_inflation_function': lambda x: x}
    return params
    

def init_conds() -> tuple[float]:
    pass

