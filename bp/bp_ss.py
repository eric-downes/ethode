
# definiton of market cap

def market_cap(*, **params) -> U.ETH:
    return price(**params) * supply(**params)

def price(*, **params): pass

def supply(*, **params): pass

def bp_market_cap(*, **params) -> U.ETH:
    # valuation ~= market_cap
    return valuation(**params)

# valuation

def valuation(*, **params) -> U.ETH: 
    return investor(**params) + consumer(**params)

def investor(*, **params) -> U.ETH:
    return (fees(**params) + yield_(**params)) * \
        (1 + growth(**params)) / risk(**params)

def consumer(*, **params) -> U.ETH: # "convenience / utility"
    return theta(**params) * (fees(**params) + demand(**params))

# shared investor & consumer



def fees(*, **params) -> U.ETH: # Ft^$
    return 

def theta(*, **params) -> float:
    # ~ inflation_rate * timescale ~ risk_premium + risk_free_rate
    pass

# consumer

def demand(*, **params) -> U.ETH: # Nt^$
    pass

def yield_(*, **params) -> U.ETH:
    return staked_eth(**params) * pct_yield(**params) \
        * time_discounting(**params)

def bp_yield(*, **params) -> U.ETH:
    return delta_unstaked_eth(**params) - inflation_tax(**params)

def inflation_tax(*, **params) -> U.ETH:
    
    pass

def delta_unstaked_holdings(*, **params) -> U.ETH:
    return consumer(**params) * growth(**params)
    
