
# definiton of market cap

def market_cap(*, **params) -> U.ETH:
    return price(**params) * supply(**params)

def price(*, **params): pass

def supply(*, **params): pass

# valuation ~= market_cap

def valuation(*, **params) -> U.ETH: 
    return investor(**params) + consumer(**params)

def investor(*, **params) -> U.ETH:
    return (fees(**params) + yield_(**params)) * \
        (1 + growth(**params)) / risk(**params)

def consumer(*, **params) -> U.ETH: # "convenience / utility"
    return theta(**params) * (fees(**params) + demand(**params))

# shared investor & consumer

def fees(*, **params) -> U.ETH: # Ft^$
    pass

def theta(*, **params) -> float:
    # ~ inflation_rate * timescale ~ risk_premium + risk_free_rate
    pass

# consumer

def demand(*, **params) -> U.ETH: # Nt^$
    pass

def yield_(*, **params) -> U.ETH:
    return delta_unstaked_holdings(**params) - inflation_tax(**params)

def inflation_tax(*, **params) -> U.ETH:

    pass

def delta_unstaked_holdings(*, **params) -> U.ETH:

    pass
    
