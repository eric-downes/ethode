from contextlib import 

from pint import UnitRegistry

U = UnitRegistry()
U.load_definitions('eth_units.txt')

def total_issuance(eth_supply: U.ETH,
                 epochs_per_year: U.year = None) -> U.ETH:
    if not epochs_per_year:
        epochs_per_year = U.ave_epoch.to(U.year)
    return U.base_reward_factor * epochs_per_year * \
        math.sqrt(eth_supply / U.giga_eth) * U.ETH / U.year

