from __future__ import annotations
import random
from parallelized_algorithmic_trader.trading.simulated_broker import SimulatedAccount


def get_random_fitness_func_type(weights:dict[function,float]) -> function:
    return random.choices(list(weights.keys()), list(weights.values()))[0]


def score_trading_frequency(account: SimulatedAccount, min_n_trades_to_penalize:float=40, scale:float=100) -> float:
    """Scores an account based on how frequently it trades on a scale from 0 to 100, penalizing infrequent trades.
    
    min_n_trades_to_penalize: should be in units of trades/unit time of the simulation data
    
    The score is calculated linearly from 0 to 100, with 0 being 0 trades/time and 100 being >= min_n_trades_to_penalize.

    The idea here is that if youre algorithm only trades a few times, it is almost certainly overfitting the data, not learning 
    generalization. It doesn't actually matter how often we are trading, just so long as we have enough trades happen in our 
    training dataset. Therefore, agents who trade infrequently are penalized, but agents who trade very frequently are scored
    similarly to agents who trade moderately frequently."""
    orders = account._order_history
    n_orders = len(orders)
    return scale if n_orders >= min_n_trades_to_penalize else scale*(n_orders / min_n_trades_to_penalize)

