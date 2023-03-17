from __future__ import annotations
import uuid
import neat 
import pandas as pd
from typing import List, Dict

from parallelized_algorithmic_trader.orders import MarketOrder, OrderBase, OrderSide
from parallelized_algorithmic_trader.indicators import IndicatorMapping
from parallelized_algorithmic_trader.strategy import StrategyBase 
from parallelized_algorithmic_trader.trading.simulated_broker import SimulatedAccount


class NEATEqualAllocation(StrategyBase):
    """
    This strategy is a simple equal allocation strategy that uses a neural net to determine which assets to buy and sell, trading an fraction of the portfolio.
    """
    
    def __init__(
        self, 
        account_number:uuid.UUID, 
        indicator_mapping:IndicatorMapping, 
        tickers:List[str],
        net:neat.nn.FeedForwardNetwork, 
        log_level:int=None):
        name = __name__ + '.' + self.__class__.__name__ + '.' + str(account_number)[:8]
        if not "parallelized_algorithmic_trader" in name:
            name = "parallelized_algorithmic_trader." + name
        super().__init__(account_number, name, tickers, indicator_mapping)

        if log_level is not None: self.logger.setLevel(log_level)
        
        self.neural_net = net
        self.logger.debug(f'ordered_feature_names: {self.ordered_feature_names}')

    def _convert_nn_output_to_action_multi_node(self, account:SimulatedAccount, net_output:List[float]) -> List[MarketOrder] | None:
        orders = []
        for node in net_output:
            corresponding_ticker = self._tickers[net_output.index(node)]
            if node > 0.5:          # we want exposure to the corresponding asset
                self.logger.debug(f'The node for {corresponding_ticker} is firing: nodes: {net_output}, tickers: {self._tickers}')
 
                if account.check_for_exposure(corresponding_ticker):
                    self.logger.debug(f'Already have exposure to {corresponding_ticker}')
                    continue
                else:
                    self.logger.debug(f'Buying {corresponding_ticker}')
                    orders.append(self._get_sized_market_buy_order(corresponding_ticker, account))

            else:
                self.logger.debug(f'The node for {corresponding_ticker} is NOT firing: nodes: {net_output}, tickers: {self._tickers}')
                if account.check_for_exposure(corresponding_ticker):
                    self.logger.debug(f'Selling {corresponding_ticker}')
                    orders.append(MarketOrder(self.account_number, corresponding_ticker, OrderSide.SELL))
        return orders

    def act(self, account:SimulatedAccount, state:Dict[str, float]) -> OrderBase | None:
        """This is where the magic happens"""
        
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if account.has_pending_order:
            self.logger.debug(f'waiting for order to execute...')
            return None
        # get just the items out of the series that are needed for the neural net: drop the timestamp
        if 'timestamp' in state.keys(): state.pop('timestamp')
        output = self.neural_net.activate(state.values())  # the item here just has to be some sort of collection
        self.logger.debug(f'Net output: {output}')
        return self._convert_nn_output_to_action_multi_node(account, output)
