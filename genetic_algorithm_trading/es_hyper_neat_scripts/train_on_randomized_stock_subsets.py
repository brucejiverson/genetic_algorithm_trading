"""
The idea here is to increase the amount and variety of data that the NNs are trained on.
"""

from __future__ import annotations
import neat
import logging
import datetime
import numpy as np
from typing import List
from enum import Enum

import pureples

from parallelized_algorithmic_trader.data_management.historical_data import CandleData, get_candle_data
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.backtest import StrategyConfig, run_simulation_on_candle_data
import parallelized_algorithmic_trader.performance_analysis as pf_anal
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
import parallelized_algorithmic_trader.indicators as indicators

from genetic_algorithm_trading.utils.fitness import get_random_fitness_func_type
from genetic_algorithm_trading.agent import NEATEqualAllocation
import genetic_algorithm_trading.utils.utils as utils


logger = logging.getLogger('parallelized_algorithmic_trader.neat')
root_logger = logging.getLogger('parallelized_algorithmic_trader')
root_logger.setLevel(logging.WARNING)


# Global variables
GENERATIONS_PER_TICKER = 10
GENERATIONS_RAN_ON_CURRENT_TICKER = 0
CURRENT_TICKER = None


class SubstrateSize(Enum):
    """Analogous to the version parameter in the ES-HyperNEAT PUREPles examples"""
    SMALL = 'S'
    MEDIUM = 'M'
    LARGE = 'L'


def produce_cppn_params(version:SubstrateSize):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version.value == "S" else 1 if version.value == "M" else 2,
            "max_depth": 1 if version.value == "S" else 2 if version.value == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 8.0,
            "activation": "sigmoid"}


# Config for CPPN.
CONFIG = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            './genetic_algorithm_trading/es_hyper_neat/cppn_config.txt')


# Network input and output coordinates.
NODE_SPACING = 1
NODE_SHIFT = NODE_SPACING/2 if NODE_SPACING%2 == 1 else 0
SUBSTRATE = None


def set_substrate(tickers, indicator_mapping):
    all_feature_names = [name for ind_config in indicator_mapping for name in ind_config.names]
    N_INPUTS = len(all_feature_names)//len(tickers)
    assert N_INPUTS%1 == 0, "Number of inputs must be divisible by number of tickers"
    N_OUTPUTS = len(tickers)

    # input coordinates spaced evenly across the x axis 
    INPUT_COORDINATES = [(i*NODE_SPACING + NODE_SHIFT, -1.) for i in range(-N_INPUTS // 2, N_INPUTS // 2)]
    OUTPUT_COORDINATES = [(i*NODE_SPACING + NODE_SHIFT, 1.) for i in range(-N_OUTPUTS // 2, N_OUTPUTS // 2)]
    global SUBSTRATE
    SUBSTRATE = pureples.shared.Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def produce_eshyperneat_net(genome, config, filename=None):
    
        global SUBSTRATE
        global CPPN_PARAMS
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = pureples.es_hyperneat.ESNetwork(SUBSTRATE, cppn, CPPN_PARAMS)
        net = network.create_phenotype_network(filename=filename)
        return net


def get_random_ticker() -> str:
    """Fetches a ticker at random (uniform distribution)"""
    
    global tickers
    
    ticker = np.random.choice(tickers)
    return ticker


def eval_genomes(genomes:list[neat.DefaultGenome], config:neat.Config):
    """This function expects there to be 3 variables in the global scope:
    - candle_data_train: a CandleData object
    - indicator_mapping: an IndicatorMapping object used for all of the neural nets
    - fitness_func_weights: a dict of functions and weights indicating probability of selection"""

    # 'genomes' are really a bunch of NNs
    global indicators_for_bot
    global fitness_func_weights
    global GENERATIONS_PER_TICKER
    global GENERATIONS_RAN_ON_CURRENT_TICKER

    fitness_func = get_random_fitness_func_type(fitness_func_weights)

    print(f'Using fitness function: {fitness_func.__name__}')
    
    bot_configs:List[StrategyConfig] = []

    if GENERATIONS_RAN_ON_CURRENT_TICKER == GENERATIONS_PER_TICKER or CURRENT_TICKER is None:
        CURRENT_TICKER = get_random_ticker()
        GENERATIONS_RAN_ON_CURRENT_TICKER = 0

    indicators_for_bot = indicators_for_bot[CURRENT_TICKER]
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
 
        bot_config = StrategyConfig(
            indicator_mapping=indicators_for_bot,
            strategy=NEATEqualAllocation,
            tickers=[CURRENT_TICKER],
            args=[],
            kwargs={'net':net, 'log_level': logging.WARNING},
            quantity=1
            )

        bot_configs.append(bot_config)

    results = run_simulation_on_candle_data(
        algorithm_configs=bot_configs, 
        compute_state_for_each_strategy=False, 
        use_test_data=False, 
        display_progress_bar=False
        )

    # update the fitness of each genome
    for i, (genome_id, genome) in enumerate(genomes):
        # print(f'Ensure fitness is assigned appropriately. {id(genome)} == {id(results[i].strategy.genome)}')
        fitness = fitness_func(results[i].account.get_history_as_list())
        genome.fitness = fitness


def test_mean_performance():
    """Evaluate the genomes performance by testing it on all of the securities in the data set and take the mean of the fitness"""
    pass


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    import os
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, './neat_config.txt')
    
    tickers = ['MSFT', 'AAPL', 'NVDA', 'TSLA', 'AMZN', 'GOOG', 'FB', 'NFLX', 'BABA', 'CSCO', 'PEP']
    #, 'JPM', 'JNJ', 'V', 'MA', 'UNH', 'PG', 'HD', 'DIS', 'VZ', 'INTC', 'ADBE', 'PYPL', 'CMCSA', 'CRM', 'T', 'KO', 'AVGO', 'ABT', 'MRK', 'ACN', 'MCD', 'ORCL', 'NKE', 'NEE', 'WFC', 'BAC', 'TXN', 'COST', 'PM', 'CVX', 'IBM', 'QCOM', 'AMGN', 'DHR', 'MDT', 'LIN', 'UPS', 'LOW', 'GS', 'BA', 'CAT', 'MMM', 'UNP', 'AMT', 'AXP', 'TMO', 'HON', 'WMT', 'XOM', 'CVS', 'BKNG', 'SBUX', 'BLK', 'INTU', 'MDLZ', 'GILD', 'MU', 'DUK', 'SO', 'GE', 'TGT', 'FIS', 'PFE', 'USB', 'CHTR', 'DE', 'MS', 'MDLZ', 'ANTM', 'SYK', 'ISRG', 'LMT', 'TJX', 'C', 'CME', 'ZTS', 'WBA', 'AMAT', 'ADP', 'GPN', 'BIIB', 'MET', 'AIG', 'PLD', 'CI', 'EL', 'COP', 'COF', 'CL', 'AAL', 'CNC', 'CMI', 'CB', 'CAH', 'BMY', 'BDX', 'BAX', 'BLL', 'BHF', 'BEN', 'BWA', 'BXP', 'BSX', 'BRK.B', 'BR', 'BK', 'BMY', 'BIO', 'BAX', 'BLL', 'BHF', 'BEN', 'BWA', 'BXP', 'BSX', 'BRK.B', 'BR', 'BK', 'BMY', 'BIO', 'BAX', 'BLL', 'BHF', 'BEN', 'BWA]

    indicator_mapping:IndicatorMapping = []
    mapping_for_bots = {}
    for t in tickers:
        new_inds = [
            IndicatorConfig(t+'_close', indicators.MACD, args=[12, 26, 9]),
            IndicatorConfig(t, indicators.RSI, scaling_factor=0.01),
            IndicatorConfig(t+'_close', indicators.BB, args=(15,), desired_output_name_keywords=['BBP']),
            IndicatorConfig(t+'_close', indicators.BB, args=(45,), desired_output_name_keywords=['BBP']),
        ]

        indicator_mapping.extend(new_inds)
        mapping_for_bots[t] = new_inds

    # set up the data to be used as global variables
    n_days = 30*2
    end = datetime.datetime.now()
    
    data = get_candle_data(tickers, TemporalResolution.HOUR, end-datetime.timedelta(days=n_days), end)
    pf_anal.set_benchmark_score(data.df[tickers[0] + '_close'], pf_anal.get_curve_fit_vwr)
    fitness_func_weights = {
        pf_anal.get_vwr_curve_fit_difference: 1
    }
    
    # hyperparams
    GENERATIONS_PER_TICKER = 5
    VERSION = SubstrateSize.SMALL
    CPPN_PARAMS = produce_cppn_params(VERSION)
    set_substrate(tickers, indicator_mapping)
    

    winning_genome = utils.run(
        neat_config_path, 
        eval_genomes, 
        test_mean_performance, 
        indicator_mapping, 
        data, 
        max_generations=50)


    CPPN = neat.nn.FeedForwardNetwork.create(winning_genome, CONFIG)
    pureples.shared.draw_net(CPPN, filename=f"images/es_hyperneat_mountain_car_{VERSION.name}_cppn")
    produce_eshyperneat_net(winning_genome, CONFIG, filename=f"images/es_hyperneat_mountain_car_{VERSION.name}_winner")
    
    
    utils.save_model(
        utils.GeneticAlgorithmModelType.ES_HYPERNEAT,
        winning_genome,
        CONFIG,
        indicator_mapping,
        data.resolution)
    
    