"""
An experiment using a variable-sized ES-HyperNEAT network 
"""
from typing import List, Callable
import logging
import pickle
import neat
from enum import Enum

import pureples

from parallelized_algorithmic_trader.data_management.polygon_io import get_candle_data
from parallelized_algorithmic_trader.broker import TemporalResolution
from parallelized_algorithmic_trader.backtest import *
import parallelized_algorithmic_trader.performance_analysis as pf_anal
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
import parallelized_algorithmic_trader.indicators as indicators

from genetic_algorithm_trading.neat.agent import NEATEqualAllocation
from genetic_algorithm_trading.utils.fitness import get_random_fitness_func_type
import genetic_algorithm_trading.utils.utils as utils
import genetic_algorithm_trading.utils.visualize as visualize


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
NEAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'cppn_config.txt')
CONFIG = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, NEAT_CONFIG_PATH)


# Network input and output coordinates.
NODE_SPACING = 1
NODE_SHIFT = NODE_SPACING/2 if NODE_SPACING%2 == 1 else 0
SUBSTRATE = None    # by placing the nodes carefully, one can insert knowledge about the system


def set_substrate(tickers, indicator_mapping):
    all_feature_names = [name for ind_config in indicator_mapping for name in ind_config.names]
    N_INPUTS = len(all_feature_names)
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


def eval_genomes(genomes:List[Tuple[int, neat.DefaultGenome]], config:neat.Config):
    """
    This function expects there to be 3 variables in the global scope:
    - candle_data_train: a CandleData object
    - indicator_mapping: an IndicatorMapping object used for all of the neural nets
    - fitness_func_weights: a dict of functions and weights indicating probability of selection
    """

    global indicator_mapping
    global tickers

    fitness_func = get_random_fitness_func_type(fitness_func_weights)

    # print(f'Using fitness function: {fitness_func.__name__}')
    bot_configs:List[StrategyConfig] = []
    
    for genome_id, genome in genomes:
        net = produce_eshyperneat_net(genome, config)
        
        genome.fitness = 0
        bot_config = StrategyConfig(
            indicator_mapping=indicator_mapping,
            strategy=NEATEqualAllocation,
            tickers=tickers,
            args=[],
            kwargs={'net':net, 'log_level': logging.WARNING},
            quantity=1
            )
        bot_configs.append(bot_config)

    results = run_simulation_on_candle_data(
        algorithm_configs=bot_configs, 
        compute_state_for_each_strategy=False, 
        use_test_data=False, 
        display_progress_bar=False,
        )

    # update the fitness of each genome
    for i, (genome_id, genome) in enumerate(genomes):
        # print(f'Ensure fitness is assigned appropriately. {id(genome)} == {id(results[i].strategy.genome)}')
        fitness = fitness_func(results[i].account.get_history_as_list())
        # results[i].strategy.genome.fitness = fitness # commented out as the strategy no longer holds onto the genome, just the net
        genome.fitness = fitness


if __name__ == "__main__":
    tickers = ['SPY']
    
    # set up the data to be used as global variables
    n_days = 30*24
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=n_days)
    data = get_candle_data(os.environ['POLYGON_IO'], tickers, start, end, TemporalResolution.HOUR)
    
    indicator_mapping:IndicatorMapping= []
    for t in tickers:
        new_inds = [
            IndicatorConfig(t+'_close', indicators.MACD, args=[12, 26, 9]),
            IndicatorConfig(t, indicators.RSI, scaling_factor=0.1),
            # IndicatorConfig(t, indicators.SUPERTREND, args=[25,3], desired_output_name_keywords=['SUPERTRENDd']),
            # IndicatorConfig(t, indicators.SUPERTREND, args=[60,3], desired_output_name_keywords=['SUPERTRENDd']),
            IndicatorConfig(t+'_close', indicators.BB, args=(30,), desired_output_name_keywords=['BBB']),
            IndicatorConfig(t+'_close', indicators.PercentBB, args=(30,)),
            IndicatorConfig(t+'_close', indicators.BB, args=(90,), desired_output_name_keywords=['BBB']),
            IndicatorConfig(t+'_close', indicators.PercentBB, args=(90,)),
        ]
        
        indicator_mapping.extend(new_inds)
        break   # only do it for the first ticker for now

    # {
        # 'SPY_close_MACD(12, 26, 9)': 0.3755536627253946, 
        # 'SPY_close_MACDh(12, 26, 9)': 0.18033365613747332, 
        # 'SPY_close_MACDs(12, 26, 9)': 0.19522000658792127, 
        # 'SPY_RSI()': 6.938977138578355, 
        # 'SPY_close_BBB(30,)': 0.8188888831255183, 
        # 'SPY_close_PercentBB(30,)': 1.0635814982050085, 
        # 'SPY_close_BBB(90,)': 3.084837385703649, 
        # 'SPY_close_PercentBB(90,)': 0.8780126397909496}

    pf_anal.set_benchmark_score(data.df[tickers[0] + '_close'], pf_anal.get_curve_fit_vwr)
    fitness_func_weights = {
        pf_anal.get_vwr_curve_fit_difference: 1
    }
    
    VERSION = SubstrateSize.SMALL
    CPPN_PARAMS = produce_cppn_params(VERSION)
    set_substrate(tickers, indicator_mapping)

    winning_genome = utils.run(
        NEAT_CONFIG_PATH, 
        eval_genomes, 
        utils.test_genome_single_stock_set, 
        produce_eshyperneat_net,
        indicator_mapping, 
        data, 
        max_generations=100)

    CPPN = neat.nn.FeedForwardNetwork.create(winning_genome, CONFIG)
    pureples.shared.draw_net(CPPN, filename=f"images/es_hyperneat_mountain_car_{VERSION.name}_cppn")
    produce_eshyperneat_net(winning_genome, CONFIG, filename=f"images/es_hyperneat_mountain_car_{VERSION.name}_winner")
    
    utils.save_model(
        utils.GeneticAlgorithmModelType.ES_HYPERNEAT,
        winning_genome,
        CONFIG,
        indicator_mapping,
        data.resolution)
    
    