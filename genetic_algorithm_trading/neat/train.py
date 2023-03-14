from __future__ import annotations
import neat
import logging
import datetime
from typing import Tuple

from parallelized_algorithmic_trader.data_management.polygon_io import get_candle_data
from parallelized_algorithmic_trader.broker import TemporalResolution
from parallelized_algorithmic_trader.backtest import StrategyConfig, run_simulation_on_candle_data
import parallelized_algorithmic_trader.performance_analysis as pf_anal
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
import parallelized_algorithmic_trader.indicators as indicators

from genetic_algorithm_trading.utils.fitness import get_random_fitness_func_type
from genetic_algorithm_trading.utils.utils import *


# inherits all properties from the pat library
logger = logging.getLogger('pat.neat')
# root_logger = logging.getLogger('pat')
# root_logger.setLevel(logging.WARNING)


def eval_genomes(genomes:List[Tuple[int, neat.DefaultGenome]], config:neat.Config):
    """This function expects there to be 3 variables in the global scope:
    - candle_data_train: a CandleData object
    - indicator_mapping: an IndicatorMapping object used for all of the neural nets
    - fitness_func_weights: a dict of functions and weights indicating probability of selection"""

    # 'genomes' are really a bunch of NNs
    global indicator_mapping
    global fitness_func_weights
    global tickers

    fitness_func = get_random_fitness_func_type(fitness_func_weights)

    print(f'Using fitness function: {fitness_func.__name__}')
    bot_configs:List[StrategyConfig] = []
       
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
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
        genome.fitness = fitness


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    import os
    local_dir = os.path.dirname(__file__)
    NEAT_CONFIG_PATH = os.path.join(local_dir, './neat_config.txt')

    tickers = ['SPDN', 'SPY']

    indicator_mapping:IndicatorMapping= []
    for t in tickers:
        new_inds = [
            # IndicatorConfig(t+'_close', MACD_INTERPRETATION_1, args=[12, 26, 9]),
            # IndicatorConfig(t+'_close', MACD_INTERPRETATION_1, args=[3, 6.5, 2.25]),
            IndicatorConfig(t+'_close', indicators.MACD, args=[12, 26, 9]),
            IndicatorConfig(t, indicators.RSI, scaling_factor=0.1),
            # IndicatorConfig(t, indicators.SUPERTREND, args=[10,3], desired_output_name_keywords=['SUPERTRENDd']),
            IndicatorConfig(t, indicators.SUPERTREND, args=[25,3], desired_output_name_keywords=['SUPERTRENDd']),
            IndicatorConfig(t, indicators.SUPERTREND, args=[60,3], desired_output_name_keywords=['SUPERTRENDd']),

            # IndicatorConfig(
            #     IndicatorConfig(t, indicators.VWAP, args=[], kwargs={}),
            #     indicators.DIFF,args=(t+'_close',)
            #     ),

            IndicatorConfig(t+'_close', indicators.PercentBB, args=(15,)),
            IndicatorConfig(t+'_close', indicators.PercentBB, args=(45,)),
            IndicatorConfig(t+'_close', indicators.PercentBB, args=(60,)),
        ]
        indicator_mapping.extend(new_inds)
        break   # only do it for the first ticker for now

    fitness_func_weights = {
        pf_anal.get_curve_fit_vwr: 1
        # pf_anal.get_ROI: 1
    }

    # set up the data to be used as global variables
    n_days = 30*1
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=n_days)
    
    data = get_candle_data(os.environ['POLYGON_IO'], tickers, start, end, TemporalResolution.HOUR)
    configure_neat_inputs_and_outputs(NEAT_CONFIG_PATH, indicator_mapping, data.tickers)
    
    run(
        NEAT_CONFIG_PATH, 
        eval_genomes, 
        test_genome_single_stock_set, 
        neat.nn.FeedForwardNetwork.create,
        indicator_mapping, 
        data, 
        max_generations=4)

