import logging
import pickle
import datetime
from typing import Set
import click

from parallelized_algorithmic_trader.data_management.historical_data import get_candle_data
from parallelized_algorithmic_trader.indicators import IndicatorMapping
from parallelized_algorithmic_trader.backtest import build_features, set_train_test_true

import genetic_algorithm_trading.utils.visualize as neat_visualize
from genetic_algorithm_trading.utils.utils import *


logger = logging.getLogger('pat.'  + __file__)
logger.setLevel(logging.DEBUG)


def get_all_tickers_from_mapping(mappings:IndicatorMapping) -> Set[str]:
    tickers = set()
    for m in mappings:
        # search down through the targets until we find a ticker
        target = m.target
        while not isinstance(target, str):
            target = target.target
        tickers.add(target.split('_')[0])
    print(f'Found following tickers feature mapping associated with training: {tickers}')
    return tickers

# the main function to run the saved model using the click library for command line arguments for the file name

@click.command()
@click.option('--model', '-m', default='es_hyperneat', help='the type of model to run')
def run_saved_model(model):
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    model_data = NEATModelMetaInfo.read_pickle(f'{model}_winner.pkl')
    
    # build the FeedForwardNetwork from the genome
    # match model_data.model_type:
    #     case GeneticAlgorithmModelType.NEAT:
    #         ff_net = neat.nn.FeedForwardNetwork.create(model_data.genome, model_data.config)
    #     case GeneticAlgorithmModelType.HYPERNEAT:
    #         raise NotImplementedError('HyperNEAT is not yet implemented')
    #     case GeneticAlgorithmModelType.ES_HYPERNEAT:
    #         raise NotImplementedError('ES-HyperNEAT is not yet implemented')
    
    feature_names = [name for ind_config in model_data.indicator_mapping for name in ind_config.names]
    logger.info(f'Indicator mappings ({len(feature_names)}): {feature_names}, Tickers: {model_data.tickers}')

    # set up the data to be used as global variables
    now = datetime.datetime.now()
    train_test_split = False
    if now - model_data.training_end > datetime.timedelta(days=4): # then we will test on all data up until now
        train_test_split = True
        data_end = now - datetime.timedelta(days=3)
    else: data_end = model_data.training_end
    logger.info(f'Running on the training data from {model_data.training_start.date()} to {model_data.training_end.date()}')

    # PREPARE THE DATA
    candle_data = get_candle_data(model_data.tickers, model_data.resolution, model_data.training_start, data_end)
    # build the features
    build_features(candle_data, [model_data.indicator_mapping])

    # infer the rough split fraction for the training data
    if train_test_split:
        split_frac = (model_data.training_end - model_data.training_start) / (data_end - model_data.training_start)
        split_frac = round(split_frac, 2) # round to nearest 0.1
        logger.debug(f'Inferred split fraction: {split_frac}')
        set_train_test_true(split_frac)
    
    logger.debug(f'Configured num inputs: {model_data.config.genome_config.num_inputs}.')
    logger.debug(f'Output: {model_data.config.genome_config.num_outputs}')
    
    neat_visualize.draw_neat_net(model_data.config, model_data.genome, True)
    # play through the winning genome and do a full performance analysis incl. visualizations
    if train_test_split:
        print('Running on the training data')
        test_genome_single_stock_set(model_data.neural_net, candle_data.tickers, model_data.indicator_mapping, use_test_data=False)
        print(f'Test data play through:')
        test_genome_single_stock_set(model_data.neural_net, candle_data.tickers, model_data.indicator_mapping, use_test_data=True)
    else:
        test_genome_single_stock_set(model_data.neural_net, candle_data.tickers, model_data.indicator_mapping)


if __name__ == '__main__':
    run_saved_model()