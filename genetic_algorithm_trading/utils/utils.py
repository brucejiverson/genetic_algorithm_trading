from typing import Any, List, Callable, Dict
import datetime
import neat
import pickle 
from enum import Enum
import os
from dataclasses import dataclass, field

from parallelized_algorithmic_trader.data_management.historical_data import CandleData
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.indicators import IndicatorMapping
from parallelized_algorithmic_trader.strategy import StrategyConfig
from parallelized_algorithmic_trader.backtest import build_features, run_simulation_on_candle_data, set_train_test_true, get_training_start_end_dates

import genetic_algorithm_trading.utils.visualize as neat_visualize
from genetic_algorithm_trading.agent import NEATEqualAllocation


ROOT_DIR = os.path.dirname(__file__).split('genetic_algorithm_trading')[0] 
MODEL_DIR = os.path.join(ROOT_DIR, 'genetic_algorithm_trading/models')
IMAGES_DIR = os.path.join(ROOT_DIR, 'genetic_algorithm_trading/images')


class GeneticAlgorithmModelType(Enum):
    NEAT = 'neat'
    HYPERNEAT = 'hyperneat'
    ES_HYPERNEAT = 'es_hyperneat'


@dataclass
class NEATModelMetaInfo:
    """A class for containing all of the information about the NEAT model including training information"""
    
    model_type:GeneticAlgorithmModelType
    genome:neat.DefaultGenome
    config:neat.Config
    neural_net:neat.nn.FeedForwardNetwork
    indicator_mapping:IndicatorMapping
    tickers:List[str]
    resolution:TemporalResolution
    training_start:datetime.datetime|None=None
    training_end:datetime.datetime|None=None
    other_training_hyperparameters:Dict[str, Any]=field(default_factory=dict)
    
    def __post_init__(self):
        if not self.training_start or not self.training_end:
            self.training_start, self.training_end = get_training_start_end_dates()
    
    def to_pickle(self, file_name:str=None):
        """Save the model meta info to a pickle file in the models directory. If no file name is provided, the default"""
        if not file_name:
            file_name = f'{self.model_type.value}_winner.pkl'
            
        with open(os.path.join(MODEL_DIR, file_name), 'wb') as f:
            pickle.dump(self, f)
                
        print(f'Saved genome for neural net to {MODEL_DIR}')

    @classmethod
    def read_pickle(cls, file_name:str):
        """Read the model meta info from a pickle file in the models directory"""
        with open(os.path.join(MODEL_DIR, file_name), 'rb') as f:
            obj = pickle.load(f)
            
        assert isinstance(obj, cls), f'Object read from {file_name} is not of type {cls.__name__}: {type(obj)}'
        print(f'Loaded a {obj.model_type.value} model from {MODEL_DIR}')
        return obj
    

def get_neat_config_param(config_path:str, param_name:str) -> int:
    """Get the number of inputs for the NEAT network"""
    with open(config_path, 'r') as f:
        config = f.read()
    lines = config.split('\n')
    for l in lines:
        if param_name in l:
            return int(l[-1])
    raise ValueError(f'Could not find parameter {param_name} in config file')


def set_neat_config_param(config_path:str, param_name:str, param_value:Any):
    """Set the number of inputs for the NEAT network"""
    with open(config_path, 'r') as f:
        config = f.read()
    
    # find the line with
    lines = config.split('\n')
    for i, l in enumerate(lines):
        if param_name in l:
            # search backwards through the line for the space to the left of the number
            for j in range(len(l)):
                if l[-j] == ' ':
                    new_line = l[:-(j-1)] + str(param_value)
                    break
            config = config.replace(l, new_line)

    with open(config_path, 'w') as f:
        f.write(config)


def configure_neat_inputs_and_outputs(config_path:str, indicator_mapping:IndicatorMapping, tickers:List[str]):
    all_feature_names = [n for config in indicator_mapping for n in config.names]
    n_features = len(all_feature_names)
    n_assets = len(tickers)

    configured_n_inputs = get_neat_config_param(config_path, 'num_inputs')
    if configured_n_inputs != n_features:
        print(f'Overriding num_inputs which was set by the neat_config file at {configured_n_inputs} to {n_features}')
        set_neat_config_param(config_path, 'num_inputs', n_features)
    else:
        print(f'num_inputs is already set to {n_features} as expected')
    
    configured_n_outputs = get_neat_config_param(config_path, 'num_outputs')
    n_outputs = n_assets
    # get the number of assets we are trading
    if configured_n_outputs != n_assets:
        print(f'Overriding num_outputs which was set by the neat_config file at {configured_n_outputs} to {n_outputs}')
        set_neat_config_param(config_path, 'num_outputs', n_outputs)
    else:
        print(f'num_outputs is already set to {n_outputs} as expected')


def test_genome_single_stock_set(net:neat.nn.FeedForwardNetwork, tickers:List[str], indicator_mapping:IndicatorMapping, use_test_data:bool=True):
    """Runs a backtest simulation giving the Agent """
    import logging
    bot_config = StrategyConfig(
        indicator_mapping=indicator_mapping,
        strategy=NEATEqualAllocation,
        tickers=tickers,
        args=[],
        kwargs={'net':net, 'log_level':logging.WARNING},
        quantity=1
        )

    # backtest the entire generation simultaneously
    result = run_simulation_on_candle_data(
        algorithm_configs=[bot_config],
        compute_state_for_each_strategy=False,
        use_test_data=use_test_data,
        verbose=True,
        plot=True,
        folder_to_save_plots=os.path.join(IMAGES_DIR, 'test' if use_test_data else 'train'),
    )
    return result

    
def run(
    CONFIG_PATH:str, 
    eval_genomes:Callable, 
    test_genome_single_stock_sets:Callable,
    produce_net_from_genome:Callable,  # this is a function that takes a genome and config and produces a net
    indicator_mapping:IndicatorMapping, 
    data:CandleData, 
    train_test_split:float=0.8,
    max_generations:int=80):
    
    print(f'Using NEAT config file: {CONFIG_PATH}')
    
    # set up the data
    build_features(data, [indicator_mapping])
    set_train_test_true(train_test_split)

    # Load configuration.
    CONFIG = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(CONFIG)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    winning_genome:neat.DefaultGenome = p.run(eval_genomes, max_generations)

    print('\nBest genome:\n{!s}'.format(winning_genome))

    node_names = {'A':0}
    
    neat_visualize.draw_neat_net(CONFIG, winning_genome, True, node_names=node_names)
    neat_visualize.plot_stats(stats, ylog=False, view=True)
    neat_visualize.plot_species(stats, view=True)

    # play through the winning genome and do a full performance analysis incl. visualizations
    print(f'Running winning genome on train data with full output')
    # results = run_simulation_on_candle_data(algorithm_configs=bot_configs, compute_state_for_each_strategy=False,)
    net = produce_net_from_genome(winning_genome, CONFIG)
    
    test_genome_single_stock_sets(net, data.tickers, indicator_mapping, use_test_data=False)
    print(f'Playing through on test data')
    test_genome_single_stock_sets(net, data.tickers, indicator_mapping, use_test_data=True)
    return winning_genome


def run_k_fold_training(
    config_path:str, 
    eval_genomes:Callable, 
    indicator_mapping:IndicatorMapping, 
    data:CandleData, 
    test_data:CandleData=None, 
    max_generations:int=100, 
    k:int=5):

    n_features = 0
    for i_config in indicator_mapping:
        if i_config.desired_output_name_keywords is not None:
            n_features += len(i_config.desired_output_name_keywords)
        else:
            n_features += 1
    n_assets = len(data.tickers)
    configure_neat_inputs_and_outputs(config_path, n_features, n_assets)

    # Load configuration.
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # break the data up into k folds
    train_data_splits = data.k_fold_split(k)
    MAX_GENERATIONS_PER_FOLD = max_generations // k
    for fold in train_data_splits:
        data = fold           # a global variable referenced by eval_genomes
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(neat_config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winning_genome:neat.DefaultGenome = p.run(eval_genomes, MAX_GENERATIONS_PER_FOLD)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winning_genome))

    with open(MODEL_DIR, 'wb') as f:
        pickle.dump({'winning_genome':winning_genome, 'config':neat_config, 'indicator_mapping':indicator_mapping}, f)
    print(f'Saved winning genome to {MODEL_DIR}')

    # node_names = {0: 'output'}
    # for i in range(7): 
    #     num = i + 1
    #     node_names[-num] = 'X' + str(num)
    node_names = None
    
    neat_visualize.draw_neat_net(neat_config, winning_genome, True, node_names=node_names)
    neat_visualize.plot_stats(stats, ylog=False, view=True)
    neat_visualize.plot_species(stats, view=True)

    # play through the winning genome and do a full performance analysis incl. visualizations
    if test_data is None: 
        print(F'Warning! No test data provided. Doing a backtest on the training data.')
        d=data
    else: d=test_data
    test_genome_single_stock_set(winning_genome, neat_config, d, indicator_mapping)


def delete_all_images():
    """Deletes all the files in the neat_images folder"""
    
    def delete_all_files_in_folder(folder_path):
        # walk over the subdirectories and files and delete
        for root, dirs, files in os.walk(folder_path):
            if root == folder_path:
                continue
            
            for file in files:
                if file == '.gitignore':
                    continue
                os.remove(os.path.join(root, file))
    
    for folder in ['.', 'train', 'test', 'model_and_training']:
    
        delete_all_files_in_folder(os.path.join(os.curdir, 'images'+'/'+folder))
    
    