import datetime
import matplotlib.pyplot as plt

from parallelized_algorithmic_trader.data_management.historical_data import get_candle_data
from parallelized_algorithmic_trader.backtest import build_features
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
import parallelized_algorithmic_trader.performance_analysis as pf_anal
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
import parallelized_algorithmic_trader.indicators as indicators
import parallelized_algorithmic_trader.visualizations as viz
from genetic_algorithm_trading.indicators import *


if __name__ == '__main__':
    import os
    NEAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')

    tickers = ['SPY']

    indicator_mapping:IndicatorMapping= []
    for t in tickers:
        new_inds = [
            # IndicatorConfig(t+'_close', indicators.ZBB, args=(50,2), desired_output_name_keywords=['ZBBB']),
            # IndicatorConfig(t+'_close', indicators.ZBB, args=(200,2), desired_output_name_keywords=['ZBBB']),
            
            ZBB_change_in_stddev(t+'_close', 25, 50, 2),
            ZBB_change_in_stddev(t+'_close', 100, 200, 2),
            
            slope_ZBB(t+'_close', 50, 2, 1, 5),
            slope_ZBB(t+'_close', 200, 2, 1, 10),
            
            IndicatorConfig(t+'_close', indicators.MACD, args=[12, 26, 9]),
            IndicatorConfig(t+'_close', indicators.RSI, scaling_factor=0.05, bias=-50),
            
            # IndicatorConfig(t+'_close', MACD_INTERPRETATION_1, args=[12, 26, 9]),
            # IndicatorConfig(t+'_close', MACD_INTERPRETATION_1, args=[3, 6.5, 2.25]),
        #     IndicatorConfig(t+'_close', indicators.MACD, args=[12, 26, 9]),
        #     IndicatorConfig(IndicatorConfig(t, indicators.RSI), indicators.MIN_MAX_SCALER, kwargs={'min_val':-1, 'max_val':1}),

        #     IndicatorConfig(t, indicators.SUPERTREND, args=[25,3], desired_output_name_keywords=['SUPERTRENDd']),
        #     IndicatorConfig(t, indicators.SUPERTREND, args=[60,3], desired_output_name_keywords=['SUPERTRENDd']),

        #     IndicatorConfig(t+'_close', indicators.PercentBB, args=(15,)),
        #     IndicatorConfig(t+'_close', indicators.PercentBB, args=(45,)),
        #     IndicatorConfig(t+'_close', indicators.PercentBB, args=(60,)),
        ]
        indicator_mapping.extend(new_inds)
        break   # only do it for the first ticker for now

    # set up the data to be used as global variables
    start = datetime.datetime(year=2021, month=1, day=1)
    end = datetime.datetime(year=2023, month=1, day=1)
    
    candle_data = get_candle_data(tickers, TemporalResolution.HOUR, start, end)
    data = build_features(candle_data, [indicator_mapping])

    fig, (ax1, ax2) = viz.plot_backtest_results(data.df, tickers=candle_data.tickers)
    plt.show()
    
    