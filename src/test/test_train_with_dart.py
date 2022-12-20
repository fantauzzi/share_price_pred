import sys
import os
from pathlib import Path
import numpy as np
import hydra
from hydra import compose, initialize
import mlflow as mf

sys.path.append('..')
from utils import get_data_filename, load_daily_price_adjusted


def test_train_with_dart():
    initialize(config_path='../../config', job_name="test_app")
    params = compose(config_name="params")
    os.chdir('../..')

    experiment_name = params.run.main.experiment_name
    train_params_override = 'run=params_test'

    mf.run(uri=str(Path('src').absolute()),
           entry_point='train_with_darts',
           experiment_name=experiment_name,
           parameters={'overrides': train_params_override})


"""
def test_main():
    initialize(config_path='../../config', job_name="test_app")
    params = compose(config_name="params")
    os.chdir('..')
    symbol = 'CSCO'
    data_filename = get_data_filename(params, symbol)
    df = load_daily_price_adjusted(data_filename)
    values = np.around([10. + 90 * i / len(df) for i in df.index], decimals=2)
    df.open = values
    df.high = values
    df.low = values
    df.close = values
    df.adjusted_close = values
    df.volume = [int(value) for value in values]
    df.dividend_amount = 0
    df.split_coefficient = 1.

    df.to_csv('../data/daily_price-TEST.csv', index=False)
"""
