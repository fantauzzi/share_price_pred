import requests
import shutil
import json
import os
from pathlib import Path
import logging
from logging import info
from dotenv import load_dotenv
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig


def download_time_series_daily_adjusted(symbol: str, full_output: bool, api_key: str, filename: str):
    """
    Downloads into a CSV file all the available daily adjusted closure prices for a given stock
    :param symbol:
    :param full_output:
    :param api_key:
    :param filename:
    :return:
    """
    outputsize = 'full' if full_output else 'compact'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={outputsize}&datatype=csv&apikey={api_key}'
    with requests.get(url, stream=True) as req:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(req.raw, f)


def time_series_daily_adjusted(symbol: str, full_output: bool, api_key: str) -> pd.DataFrame:
    """
    Returns a dataframe with all the available daily adjusted closure prices for a given stock.
    :param symbol:
    :param full_output:
    :param api_key:
    :return:
    """
    outputsize = 'full' if full_output else 'compact'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={outputsize}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()['Time Series (Daily)']
    # noinspection PyTypeChecker
    df = pd.read_json(json.dumps(data), orient='index', dtype=False, convert_axes=False)
    df.rename({'1. open': 'open',
               '2. high': 'high',
               '3. low': 'low',
               '4. close': 'close',
               '5. adjusted close': 'adjusted_close',
               '6. volume': 'volume',
               '7. dividend amount': 'dividend_amount',
               '8. split coefficient': 'split_coefficient'},
              inplace=True, axis=1)

    df = df.astype({'open': 'float',
                    'high': 'float',
                    'low': 'float',
                    'close': 'float',
                    'adjusted_close': 'float',
                    'volume': 'int',
                    'dividend_amount': 'float',
                    'split_coefficient': 'float'})
    df.rename_axis('timestamp', inplace=True)
    return df


def load_daily_price_adjusted(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df.astype({'timestamp': 'datetime64'})
    return df


@hydra.main(version_base=None, config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

    info(f'Working directory: {Path.cwd()}')
    log_file = HydraConfig.get().job_logging.handlers.file.filename
    info(f'Log file: {log_file}')
    dot_hydra = f'{HydraConfig.get().run.dir}/{HydraConfig.get().output_subdir}'
    info(f'Hydra output sub-directory: {dot_hydra}')

    load_dotenv()
    api_key = os.getenv('API_KEY')

    symbol = params.main.stock_symbol
    data_filename = f'../{params.main.data_path}/daily_price-{symbol}.csv'
    autogluon_dir = f'../{params.main.autogluon_path}/autogluon'
    if not Path(data_filename).exists():
        info(f'Downloading data for stock {symbol} into file {data_filename}')
        download_time_series_daily_adjusted(symbol, full_output=True, api_key=api_key, filename=data_filename)

    info(f'Loading data from file {data_filename}')
    df = load_daily_price_adjusted(data_filename)
    df.drop(['open', 'high', 'low', 'close', 'volume', 'dividend_amount', 'split_coefficient'], axis=1, inplace=True)
    df['symbol'] = symbol
    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='symbol', timestamp_column='timestamp')

    prediction_length = 7
    test_data = ts_df
    train_data = ts_df.slice_by_timestep(None, -prediction_length)
    predictor = TimeSeriesPredictor(path=autogluon_dir,
                                    target="adjusted_close",
                                    prediction_length=prediction_length,
                                    eval_metric="MAPE",
                                    ignore_time_index=True
                                    )

    predictor.fit(train_data,
                  presets="fast_training",
                  time_limit=60,
                  )

    leaderboard = predictor.leaderboard(test_data, silent=False)
    # print(leaderboard)


if __name__ == '__main__':
    main()

# TODO use a multi-window backtest
# Draw charts, understand what is going on (MLFlow? Tensorboard?)
# Compare results with QQQ and indices with those of individual stocks
# Try much longer time limit, or allocate additional time to time-consuming models, if AutoGluon allows it
# Understand how AutoGluon uses metadata and covariates
# How to explain the results?
# Check AutoTS to do multivariate stuff https://winedarksea.github.io/AutoTS/build/html/index.html
