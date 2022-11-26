import requests
import shutil
import json
import os
from pathlib import Path
from logging import info
from dotenv import load_dotenv
import pandas as pd
from omegaconf import DictConfig
import hydra
import mlflow as mf
from utils import boostrap_pipeline_component, get_data_filename, get_run_name


# mlflow run src -e source_data --experiment-name default_experiment


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


@hydra.main(config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    # Using an old version of hydra-core because of compatibility with autogluon;
    # the old version changes current directory right after reading the params.yaml file.
    # Here we undo that, and set the current directory back to the one containing this script
    os.chdir('../../..')
    info(f'Running {Path(__file__).name} ############################################################################')
    boostrap_pipeline_component(params)
    mf.start_run()
    run_name = get_run_name()
    info(f'Started run {run_name}')

    load_dotenv()
    api_key = os.getenv('API_KEY')

    symbol = params.main.stock_symbol
    data_filename = get_data_filename(params, symbol)
    if not Path(data_filename).exists():
        info(f'Downloading data for stock {symbol} into file {data_filename}')
        download_time_series_daily_adjusted(symbol, full_output=True, api_key=api_key, filename=data_filename)
    else:
        info(f'Data for stock {symbol} already available in {data_filename} -Not going to download data')


if __name__ == '__main__':
    main()
