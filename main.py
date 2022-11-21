import requests
import shutil
import json
import os
from dotenv import load_dotenv
import pandas as pd


def download_time_series_daily_adjusted(symbol: str, full_output: bool, api_key: str, filename: str):
    outputsize = 'full' if full_output else 'compact'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize={outputsize}&datatype=csv&apikey={api_key}'
    with requests.get(url, stream=True) as req:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(req.raw, f)


def time_series_daily_adjusted(symbol: str, full_output: bool, api_key: str) -> pd.DataFrame:
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
                    'adjusted close': 'float',
                    'volume': 'int',
                    'dividend amount': 'float',
                    'split coefficient': 'float'})
    df.rename_axis('timestamp', inplace=True)
    return df


def load_daily_price_adjusted(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df


def main():
    load_dotenv()
    api_key = os.getenv('API_KEY')

    symbol = 'CSCO'
    """df = time_series_daily_adjusted(symbol, full_output=True, api_key=api_key)
    df.to_csv(f'daily_price-{symbol}.csv')"""
    filename = f'daily_price-{symbol}.csv'
    # download_time_series_daily_adjusted(symbol, full_output=True, api_key=api_key, filename=filename)

    df = load_daily_price_adjusted(filename)
    ...


if __name__ == '__main__':
    main()
