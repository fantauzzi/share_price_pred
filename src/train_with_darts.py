import os
from logging import info

import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA
from omegaconf import DictConfig
import hydra
import mlflow as mf
from utils import boostrap_pipeline_component, get_data_filename, load_daily_price_adjusted, log_outputs
import numpy as np


# mlflow run src -e train_with_darts --experiment-name default_experiment


@hydra.main(config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    try:
        os.chdir('../../..')
        boostrap_pipeline_component(params)

        symbol = params.main.stock_symbol
        if params.main.dataset is not None:
            info(f'Downloading artifact {params.main.dataset} as dataset')
            data_filename = mf.artifacts.download_artifacts(artifact_uri=params.main.dataset)
        else:
            data_filename = get_data_filename(params, symbol)
        info(f'Loading data from file {data_filename}')
        df = load_daily_price_adjusted(data_filename)
        df.drop(['open', 'high', 'low', 'close', 'volume', 'dividend_amount', 'split_coefficient'], axis=1,
                inplace=True)
        df['symbol'] = symbol
        df_train_val = df.iloc[:-params.training.test_set_size]
        ts_train_val = TimeSeries.from_dataframe(df=df_train_val, value_cols='adjusted_close')

        prediction_length = params.training.prediction_length
        model = ARIMA()
        saved_model = f'../{params.main.saved_model}'
        # noinspection PyArgumentList
        forecasts = model.historical_forecasts(series=ts_train_val,
                                               start=params.training.start_training_at,
                                               forecast_horizon=prediction_length,
                                               stride=params.training.stride,
                                               retrain=True,
                                               overlap_end=False,
                                               last_points_only=False,
                                               verbose=True)

        """ Compute the difference (with sign) forecast-actual and stores it in a DataFrame; every line corresponds
        to the set of forecasts made at a given time (in the line index) for the forecasting horizon (one forecast
        per column) """

        differences_dict = {i: [] for i in range(prediction_length)}
        forecasts_timestamp = []
        for metrics_series in forecasts:
            metrics_series_pd = metrics_series.pd_series()
            time_at_forecast = metrics_series_pd.index[0] - 1
            forecasts_timestamp.append(time_at_forecast)
            for i in range(time_at_forecast, time_at_forecast + prediction_length):
                diff = (metrics_series_pd[i + 1] - df_train_val.adjusted_close[i + 1])
                differences_dict[i - time_at_forecast].append(diff)

        differences_df = pd.DataFrame(differences_dict, index=forecasts_timestamp)

        model.save(saved_model)
        info(f'Logging artifact {saved_model} in current run')
        mf.log_artifact(saved_model)

        info('Logging validation metrics in current run')
        for idx, row in differences_df.iterrows():
            # noinspection PyTypeChecker
            se_to_log = {f'SE at T{item_idx + 1}': value ** 2 for item_idx, value in row.items()}
            # noinspection PyTypeChecker
            ae_to_log = {f'AE at T{item_idx + 1}': np.abs(value) for item_idx, value in row.items()}
            # noinspection PyTypeChecker
            mf.log_metrics(se_to_log, step=idx)
            # noinspection PyTypeChecker
            mf.log_metrics(ae_to_log, step=idx)

        mse_metrics = (differences_df ** 2).mean()
        mae_metrics = np.abs(differences_df).mean()
        mse_to_log = {f'MSE at T{item_idx + 1}': value for item_idx, value in mse_metrics.items()}
        mae_to_log = {f'MAE at T{item_idx + 1}': value for item_idx, value in mae_metrics.items()}
        mf.log_metrics(mse_to_log)
        mf.log_metrics(mae_to_log)


    except Exception as ex:
        raise ex
    finally:
        log_outputs()


if __name__ == '__main__':
    main()

""" TODO
Add the MAPE metric (% error) 
Automate experiments on a list of symbols. Should those be uni or multi variate?
Throw in an index as an exogenous variable
from source_data, save two different csv files for train/val and test
Implement the testing, draw charts of predictions vs. actual
Try different models, do hyperparameters tuning (Optuna?)
Explain the model
"""
