import os
from logging import info

import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA, forecasting
from omegaconf import DictConfig
import hydra
import mlflow as mf
from utils import boostrap_pipeline_component, get_data_filename, load_daily_price_adjusted, log_outputs, get_run_name
import numpy as np


# mlflow run src -e train_with_darts --experiment-name default_experiment


def rolling_forecast(model: forecasting,
                     df_dataset: pd.DataFrame,
                     start_at: int,
                     prediction_length: int,
                     stride: int,
                     value_cols: str,
                     saved_model: str) -> (forecasting, pd.DataFrame):
    """
    Trains and backtests the model, saves the resulting model and logs the model and metrics with MLFlow. Before
    being saved, the model is re-trained on the whole dataset.
    :param model: the model to be trained and backtested.
    :param df_dataset: the dataset to be used for training and backtasting.
    :param start_at: the position of the first sample in the dataset that will be used to train the model; training
    and backtest will start from that position and proceed toward the end of the dataset.
    :param prediction_length: the temporal horizon for the forecast; a forecast at time T will predict times
    T+1,...,T+prediction_length
    :param stride: the number of temporal steps between two consecutive forecasts
    :param value_cols: the name of the column in the df_dataset dataframe that contain the variable to be forecasted
    :param saved_model: file name with path where the trained model will be saved
    """
    ts_train_val = TimeSeries.from_dataframe(df=df_dataset, value_cols=value_cols)

    forecasts = model.historical_forecasts(series=ts_train_val,
                                           start=start_at,
                                           forecast_horizon=prediction_length,
                                           stride=stride,
                                           retrain=True,
                                           overlap_end=False,
                                           last_points_only=False,
                                           verbose=True)

    # Store the forecasts in a dataframe, to be returned

    forecasts_dict = {i: [] for i in range(prediction_length)}
    forecasts_timestamp = []
    for metrics_series in forecasts:
        metrics_series_pd = metrics_series.pd_series()
        time_at_forecast = metrics_series_pd.index[0] - 1
        forecasts_timestamp.append(time_at_forecast)
        for i in range(time_at_forecast, time_at_forecast + prediction_length):
            forecasts_dict[i - time_at_forecast].append(metrics_series_pd[i + 1])

    forecasts_df = pd.DataFrame(forecasts_dict, index=forecasts_timestamp)

    info('Re-fitting model on whole dataset')
    model.fit(series=ts_train_val)
    info(f'Saving traiend model into {saved_model}')
    model.save(saved_model)
    info(f'Logging artifact {saved_model} in current run')
    mf.log_artifact(saved_model)

    return model, forecasts_df


def log_rolling_forecasts(forecasts_df: pd.DataFrame, df_dataset: pd.DataFrame, value_cols: str) -> None:
    for t_index in forecasts_df.columns:
        for time_at_forecast in forecasts_df.index:
            mf.log_metrics({f'Forecast T{t_index + 1}': forecasts_df[t_index][time_at_forecast],
                            f'Ground truth T{t_index + 1}': df_dataset[value_cols][time_at_forecast + t_index + 1]},
                           step=time_at_forecast + t_index + 1)


def rolling_metrics(forecasts_df: pd.DataFrame, df_dataset: pd.DataFrame, value_cols):
    prediction_length = len(forecasts_df.columns)
    ground_truth_dict = {i: [] for i in range(prediction_length)}
    for t_index in forecasts_df.columns:
        for time_at_forecast in forecasts_df.index:
            ground_truth_dict[t_index].append(df_dataset[value_cols][time_at_forecast + 1 + t_index])

    ground_truth_df = pd.DataFrame(ground_truth_dict, index=forecasts_df.index)

    info('Logging validation metrics in current run')
    assert ground_truth_df.columns.equals(forecasts_df.columns)

    se_df = (forecasts_df - ground_truth_df) ** 2
    ae_df = np.abs(forecasts_df - ground_truth_df)
    ape_df = 100 * ae_df / ground_truth_df

    for label, df in zip(('SE', 'AE', 'APE'), (se_df, ae_df, ape_df)):
        for time_idx, value in df.mean().items():
            mf.log_metric(key=f'M{label}', value=value, step=time_idx)
        for time_idx, row in df.iterrows():
            metrics_to_log = {f'{label} at T{item_idx + 1}': value for item_idx, value in row.items()}
            mf.log_metrics(metrics=metrics_to_log, step=time_idx)


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
        columns_to_drop = set(df.columns) - {'timestamp', 'symbol', params.main.column}
        df.drop(columns_to_drop, axis=1, inplace=True)
        df['symbol'] = symbol

        prediction_length = params.training.prediction_length
        model = ARIMA()
        trained_saved_model = f'../{params.main.trained_saved_model}'
        df_train_val = df.iloc[:-params.training.test_set_size]

        with mf.start_run(nested=True) as _:
            run_name = get_run_name()
            info(f'Started nested run {run_name} for training and validation via backtesting on model with '
                 f'{len(df_train_val)} samples')
            model, forecast_df = rolling_forecast(model=model,
                                                  df_dataset=df_train_val,
                                                  start_at=params.training.start_training_at,
                                                  prediction_length=prediction_length,
                                                  stride=params.training.stride,
                                                  value_cols=params.main.column,
                                                  saved_model=trained_saved_model)

            log_rolling_forecasts(forecasts_df=forecast_df, df_dataset=df_train_val, value_cols=params.main.column)

            rolling_metrics(forecasts_df=forecast_df, df_dataset=df_train_val, value_cols=params.main.column)

        tested_saved_model = f'../{params.main.tested_saved_model}'
        with mf.start_run(nested=True) as _:
            run_name = get_run_name()
            info(f'Started nested run {run_name} for testing on model with '
                 f'{params.training.test_set_size} samples to be forecasted')
            rolling_forecast(model=model,
                             df_dataset=df,
                             start_at=len(df) - params.training.test_set_size,
                             prediction_length=prediction_length,
                             stride=params.training.stride,
                             value_cols=params.main.column,
                             saved_model=tested_saved_model)

    except Exception as ex:
        raise ex
    finally:
        log_outputs()


if __name__ == '__main__':
    main()

""" TODO
Why the spike in validation metrics toward the end of the training/val dataset?
Implement the testing without rolling forecast (model is frozen, no retrained), draw charts of predictions vs. actual
Add the MAPE metric (% error) 
Try different models, do hyperparameters tuning (Optuna?). How do you track the tuning with MLFlow? Check examples.
Automate experiments on a list of symbols. Should those be uni or multi variate?
Throw in an index as an exogenous variable
From source_data, save two different csv files for train/val and test
Explain the model
Dataset versoining
Deployment and inference
"""
