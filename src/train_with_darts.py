import os
from logging import info
from pathlib import Path
# from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
# from autogluon.timeseries.splitter import MultiWindowSplitter
from darts import TimeSeries, metrics
from darts.models import NaiveSeasonal
from omegaconf import DictConfig
import hydra
import mlflow as mf
from utils import boostrap_pipeline_component, get_data_filename, load_daily_price_adjusted, \
    get_autogluon_dir, log_outputs, log_dataframe


# mlflow run src -e train --experiment-name default_experiment

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
        model = NaiveSeasonal()
        res = model.backtest(series=ts_train_val,
                             start=prediction_length * 52,
                             forecast_horizon=prediction_length,
                             stride=1,
                             retrain=True,
                             overlap_end=False,
                             last_points_only=False,
                             metric=metrics.mse,
                             reduction=None,
                             verbose=True)
        print(res)

        """
        # noinspection PyArgumentList
        res = model.historical_forecasts(series+=ts_train_val,
                                         start=5 * 52,
                                         forecast_horizon=5,
                                         stride=5,
                                         retrain=True,
                                         overlap_end=False,
                                         last_points_only=False,
                                         verbose=True)
                                         """

        # print(res)

        """     
        num_windows = params.training.num_windows
        test_data = ts_df
        train_data = ts_df.slice_by_timestep(None, -prediction_length * num_windows)
        autogluon_dir = get_autogluon_dir(params)

        splitter = MultiWindowSplitter(num_windows=num_windows)
        predictor = TimeSeriesPredictor(path=autogluon_dir,
                                        target="adjusted_close",
                                        prediction_length=prediction_length,
                                        eval_metric=params.training.eval_metric,
                                        ignore_time_index=True,
                                        validation_splitter=splitter
                                        )

        predictor.fit(train_data,
                      presets=params.training.presets,
                      time_limit=params.training.time_limit,
                      )

        mf.log_artifact(autogluon_dir)
        artifact_uri = mf.get_artifact_uri()
        info(f'Logged artifact {autogluon_dir} in current run with URI {artifact_uri}/{Path(autogluon_dir).name}')

        # score = predictor.evaluate(test_data)
        # summary = predictor.fit_summary()

        leaderboard = predictor.leaderboard(silent=False)
        # predictor.leaderboard(data=train_data, silent=False)
        leaderboard.set_index('model', inplace=True)
        info('Logging the leaderboard to the current run')
        log_dataframe(leaderboard)
        if params.training.selected_model is None:
            selected_model = predictor.get_model_best()
            info(f'Selected model based on cross-validations: {selected_model}')
        else:
            selected_model = params.training.selected_model
            info(f'Selected model based configuration parameters: {selected_model}')
            best_model = predictor.get_model_best()
            if selected_model != best_model:
                info(f'Note: selected model is not the best based on cross-validation, which would be {best_model}')

        # Now chose the model of choice with the best evaluation score, and re-fit it over the whole train-val dataset


        """
    except Exception as ex:
        raise ex
    finally:
        log_outputs()


if __name__ == '__main__':
    main()

""" TODO Open questions
When using multi-window backtesting, is the validation metric the average of the validation metrics?
When using multi-window backtesting, is the model re-trained on all the available data?

"""
