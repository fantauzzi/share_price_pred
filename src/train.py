import os
from logging import info
from pathlib import Path
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from omegaconf import DictConfig
import hydra
import mlflow as mf
from utils import boostrap_pipeline_component, get_data_filename, load_daily_price_adjusted, \
    get_autogluon_dir


# mlflow run src -e train --experiment-name default_experiment

@hydra.main(config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
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
    df.drop(['open', 'high', 'low', 'close', 'volume', 'dividend_amount', 'split_coefficient'], axis=1, inplace=True)
    df['symbol'] = symbol
    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='symbol', timestamp_column='timestamp')

    prediction_length = params.training.prediction_length
    test_data = ts_df
    train_data = ts_df.slice_by_timestep(None, -prediction_length)
    autogluon_dir = get_autogluon_dir(params)
    predictor = TimeSeriesPredictor(path=autogluon_dir,
                                    target="adjusted_close",
                                    prediction_length=prediction_length,
                                    eval_metric=params.training.eval_metric,
                                    ignore_time_index=True
                                    )

    predictor.fit(train_data,
                  presets=params.training.presets,
                  time_limit=params.training.time_limit,
                  )

    mf.log_artifact(autogluon_dir)
    artifact_uri = mf.get_artifact_uri()
    info(f'Logged artifact {autogluon_dir} in current run with URI {artifact_uri}/{Path(autogluon_dir).name}')

    leaderboard = predictor.leaderboard(test_data, silent=False)


if __name__ == '__main__':
    main()

# TODO use a multi-window backtest
# Draw charts, understand what is going on (MLFlow? Tensorboard?)
# Compare results with QQQ and indices with those of individual stocks
# Try much longer time limit, or allocate additional time to time-consuming models, if AutoGluon allows it
# Understand how AutoGluon uses metadata and covariates
# How to explain the results?
# Check AutoTS to do multivariate stuff https://winedarksea.github.io/AutoTS/build/html/index.html
