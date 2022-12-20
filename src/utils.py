from pathlib import Path
import logging
from logging import info
from sys import argv
from typing import Union
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import mlflow as mf
from mlflow.entities import Metric


def boostrap_pipeline_component(params: DictConfig) -> None:
    """
    Boilerplate actions a pipeline component should execute when it starts:
     - configure the logging;
     - log basic information about the component that has started
     - set the tracking URI for MLFLow, between a local directory and Databricks
     - set the MLFlow experiment name and log the name
     - start an MLFlow run and log its name
    :param params: The parameters for the configuration of the component, they are meant to have been filled in with Hydra
    """
    info(f'Running {Path(argv[0]).name} ############################################################################')
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    info(f'Working directory: {Path.cwd()}')
    log_file = HydraConfig.get().job_logging.handlers.file.filename
    info(f'Log file: {log_file}')
    dot_hydra = f'{HydraConfig.get().run.dir}/{HydraConfig.get().output_subdir}'
    info(f'Hydra output sub-directory: {dot_hydra}')

    tracking_uri = 'databricks' if params.run.main.use_databricks else str(Path('../' + params.run.main.mlruns_path).absolute())
    info(f'Tracking info will go to: {tracking_uri}')
    mf.set_tracking_uri(tracking_uri)
    experiment_name = params.run.main.experiment_name  # TODO Should I allow to set it only from CLI, and here only read it?
    # Therefore remove it from params.yaml
    mf.set_experiment(experiment_name)
    info(f'Experiment name is: {experiment_name}')

    mf.start_run()  # TODO Should I check if a run is already going on, before starting a new one?
    run_name = get_run_name()
    info(f'Started run {run_name}')

    unfolded_params = unfold_config(params)
    info(f'Logging parameters into current run: {unfolded_params}')
    mf.log_params(unfold_config(params))


def get_data_filename(params: DictConfig, symbol: str) -> str:
    """
    Returns the file name to be used to save the dataset locally, with a path relative to the root of the project
    :param params: configuration parameters, as filled in by Hydra.
    :param symbol: ticker symbol of the stock the dataset relates to.
    :return: the requested file name, with relative path.
    """
    # TODO Should I return an absolute path here instead?
    data_filename = f'../{params.run.main.data_path}/daily_price-{symbol}.csv'
    return data_filename


def get_autogluon_dir(params: DictConfig) -> str:
    """
    Returns the directory for Autogluon output (models, etc.). It is a relative path from the root of the project
    :param params: configuration parameters, as filled in by Hydra.
    :return: the requested directory name, with a relative path.
    """
    autogluon_dir = f'../{params.run.main.autogluon_path}/autogluon'
    return autogluon_dir


def get_run_name() -> str:
    """
    Returns the name of the current MLFlow run.
    :return: the requested name.
    """
    # TODO what happens if there is no run ongoing?
    mlflow_client = mf.MlflowClient()
    mlflow_run_data = mlflow_client.get_run(mf.active_run().info.run_id).data
    run_name = mlflow_run_data.tags["mlflow.runName"]
    return run_name


def load_daily_price_adjusted(filename: str) -> pd.DataFrame:
    """
    Loads from a file, and returns a Pandas dataframe, with the daily information for a given stock, including its
    adjusted prices.
    :param filename: the name of the file with the information.
    :return: the requested dataset in a dataframe.
    """
    df = pd.read_csv(filename)
    df = df.astype({'timestamp': 'datetime64'})
    return df


def log_outputs() -> None:
    dot_hydra = f'{HydraConfig.get().run.dir}'
    info(f'Logging artifact {dot_hydra} into current run')
    mf.log_artifact(dot_hydra)


def unfold_config(config: Union[DictConfig, dict]) -> dict[str, str]:
    """
    Takes a DictConfig, or a dict obtained from a DictConfig, and converts it to a dict with one key-value pair
    for every parameter, where the grouping of keys from the DictConfig is replaced by concatenating all the keys
    with a dot.
    :param config: the given DictConfig, or the given DictConfig cast to a dict.
    :return: a dictionary with the result of the translation.
    """

    def unfold_config_as_list(config: Union[DictConfig, dict]) -> list[str]:
        res = []
        for key, value in config.items():
            if isinstance(value, dict) or isinstance(value, DictConfig):
                embedded_res = unfold_config_as_list(value)
                res.extend([f'{key}.{item}' for item in embedded_res])
            else:
                res.append(f'{key} {value}')
        return res

    unfolded = unfold_config_as_list(config)
    unfolded = {item[:item.rfind(' ')]: item[item.rfind(' ') + 1:] for item in unfolded}
    unfolded = dict(sorted(unfolded.items()))
    return unfolded


def log_dataframe(df: pd.DataFrame, run_id: str = None) -> None:
    """
    Log a dataframe as metrics under a given run, or the current run. For every value in the dataframe, its
    metric name is obtained from the value of the dataframe index on the same row and the name of the column.
    :param df: The dataframe to be logged as metrics.
    :param run_id: The ID of the run under which to log the metrics. If missing, it defaults to the current run.
    """
    if run_id is None:
        run_id = mf.active_run().info.run_id

    log_this = [Metric(key=f'{row_name}/{col_name}', value=item, step=0, timestamp=0)
                for col_name in df
                for row_name, item in zip(df.index, df[col_name])]

    client = mf.MlflowClient()
    client.log_batch(run_id=run_id, metrics=log_this)


def main():
    df = pd.DataFrame({'model': ['ARIMA', 'ETS', 'Theta'], 'score_test': [1, 2, 3], 'score_val': [-1, -2, -3]})
    df.set_index('model', inplace=True)


if __name__ == '__main__':
    main()
