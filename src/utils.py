from pathlib import Path
import logging
from logging import info
from sys import argv
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import mlflow as mf


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

    tracking_uri = 'databricks' if params.main.use_databricks else str(Path('../' + params.main.mlruns_path).absolute())
    info(f'Tracking info will go to: {tracking_uri}')
    mf.set_tracking_uri(tracking_uri)
    experiment_name = params.main.experiment_name  # TODO Should I allow to set it only from CLI, and here only read it?
    # Therefore remove it from params.yaml
    mf.set_experiment(experiment_name)
    info(f'Experiment name is: {experiment_name}')

    mf.start_run()  # TODO Should I check if a run is already going on, before starting a new one?
    run_name = get_run_name()
    info(f'Started run {run_name}')


def get_data_filename(params: DictConfig, symbol: str) -> str:
    """
    Returns the file name to be used to save the dataset locally, with a path relative to the root of the project
    :param params: configuration parameters, as filled in by Hydra.
    :param symbol: ticker symbol of the stock the dataset relates to.
    :return: the requested file name, with relative path.
    """
    # TODO Should I return an absolute path here instead?
    data_filename = f'../{params.main.data_path}/daily_price-{symbol}.csv'
    return data_filename


def get_autogluon_dir(params: DictConfig) -> str:
    """
    Returns the directory for Autogluon output (models, etc.). It is a relative path from the root of the project
    :param params: configuration parameters, as filled in by Hydra.
    :return: the requested directory name, with a relative path.
    """
    autogluon_dir = f'../{params.main.autogluon_path}/autogluon'
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
    adjausted prices.
    :param filename: the name of the file with the information.
    :return: the requested dataset in a dataframe.
    """
    df = pd.read_csv(filename)
    df = df.astype({'timestamp': 'datetime64'})
    return df