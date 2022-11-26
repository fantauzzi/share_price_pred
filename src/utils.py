from pathlib import Path
import logging
from logging import info
from hydra.core.hydra_config import HydraConfig
import mlflow as mf


def boostrap_pipeline_component(params):
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


def get_data_filename(params, symbol):
    data_filename = f'../{params.main.data_path}/daily_price-{symbol}.csv'
    return data_filename


def get_autogluon_dir(params):
    autogluon_dir = f'../{params.main.autogluon_path}/autogluon'
    return autogluon_dir


def get_run_name():
    mlflow_client = mf.MlflowClient()
    mlflow_run_data = mlflow_client.get_run(mf.active_run().info.run_id).data
    run_name = mlflow_run_data.tags["mlflow.runName"]
    return run_name
