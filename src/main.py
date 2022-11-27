import os
from pathlib import Path
from logging import info, warning
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import mlflow as mf
from utils import boostrap_pipeline_component, get_data_filename


# mlflow run src -e main --experiment-name default_experiment

@hydra.main(config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    os.chdir('../../..')
    boostrap_pipeline_component(params)

    available_steps = ('source_data', 'train')
    required_steps = available_steps if params.main.steps in ('all', None) else params.main.steps.split(',')
    for step in required_steps:
        assert step in available_steps
    experiment_name = params.main.experiment_name
    task = HydraConfig.get().overrides.task
    params_override = ' '.join(task)

    dataset_artifact_uri = None
    # If required to source the dataset and no dataset is provided in the parameters...
    if 'source_data' in required_steps and params.main.dataset is None:
        # ... then download the dataset and log it with MLFlow
        symbol = params.main.stock_symbol
        data_filename = get_data_filename(params, symbol)
        if Path(data_filename).exists():
            warning(f'Dataset already available in file {data_filename} -Will be overwritten with new download')
        source_data_run = mf.run(uri=str(Path(hydra.utils.get_original_cwd()) / '.'),
                                 entry_point='source_data',
                                 experiment_name=experiment_name,
                                 parameters={'overrides': params_override})
        # Given the URI to the root of the artifacts repository, figure out the URI to the dataset that has just been downloaded and logged
        artifact_uri = mf.get_artifact_uri()
        last_slash_pos = artifact_uri.rfind('/')
        second_last_slash_pos = artifact_uri[:last_slash_pos].rfind('/')
        dataset_artifact_uri = f'{artifact_uri[:second_last_slash_pos + 1]}{source_data_run.run_id}{artifact_uri[last_slash_pos:]}/{Path(data_filename).name}'
        info(f'Dataset artifact has been logged as {dataset_artifact_uri}')

    if 'train' in required_steps:
        train_params_override = params_override if dataset_artifact_uri is None else f'{params_override} main.dataset={dataset_artifact_uri}'.lstrip()

        mf.run(uri=str(Path(hydra.utils.get_original_cwd()) / '.'),
               entry_point='train',
               experiment_name=experiment_name,
               parameters={'overrides': train_params_override})


if __name__ == '__main__':
    main()
