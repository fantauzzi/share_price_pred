import os
from pathlib import Path
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import mlflow as mf
from utils import boostrap_pipeline_component


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

    if 'source_data' in required_steps:
        mf.run(uri=str(Path(hydra.utils.get_original_cwd()) / '.'),
               entry_point='source_data',
               experiment_name=experiment_name,
               parameters={'overrides': params_override})
    if 'train' in required_steps:
        mf.run(uri=str(Path(hydra.utils.get_original_cwd()) / '.'),
               entry_point='train',
               experiment_name=experiment_name,
               parameters={'overrides': params_override})


if __name__ == '__main__':
    main()
