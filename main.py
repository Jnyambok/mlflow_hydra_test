import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


#The @hydra.main() decorator tells Hydra that this is the main function of the script. This decorator is required for Hydra to work.
@hydra.main(config_path="conf/config.yaml",version_base=None)
def go(config: DictConfig)-> None:

    #You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    #serialize decision tree configuration
    model_config = os.path.abspath("random_forest_config.yaml")

    #The with mlflow.start_run() statement creates a new MLflow run. All MLflow logging calls made within this block will be associated with the run.
    with open(model_config, 'w+') as fp:
        fp.write(OmegaConf.to_yaml(config))

    _ = mlflow.run(
        os.path.join(
            root_path, 'random_forest'
            ), #run the subdirectory
        'main',
        parameters={
            'model_config': model_config
        },
    )

if __name__ == '__main__':
    go()