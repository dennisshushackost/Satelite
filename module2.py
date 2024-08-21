import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from helpers.train import TrainNetwork


def experiment_1(models, data_set_path, data_set_path_upscaled):
    for model_name in models:
        for augmentation in [True, False]:
            for upscale in [True, False]:
                dataset_path = data_set_path_upscaled if upscale else data_set_path
                experiment_name = f'{model_name}_upscale_{upscale}_augmentation_{augmentation}'
                print(f"Running experiment: {experiment_name}")
                
                TrainNetwork(
                    dataset_path=dataset_path,
                    modelname=model_name,
                    augmentation=augmentation,
                    experiment_name=experiment_name,
                    experiment=1,
                    upscale=upscale
                )

# Example usage:
models_to_test = ['unet']  
data_set_path = "/workspaces/Satelite/data/dataset_upscaled_False"
data_set_path_upscaled = "/workspaces/Satelite/data/dataset_upscaled_True"

experiment_1(models_to_test, data_set_path, data_set_path_upscaled)