import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from helpers.train import TrainNetwork

# Path to non-upscaled dataset
data_set_path = "/workspaces/Satelite/data/dataset_upscaled_False"
# Path to upscaled dataset (not used in this experiment as upscale=False)
data_set_path_upscaled = "/workspaces/Satelite/data/dataset_upscaled_True"
models = ['resunet']

# Function to run experiments
def run_experiments(data_set_path, models):
    for model_name in models:
        for augmentation in [False]:
            # We are only using the non-upscaled dataset as per the requirements
            upscale = False
            dataset_path = data_set_path
            experiment_name = f'{model_name}_augmentation_{augmentation}'
            print(f"Running experiment: {experiment_name}")
            TrainNetwork(dataset_path, augmentation=augmentation, modelname=model_name, experiment_name=experiment_name, upscale=upscale)

if __name__ == "__main__":
    run_experiments(data_set_path, models)
