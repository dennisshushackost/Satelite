import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from numba import cuda
from pathlib import Path
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from helpers.load import LoadandAugment
import helpers.model
import helpers.modelup

# Reset the GPU memory:
device = cuda.get_current_device()
device.reset()


class TrainNetwork:
    """ 
    This class trains and evaluates the network.
    """
    def __init__(self, dataset_path, augmentation, modelname, experiment_name, experiment, upscale=False):
        self.experiment = experiment
        self.dataset_path = dataset_path
        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")
        self.val_path = os.path.join(self.dataset_path, "val")
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model_name = modelname
        self.augmentation = augmentation
        self.num_classes = 1
        self.dropout_rate = 0.0
        self.batch_norm = True
        self.model_path = self.create_folders(experiment_name)
        self.upscale = upscale
        self.experiment_name = experiment_name
        if self.upscale:
            self.input_shape = (512, 512, 4)
            self.batch_size = 8
        else:
            self.input_shape = (256, 256, 4)
            self.batch_size = 16

        if modelname == 'unet':
            if self.upscale:
                self.model = helpers.modelup.unet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm) 
            else:
                self.model = helpers.model.unet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        elif modelname == 'attunet':
            if self.upscale:
                self.model = helpers.modelup.attunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
            else:
                self.model = helpers.model.attunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        elif modelname == 'resunet':
            if self.upscale:
                self.model = helpers.modelup.resunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
            else:
                self.model = helpers.model.resunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        else:
            raise ValueError(f"Model {modelname} not implemented")
        
        self.train_network()
        self.test_accuracy = self.evaluate_network()
        self.record_experiment()

    def create_folders(self, experiment_name):
        base_path = Path(self.dataset_path).parent
        experiment_path = base_path / 'experiments' / experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)
        self.model_path = experiment_path
        return self.model_path

    def train_network(self):
        self.train_data = LoadandAugment(dataset_path=self.train_path, data_type="train", batch=self.batch_size, augmentation=self.augmentation, upscale=self.upscale)
        self.val_data = LoadandAugment(dataset_path=self.val_path, data_type="val", batch=self.batch_size, augmentation=False, upscale=self.upscale)
        self.test_data = LoadandAugment(dataset_path=self.test_path, data_type="test", batch=self.batch_size, augmentation=False, upscale=self.upscale)

        self.model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
        checkpoint = ModelCheckpoint(str(self.model_path / f"{self.experiment_name}.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        self.model.fit(self.train_data.dataset, 
                       validation_data=self.val_data.dataset, 
                       epochs=120, 
                       callbacks=[checkpoint])
        
    def evaluate_network(self):
        _, test_accuracy = self.model.evaluate(self.test_data.dataset)
        return test_accuracy
    
    def record_experiment(self):
        base_path = Path(self.dataset_path).parent
        eval_file = base_path / f'{self.experiment}_evaluation.csv'
        
        new_entry = {
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'augmentation': self.augmentation,
            'upscaled': self.upscale,
            'test_accuracy': self.test_accuracy
        }
        
        if not eval_file.exists():
            df = pd.DataFrame(columns=['experiment_name', 'model_name', 'augmentation', 'upscaled', 'test_accuracy'])
        else:
            df = pd.read_csv(eval_file)
        
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(eval_file, index=False)
        
if __name__ == "__main__":
    data_set_path = "/workspaces/Satelite/data/dataset_upscaled_False"
    augmentation = False
    modelname = 'attunet'
    experiment_name = 'attunet_experiment'
    upscale = False
    train = TrainNetwork(data_set_path, augmentation, modelname, experiment_name, upscale)
