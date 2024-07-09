import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from numba import cuda
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from load import LoadandAugment
import model
import modelup

# Reset the GPU memory:
device = cuda.get_current_device()
device.reset()


class TrainNetwork:
    """ 
    This class trains and evaluates the network.
    """
    def __init__(self, dataset_path, augmentation, modelname, experiment_name, upscale=False):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")
        self.val_path = os.path.join(self.dataset_path, "val")
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self.augmentation = augmentation
        self.num_classes = 1
        self.dropout_rate = 0.0
        self.batch_norm = True
        self.model_path = self.create_folders()
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
                self.model = modelup.unet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm) 
            else:
                self.model = model.unet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        elif modelname == 'attunet':
            if self.upscale:
                self.model = modelup.attunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
            else:
                self.model = model.attunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        elif modelname == 'resunet':
            if self.upscale:
                self.model = modelup.resunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
            else:
                self.model = model.resunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        else:
            raise ValueError(f"Model {modelname} not implemented")
        
        self.train_network()

    def create_folders(self):
        base_path = Path(self.dataset_path).parent
        self.model_path = base_path / 'experiment'
        if not self.model_path.exists():
            self.model_path.mkdir()
        return self.model_path

    def train_network(self):
        self.train_data = LoadandAugment(dataset_path=self.train_path, data_type="train", batch=self.batch_size, augmentation=self.augmentation, upscale=self.upscale)
        self.val_data = LoadandAugment(dataset_path=self.val_path, data_type="val", batch=self.batch_size, augmentation=False, upscale=self.upscale)
        self.test_data = LoadandAugment(dataset_path=self.test_path, data_type="test", batch=self.batch_size, augmentation=False, upscale=self.upscale)

        self.model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
        checkpoint = ModelCheckpoint(str(self.model_path / f"{self.experiment_name}.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        self.model.fit(self.train_data.dataset, 
                       validation_data=self.val_data.dataset, 
                       epochs=100, 
                       callbacks=[checkpoint])
        
    def evaluate(self):
        self.model.evaluate(self.test_data.dataset)
        

if __name__ == "__main__":
    data_set_path = "/workspaces/Satelite/data/dataset_upscaled_False"
    augmentation = False
    modelname = 'attunet'
    experiment_name = 'resunet_experiment_up'
    upscale = False
    train = TrainNetwork(data_set_path, augmentation, modelname, experiment_name, upscale)