import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from numba import cuda 
from  pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from helpers.load import LoadandAugment
from helpers import model


class TrainNetwork:
    """
    This class trains the network using the loaded dataset
    """
    def __init__(self, dataset_path, augmentation, modelname, upscale=False):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")
        self.val_path = os.path.join(self.dataset_path, "val")
        self.augmentation = augmentation
        self.num_classes = 1
        self.dropout_rate = 0.0
        self.batch_norm = True
        self.model_path = self.create_folders()
        if upscale:
            self.input_shape = (512, 512, 3)
        else:
            self.input_shape = (256, 256, 3)
   
        if modelname == 'unet':
            self.model = model.deepunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        elif modelname == 'attunet':
            self.model = model.attunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        elif modelname == 'resunet':
            self.model = model.resunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)  
        elif modelname == 'resattunet':
            self.model = model.resattunet(self.input_shape, self.num_classes, self.dropout_rate, self.batch_norm)
        else:
            raise ValueError(f"Model {modelname} not implemented")  
        

    def create_folders(self):
        """
        This function creates the folders for the model and the logs
        """
        base_path = Path(self.dataset_path).parent
        self.model_path = base_path / 'models'
        # Create the model folder
        if not self.model_path.exists():
            self.model_path.mkdir()
        return self.model_path
            
    def train_network(self):
        """
        This function loads the data and trains the network
        """
        if self.augmentation:
            train_data = LoadandAugment(self.train_path, self.input_shape)
        else:
            train_data = LoadandAugment(self.train_path, self.input_shape, augment=False)
        