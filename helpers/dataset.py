import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from pathlib import Path
import numpy as np
import rasterio


class CreateTensorflowDataset:
    """
    This class prepares the tensorflow dataset for training:
    - Loads and processes the images and masks in order to be suiting for 
    a tensorflow dataset.
    """

    def __init__(self, data_path, train=0.8, test=0.1, val=0.1):
        self.data_path = Path(data_path)
        self.base_path = self.data_path.parent
        self.satellite_dir = self.base_path / 'satellite'
        self.mask_dir = self.base_path / 'mask'
        self.train = train
        self.test = test
        self.val = val
        self.prepare_dataset()

    def process_image(self, image_path):
        """
        Tensorflow function to process the satellite images for tensorflow datasets.
        :param image_path: A string of the image path
        :return: tensorflow compatible float32 image
        """

        def _load_image(path):
            with rasterio.open(path.decode("utf-8")) as src:
                image = src.read().transpose((1, 2, 0))
                return image.astype(np.float32)

        tensor = tf.numpy_function(_load_image, [image_path], tf.float32)
        tensor.set_shape([512, 512, 4])
        return tensor

    def process_mask(self, mask_path):
        """
        Tesnorflow function to process the masks for tensorflow datasets
        :param mask_path: A string of the mask path
        :return: tensorflow compatible uint8 image
        """

        def _load_mask(path):
            with rasterio.open(path.decode("utf-8")) as src:
                mask = src.read(1)
                mask = np.expand_dims(mask,
                                      -1)  # Adds a new dimension in the end of the array (height, width, channels=1)
                return mask.astype(np.uint8)

        tensor = tf.numpy_function(_load_mask, [mask_path], tf.uint8)
        tensor.set_shape([512, 512, 1])
        return tensor
    
    def prepare_dataset(self):
        print("Preparing the tensorflow dataset...")
        image_paths = sorted(self.satellite_dir.glob('*.tif'), key=lambda x: int(x.stem.split('_')[-1]))
        mask_paths = sorted(self.mask_dir.glob('*.tif'), key=lambda x: int(x.stem.replace('_mask', '').split('_')[-1]))
        image_paths = [str(path) for path in image_paths]
        mask_paths = [str(path) for path in mask_paths]
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(lambda x, y: (self.process_image(x), self.process_mask(y)),
                              num_parallel_calls=tf.data.AUTOTUNE)
        
        # Calculate the number of examples in each split
        dataset_size = len(list(dataset))
        train_size = int(self.train * dataset_size)
        val_size = int(self.val * dataset_size)
        print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {dataset_size - train_size - val_size}")
        
        # Shuffle the dataset
        dataset = dataset.shuffle(1000)
        
        # Split the dataset into training, validation, and testing
        train_set = dataset.take(train_size)
        remaining = dataset.skip(train_size)
        val_set = remaining.take(val_size)
        test_set = remaining.skip(val_size)
        
        # Save the three datasets as train, test and val
        save_path = self.base_path / 'dataset'
        if not save_path.exists():
            save_path.mkdir()
        
        tf.data.Dataset.save(
            train_set, str(save_path / 'train'), compression=None, shard_func=None, checkpoint_args=None
        )
        tf.data.Dataset.save(
            val_set, str(save_path / 'val'), compression=None, shard_func=None, checkpoint_args=None
        )
        tf.data.Dataset.save(
            test_set, str(save_path / 'test'), compression=None, shard_func=None, checkpoint_args=None
        )
        print("Done preparing the dataset.")
        

if __name__ == '__main__':
    data_path = '/home/tfuser/project/Satelite/data'
    loader = CreateTensorflowDataset(data_path)
    loader.prepare_dataset()
