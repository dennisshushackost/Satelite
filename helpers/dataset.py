import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from pathlib import Path
import numpy as np
import rasterio
import random
import json

class CreateTensorflowDataset:
    """
    This class prepares the tensorflow dataset for training:
    - Loads and processes the images and masks in order to be suiting for 
    a tensorflow dataset. The sizes of the different images are given by 
    (512, 512, 4) for the images if they are upscaled and (256, 256, 4) if they are not.
    The masks are always (512, 512, 1).
    """

    def __init__(self, data_path, cantons, upscaled, train=0.8, test=0.1, val=0.1):
        self.base_path = Path(data_path)
        self.list_of_cantons = cantons
        self.upscaled = upscaled
        self.satellite_dir = self.base_path / 'satellite'
        self.mask_dir = self.base_path / 'mask'
        self.train = train
        self.test = test
        self.val = val
        if upscaled:
            self.image_shape = [512, 512, 4]
            self.mask_shape = [512, 512, 1]
        else:
            self.image_shape = [256, 256, 4]
            self.mask_shape = [512, 512, 1]
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
        tensor.set_shape(self.image_shape)
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
                                      -1)  # Adds a new dimension in the end of   thearray (height, width, channels=1)
                return mask.astype(np.uint8)

        tensor = tf.numpy_function(_load_mask, [mask_path], tf.uint8)
        tensor.set_shape(self.mask_shape)
        return tensor
    
    def process_mask_special(self, mask_path):
        """
        Tensorflow function to process the masks for tensorflow datasets
        :param mask_path: A string of the mask path
        :return: tensorflow compatible uint8 image
        """

        def _load_mask(path):
            with rasterio.open(path.decode("utf-8")) as src:
                mask = src.read()  # Read all channels
                mask = np.transpose(mask, (1, 2, 0))  # Reshape to (height, width, channels)
                return mask.astype(np.uint8)

        tensor = tf.numpy_function(_load_mask, [mask_path], tf.uint8)
        tensor.set_shape(self.mask_shape)
        return tensor
       
    def save_file_mapping(self, indices, images, masks, filename):
        mapping = [{'index': index, 'image': image, 'mask': mask} for index, image, mask in zip(indices, images, masks)]
        with open(self.base_path / filename, 'w') as f:
            json.dump(mapping, f)
            
    def prepare_dataset(self):
        print("Preparing the dataset...")
        images = []
        masks = []
        for canton in self.list_of_cantons:
            if self.upscaled:
                image_paths = sorted(self.satellite_dir.glob(f'{canton}_*_upscaled_parcel_*.tif'))
                images += image_paths
                mask_paths = sorted(self.mask_dir.glob(f'{canton}_*_upscaled_parcel_*.tif'))
                masks += mask_paths
            else:
                image_paths = sorted(self.satellite_dir.glob(f'{canton}_*_parcel_*.tif'))
                image_paths = [str(path) for path in image_paths if 'upscaled' not in str(path)] 
                images += image_paths
                mask_paths = sorted(self.mask_dir.glob(f'{canton}_*_parcel_*.tif'))
                masks += mask_paths
        print(f"Found {len(images)} images and {len(masks)} masks.")
        
        
        # Shuffle the dataset:
        images = [str(path) for path in images]
        masks = [str(path) for path in masks]
        combined = list(zip(images, masks))
        random.shuffle(combined)
        images, masks = zip(*combined)

        dataset_size = len(images)
        train_size = int(self.train * dataset_size)
        val_size = int(self.val * dataset_size)

        # Split the dataset into training, validation, and testing
        train_images, train_masks = images[:train_size], masks[:train_size]
        val_images, val_masks =  images[train_size:train_size + val_size], masks[train_size:train_size + val_size]
        test_images, test_masks = images[train_size + val_size:], masks[train_size + val_size:]
        
        # Save the three datasets as train, test and val
        save_path = self.base_path / f'dataset_upscaled_{self.upscaled}'
        if not save_path.exists():
            save_path.mkdir()

        # Create list of the 
        self.save_file_mapping(list(range(train_size)), train_images, train_masks, save_path / 'train_file_mapping.json')
        self.save_file_mapping(list(range(val_size)), val_images, val_masks, save_path / 'val_file_mapping.json')
        self.save_file_mapping(list(range(dataset_size-train_size-val_size)), test_images, test_masks, save_path / 'test_file_mapping.json')
        
        train_dataset = tf.data.Dataset.from_tensor_slices((list(train_images), list(train_masks)))
        val_dataset = tf.data.Dataset.from_tensor_slices((list(val_images), list(val_masks)))
        test_dataset = tf.data.Dataset.from_tensor_slices((list(test_images), list(test_masks)))
        
        # Process the images and masks
        train_dataset = train_dataset.map(lambda image, mask: (self.process_image(image), self.process_mask(mask)),
                                          num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(lambda image, mask: (self.process_image(image), self.process_mask(mask)),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.map(lambda image, mask: (self.process_image(image), self.process_mask(mask)),
                                        num_parallel_calls=tf.data.AUTOTUNE)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        tf.data.Dataset.save(
            train_dataset, str(save_path / 'train'), compression=None, shard_func=None, checkpoint_args=None
        )
        tf.data.Dataset.save(
            val_dataset, str(save_path / 'val'), compression=None, shard_func=None, checkpoint_args=None
        )
        tf.data.Dataset.save(
            test_dataset, str(save_path / 'test'), compression=None, shard_func=None, checkpoint_args=None
        )
        print("Done preparing the dataset.")
        
     
if __name__ == '__main__':
    data_path = '/workspaces/Satelite/data'
    upscaled = False
    list_of_cantons = ['ZH']
    loader = CreateTensorflowDataset(data_path, list_of_cantons, upscaled)


