import os
import tensorflow as tf
from pathlib import Path
import numpy as np
import rasterio
import random
import json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')  # Only show errors, not warnings

class CreateTensorflowDataset:
    def __init__(self, data_path, upscaled, train=0.8, test=0.1, val=0.1):
        self.base_path = Path(data_path).parent
        self.upscaled = upscaled
        self.satellite_dir = self.base_path / 'satellite'
        self.mask_dir = self.base_path / 'mask'
        self.train = train
        self.test = test
        self.val = val
        self.image_shape = [512, 512, 4] if upscaled else [256, 256, 4]
        self.mask_shape = [512, 512, 1]
        self.split_info_file = self.base_path / 'dataset_split_info.json'
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
                                      -1)  # Adds a new dimension in the end of the array (height, width, channels=1)
                return mask.astype(np.uint8)

        tensor = tf.numpy_function(_load_mask, [mask_path], tf.uint8)
        tensor.set_shape(self.mask_shape)
        return tensor

    def save_file_mapping(self, indices, images, masks, filename):
        mapping = [{'index': index, 'image': image, 'mask': mask} for index, image, mask in zip(indices, images, masks)]
        with open(filename, 'w') as f:
            json.dump(mapping, f)

    def create_dataset_split(self):
        images = sorted(self.satellite_dir.glob('*_CH_parcel_*.tif'))
        images = [str(path) for path in images if 'upscaled' not in str(path)]
        masks = [str(path).replace('satellite', 'mask') for path in images]

        combined = list(zip(images, masks))
        random.shuffle(combined)
        images, masks = zip(*combined)

        dataset_size = len(images)
        train_size = int(self.train * dataset_size)
        val_size = int(self.val * dataset_size)

        split_info = {
            'train': images[:train_size],
            'val': images[train_size:train_size + val_size],
            'test': images[train_size + val_size:]
        }

        with open(self.split_info_file, 'w') as f:
            json.dump(split_info, f)

        return split_info

    def prepare_dataset(self):
        print("Preparing the dataset...")
        
        if not self.split_info_file.exists():
            split_info = self.create_dataset_split()
        else:
            with open(self.split_info_file, 'r') as f:
                split_info = json.load(f)

        datasets = {}
        for split, image_paths in split_info.items():
            if self.upscaled:
                images = [path.replace('_CH_parcel_', '_CH_upscaled_parcel_') for path in image_paths]
            else:
                images = image_paths
            masks = [path.replace('satellite', 'mask') for path in images]

            # Check if images and masks are not empty
            if not images or not masks:
                print(f"Warning: No images or masks found for {split} split.")
                continue

            # Verify that all files exist
            images = [path for path in images if os.path.exists(path)]
            masks = [path for path in masks if os.path.exists(path)]

            if not images or not masks:
                print(f"Warning: No valid image or mask files found for {split} split.")
                continue

            # Print some debugging information
            print(f"Number of images for {split}: {len(images)}")
            print(f"Number of masks for {split}: {len(masks)}")
            print(f"First image path: {images[0]}")
            print(f"First mask path: {masks[0]}")

            try:
                dataset = tf.data.Dataset.from_tensor_slices((images, masks))
                dataset = dataset.map(lambda image, mask: (self.process_image(image), self.process_mask(mask)),
                                    num_parallel_calls=tf.data.AUTOTUNE)
                datasets[split] = dataset

                save_path = self.base_path / f'dataset_upscaled_{self.upscaled}'
                if not save_path.exists():
                    save_path.mkdir()
                
                self.save_file_mapping(list(range(len(images))), images, masks, save_path / f'{split}_file_mapping.json')
                
                tf.data.Dataset.save(
                    dataset, str(save_path / split), compression=None, shard_func=None, checkpoint_args=None
                )
            except ValueError as e:
                print(f"Error creating dataset for {split} split: {str(e)}")
                continue

        if not datasets:
            raise ValueError("No valid datasets could be created. Please check your input data.")

        self.train_dataset = datasets.get('train')
        self.val_dataset = datasets.get('val')
        self.test_dataset = datasets.get('test')

        # Create combined dataset
        combined_images = []
        combined_masks = []
        for split in ['train', 'val', 'test']:
            if split in split_info:
                combined_images.extend(split_info[split])
                combined_masks.extend([path.replace('satellite', 'mask') for path in split_info[split]])

        if self.upscaled:
            combined_images = [path.replace('_CH_parcel_', '_CH_upscaled_parcel_') for path in combined_images]
            combined_masks = [path.replace('_CH_parcel_', '_CH_upscaled_parcel_') for path in combined_masks]

        if not combined_images or not combined_masks:
            print("Warning: No valid combined dataset could be created.")
            return

        try:
            combined_dataset = tf.data.Dataset.from_tensor_slices((combined_images, combined_masks))
            combined_dataset = combined_dataset.map(lambda image, mask: (self.process_image(image), self.process_mask(mask)),
                                                    num_parallel_calls=tf.data.AUTOTUNE)

            save_path = self.base_path / f'dataset_upscaled_{self.upscaled}'
            self.save_file_mapping(list(range(len(combined_images))), combined_images, combined_masks, save_path / 'combined_file_mapping.json')
            tf.data.Dataset.save(
                combined_dataset, str(save_path / 'combined'), compression=None, shard_func=None, checkpoint_args=None
            )
        except ValueError as e:
            print(f"Error creating combined dataset: {str(e)}")

        print("Done preparing the dataset.")