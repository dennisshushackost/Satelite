import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from pathlib import Path
import numpy as np
import rasterio


class ImageDataLoader:
    """
    This class prepares the tensorflow dataset for training:
    - Loads and processes the images and masks
    - Applies data augmentation: Flipping, Rotating, Adds noise, gaussian blur and zooming in.
    """

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.base_path = self.data_path.parent
        self.satellite_dir = self.base_path / 'satellite'
        self.mask_dir = self.base_path / 'mask'

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
        tensor.set_shape([1024, 1024, 4])
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
        tensor.set_shape([1024, 1024, 1])
        return tensor

    def add_speckle_noise(self, image):
        """Apply speckle noise to an already normalized image."""
        speckle_variance = np.random.uniform(0.01, 0.1)
        noise = tf.random.normal(shape=tf.shape(image), mean=1.0, stddev=tf.sqrt(speckle_variance))
        noisy_image = image * noise
        return noisy_image

    def add_gaussian_noise(self, image):
        """Apply gaussian noise to an already normalized image."""
        noise_variance = np.random.uniform(0.01, 0.1)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=tf.sqrt(noise_variance))
        noisy_image = image + noise
        return noisy_image

    def add_salt_and_pepper_noise(self, image):
        """
        Adds salt and pepper noise to an already normalized image.
        """
        noise_level = np.random.uniform(0.01, 0.1)
        shape = tf.shape(image)
        # Creates a tensors with the same size as the image with random values between 0 and 1:
        random_tensor = tf.random.uniform(shape, minval=0, maxval=1.0)

        # Creates a salt mask (white pixels) i.e. at 6% noise_level => 3% of the pixels will be white
        # Creates a mask with 0 and 1 values where 1 is the pixels that will be white
        salt_mask = tf.cast(random_tensor <= (noise_level / 2), image.dtype)
        # Creates a pepper mask (black pixels) i.e. at 6% noise_level => 3% of the pixels will be black
        # Creates a mask with 0 and 1 values where 1 is the pixels that will be black
        pepper_mask = tf.cast(random_tensor >= (1 - noise_level / 2), image.dtype)
        # Apply the masks to the image
        image_with_salt = tf.where(salt_mask == 1, tf.ones_like(image), image)
        noisy_image = tf.where(pepper_mask == 1, tf.zeros_like(image), image_with_salt)

        return noisy_image

    def add_gaussian_blur(self, image):
        """
        Apply gaussian blur to an already normalized image.
        """

    def add_rotation(self, image, mask):
        """
        Rotate image by +- 90 degrees
        """
        rotation = [+1, -1]
        random_rotation = np.random.choice(rotation)
        image = tf.image.rot90(image, k=random_rotation)
        mask = tf.image.rot90(mask, k=random_rotation)
        return image, mask

    def add_horizontal_flip(self, image, mask):
        """
        Flip image horizontally
        """
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        return image, mask

    def create_dataset(self):
        image_paths = sorted(self.satellite_dir.glob('*.tif'),
                             key=lambda x: int(x.stem.split('_')[-1]))
        mask_paths = sorted(self.mask_dir.glob('*.tif'),
                            key=lambda x: int(x.stem.replace('_mask', '').split('_')[-1]))

        # Transform into string tensors
        image_paths = tf.constant([str(path) for path in image_paths])
        mask_paths = tf.constant([str(path) for path in mask_paths])
        assert len(image_paths) == len(mask_paths)

        # Print available gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))


if __name__ == '__main__':
    data_path = '/project/Satelite/data/AG.gpkg'
    loader = ImageDataLoader(data_path)
    loader.create_dataset()
