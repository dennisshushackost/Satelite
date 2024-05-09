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
        def _load_image(path):
            with rasterio.open(path.decode("utf-8")) as src:
                image = src.read().transpose((1, 2, 0))
                return image.astype(np.float32)

    def load_and_preprocess_image(self, image_path):
        """
        Loads the satellite image into the tensorflow dataset and
        transforms it into the correct format (height, width, channels)
        """

        def _load_image(path):
            with rasterio.open(path.decode('utf-8')) as src:
                image = src.read().transpose((1, 2, 0))
                return image.astype(np.float32)

        tensor = tf.numpy_function(_load_image, [image_path], tf.float32)
        tensor.ensure_shape((1024, 1024, 4))
        return tensor

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
