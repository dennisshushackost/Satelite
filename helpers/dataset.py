import tensorflow as tf
import numpy as np
import rasterio
from pathlib import Path


class ImageDataLoader:
    """
    This class prepares the tensorflow dataset for training:
    - Loads and processes the images and masks
    - Applies data augmentation: Flipping, Rotating and adding speckle noise
    """
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.base_path = self.data_path.parent


