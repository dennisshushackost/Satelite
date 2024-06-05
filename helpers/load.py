"""
This script loads the tensorflow dataset and does on the fly augmentation 
during the training process. 
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import numpy as np

class LoadandAugment:
    """
    This class loads and augmentes the dataset for training if wanted.
    The following augmentations are available:
    1. Adding random brightness 
    2. Adding horizontal flip 
    3. Adding rotation 
    4. Adding gaussian blur 
    5. Adding speckle noise
    6. Adding gaussian noise
    7. Adding salt and pepper noise
    """
    
    def __init__(self, dataset_path, data_type, batch, augmentation):
        self.dataset_path = dataset_path
        self.data_type = data_type
        self.batch = batch
        self.augmentation = augmentation
        self.load_and_augment()
        
    # Augmentation functions:
    def add_random_brightness(self, image):
        """
        Randomly change the brightness of the image
        """
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image
    
    def add_horizontal_flip(self, image, mask):
        """
        Flip image horizontally
        """
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        return image, mask
    
    def add_rotation(self, image, mask):
        """
        Rotate image by +- 90 degrees
        """
        rotation = [+1, -1]
        random_rotation = np.random.choice(rotation)
        image = tf.image.rot90(image, k=random_rotation)
        mask = tf.image.rot90(mask, k=random_rotation)
        return image, mask

    def add_gaussian_blur(self, image):
        """
        Apply gaussian blur to an already normalized image using scipy. 
        The function expects a tensorflow tensor and returns a blurred version of the image. 
        Each band is blurred independently to maintain spectral integrity.
        """
        def _apply_blur(image):
            # Define the standard deviation for the Gaussian Kernel: (Higher values will increase the blur)
            sigma = np.random.uniform(0.5, 1.5)
            blurred_image = np.zeros_like(image)
            for i in range(4):
                blurred_image[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma)
            return blurred_image.astype(np.float32)
        
        blurred_image = tf.numpy_function(_apply_blur, [image], tf.float32)
        blurred_image.set_shape([1024, 1024, 4])
        return blurred_image
    
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
    
    def augment(self, image, mask):
        # Adds a horizontal flip to the image and mask with a 50% probability
        if np.random.rand() > 0.5:
            image, mask = self.add_horizontal_flip(image, mask)
    
        # Adds a rotation to the image and mask with a 20% probability
        if np.random.rand() < 0.2:
            image, mask = self.add_rotation(image, mask)
    
        # Adds one of the following noises/blurs with a 50 % probability (Gaussian, speckle, salt and pepper, gaussian noise)
        if np.random.rand() > 0.5:
            noise_functions = [self.add_gaussian_noise, self.add_speckle_noise, self.add_salt_and_pepper_noise, self.add_gaussian_blur, self.add_random_brightness]
            noise_function = np.random.choice(noise_functions)
            image = noise_function(image)
        
        return image, mask
         
    def load_and_augment(self):
        """
        This function loads the data and applies the augementation if wanted.
        """
        self.dataset = tf.data.Dataset.load(self.dataset_path)
        self.dataset.cache()
        # Map the training dataset with augmentation
        if self.data_type == 'train' and self.augmentation:
            self.dataset = self.dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch).prefetch(tf.data.AUTOTUNE)
        else:
            self.dataset = self.dataset.batch(self.batch).prefetch(tf.data.AUTOTUNE)
                    
        return self.dataset
        