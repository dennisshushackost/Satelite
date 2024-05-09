import tensorflow as tf
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# Create an example tensor simulating an image
image_tensor = tf.random.normal([1024, 1024, 4])


# Function to apply Gaussian blur using scipy
def add_gaussian_blur(image_np, sigma=3.0):
    """
    Apply Gaussian blur to an already normalized image using scipy.ndimage.gaussian_filter.
    Parameters:
        image_np (numpy.ndarray): Numpy array of shape (height, width, channels)
        sigma (float): The standard deviation of the Gaussian kernel
    Returns:
        numpy.ndarray: Blurred image
    """
    # Apply the Gaussian filter to each channel
    blurred_image_np = np.zeros_like(image_np)
    for i in range(image_np.shape[2]):
        blurred_image_np[..., i] = scipy.ndimage.gaussian_filter(image_np[..., i], sigma=sigma)
    return blurred_image_np


# Convert the TensorFlow tensor to a NumPy array
image_np = image_tensor.numpy()

# Apply Gaussian blur
blurred_image_np = add_gaussian_blur(image_np, sigma=3.0)

# Convert the blurred image back to a TensorFlow tensor if needed
blurred_image_tensor = tf.convert_to_tensor(blurred_image_np)

# Visualize the result using matplotlib
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_np[..., :3], cmap='gray')  # Display only the first three channels
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_image_np[..., :3], cmap='gray')  # Display only the first three channels
plt.title('Blurred Image')
plt.axis('off')
plt.show()
