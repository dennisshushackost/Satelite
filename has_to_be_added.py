import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from helpers.UNET import UNET
from helpers.load import LoadandAugment

# Base path for the dataset
path = "/home/tfuser/project/Satelite/data/dataset/"
train_path = os.path.join(path, "train")
val_path = os.path.join(path, "val")
test_path = os.path.join(path, "test")

# Initialize data loaders
train_data = LoadandAugment(train_path, "train", 8)
val_data = LoadandAugment(val_path, "val", 8)

# Initialize UNET model
unet = UNET(input_shape=(512, 512, 4))
unet.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Setup the model checkpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    'best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'
)

# Train the model
unet.model.fit(
    train_data.dataset, 
    validation_data=val_data.dataset, 
    epochs=40, 
    callbacks=[checkpoint_callback]
)

import numpy as np
import matplotlib.pyplot as plt
def plot_predictions(images, masks, predictions, num=3):
    plt.figure(figsize=(15, 5*num))
    
    for i in range(num):
        plt.subplot(num, 3, i*3+1)
        plt.imshow(images[i])
        plt.title("Satellite Image")
        plt.axis('off')
        
        plt.subplot(num, 3, i*3+2)
        plt.imshow(masks[i], cmap='gray')
        plt.title("Actual Mask")
        plt.axis('off')

        plt.subplot(num, 3, i*3+3)
        plt.imshow(predictions[i], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
# Load the best model
unet.model.load_weights('best_model.h5')

# Evaluate the model
test_data = LoadandAugment(test_path, "test", 4)

# Predict on the test data
for images, masks in test_data.dataset:
    break
predictions = unet.model.predict(images)
predictions = (predictions > 0.5).astype(np.float32) 

# Plot the predictions
plot_predictions(images.numpy(), masks, predictions)

