import os
import rasterio
from PIL import Image

# Define the folder paths
input_folder = "/workspaces/Satelite/data/satellite"
output_folder = os.path.join(input_folder, "images")

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to convert GeoTIFF to PNG
def convert_to_png(input_file, output_file):
    with rasterio.open(input_file) as src:
        # Read the image data
        img_data = src.read()
        
        # Normalize the image data to 0-255 range for PNG output
        norm_img_data = ((img_data - img_data.min()) * (255 / (img_data.max() - img_data.min()))).astype('uint8')
        
        # If the image has more than one band, it's likely RGB or RGBA
        if norm_img_data.shape[0] > 1:
            img = Image.fromarray(norm_img_data.transpose(1, 2, 0))  # Transpose to (height, width, channels)
        else:
            img = Image.fromarray(norm_img_data[0])  # Single band image
        
        img.save(output_file)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif") and "upscaled" not in filename.lower():
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
        
        # Convert the file to PNG
        convert_to_png(input_file, output_file)

print("Conversion complete!")
