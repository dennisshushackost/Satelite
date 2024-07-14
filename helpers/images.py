import os
import csv
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
from PIL import Image

class SatelliteImageProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.csv_file = os.path.join(output_folder, 'image_data.csv')

    def process_images(self):
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Prepare CSV file
        with open(self.csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['filename', 'min_lat', 'min_lon', 'max_lat', 'max_lon'])

            # Process each .tif file in the input folder
            for filename in os.listdir(self.input_folder):
                if filename.endswith('.tif'):
                    self.process_single_image(filename, csvwriter)

    def process_single_image(self, filename, csvwriter):
        tif_path = os.path.join(self.input_folder, filename)
        png_filename = filename.replace('.tif', '.png')
        png_path = os.path.join(self.output_folder, png_filename)

        with rasterio.open(tif_path) as src:
            # Read the image data
            img = src.read()

            # Convert the image to RGB if it's not already
            if img.shape[0] == 1:  # If it's a single-band image
                img = np.tile(img, (3, 1, 1))
            elif img.shape[0] == 4:  # If it's an RGBA image
                img = img[:3, :, :]

            # Normalize and convert to 8-bit
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            # Convert to PIL Image and save as PNG
            pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            pil_img.save(png_path)

            # Get the bounds and transform to EPSG:4326 (WGS84)
            bounds = src.bounds
            src_crs = src.crs
            bounds_4326 = transform_bounds(src_crs, 'EPSG:4326', *bounds)

            # Write data to CSV
            csvwriter.writerow([
                png_filename,
                bounds_4326[1],  # min_lat
                bounds_4326[0],  # min_lon
                bounds_4326[3],  # max_lat
                bounds_4326[2]   # max_lon
            ])

    def get_csv_path(self):
        return self.csv_file

# Example usage:
if __name__ == "__main__":
    input_folder = '/path/to/input/folder'
    output_folder = '/path/to/output/folder'

    processor = SatelliteImageProcessor(input_folder, output_folder)
    processor.process_images()
    print(f"Processing complete. CSV file saved at: {processor.get_csv_path()}")