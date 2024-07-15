import os
import csv
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
from PIL import Image

class SatelliteImageProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder / 'satellite'
        self.csv_file = os.path.join(self.output_folder, 'image_data.csv')

    def process_images(self):
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Prepare CSV file
        with open(self.csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['file_name', 'min_lat', 'min_lon', 'max_lat', 'max_lon', 'canton', 'path_to_image'])

            # Process each .tif file in the input folder
            for filename in os.listdir(self.input_folder):
                if filename.endswith('.tif'):
                    # Full path to the output PNG file
                    full_path = os.path.join(self.output_folder, filename.replace('.tif', '.png'))
                    # Canton name = first two characters of the filename
                    canton = filename[:2]
                    self.process_single_image(filename, csvwriter, canton, full_path)

    def process_single_image(self, filename, csvwriter, canton, full_path):
        tif_path = os.path.join(self.input_folder, filename)
        png_filename = filename.replace('.tif', '.png')
        png_path = os.path.join(self.output_folder, png_filename)

        try:
            with rasterio.open(tif_path) as src:
                # Check if CRS is None and set a default if necessary
                if src.crs is None:
                    st.warning(f"CRS is missing for {filename}. Setting default CRS (EPSG:4326).")
                    src.crs = rasterio.crs.CRS.from_epsg(4326)

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
                try:
                    bounds_4326 = transform_bounds(src_crs, 'EPSG:4326', *bounds)
                except Exception as e:
                    st.warning(f"Error transforming bounds for {filename}: {str(e)}. Using original bounds.")
                    bounds_4326 = bounds

                # Write data to CSV
                csvwriter.writerow([
                    png_filename,
                    bounds_4326[1],  # min_lat
                    bounds_4326[0],  # min_lon
                    bounds_4326[3],  # max_lat
                    bounds_4326[2],  # max_lon
                    canton,
                    full_path
                ])

        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")

    def get_csv_path(self):
        return self.csv_file