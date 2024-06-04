import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import List
import geopandas as gpd
import numpy as np
import rasterio
from eodal.config import get_settings
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs
from rasterio.windows import Window
from rasterio.enums import Resampling
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


# Ignore warnings
warnings.filterwarnings('ignore')


def preprocess_sentinel2_scenes(ds: Sentinel2, target_resolution: int) -> Sentinel2:
    """
    Resample Sentinel-2 scenes and mask clouds, shadows, and snow
        based on the Scene Classification Layer (SCL)

    Attributes:
        target_resolution: spatial target resolution to resample all bands to.
        return: resampled, cloud-masked Sentinel-2 scene
    """
    # Resample and Mask
    ds.resample(inplace=True, target_resolution=target_resolution)
    ds.mask_clouds_and_shadows(inplace=True)
    return ds


class ProcessSatellite:
    """
    Process a grid cell corresponding to the bounding box of a parcel
    to create the corresponding satellite images.

    Attributes:
        data_path (str): Path to the cantonal data.
        time_start (str): Start date for the satellite images.
        time_end  (str): End date for the satellite images
        target_resolution (int): Spatial resolution to resample all bands to.
        target_size (int): Target size of the satellite images (default 256, 256), will be padded or
            cropped to this size.
    """

    def __init__(self, data_path, time_start, time_end, target_resolution, grid_index, target_size=256):
        self.data_path = Path(data_path)
        print(self.data_path)
        self.time_start = time_start
        self.time_end = time_end
        self.target_resolution = target_resolution
        self.target_size = target_size
        self.canton_name = self.data_path.stem
        self.grid_index = grid_index
        self.mapper = None
        self.scene = None
        self.output_path_satellite = self.create_folders()
        self.grid = gpd.read_file(self.output_path_grid / f'{self.canton_name}_essential_grid.gpkg')
        self.satellite_name = f'{self.canton_name}_parcel_{self.grid_index}'
        self.crs = self.grid.crs
        print(self.crs)

        # Use cloud data not local storage
        Settings = get_settings()
        Settings.USE_STAC = True

    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.base_path = self.data_path.parent
        self.output_path_grid = self.base_path / "grid"
        self.output_path_sat = self.base_path / "satellite"
        self.output_path_sat.mkdir(exist_ok=True)
        return self.output_path_sat

    def create_satellite_mapper(self):
        """
        Create a mapper to extract the corresponding satellite images for a grid cell.
        """
        try:
            collection: str = 'sentinel2-msi'
            time_start: datetime = self.time_start
            time_end: datetime = self.time_end

            # Creating the bounding box for the grid cell:
            grid_geometry = self.grid.geometry[self.grid_index]
            geoseries = gpd.GeoSeries([grid_geometry], crs=self.crs)
            feature = Feature.from_geoseries(geoseries)

            # Metadata filters and mapper configurations:
            metadata_filters: List[Filter] = [
                Filter('cloudy_pixel_percentage', '<', 25),
                Filter('processing_level', '==', 'Level-2A')
            ]

            mapper_configs = MapperConfigs(
                collection=collection,
                time_start=time_start,
                time_end=time_end,
                feature=feature,
                metadata_filters=metadata_filters
            )

            self.mapper = Mapper(mapper_configs)
        except Exception as e:
            print(f"An error occurred: {e}")
            return

    def crop_or_pad_image(self, file_path, upscale=False):
        """
        Crops or pads the image to the target size while maintaining the image center.
        """
        if upscale:
            self.target_size = self.target_size * 2
            
        with rasterio.open(file_path) as src:
            data = src.read()
            channels, height, width = data.shape

            # Calculate padding or cropping amounts for height and width
            pad_height = max(self.target_size - height, 0)
            pad_width = max(self.target_size - width, 0)
            crop_height = max(height - self.target_size, 0)
            crop_width = max(width - self.target_size, 0)

            # Pad to target size
            if pad_height > 0 or pad_width > 0:
                data_padded = np.pad(data, 
                                    ((0, 0), 
                                    (pad_height // 2, pad_height - pad_height // 2), 
                                    (pad_width // 2, pad_width - pad_width // 2)), 
                                    mode='constant', constant_values=0)
                data = data_padded

            # If image is larger than target size, crop the center
            if crop_height > 0 or crop_width > 0:
                start_height = crop_height // 2
                start_width = crop_width // 2
                data = data[:, start_height:start_height + self.target_size, start_width:start_width + self.target_size]

            # Ensure the output is the target size
            data = data[:, :self.target_size, :self.target_size]

            # Calculate new transform to maintain the image location
            original_center_x = src.bounds.left + (src.bounds.right - src.bounds.left) / 2
            original_center_y = src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2

            new_bounds_left = original_center_x - (self.target_size / 2) * src.res[0]
            new_bounds_right = original_center_x + (self.target_size / 2) * src.res[0]
            new_bounds_bottom = original_center_y - (self.target_size / 2) * src.res[1]
            new_bounds_top = original_center_y + (self.target_size / 2) * src.res[1]

            new_transform = rasterio.transform.from_bounds(new_bounds_left, new_bounds_bottom, new_bounds_right, new_bounds_top, self.target_size, self.target_size)

            # Save the cropped or padded image
            new_meta = src.meta.copy()
            new_meta.update({"height": self.target_size, "width": self.target_size, "transform": new_transform})

            with rasterio.open(file_path, 'w', **new_meta) as dst:
                dst.write(data)


    def get_no_data_percentage(self, file_path):
        """
        Removes all satellite images, which have a no data percentage above 10%.
        This helps remove satellite images with clouds or other artifacts.
        """
        # Open the raster image
        with rasterio.open(file_path) as src:
            # Read all bands
            data = src.read(masked=True)
            # Calculate no data percentage
            no_data_count = data.mask.sum()
            total_pixels = data.size
            no_data_percentage = (no_data_count / total_pixels) * 100
            data = src.read(1)
            nan_count = np.sum(np.isnan(data))
   
            # Delete non-padded and padded images if the no data percentage is above 10%
            if no_data_percentage > 10 or nan_count > 0:
                print(f"Removing satellite image {file_path} with no data percentage {no_data_percentage:.2f}%")
                os.remove(str(file_path).replace(self.canton_name, f"{self.canton_name}_upscaled"))  
                os.remove(file_path)
                return True

    def normalize_bands(self, src):
        """ 
        This function normalizes the bands of the satellite image.
        It clips the pixel values between the 1st and 99th percentile and scales them between 0 and 1.
        Applies min-max normalization to scale pixel values between 0 and 1.
        """
        bands_normalized = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            # Calculates the first and 99th percentile in the image:
            p1, p99 = np.percentile(band, [1, 99])
            # Values are clipped to the 1st and 99th percentile:
            band_clipped = np.clip(band, p1, p99)
            # Apply min-max normalization:
            band_normalized = (band_clipped - p1) / (p99 - p1)
            bands_normalized.append(band_normalized)
        return bands_normalized

    def process_and_save_normalized_image(self, input_path):
        """ 
        This function processes the normalized bands and saves the image to a new file.
        """
        with rasterio.open(input_path, 'r') as src:
            # Read and normalize all bands first
            bands_normalized = self.normalize_bands(src)

            # Prepare new metadata for the output file
            new_meta = src.meta.copy()
            new_meta.update(dtype=rasterio.float32)

            # Write the normalized data to a new output file
            with rasterio.open(input_path, 'w', **new_meta) as dst:
                for i, band in enumerate(bands_normalized, start=1):
                    dst.write(band, i)
                
    def resample_image(self, path_file):
        """
        Resamples the image to the new resolution of 2.5m using bicubic interpolation.
        In other words, this means decreasing the pixel size to 1/4 of the original size.
        """
        upscale_path = str(path_file).replace(self.canton_name, f"{self.canton_name}_upscaled")
        with rasterio.open(path_file) as src:
            # Define new dimensions based on scale factor
            scale = 2
            new_height, new_width = int(src.height * scale), int(src.width * scale)

            # Resample data to target shape:
            data = src.read(
                out_shape=(src.count,
                           new_height,
                           new_width),
                resampling=Resampling.cubic
            )

            # Scale image transform:
            new_transform = src.transform * src.transform.scale(
                (src.width / (src.width * scale)),
                (src.height / (src.height * scale))
            )

            # Prepare the new metadata
            new_meta = src.meta.copy()
            new_meta.update({
                'driver': 'GTiff',
                'height': new_height,
                'width': new_width,
                'transform': new_transform
            })

            # Write the upscaled image to disk:
            with rasterio.open(upscale_path, 'w', **new_meta) as dst:
                dst.write(data)
            return upscale_path


    def select_min_coverage_scene(self):
        """
        Load and sort the satellite images based on the grid cell.

        Attributes:
            mapper (Mapper): Mapper to extract the corresponding satellite images for a grid cell.
        """
        try:
            self.mapper.query_scenes()
            self.mapper.metadata

            # Choose the scene with the least cloud coverage:
            self.mapper.metadata = self.mapper.metadata[
                self.mapper.metadata.cloudy_pixel_percentage ==
                self.mapper.metadata.cloudy_pixel_percentage.min()].copy()

            # Load the scene: Mask out clouds, bands: B02 (blue), B03 (green), B04 (red), B08 (nir)
            scene_kwargs = {
                'scene_constructor': Sentinel2.from_safe,
                'scene_constructor_kwargs': {'band_selection':
                                                    ['B04', 'B03', 'B02', 'B08'],
                                                'apply_scaling': False,
                                                'read_scl': True},
                'scene_modifier': preprocess_sentinel2_scenes,
                'scene_modifier_kwargs': {'target_resolution': self.target_resolution}
            }
            self.mapper.load_scenes(scene_kwargs=scene_kwargs)
            self.scene = self.mapper.data

            # Output path for the scene: 
            original_path = self.output_path_satellite / f'{self.satellite_name}.tif'

            # Save the scene: 
            for timestamp, scene in self.scene:
                # Set to uint16
                scene.to_rasterio(
                    original_path,
                    band_selection=['red', 'green', 'blue', 'nir_1'],
                    as_cog=True)
            
            # Remove all satelite images, which have a no data percentage above 10%
            noData = self.get_no_data_percentage(original_path)
            
            if noData:
                return
            else:   
                # Upscale the image to 2.5m resolution
                upscale_path = self.resample_image(original_path)
        
                # Crop or pad the image to the target size
                self.crop_or_pad_image(original_path, upscale=False)
                self.crop_or_pad_image(upscale_path, upscale=True)
                
                # Normalize the bands of the satellite image
                self.process_and_save_normalized_image(original_path)
                self.process_and_save_normalized_image(upscale_path)
                
        except Exception as e: 
            print(f"An error occurred: {e}")
            return

if __name__ == "__main__":
    # Define the path to the cantonal data
    data_path = Path("/workspaces/Satelite/data/CH.gpkg")
    # Define the start and end date for the satellite images
    time_start = datetime(2021, 1, 1)
    time_end = datetime(2021, 12, 31)
    # Define the spatial resolution to resample all bands to
    target_resolution = 10
    # Define the grid index
    grid_index = 285
    # Create an instance of the ProcessSatellite class
    process_satellite = ProcessSatellite(data_path, time_start, time_end, target_resolution, grid_index)
    process_satellite.create_satellite_mapper()
    process_satellite.select_min_coverage_scene()