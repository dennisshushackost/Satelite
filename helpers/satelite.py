from pathlib import Path
from typing import List
import geopandas as gpd
import numpy as np 
import rasterio
import warnings 
import os
from rasterio.enums import Resampling
from datetime import datetime
from eodal.config import get_settings
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs

# Ignore warnings
warnings.filterwarnings('ignore')

def preprocess_sentinel2_scenes(ds: Sentinel2, target_resolution:int) -> Sentinel2:
    """
    Resample Sentinel-2 scenes and mask clouds, shadows, and snow
        based on the Scene Classification Layer (SCL)

    Attributes:
        target_resolution: spatial target resolution to resample all bands to.
        return: resampled, cloud-masked Sentinel-2 scene
    """
    # Resample and Mask
    ds.resample(inplace=True, target_resolution=target_resolution,)
    ds.mask_clouds_and_shadows(inplace=True)
    return ds

class ProcessSatelite():
    """
    Processs a grid cell corresponding to the bounding box of a parcel
    to create the corresponding satelite images. 

    Attributes:
        data_path (str): Path to the cantonal data.
        time_start (str): Start date for the satelite images.
        time_end  (str): End date for the satelit (int): Spatial target ree images.
        target_resolutionsolution to resample all bands to.
        target_size (touple): Target size of the satelite images (default 256, 256)
        parcel_index (int): Index of the the parcel
    """
    def __init__(self, data_path, time_start, time_end, target_resolution, parcel_index, target_size=256):
        self.data_path = Path(data_path)
        self.time_start = time_start
        self.time_end = time_end
        self.target_resolution = target_resolution
        self.target_size = target_size
        self.parcel_index = parcel_index
        self.canton_name = self.data_path.stem
        self.grid_index = None
        self.mapper = None
        self.scene = None
        self.output_path_satellite = self.create_folders()
        self.grid = gpd.read_file(self.output_path_grid / 'grid.gpkg')
        self.parcel_name = f'{self.canton_name}_parcel_{self.parcel_index}'
        self.parcel = gpd.read_file(self.output_path_gdf / f'{self.parcel_name}.gpkg')
        self.grid_index = self.parcel['grid_index'][0]
        self.crs = self.grid.crs

        # Use cloud data not local storage
        Settings = get_settings()
        Settings.USE_STAC = True
    
    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.base_path = self.data_path.parent
        self.output_path_gdf = self.base_path / "parcels"
        self.output_path_sat = self.base_path / "satellite"
        self.output_path_grid = self.base_path / "grid"
        self.output_path_sat.mkdir(exist_ok=True)
        return self.output_path_sat

    def create_satelite_mapper(self):
        """
        Create a mapper to extract the corresponding satelite images for a grid cell.
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
    
    def crop_or_pad_image(self, file_path):
        """
        This function either crops or pads the image to the target size.
        """
        with rasterio.open(file_path) as src:
            data = src.read()  
            height, width = data.shape[1], data.shape[2]

            if height > self.target_size or width > self.target_size:
                print("Cropping the image")
                # Center crop the image
                start_y = max(0, (height - self.target_size) // 2)
                start_x = max(0, (width - self.target_size) // 2)
                end_y = start_y + self.target_size
                end_x = start_x + self.target_size
                data_cropped = data[:, start_y:end_y, start_x:end_x]
            elif height < self.target_size or width < self.target_size:
                print("Padding the image")
                # Pad the image
                pad_height = (self.target_size - height) // 2
                pad_width = (self.target_size - width) // 2
                data_cropped = np.pad(data, pad_width=((0, 0), (pad_height, pad_height + (self.target_size - height - 2*pad_height)), (pad_width, pad_width + (self.target_size - width - 2*pad_width))), mode='constant', constant_values=0)
            else:
                print("Image is already of the target size")
                return

            # Update metadata for output file
            new_meta = src.meta.copy()
            new_meta.update({
                'driver': 'GTiff',
                'height': self.target_size,
                'width': self.target_size,
                'transform': rasterio.windows.transform(rasterio.windows.Window(start_x, start_y, self.target_size, self.target_size), src.transform)
            })

            # Write the cropped or padded image to a new file
            with rasterio.open(file_path, 'w', **new_meta) as dst:
                dst.write(data_cropped)

            
    def get_no_data_percentage(self, file_path):
        """
        Returns the no data percentage of a raster image.
        """
        # Open the raster image
        with rasterio.open(file_path) as src:
            # Read all bands
            data = src.read(masked=True)
            # Calculate no data percentage
            no_data_count = data.mask.sum()
            total_pixels = data.size
            no_data_percentage = (no_data_count / total_pixels) * 100

            # Delete non-padded and padded images if the no data percentage is above 10%
            if no_data_percentage > 10:
                print(f"Deleting {file_path} due to no data percentage of {no_data_percentage}%")
                os.remove(file_path)

    def resample_image(self, path_file):
        """
        Resample the image to the new resolution of 5m using bicubic interpolation.
        """
        with rasterio.open(path_file) as src:
            scale = 2
            new_transform = src.transform * src.transform.scale(
                (src.width / (src.width * scale)),
                (src.height / (src.height * scale))
            )

            # Define new dimensions
            new_height, new_width = int(src.height * scale), int(src.width * scale)

            # Set up the parameters for resampling
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.cubic
            )

            # Prepare the new metadata
            new_meta = src.meta.copy()
            new_meta.update({
                'driver': 'GTiff',
                'height': new_height,
                'width': new_width,
                'transform': new_transform
            })

            # Write the upscaled image
            with rasterio.open(path_file, 'w', **new_meta) as dst:
                dst.write(data)

    
    def normalize_bands(self, src):
        """ 
        This function normalises the bands of the satellite image.
        It clips the pixel values between the 1st and 99th percentile and scales them between 0 and 1.
        """
        bands_normalized = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            p1, p99 = np.percentile(band, [1, 99])
            band_normalized = np.clip((band - p1) / (p99 - p1), 0, 1)
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
            original_path = self.output_path_satellite / f'{self.parcel_name}.tif'

            # Save the scene: 
            for timestamp, scene in self.scene:
                # Set to uint16
                scene.to_rasterio(
                    original_path,
                    band_selection=['red', 'green', 'blue', 'nir_1'],
                    as_cog=True)


            # Crop or pad the image to the target size
            self.crop_or_pad_image(original_path)

            # Resample the image to the new resolution of 5m using bicubic interpolation
            self.resample_image(original_path)

            # Check the no data percentage
            #self.get_no_data_percentage(padded_path)

            self.process_and_save_normalized_image(original_path)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return

if __name__ == '__main__':
    data_path = '/workspaces/Satelite/data/aargau.gpkg'
    time_start: datetime = datetime(2023,6,1) 
    time_end: datetime = datetime(2023,7,31)   
    target_resolution = 10
    parcel_index = 53
    process = ProcessSatelite(data_path, time_start, time_end, target_resolution, parcel_index)
    process.create_satelite_mapper()
    process.select_min_coverage_scene()
  