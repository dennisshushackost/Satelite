import os
import geopandas as gpd
from pathlib import Path
import rasterio
from tqdm import tqdm
from rasterio.features import shapes
from shapely.geometry import shape, box
import numpy as np
import logging

import os
import geopandas as gpd
from pathlib import Path
import rasterio
from tqdm import tqdm
from rasterio.features import shapes
from shapely.geometry import shape, box
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessParcels:
    """
    This class processes parcels based on satellite image data masks and extracts the largest data-rich areas.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.canton_name = data_path.split('/')[-1].split('.')[0]
        self.canton_name_simplified = f"{self.canton_name}_simplified"
        self.data_path = self.data_path.replace(self.canton_name, self.canton_name_simplified)
        self.data_path = Path(self.data_path)
        self.parcel_data_path, self.satellite_images_folder = self.create_folders()
        self.canton = gpd.read_file(self.data_path)
        self.crs = self.canton.crs
        self.process_parcels()
        
    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.base_path = self.data_path.parent
        self.satellite_images_folder = self.base_path / "satellite"
        self.parcel_data_path = self.base_path / "parcels"
        self.parcel_data_path.mkdir(exist_ok=True)
        return self.parcel_data_path, self.satellite_images_folder
    
    def get_data_mask(self, image_path):
        """
        Extracts the data mask from a satellite image and converts it to a GeoDataFrame.
        Only the largest area with data is included.
        """
        logging.info(f"Extracting data mask from {image_path}")
        with rasterio.open(image_path) as src:
            mask = src.read_masks(1) != 0
            mask_shapes = shapes(mask.astype(np.uint8), mask=mask, transform=src.transform)
            gdf = gpd.GeoDataFrame.from_features(
                [{"geometry": shape(s), "properties": {}} for s, value in mask_shapes if value == 1],
                crs=src.crs
            )
            if not gdf.empty:
                gdf = gdf.to_crs(self.crs)  # Convert CRS to match parcels
                gdf['area'] = gdf['geometry'].area
                largest_polygon = gdf.loc[gdf['area'].idxmax()]
                logging.info(f"Data mask extracted with area {largest_polygon['area']}")
                return gpd.GeoDataFrame([largest_polygon], crs=self.crs)
            logging.warning(f"No valid data mask found in {image_path}")
            return gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)

    def get_image_extent_with_mask(self, image_path):
        """
        Extract the image extent and no-data mask.
        """
        logging.info(f"Extracting image extent and mask from {image_path}")
        with rasterio.open(image_path) as img:
            bounds = img.bounds
            meta = img.meta
            mask = self.get_data_mask(image_path)
            if mask.empty:
                logging.warning(f"Empty mask for image {image_path}")
                return None, None, None
            # Create a shapely box (polygon) for the extent of the image
            image_extent = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            # Convert the shapely box to a GeoDataFrame for CRS transformation
            image_extent_gdf = gpd.GeoDataFrame([{'geometry': image_extent}], crs=meta['crs'])
            # Convert the CRS of the image extent to match the parcels' CRS
            image_extent_gdf = image_extent_gdf.to_crs(self.crs)
            return image_extent_gdf.iloc[0]['geometry'], meta, mask

    def trim_parcels_to_data_areas(self, canton, image_extent, data_mask_gdf):
        """
        Trim parcels to the data-rich areas identified. Ensure parcels smaller than 5000 square meters are removed after all operations.
        Ensures that all multi-part geometries are exploded to single-part geometries for accurate area calculation and filtering.
        """
        logging.info("Trimming parcels to data areas")
        if data_mask_gdf is None or data_mask_gdf.empty:
            logging.warning("Data mask is empty or None, returning empty GeoDataFrame")
            return gpd.GeoDataFrame(columns=canton.columns, crs=canton.crs)
        
        image_extent_gdf = gpd.GeoDataFrame([{'geometry': image_extent}], crs=self.crs)
        logging.info(f"Image extent GeoDataFrame: {image_extent_gdf}")

        # Perform spatial join to find parcels intersecting the image extent
        trimmed_parcels = gpd.sjoin(canton, image_extent_gdf, how='inner', predicate='intersects')
        logging.info(f"Trimmed parcels after spatial join: {len(trimmed_parcels)} records")

        # Perform overlay to trim parcels to data mask area
        trimmed_parcels = gpd.overlay(trimmed_parcels, data_mask_gdf, how='intersection')
        logging.info(f"Trimmed parcels after overlay: {len(trimmed_parcels)} records")

        # Explode MultiPolygons to handle individual geometries
        if any(trimmed_parcels.geometry.type == 'MultiPolygon'):
            trimmed_parcels = trimmed_parcels.explode(index_parts=True).reset_index(drop=True)
            logging.info(f"Trimmed parcels after exploding MultiPolygons: {len(trimmed_parcels)} records")

        # Recalculate the area after overlay and exploding MultiPolygons to catch any new, smaller polygons
        trimmed_parcels['area'] = trimmed_parcels.geometry.area

        # Filter parcels by area again after performing the intersection and exploding
        trimmed_parcels = trimmed_parcels[trimmed_parcels['area'] > 5000]
        logging.info(f"Trimmed parcels after area filtering: {len(trimmed_parcels)} records")

        return trimmed_parcels

    def process_parcels(self):
        """
        Processes parcels for each satellite image by trimming them to the data-rich areas identified.
        """
        satellite_images = [f for f in os.listdir(self.satellite_images_folder) if f.endswith('.tif')]
        logging.info(f"Found {len(satellite_images)} satellite images for processing")
        for image_file in tqdm(satellite_images, desc='Processing parcels'):
            image_path = os.path.join(self.satellite_images_folder, image_file)
            image_extent, meta, data_mask_gdf = self.get_image_extent_with_mask(image_path)
            if image_extent is None or data_mask_gdf is None:
                logging.warning(f"Skipping {image_file} due to empty mask or extent")
                continue
            trimmed_parcels = self.trim_parcels_to_data_areas(self.canton, image_extent, data_mask_gdf)
            if trimmed_parcels.empty:
                logging.warning(f"No parcels to save for {image_file}")
                continue
            trimmed_parcels = trimmed_parcels.to_crs(meta['crs'])  # Convert to satellite image CRS before saving
            output_path = os.path.join(self.parcel_data_path, os.path.splitext(image_file)[0] + ".gpkg")
            logging.info(f"Saving trimmed parcels to {output_path}")
            trimmed_parcels.to_file(output_path, driver="GPKG")

if __name__ == "__main__":
    processor = ProcessParcels("/workspaces/Satelite/data/AI.gpkg")
