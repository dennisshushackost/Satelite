import os
import geopandas as gpd
from pathlib import Path
import rasterio
from tqdm import tqdm
from rasterio.features import shapes
from shapely.geometry import shape, box
import numpy as np

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
        with rasterio.open(image_path) as src:
            mask = src.read_masks(1) != 0
            mask_shapes = shapes(mask.astype(np.uint8), mask=mask, transform=src.transform)
            gdf = gpd.GeoDataFrame.from_features(
                [{"geometry": shape, "properties": {}} for shape, value in mask_shapes if value == 1],
                crs=src.crs
            )
            if not gdf.empty:
                gdf = gdf.to_crs(self.crs)  # Convert CRS to match parcels
                gdf['area'] = gdf['geometry'].area
                largest_polygon = gdf.loc[gdf['area'].idxmax()]
                return gpd.GeoDataFrame([largest_polygon], crs=self.crs)
            return gdf

    def get_image_extent_with_mask(self, image_path):
        """
        Extract the image extent and no-data mask.
        """
        with rasterio.open(image_path) as img:
            bounds = img.bounds
            meta = img.meta
            mask = self.get_data_mask(image_path)
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
        image_extent_gdf = gpd.GeoDataFrame([{'geometry': image_extent}], crs=self.crs)
        trimmed_parcels = gpd.sjoin(canton, image_extent_gdf, how='inner', op='intersects')
        trimmed_parcels = gpd.overlay(trimmed_parcels, data_mask_gdf, how='intersection')

        # Explode MultiPolygons to handle individual geometries
        if any(trimmed_parcels.geometry.type == 'MultiPolygon'):
            trimmed_parcels = trimmed_parcels.explode().reset_index(drop=True)

        # Recalculate the area after overlay and exploding MultiPolygons to catch any new, smaller polygons
        trimmed_parcels['area'] = trimmed_parcels.geometry.area

        # Filter parcels by area again after performing the intersection and exploding
        trimmed_parcels = trimmed_parcels[trimmed_parcels['area'] > 5000]

        return trimmed_parcels


    def process_parcels(self):
        """
        Processes parcels for each satellite image by trimming them to the data-rich areas identified.
        """
        satellite_images = [f for f in os.listdir(self.satellite_images_folder) if f.endswith('.tif')]
        for image_file in tqdm(satellite_images, desc='Processing parcels'):
            image_path = os.path.join(self.satellite_images_folder, image_file)
            image_extent, meta, data_mask_gdf = self.get_image_extent_with_mask(image_path)
            trimmed_parcels = self.trim_parcels_to_data_areas(self.canton, image_extent, data_mask_gdf)
            trimmed_parcels = trimmed_parcels.to_crs(meta['crs'])  # Convert to satellite image CRS before saving
            output_path = os.path.join(self.parcel_data_path, os.path.splitext(image_file)[0] + ".gpkg")
            trimmed_parcels.to_file(output_path, driver="GPKG")


if __name__ == "__main__":
    processor = ProcessParcels("/home/tfuser/project/Satelite/data/AG.gpkg")
