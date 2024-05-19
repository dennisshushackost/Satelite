import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from rasterio.features import rasterize
from shapely.geometry import LineString, MultiPolygon, Polygon

# ignore warnings
warnings.filterwarnings('ignore')

class ProcessMask:
    """
    Processes the satellite image and the geodataframe to create a
    parcel border mask for deep learning,
    with borders marked in white and interiors in black.
    """
    def __init__(self, data_path, parcel_index, upscaled=False):
        self.data_path = Path(data_path)
        self.parcel_index = parcel_index
        self.base_path = self.data_path.parent.parent
        self.canton = self.data_path.stem
        
        if not upscaled:
            self.parcel_path = (f'{self.base_path}/parcels/'
                            f'{self.canton}_parcel_{self.parcel_index}.gpkg')
            self.satellite_path = (f"{self.base_path}/satellite/"
                               f"{self.canton}_parcel_{self.parcel_index}.tif")
            self.mask_name = f"{self.canton}_parcel_{self.parcel_index}.tif"
        else:
            self.parcel_path = (f'{self.base_path}/parcels/'
                            f'{self.canton}_upscaled_parcel_{self.parcel_index}.gpkg')
            self.satellite_path = (f"{self.base_path}/satellite/"
                               f"{self.canton}_upscaled_parcel_{self.parcel_index}.tif")
            self.mask_name = f"{self.canton}_upscaled_parcel_{self.parcel_index}.tif"
            
        self.parcel = gpd.read_file(self.parcel_path)
        # Make a copy of the parcel data to avoid modifying the original
        self.parcel = self.parcel.copy()
        self.create_folders()

    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.mask_path = self.base_path / 'mask'
        self.mask_path.mkdir(parents=True, exist_ok=True)
        return

    def create_border_mask(self, border_width):
        """
        Creates a mask with parcel borders in white and interiors in black.
        """
        try:
            with rasterio.open(self.satellite_path) as src:
                meta = src.meta.copy()
                # Update metadata for high resolution
                meta.update({
                    'count': 1,
                    'dtype': 'uint8'
                })

            # Make sure the vector data is in the same CRS as the raster data
            if self.parcel.crs != meta['crs']:
                self.parcel = self.parcel.to_crs(meta['crs'])

            # Rasterize the parcel fills
            fill_masks = rasterize(
                shapes=((geom, 0) for geom in self.parcel.geometry),
                out_shape=(meta['height'], meta['width']),
                fill=1,  # Fill outside the parcels
                transform=meta['transform'],
                all_touched=True,
                dtype=rasterio.uint8
            )

            # Rasterize the borders
            border_masks = rasterize(
                shapes=self._extract_and_buffer_borders(border_width),
                out_shape=(meta['height'], meta['width']),
                fill=0,  # Keep existing fills
                transform=meta['transform'],
                all_touched=True,
                dtype=rasterio.uint8
            )

            # Combine fill and border masks
            combined_mask = np.maximum(fill_masks, border_masks)

            # Write the mask to a file
            with rasterio.open(self.mask_path / self.mask_name, 'w', **meta) as out:
                out.write(combined_mask, 1)
        except Exception as e:
            print(f'Error creating border mask: {e}')

    def _extract_and_buffer_borders(self, buffer_width):
        """
        Extracts borders from the parcel geometries and buffers them slightly to
        enhance visibility, including internal edges.
        """
        border_shapes = []
        for geom in self.parcel.geometry:
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    # Buffer the exterior
                    buffered_exterior = (LineString(list(poly.exterior.coords)).
                                         buffer(buffer_width))
                    border_shapes.append((buffered_exterior, 1))
                    # Buffer each interior boundary
                    for interior in poly.interiors:
                        buffered_interior = (LineString(list(interior.coords)).
                                             buffer(buffer_width))
                        border_shapes.append((buffered_interior, 1))
            elif isinstance(geom, Polygon):
                # Buffer the exterior
                buffered_exterior = (LineString(list(geom.exterior.coords)).
                                     buffer(buffer_width))
                border_shapes.append((buffered_exterior, 1))
                # Buffer each interior boundary
                for interior in geom.interiors:
                    buffered_interior = (LineString(list(interior.coords)).
                                         buffer(buffer_width))
                    border_shapes.append((buffered_interior, 1))
        return border_shapes


