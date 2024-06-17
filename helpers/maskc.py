
import warnings
from pathlib import Path
import glob
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from rasterio.features import rasterize
from shapely.geometry import LineString, MultiPolygon, Polygon

class ProcessMask:
    """
    Processes the satellite image and the geodataframe to create a
    parcel border mask for deep learning,
    with borders marked in white and interiors in black. This class uses the interior and 
    the exterior mask to create the final mask.
    """
    BORDER_VALUE = 255

    def __init__(self, data_path, parcel_index, upscaled=False, all_classes=None):
        self.data_path = Path(data_path)
        self.parcel_index = parcel_index
        self.base_path = self.data_path.parent
        self.canton = self.data_path.stem

        if not upscaled:
            parcel_pattern = f'{self.base_path}/parcels/*_{self.canton}_parcel_{self.parcel_index}.gpkg'
            satellite_pattern = f"{self.base_path}/satellite/*_{self.canton}_upscaled_parcel_{self.parcel_index}.tif"
        else:
            parcel_pattern = f'{self.base_path}/parcels/*_{self.canton}_upscaled_parcel_{self.parcel_index}.gpkg'
            satellite_pattern = f"{self.base_path}/satellite/*_{self.canton}_upscaled_parcel_{self.parcel_index}.tif"

        # Use glob to find the files matching the pattern
        parcel_files = glob.glob(parcel_pattern)
        satellite_files = glob.glob(satellite_pattern)

        if not parcel_files or not satellite_files:
            raise FileNotFoundError('Parcel or satellite file not found.')

        self.parcel_path = parcel_files[0]
        self.satellite_path = satellite_files[0]
        # Mask name = parcel name
        self.mask_name = self.parcel_path.split('/')[-1].replace('.gpkg', '.tif')

        self.parcel = gpd.read_file(self.parcel_path)
        # Make a copy of the parcel data to avoid modifying the original
        self.parcel = self.parcel.copy()
        self.create_folders()

        # Define all possible classes if provided, else infer from data
        self.all_classes = all_classes if all_classes is not None else self.parcel['class_id'].unique()

    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.mask_path = self.base_path / 'mask'
        self.mask_path.mkdir(parents=True, exist_ok=True)

    def create_border_mask(self, border_width):
        """
        Creates a multiclass mask with parcel borders in white and interiors filled with class IDs.
        Each class is represented in a separate channel.
        """
        try:
            with rasterio.open(self.satellite_path) as src:
                meta = src.meta.copy()
                # Update metadata for high resolution
                meta.update({
                    'count': len(self.all_classes),  # One band per class
                    'dtype': 'uint8'
                })

            # Make sure the vector data is in the same CRS as the raster data
            if self.parcel.crs != meta['crs']:
                self.parcel = self.parcel.to_crs(meta['crs'])

            class_masks = []

            for class_id in self.all_classes:
                # Filter parcels by class_id
                class_parcels = self.parcel[self.parcel['class_id'] == class_id]

                if class_parcels.empty:
                    # If there are no parcels for this class, create an empty mask
                    class_fill_mask = np.zeros((meta['height'], meta['width']), dtype=np.uint8)
                    class_border_mask = np.zeros((meta['height'], meta['width']), dtype=np.uint8)
                else:
                    if class_parcels.is_empty.all():
                        class_fill_mask = np.zeros((meta['height'], meta['width']), dtype=np.uint8)
                        class_border_mask = np.zeros((meta['height'], meta['width']), dtype=np.uint8)
                    else:
                        # Rasterize the parcel fills for the current class
                        class_fill_mask = rasterize(
                            shapes=((geom, class_id) for geom in class_parcels.geometry if geom.is_valid),
                            out_shape=(meta['height'], meta['width']),
                            fill=0,  # Fill outside the parcels
                            transform=meta['transform'],
                            all_touched=True,
                            dtype=rasterio.uint8
                        )

                        # Rasterize the borders for the current class
                        class_border_mask = rasterize(
                            shapes=self._extract_and_buffer_borders(class_parcels, border_width),
                            out_shape=(meta['height'], meta['width']),
                            fill=0,  # Keep existing fills
                            transform=meta['transform'],
                            all_touched=True,
                            dtype=rasterio.uint8
                        )

                # Combine fill and border masks for the current class
                class_mask = np.maximum(class_fill_mask, class_border_mask)
                class_masks.append(class_mask)

            # Convert the list of class masks to a 3D array (channels, height, width)
            combined_mask = np.stack(class_masks, axis=0)

            # Write the mask to a file
            with rasterio.open(self.mask_path / self.mask_name, 'w', **meta) as out:
                for i in range(combined_mask.shape[0]):
                    out.write(combined_mask[i], i + 1)
        except FileNotFoundError as e:
            print(f'File not found: {e}')
        except Exception as e:
            print(f'Error creating border mask: {e}')

    def _extract_and_buffer_borders(self, parcel_subset, buffer_width):
        """
        Extracts borders from the parcel geometries and buffers them slightly to
        enhance visibility, including internal edges.
        """
        border_shapes = []
        for geom in parcel_subset.geometry:
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    if poly.is_valid:
                        # Buffer the exterior
                        buffered_exterior = LineString(list(poly.exterior.coords)).buffer(buffer_width)
                        border_shapes.append((buffered_exterior, self.BORDER_VALUE))  # Use white (255) for borders
                        # Buffer each interior boundary
                        for interior in poly.interiors:
                            buffered_interior = LineString(list(interior.coords)).buffer(buffer_width)
                            border_shapes.append((buffered_interior, self.BORDER_VALUE))  # Use white (255) for borders
            elif isinstance(geom, Polygon) and geom.is_valid:
                # Buffer the exterior
                buffered_exterior = LineString(list(geom.exterior.coords)).buffer(buffer_width)
                border_shapes.append((buffered_exterior, self.BORDER_VALUE))  # Use white (255) for borders
                # Buffer each interior boundary
                for interior in geom.interiors:
                    buffered_interior = LineString(list(interior.coords)).buffer(buffer_width)
                    border_shapes.append((buffered_interior, self.BORDER_VALUE))  # Use white (255) for borders
        return border_shapes
