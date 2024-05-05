import os
import warnings
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

# ignore warnings
warnings.filterwarnings('ignore')

class Cantons():
    """
    Processes the cantonal data by dividing the given canton
    into 2km x 2km grids and extracting the corresponding parcels.

    Attributes:
        data_path (str): Path to the cantonal data (full path to the shapefile/geopandas dataframe)
        cell_size (int): Size of the grid cells in meters (default=1500).
        threshold (int): Minimum number of percentage covered by the polygons in the parcels.
    """

    def __init__(self, data_path, cell_size=2500, threshold=0.1):
        
        self.data_path = Path(data_path)
        self.canton_name = self.data_path.stem
        self.cell_size = cell_size
        self.threshold = threshold
        self.data = gpd.read_file(self.data_path)
        self.data = self.simplify_data()
        self.crs = self.data.crs
        self.xmin, self.ymin, self.xmax, self.ymax = self.data.total_bounds
        self.grid = None
        self.create_folders()
        
    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        # Base path = self.data_path.parent
        self.base_path = self.data_path.parent
        self.parcels_path = self.base_path / "parcels"
        self.grid_path = self.base_path / "grid"
        self.parcels_path.mkdir(exist_ok=True)
        self.grid_path.mkdir(exist_ok=True)

    def simplify_data(self):
        """
        Simplifies and validates the geometries of the cantonal data.
        """
        self.data = self.data.copy()
        # Simplify the geometries
        self.data['geometry'] = self.data['geometry'].simplify(tolerance=5, preserve_topology=True)
        # Ensure all geometries are valid, fix if not
        self.data['geometry'] = (self.data['geometry'].
                                 apply(lambda geom: geom if geom.is_valid else geom.buffer(0)))
        return self.data 

    def create_grid(self):
        """
        Creates a grid based on the defined width and height, covering the extend of the canton data.
        """
        # Calculates the number of full 1.5km x 1.5km cells that fit in the canton
        cols = int((self.xmax - self.xmin) / self.cell_size)
        rows = int((self.ymax - self.ymin) / self.cell_size)

        grid_cells = []
        for col in range(cols):
            for row in range(rows):
                cell_xmin = self.xmin + col * self.cell_size
                cell_xmax = cell_xmin + self.cell_size
                cell_ymin = self.ymin + row * self.cell_size
                cell_ymax = cell_ymin + self.cell_size
                cell = box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)
                grid_cells.append(cell)

        self.grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=self.crs)
        print(f'Created a grid with {len(self.grid)} cells.')

    def process_and_save_grid(self):
        """
        Processes the grid, extracts parcels from the cantonal data, assigns them to
        the corresponding grid cell, and saves the grid and parcels as GeoDataFrames.
        """
        # Initalise an empty list to store the GeoDataFrames:
        parcel_gdfs = []

        for i, cell in self.grid.iterrows():       
            # Extract the parcels that are within the grid cell using spatial join: (inner = only the ones that
            # intersect with the cell)
            parcel_data = gpd.sjoin(self.data, gpd.GeoDataFrame([cell],
                                                                columns=['geometry'], crs=self.crs),
                                    how='inner', predicate='intersects')
            # Clip the polygons to the cell boundaries:
            parcel_data['geometry'] = parcel_data.geometry.intersection(cell.geometry)
            # Remove the rows with empty geometries:
            parcel_data = parcel_data[~parcel_data.geometry.is_empty]
            # Add the resulting GeoDataFrame to the list of it's not empty:
            if not parcel_data.empty:
                parcel_data['grid_index'] = i
                parcel_gdfs.append(parcel_data)

        for index, gdf in enumerate(parcel_gdfs):
            gdf.to_file(self.parcels_path/ f'{self.canton_name}_parcel_{index}.gpkg', driver="GPKG")
        # Save the grid as a GeoDataFrame:
        self.grid['grid_index'] = range(len(self.grid))
        self.grid.to_file(self.grid_path / f'{self.canton_name}_grid.gpkg', driver="GPKG")

    def remove_non_significant_geodataframes(self):
        """
        Removes GeoDataFrames that are not significant. This includes removing any parcels within
        each GeoDataFrame that are smaller than 5000 square meters, and then removing any GeoDataFrames
        that do not meet the area threshold or contain no significant parcels.
        """
        cell_area = self.cell_size ** 2
        min_area_threshold = cell_area * self.threshold
    
        gdf_files = list(self.parcels_path.glob(f"{self.canton_name}_parcel_*.gpkg"))
        significant_files = []

        for gdf_file in gdf_files:
            gdf = gpd.read_file(gdf_file)
            # Filter out small parcels within each GeoDataFrame
            gdf = gdf[gdf['geometry'].area >= 5000]
            
            if not gdf.empty:
                parcel_area = gdf.geometry.area.sum()  # Sum area of all geometries in the GeoDataFrame

                if parcel_area >= min_area_threshold:
                    # Save the modified GeoDataFrame back to file
                    gdf.to_file(gdf_file, driver="GPKG")
                    significant_files.append(gdf_file)
                else:
                    os.remove(gdf_file)  # Delete the file if it's not significant
            else:
                os.remove(gdf_file)  # Delete the file if all parcels were too small
        
        print(f"Kept {len(significant_files)} significant GeoDataFrames and deleted the rest.")

if __name__ == "__main__":
    cantons = Cantons(data_path="/workspaces/Satelite/data/aargau.gpkg", cell_size=2500, threshold=0.1)
    cantons.create_grid()
    cantons.process_and_save_grid()
    cantons.remove_non_significant_geodataframes()
