import os
import warnings
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import numpy as np


# Ignore warnings
warnings.filterwarnings('ignore')


class CreateGrid:
    """
    This class creates a grid of the canton of a given cell_size and simplifies the geometries of the cantonal data.
    """
    def __init__(self, data_path, cell_size=2500, non_essential_cells=0.1, area_to_ignore=5000):
        self.data_path = Path(data_path)
        self.canton_name = self.data_path.stem
        self.area_to_ignore = area_to_ignore
        self.cell_size = cell_size
        self.data = gpd.read_file(self.data_path)
        self.data = self.simplify_data()
        self.non_essential_cells = non_essential_cells
         # Optionally save to a new file to preserve the original data
        new_data_path = self.data_path.parent / f"{self.canton_name}_simplified.gpkg"
        self.data.to_file(new_data_path, driver="GPKG")
        self.crs = self.data.crs
        self.xmin, self.ymin, self.xmax, self.ymax = self.data.total_bounds
        self.create_folders()
        self.create_grid()

    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.base_path = self.data_path.parent.parent
        self.grid_path = self.base_path + self.canton_name / "grid"
        self.grid_path.mkdir(exist_ok=True)

    def simplify_data(self):
        """
        Simplifies and validates the geometries of the cantonal data.
        Removes parcels that are too small: Under 5000 square meters as we have a 10m resolution satellite data. Further, it explodes MultiPolygons to handle individual geometries.
        """
        self.data = self.data.copy()
        self.data['geometry'] = self.data['geometry'].simplify(tolerance=5, preserve_topology=True)
        self.data['geometry'] = self.data['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
        self.data['area'] = self.data['geometry'].area
        self.data = self.data[self.data['area'] > self.area_to_ignore]       
        return self.data

    def create_grid(self):
        """
        Creates a grid based on the defined width and height,
        covering the extent of the cantonal geodataframe.
        """
        cols = int((self.xmax - self.xmin) / self.cell_size)
        rows = int((self.ymax - self.ymin) / self.cell_size)
        grid_cells = []
        for col in tqdm(range(cols), desc='Generating grid cells'):
            for row in range(rows):
                cell_xmin = self.xmin + col * self.cell_size
                cell_xmax = cell_xmin + self.cell_size
                cell_ymin = self.ymin + row * self.cell_size
                cell_ymax = cell_ymin + self.cell_size
                cell = box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)
                grid_cells.append(cell)
        grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=self.crs)
        grid["cell_id"] = np.arange(1, len(grid_cells) + 1)
        grid.to_file(self.grid_path / f'{self.canton_name}_grid.gpkg', driver="GPKG")
        self.remove_non_essential_grid_cells(grid)
        
    def remove_non_essential_grid_cells(self, grid):
        print('Removing non-essential grid cells...')
        grid['cell_area'] = grid['geometry'].area
        joined = gpd.sjoin(grid, self.data, how='inner', op='intersects')

        # Ensure 'area' from self.data is used
        grouped = joined.groupby('cell_id').agg({
            'geometry': 'first', 
            'area': 'sum', 
            'cell_area': 'first'
        }).reset_index()

        grouped['coverage_ratio'] = grouped['area'] / grouped['cell_area']
        essential_cells = grouped[grouped['coverage_ratio'] >= self.non_essential_cells]
        essential_cells['cell_id'] = np.arange(1, len(essential_cells) + 1)
        essential_cells = gpd.GeoDataFrame(essential_cells, geometry='geometry', crs=self.crs)
        essential_cells.to_file(self.grid_path / f'{self.canton_name}_essential_grid.gpkg', driver="GPKG")
        print('Removed non-essential grid cells.')


 