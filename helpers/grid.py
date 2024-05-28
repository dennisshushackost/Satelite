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
    This class creates a grid of the swiss data of a given cell_size and simplifies the geometries of the cantonal data.
    """
    def __init__(self, data_path, boundary_path, cell_size=2500, non_essential_cells=0.1):
        self.data_path = Path(data_path)
        self.boundary_path = Path(boundary_path)
        self.canton_name = self.data_path.stem
        self.cell_size = cell_size
        self.create_folders()
        self.data = gpd.read_file(self.data_path)
        self.border = gpd.read_file(self.boundary_path)
        self.non_essential_cells = non_essential_cells
        
        # Ensure all data has the same CRS
        if self.data.crs != self.border.crs:
            self.border = self.border.to_crs(self.data.crs)
        
        # Simplify data
        self.data = self.simplify_data()
        
        # Save simplified data to a new file to preserve the original data
        new_data_path = self.data_path.parent / f"{self.canton_name}_simplified.gpkg"
        self.data.to_file(new_data_path, driver="GPKG")
        
        self.crs = self.data.crs
        self.xmin, self.ymin, self.xmax, self.ymax = self.data.total_bounds
        self.create_grid()

    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.base_path = self.data_path.parent
        print(self.base_path)
        # Create grid folder
        self.grid_path = self.base_path / "grid"
        self.grid_path.mkdir(exist_ok=True)

    def simplify_data(self):
        """
        Simplifies and validates the geometries of the cantonal data.
        """
        self.data = self.data.copy()
        # Explode MultiPolygons
        if any(self.data.geometry.type == 'MultiPolygon'):
            self.data = self.data.explode().reset_index(drop=True)
        self.data['geometry'] = self.data['geometry'].simplify(tolerance=5, preserve_topology=True)
        self.data['geometry'] = self.data['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
        self.data['area'] = self.data['geometry'].area
        return self.data

    def create_grid(self):
        """
        Creates a grid based on the defined width and height, covering the extent of the cantonal geodataframe.
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

        # Perform spatial join between grid and border to find cells intersecting the border
        print('Performing spatial join to find intersecting cells...')
        joined = gpd.sjoin(grid, self.border, how='inner', op='intersects')
        print(f'Number of intersecting cells: {len(joined)}')

        # Filter out cells that are not fully within the border
        print('Filtering cells that are fully within the border...')
        fully_within_cells = joined[joined.apply(lambda row: self.border.contains(row.geometry).all(), axis=1)]
        print(f'Number of fully within cells: {len(fully_within_cells)}')

        # Reset index to avoid conflicts in spatial join
        fully_within_cells = fully_within_cells.reset_index(drop=True)
        self.data = self.data.reset_index(drop=True)

        # Drop index_left and index_right columns if they exist
        if 'index_left' in fully_within_cells.columns:
            fully_within_cells = fully_within_cells.drop(columns=['index_left'])
        if 'index_right' in fully_within_cells.columns:
            fully_within_cells = fully_within_cells.drop(columns=['index_right'])
        if 'index_left' in self.data.columns:
            self.data = self.data.drop(columns=['index_left'])
        if 'index_right' in self.data.columns:
            self.data = self.data.drop(columns=['index_right'])

        # Perform spatial join between fully within cells and simplified parcel data
        print('Performing spatial join with simplified data...')
        joined_simplified = gpd.sjoin(fully_within_cells, self.data, how='inner', predicate='intersects')
        print(f'Number of joined cells with simplified data: {len(joined_simplified)}')

        # Ensure 'area' from the simplified data is used for calculating coverage ratio
        print('Grouping joined cells by cell_id...')
        grouped = joined_simplified.groupby('cell_id').agg({
            'geometry': 'first',
            'area': 'sum',
            'cell_area': 'first'
        }).reset_index()

        # Calculate the coverage ratio
        grouped['coverage_ratio'] = grouped['area'] / grouped['cell_area']

        # Filter essential cells based on coverage ratio
        essential_cells = grouped[grouped['coverage_ratio'] >= self.non_essential_cells]
        print(f'Number of essential cells: {len(essential_cells)}')

        # Further filter out cells that are smaller than 2500m x 2500m (6,250,000 square meters)
        cell_size_threshold = 6250000
        essential_cells = essential_cells[essential_cells['cell_area'] >= cell_size_threshold]
        print(f'Number of essential cells after size filtering: {len(essential_cells)}')

        # If there are essential cells, create the GeoDataFrame and save to file
        if not essential_cells.empty:
            essential_cells['cell_id'] = np.arange(1, len(essential_cells) + 1)
            essential_cells = gpd.GeoDataFrame(essential_cells, geometry='geometry', crs=grid.crs)
            essential_cells.to_file(self.grid_path / f'{self.canton_name}_essential_grid.gpkg', driver="GPKG")
            print('Saved essential grid cells.')
        else:
            print('No essential cells to save.')

if __name__ == "__main__":
    data_path = "/workspaces/Satelite/data/CH.gpkg"
    boundary_path = "/workspaces/Satelite/data/borders.geojson"
    grid = CreateGrid(data_path, boundary_path, cell_size=2500, non_essential_cells=0.1)
