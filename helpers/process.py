import os
import warnings
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import numpy as np

# Ignore warnings
warnings.filterwarnings('ignore')

class Process:
    """
    Processes the geodataframe for further use. Saved in the folder called the same as the geodataframe.
    Boundaries is an array of the names of the cantons, which should be included.
    """
    def __init__(self, data_path, boundaries, cell_size=2500, non_essential_cells=0.1):
        self.data_path = Path(data_path)
        self.boundaries = boundaries
        self.cell_size = cell_size
        self.non_essential_cells = non_essential_cells
        self.name = self.data_path.stem
        self.create_folders()
        self.data = gpd.read_file(self.data_path)
        self.canton_name = self.data_path.stem
        
        # Simplifies the data:
        self.data = self.simplify_data()
        new_data_path = self.data_path.parent / f"{self.canton_name}_simplified.gpkg"
        self.data.to_file(new_data_path, driver="GPKG")
        self.crs = self.data.crs
        
        # Grid creation:
        self.xmin, self.ymin, self.xmax, self.ymax = self.data.total_bounds
        self.grid = self.create_grid()
        
        # creating the essential grids:
        for boundary in self.boundaries:
            self.boundary_name = boundary
            self.boundary = gpd.read_file(self.base_path / f"{boundary}.geojson")
            self.boundary = self.boundary.to_crs(self.crs)
            self.remove_non_essential_grid_cells()
        
    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.base_path = self.data_path.parent
        self.canton_folder = self.base_path / self.name
        self.canton_folder.mkdir(exist_ok=True)
        
        # Create grid folder
        self.grid_path = self.canton_folder / "grid"
        self.grid_path.mkdir(exist_ok=True)   
        
    def simplify_data(self):
        """
        Simplifies and validates the geometries of the cantonal data.
        """
        self.data = self.data.copy()
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
        return grid
        
    def remove_non_essential_grid_cells(self):
        
        print('Removing non-essential grid cells...')
        self.grid['cell_area'] = self.grid['geometry'].area

        # Perform spatial join between grid and border to find cells intersecting the border
        joined = gpd.sjoin(self.grid, self.boundary, how='inner', op='intersects')

        # Filter out cells that are not fully within the border
        fully_within_cells = joined[joined.apply(lambda row: self.boundary.contains(row.geometry).all(), axis=1)]

        # Reset index to avoid conflicts in spatial join
        fully_within_cells = fully_within_cells.reset_index(drop=True)
        self.data = self.data.copy()
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

        # If there are essential cells, create the GeoDataFrame and save to file
        if not essential_cells.empty:
            essential_cells['cell_id'] = np.arange(1, len(essential_cells) + 1)
            essential_cells = gpd.GeoDataFrame(essential_cells, geometry='geometry', crs=self.grid.crs)
            essential_cells.to_file(self.grid_path / f'{self.boundary_name}_essential_grid.gpkg', driver="GPKG")
            print('Saved essential grid cells.')
        else:
            print('No essential cells to save.')
            
if __name__ == "__main__":
    data_path = "/workspaces/Satelite/data/CH.gpkg"
    boundaries = ['AG', 'ZH', 'ZG']
    grid = Process(data_path, boundaries, cell_size=2500, non_essential_cells=0.1)
