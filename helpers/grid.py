import warnings
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

# Ignore warnings
warnings.filterwarnings('ignore')


class CreateGrid:
    """
    This class creates a grid of the swiss data of a given cell_size and removes non-essential grid cells.
    Following operations are performed:
    1. Creates a grid of the cantonal data.
    2. Removes non-essential grid cells.
        - Grid cells that are not fully within the cantonal/swiss border.
        - Grid cells that do not have a coverage ratio of at least 0.1 (can be changed).
    """

    def __init__(self, data_path, boundary_path, cell_size=2500, non_essential_cells=0.3):
        self.data_path = Path(data_path)
        self.canton_name = self.data_path.stem
        self.simplified_data_path = Path(self.data_path.parent / f"{self.canton_name}_simplified.gpkg")
        self.data = gpd.read_file(self.simplified_data_path)
        self.boundary_path = Path(boundary_path)
        self.border = gpd.read_file(self.boundary_path)
        self.non_essential_cells = non_essential_cells
        self.cell_size = cell_size
        self.to_remove_nutzungsflächen = [
            "Übrige unproduktive Flächen (z.B. gemulchte Flächen, stark verunkrautete Flächen, Hecken ohne Pufferstreifen)",
            "Hecken-, Feld- und Ufergehölze (mit Krautsaum)",
            "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen)",
            "Wald",
            "Waldweiden (ohne bewaldete Fläche)",
            "Wassergräben, Tümpel, Teiche",
            "Ruderalflächen, Steinhaufen und -wälle",
            "Unbefestigte, natürliche Wege",
            "Hausgärten",
            "Sömmerungsweiden", # In höheren Lagen
            "Heuwiesen im Sömmerungsgebiet, Übrige Wiesen",
            "Heuwiesen im Sömmerungsgebiet, Typ extensiv genutzte Wiese",
            "Heuwiesen im Sömmerungsgebiet, Typ wenig intensiv genutzte Wiese",
            "Heuwiesen mit Zufütterung während der Sömmerung",
            "Streueflächen im Sömmerungsgebiet",
            "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen) (regionsspezifische Biodiversitätsförderfläche)", # Sehr kleine Flächen
            "Übrige Flächen ausserhalb der LN und SF", # Z.B. Waldränder
            "Flächen ohne landwirtschaftliche Hauptzweckbestimmung (erschlossenes Bauland, Spiel-, Reit-, Camping-, Golf-, Flug- und Militärplätze oder ausgemarchte Bereiche von Eisenbahnen, öffentlichen Strassen und Gewässern)",
            "Landwirtschaftliche Produktion in Gebäuden (z. B. Champignon, Brüsseler)",
            "Einheimische standortgerechte Einzelbäume und Alleen (Punkte oder Flächen)",
            "Andere Bäume",
            "Baumschule von Forstpflanzen ausserhalb der Forstzone",     
        ]
        self.create_folders()

        # Ensure all data has the same CRS
        if self.data.crs != self.border.crs:
            self.border = self.border.to_crs(self.data.crs)

        # Simplify data
        self.crs = self.data.crs
        self.xmin, self.ymin, self.xmax, self.ymax = self.data.total_bounds
        self.create_grid()

    def create_folders(self):
        """
        Creates the necessary folders for the data.
        """
        self.base_path = self.data_path.parent
        # Create grid folder
        self.grid_path = self.base_path / "grid"
        self.grid_path.mkdir(exist_ok=True)

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
        self.remove_parcels_with_high_nonessential_use()


    def remove_non_essential_grid_cells(self, grid):
        """
        It removes:
        1. Cells within 5 km of the border
        2. Cells with low coverage ratio (<30%)
        3. Cells where the majority (>50%) of parcels are smaller than 5000m2
        """
        print('Removing non-essential grid cells...')
        grid['cell_area'] = grid['geometry'].area

        border_buffer = self.border.buffer(-500)
        border_buffer_gdf = gpd.GeoDataFrame(geometry=border_buffer, crs=self.border.crs)

        cells_within_buffer = gpd.sjoin(grid, border_buffer_gdf, how='inner', predicate='within')

        cells_within_buffer = cells_within_buffer.reset_index(drop=True)
        self.data = self.data.reset_index(drop=True)

        if 'index_right' in cells_within_buffer.columns:
            cells_within_buffer = cells_within_buffer.drop(columns=['index_right'])

        print('Performing spatial join with simplified data...')
        joined_simplified = gpd.sjoin(cells_within_buffer, self.data, how='inner', predicate='intersects')
        print(f'Number of joined cells with simplified data: {len(joined_simplified)}')

        print("Columns in joined_simplified:", joined_simplified.columns)

        # Use 'geometry' column directly
        geometry_column = 'geometry'

        # Calculate parcel sizes and flag small parcels
        joined_simplified['parcel_size'] = joined_simplified[geometry_column].area
        joined_simplified['is_small_parcel'] = joined_simplified['parcel_size'] < 5000

        print('Grouping joined cells by cell_id...')
        grouped = joined_simplified.groupby('cell_id').agg({
            'geometry': 'first',
            'area': 'sum',
            'cell_area': 'first',
            'kanton': lambda x: x.value_counts().index[0],
            'is_small_parcel': ['count', 'sum']
        }).reset_index()

        grouped.columns = ['cell_id', 'geometry', 'area', 'cell_area', 'kanton', 'total_parcels', 'small_parcels']

        grouped['coverage_ratio'] = grouped['area'] / grouped['cell_area']
        grouped['small_parcel_percentage'] = grouped['small_parcels'] / grouped['total_parcels']

        essential_cells = grouped[
            (grouped['coverage_ratio'] >= self.non_essential_cells) & 
            (grouped['small_parcel_percentage'] <= 0.5)
        ]
        print(f'Number of essential cells: {len(essential_cells)}')

        if not essential_cells.empty:
            essential_cells['cell_id'] = np.arange(1, len(essential_cells) + 1)
            essential_cells = gpd.GeoDataFrame(essential_cells, geometry='geometry', crs=grid.crs)
            essential_cells.to_file(self.grid_path / f'{self.canton_name}_essential_grid_without_removal.gpkg', driver="GPKG")
            print('Saved essential grid cells.')
        else:
            print('No essential cells to save.')
            
    def remove_parcels_with_high_nonessential_use(self):
        """
        Removes grid cells with high non-essential land use.
        """
        print('Removing parcels with high non-essential land use...')
        
        # Load the original data and essential grid
        original_data = gpd.read_file(self.data_path)
        essential_grid_path = self.grid_path / f'{self.canton_name}_essential_grid_without_removal.gpkg'
        essential_grid = gpd.read_file(essential_grid_path)
        
        # Ensure CRS matches
        if original_data.crs != self.crs:
            original_data = original_data.to_crs(self.crs)
        
        # Identify non-essential land use
        original_data['is_nonessential'] = original_data['nutzung'].isin(self.to_remove_nutzungsflächen)
        
        # Perform a spatial join between the grid and the original data
        joined = gpd.sjoin(essential_grid, original_data, how='left', predicate='intersects')
        
        # Calculate intersection areas
        joined['intersection_area'] = joined.apply(lambda row: row['geometry'].intersection(original_data.loc[row['index_right'], 'geometry']).area, axis=1)
        joined['nonessential_area'] = joined.apply(lambda row: row['intersection_area'] if row['is_nonessential'] else 0, axis=1)
        
        # Calculate areas and non-essential percentages
        grouped = joined.groupby('cell_id').agg({
            'geometry': 'first',
            'intersection_area': 'sum',
            'nonessential_area': 'sum'
        })
        
        grouped['nonessential_percentage'] = (grouped['nonessential_area'] / grouped['intersection_area']) * 100
        
        # Filter cells with less than 10% non-essential use
        updated_grid = grouped[grouped['nonessential_percentage'] < 10].copy()
        
        # Reset index to keep the old cell_id as a column
        updated_grid = updated_grid.reset_index()
        
        # Create a new cell_id column with updated numbering
        updated_grid['new_cell_id'] = range(1, len(updated_grid) + 1)
        
        # Get the kanton from the essential_grid_without_removal
        updated_grid['kanton'] = updated_grid['cell_id'].map(essential_grid.set_index('cell_id')['kanton'])
        
        # Ensure we keep all relevant columns that are actually present
        columns_to_keep = ['new_cell_id', 'cell_id', 'geometry', 'kanton', 'nonessential_percentage']
        for col in essential_grid.columns:
            if col in updated_grid.columns and col not in columns_to_keep:
                columns_to_keep.append(col)
        
        updated_grid = updated_grid[columns_to_keep]
        
        # Convert to GeoDataFrame
        updated_grid = gpd.GeoDataFrame(updated_grid, geometry='geometry', crs=essential_grid.crs)
        
        # Save updated grid
        output_path = self.grid_path / f'{self.canton_name}_essential_grid.gpkg'
        updated_grid.to_file(output_path, driver="GPKG")
        print(f'Saved refined essential grid with {len(updated_grid)} cells to {output_path}')
        
        return updated_grid

if __name__ == "__main__":
    data_path = "/workspaces/Satelite/data/ZH.gpkg"
    boundary_path = "/workspaces/Satelite/data/ZH.geojson"
    grid = CreateGrid(data_path, boundary_path, cell_size=2500, non_essential_cells=0.1)
