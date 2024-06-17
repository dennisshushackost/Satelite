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
    Following operations are performed:
    1. Simplifies the geometries of the cantonal data / swiss data.
        - Explodes MultiPolygons
        - Simplifies the geometries
        - Validates the geometries
    2. Removes non-essential usable agricultural land from the data.
        - List given in the code.
    3. Optionally merges adjacent parcels of the same 'nutzung' type.
    4. Creates a grid of the cantonal data.
    5. Removes non-essential grid cells.
        - Grid cells that are not fully within the cantonal/swiss border.
        - Grid cells that do not have a coverage ratio of at least 0.1 (can be changed).
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
        self.data = self.remove_nutzungsflächen()

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
            self.data = self.data.explode(index_parts=False).reset_index(drop=True)
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


    def remove_nutzungsflächen(self):
        """
        This removes certain land use types from the data.
        """
        to_remove = [
            "Buntbrache",
            "Rotationsbrache",
            "Saum auf Ackerflächen",
            "Nützlingsstreifen auf offener Ackerfläche",
            "Übrige offene Ackerfläche, nicht beitragsberechtigt (regionsspezifische Biodiversitätsförderfläche)",
            "Christbäume",
            "Hecken-, Feld- und Ufergehölze (mit Krautsaum)",
            "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen)",
            "Wald",
            "Übrige unproduktive Flächen (z.B. gemulchte Flächen, stark verunkrautete Flächen, Hecken ohne Pufferstreifen)",
            "Wassergräben, Tümpel, Teiche",
            "Ruderalflächen, Steinhaufen und -wälle",
            "Unbefestigte, natürliche Wege",
            "Regionsspezifische Biodiversitätsförderflächen",
            "Hausgärten",
            "Sömmerungsweiden",
            "Übrige Flächen ausserhalb der LN und SF",
            "Flächen ohne landwirtschaftliche Hauptzweckbestimmung (erschlossenes Bauland, Spiel-, Reit-, Camping-, Golf-, Flug- und Militärplätze oder ausgemarchte Bereiche von Eisenbahnen, öffentlichen Strassen und Gewässern)",
            "Offene Ackerfläche, beitragsberechtigt (regionsspezifische Biodiversitätsförderfläche)",
            "Waldweiden (ohne bewaldete Fläche)",
            "Heuwiesen im Sömmerungsgebiet, Übrige Wiesen",
            "Einheimische standortgerechte Einzelbäume und Alleen (Punkte oder Flächen)",
            "Andere Bäume",
            "Andere Bäume (regionsspezifische Biodiversitätsförderfläche)",
            "Andere Elemente (regionsspezifische Biodiversitätsförderfläche)", "Quinoa",
            "Heuwiesen im Sömmerungsgebiet, Typ extensiv genutzte Wiese",
            "Heuwiesen im Sömmerungsgebiet, Typ wenig intensiv genutzte Wiese",
            "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen) (regionsspezifische Biodiversitätsförderfläche)",
            "Trockenmauern",
            "Regionsspezifische Biodiversitätsförderflächen (Weiden)",
            "Baumschule von Forstpflanzen ausserhalb der Forstzone",
            "Ackerschonstreifen",
            "Landwirtschaftliche Produktion in Gebäuden (z. B. Champignon, Brüsseler)",
            "Heuwiesen mit Zufütterung während der Sömmerung",
            "Streueflächen im Sömmerungsgebiet",
            "Regionsspezifische Biodiversitätsförderfläche (Grünflächen ohne Weiden)"

        ]
        self.data = self.data[~self.data['nutzung'].isin(to_remove)]
        # Stores the class ID of the Nutzung for later use:
        nutzung_to_class = {nutzung: idx for idx, nutzung in enumerate(self.data['nutzung'].unique(), start=1)}
        self.data['class_id'] = self.data['nutzung'].map(nutzung_to_class)
        return self.data

    def remove_non_essential_grid_cells(self, grid):
        """
        This removes the non-essential grid cells from the data. 
        As such only the essential grid cells are used for further processing.
        """
        print('Removing non-essential grid cells...')
        grid['cell_area'] = grid['geometry'].area

        # Perform spatial join between grid and border to find cells intersecting the border
        joined = gpd.sjoin(grid, self.border, how='inner', predicate='intersects')

        # Filter out cells that are not fully within the border
        fully_within_cells = joined[joined.apply(lambda row: self.border.contains(row.geometry).all(), axis=1)]

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
            'cell_area': 'first',
            'kanton': lambda x: x.value_counts().index[0]  # Majority vote for kanton
        }).reset_index()

        # Calculate the coverage ratio
        grouped['coverage_ratio'] = grouped['area'] / grouped['cell_area']

        # Filter essential cells based on coverage ratio
        essential_cells = grouped[grouped['coverage_ratio'] >= self.non_essential_cells]
        print(f'Number of essential cells: {len(essential_cells)}')

        # If there are essential cells, create the GeoDataFrame and save to file
        if not essential_cells.empty:
            essential_cells['cell_id'] = np.arange(1, len(essential_cells) + 1)
            essential_cells = gpd.GeoDataFrame(essential_cells, geometry='geometry', crs=grid.crs)
            essential_cells.to_file(self.grid_path / f'{self.canton_name}_essential_grid.gpkg', driver="GPKG")
            print('Saved essential grid cells.')
        else:
            print('No essential cells to save.')

if __name__ == "__main__":
    data_path = "/workspaces/Satelite/data/ZH.gpkg"
    boundary_path = "/workspaces/Satelite/data/ZH.geojson"
    grid = CreateGrid(data_path, boundary_path, cell_size=2500, non_essential_cells=0.1)
