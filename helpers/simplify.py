import warnings
import geopandas as gpd
from pathlib import Path

# Ignore warnings
warnings.filterwarnings('ignore')

class Simplify:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.canton_name = self.data_path.stem
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
        ]
        self.data = gpd.read_file(self.data_path)
        self.data = self.remove_nutzungsflächen()
        self.data = self.simplify_data()

    def remove_nutzungsflächen(self):
        """
        This removes certain land use types from the data. See 
        the to_remove_nutzungsflächen list for the land use types.
        """ 
        self.data = self.data[~self.data['nutzung'].isin(self.to_remove_nutzungsflächen)]
        # Stores the class ID of the Nutzung for later use:
        nutzung_to_class = {nutzung: idx for idx, nutzung in enumerate(self.data['nutzung'].unique(), start=1)}
        self.data['class_id'] = self.data['nutzung'].map(nutzung_to_class)
        return self.data
    
    def simplify_data(self):
        """
        Simplifies and validates the geometries of the cantonal data, keeping only specified attributes.
        """
        simplified_data_path = self.data_path.parent / f"{self.canton_name}_simplified.gpkg"
        
        if simplified_data_path.exists():
            print(f"Loading existing simplified data from {simplified_data_path}")
            return gpd.read_file(simplified_data_path)
        
        print("Simplified data not found. Creating new simplified data...")
        self.data = self.data.copy()
        # Explode MultiPolygons
        if any(self.data.geometry.type == 'MultiPolygon'):
            self.data = self.data.explode(index_parts=False).reset_index(drop=True)
        self.data['geometry'] = self.data['geometry'].simplify(tolerance=5, preserve_topology=True)
        self.data['geometry'] = self.data['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
        self.data['area'] = self.data['geometry'].area
        
        # Keep only specified attributes
        attributes_to_keep = ['nutzung', 'kanton', 'class_id', 'area', 'geometry']
        self.data = self.data[attributes_to_keep]
        
        # Save the simplified data
        self.data.to_file(simplified_data_path, driver="GPKG")
        print(f"Saved simplified data to {simplified_data_path}")
        return self.data
    