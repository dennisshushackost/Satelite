import re
import os 
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import logging
import time
from helpers.simplify import Simplify
from helpers.grid import CreateGrid
#from helpers.gridold import CreateGrid
from helpers.satellite import ProcessSatellite
from helpers.parcels import ProcessParcels
from helpers.mask import ProcessMask
from helpers.dataset import CreateTensorflowDataset
from helpers.delete  import RemoveImages
list_of_cantons = ['CH']
base_path = "/workspaces/Satelite/data/" # Path to the data folder containing the CH.gpkg file and the CH.geojson file
cell_size = 2500 # Size of the grid cells in meters (2500m = 2.5km)
threshold = 0.3 # Threshold for the amount of essential cells in a grid cell. If a cell has less than 30% parcels, it will be removed
target_size = 256  # Has to be divisible by 32 due to UNET architecture
time_start = datetime(2023, 6, 1) # Start date for the satellite images
time_end = datetime(2023, 7, 31) # End date for the satellite images
target_resolution = 10 # Target resolution for the satellite images
train = 0.8 # Train, test and validation split
test = 0.1
val = 0.1

def simplify_data(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    Simplify(data_path=path_gpkg)
    
def create_grid(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    boundary_path = f"{base_path}/{canton}.geojson"
    logging.info(f"Processing canton {canton}: Creating grid")
    CreateGrid(data_path=path_gpkg, boundary_path = boundary_path, cell_size=cell_size, non_essential_cells=threshold)

def create_satellite(canton: str):
        path_gpkg = f"{base_path}/{canton}.gpkg"
        path_gpkg = Path(path_gpkg)
        grid_path = str(path_gpkg.parent / "grid" / f"{canton}_essential_grid.gpkg")
        grid_length = len(list(gpd.read_file(grid_path).iterfeatures()))
        print(f"Processing {grid_length} grid cells for canton {canton}")
        for index in range(1, grid_length + 1):
            try:
                process = ProcessSatellite(path_gpkg, time_start, time_end,
                                            target_resolution, index, target_size=target_size)
                process.create_satellite_mapper()
                process.select_min_coverage_scene()
            except Exception as e:
                continue
        
def delete_satellite_images():
    path_gpkg = f"{base_path}/{canton}.gpkg"
    remover = RemoveImages(path_gpkg)
    deleted = remover.execute()
    print(f"Deleted files: {deleted}")
    
def create_parcels(canton: str, trimmed, upscaling, combine_adjacent=True):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    ProcessParcels(data_path = path_gpkg, trimmed=trimmed, combine_adjacent=combine_adjacent, upscaling=upscaling)

def create_mask(canton: str, scaled=False):
        path_gpkg = f"{base_path}/{canton}.gpkg"
        simplified = path_gpkg.replace(".gpkg", "_simplified.gpkg")
        path_gpkg_simplified = Path(simplified)
        # Get the amount of possible classes: 
        path_images = f"{str(Path(base_path).parent)}/data/satellite"
        print(path_images)
        if not scaled:
            satelite_images = list(Path(path_images).glob(f"*_{canton}_parcel_*.tif"))
            satelite_images.sort()
        else:
            # All satellite images who end with _upscaled.tif
            satelite_images = list(Path(path_images).glob(f"*_{canton}_upscaled_parcel_*.tif"))
            satelite_images.sort()
            
        for satellite_image in tqdm(satelite_images, desc="Creating masks"):
            parcel_index = int(re.findall(r'\d+', satellite_image.stem)[0])
            try:
                process = ProcessMask(path_gpkg, parcel_index, upscaled=scaled)
                process.create_border_mask(border_width=1)
            except Exception as e:
                logging.error(f"Error creating mask for canton {canton}: {e}")
        
def create_tensorflow_datasets():
    path_gpkg = f"{base_path}/{canton}.gpkg"
    CreateTensorflowDataset(data_path=path_gpkg, upscaled=False, train=train, test=test, val=val)
    CreateTensorflowDataset(data_path=path_gpkg, upscaled=True, train=train, test=test, val=val)
    

def process_switzerland(canton: str):
    #simplify_data(canton)
    #create_grid(canton)
    #create_satellite(canton)  # This will block until all satellite tasks are finished
    delete_satellite_images()
    #create_parcels(canton, trimmed=False, combine_adjacent=True, upscaling=False)   
    #create_parcels(canton, trimmed=False, combine_adjacent=True, upscaling=True)    # 
    #create_mask(canton, scaled=False)      # Create masks with upscaled satellite images
    #create_mask(canton, scaled=True)       # Create masks with upscaled satellite imagesq
    time.sleep(10)
    #create_tensorflow_datasets()

if __name__ == "__main__":
    for canton in list_of_cantons:
        print(f"Processing {canton}")
        process_switzerland(canton)



