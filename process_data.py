import re
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import concurrent.futures
import logging
import time 
from helpers.grid import CreateGrid
from helpers.satelite import ProcessSatellite
from helpers.parcels import ProcessParcels
from helpers.mask import ProcessMask
from helpers.dataset import CreateTensorflowDataset
from helpers.dataset import 

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_of_cantons = ['AI', 'GE']  # Add all your cantons here
base_path = "/workspaces/Satelite/data"
cell_size = 2500
threshold = 0.1
time_start = datetime(2023, 6, 1)
time_end = datetime(2023, 7, 31)
target_resolution = 10
border_width = 0.1
train = 0.8
test = 0.1
val = 0.1
upscale = False

def create_grid(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    logging.info(f"Processing canton: {canton}")
    CreateGrid(data_path=path_gpkg, cell_size=cell_size, non_essential_cells=threshold)
    logging.info(f"Done processing canton: {canton}")

def create_satellite(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    grid_path = f"{base_path}/grid/{canton}_essential_grid.gpkg"
    grid_length = len(list(gpd.read_file(grid_path).iterfeatures()))
    
    def process_index(index):
        process = ProcessSatellite(path_gpkg, time_start, time_end,
                                   target_resolution, index, upscale=False, grid_size=cell_size)
        process.create_satellite_mapper()
        process.select_min_coverage_scene()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_index, range(1, grid_length + 1)), total=grid_length))

def create_parcels(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    ProcessParcels(data_path=path_gpkg)

def create_mask(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    path_images = f"{base_path}/satellite"
    satelite_images = list(Path(path_images).glob(f"{canton}_parcel_*.tif"))
    satelite_images.sort()
    for satellite_image in tqdm(satelite_images, desc="Creating masks"):
        parcel_index = int(re.findall(r'\d+', satellite_image.stem)[0])
        process = ProcessMask(path_gpkg, parcel_index)
        process.create_border_mask(border_width=border_width)
        
def create_tensorflow_dataset(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    CreateTensorflowDataset(data_path=path_gpkg, train=train, test=test, val=val)

def process_canton(canton: str):
    # create_grid(canton)
    # create_satellite(canton)  # This will block until all satellite tasks are finished
    create_parcels(canton)    # Now proceed to parcels
    # Debug: Check the content of the parcels after creation
    time.sleep(10)  # Wait for the parcels to be created
    # Continue with other tasks if needed
    create_mask(canton)
    # create_tensorflow_dataset(canton)

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        list(tqdm(executor.map(process_canton, list_of_cantons), total=len(list_of_cantons)))
