import re
import os 
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import logging
import time
from helpers.grid import CreateGrid
from helpers.satellite import ProcessSatellite
from helpers.parcels import ProcessParcels
from helpers.mask import ProcessMask
from helpers.dataset import CreateTensorflowDataset
list_of_cantons = ['ZH']
base_path = "/Users/dennis/Documents/GitHub/Satelite/data"
cell_size = 2500
threshold = 0.1
target_size = 256  # Has to be divisible by 32 due to UNET architecture
time_start = datetime(2023, 6, 1)
time_end = datetime(2023, 7, 31)
target_resolution = 10
upsampling_factor = 2
border_width = 0.1
train = 0.8
test = 0.1
val = 0.1
upscale = False


def create_grid(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    logging.info(f"Processing canton {canton}: Creating grid")
    CreateGrid(data_path=path_gpkg, cell_size=cell_size, non_essential_cells=threshold)

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
    
def create_parcels(canton: str, trimmed, upscaling, combine_adjacent=True):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    ProcessParcels(data_path = path_gpkg, trimmed=trimmed, combine_adjacent=combine_adjacent, upscaling=upscaling)

def create_mask(canton: str, scaled=False):
        path_gpkg = f"{base_path}/{canton}.gpkg"
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
                process.create_border_mask(border_width=border_width)
            except Exception as e:
                logging.error(f"Error creating mask for canton {canton}: {e}")
        
def create_tensorflow_dataset(canton: str):
    try:
        path_gpkg = f"{base_path}/{canton}.gpkg"
        logging.info(f"Processing canton {canton}: Creating TensorFlow dataset")
        CreateTensorflowDataset(data_path=path_gpkg, train=train, test=test, val=val)
        logging.info(f"Done processing canton {canton}: TensorFlow dataset created")
    except Exception as e:
        logging.error(f"Error creating TensorFlow dataset for canton {canton}: {e}")

def process_canton(canton: str):
    #logging.info(f"Starting processing for canton {canton}")
    # create_grid(canton)
    # create_satellite(canton)  # This will block until all satellite tasks are finished
    #create_parcels(canton, trimmed=False, combine_adjacent=True, upscaling=False)   # Create parcels with upscaled satellite images
    #create_parcels(canton, trimmed=False, combine_adjacent=True, upscaling=True)    # Create parcels with upscaled satellite images
    create_mask(canton, scaled=False)      # Create masks with upscaled satellite images
    time.sleep(10)
    # create_tensorflow_dataset(canton)

if __name__ == "__main__":
    for canton in list_of_cantons:
        print(f"Processing canton {canton}")
        process_canton(canton)
