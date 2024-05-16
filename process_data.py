"""
This script pre-processes the data for the satellite image analysis. It creates a grid of
cells for each canton, extracts the parcels from the cantonal data, assigns them to the
corresponding grid cell, and saves the grid and parcels as GeoDataFrames.It also removes
non-significant GeoDataFrames, which includes removing any parcels within each GeoDataFrame that are smaller than 5000 square meters,
and then removing any GeoDataFrames that do not meet the area threshold or contain no significant parcels.
The script then processes the satellite images and creates a mask with parcel borders in white and interiors in black for deep learning.
"""

import re
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
import threading

from helpers.grid import CreateGrid
from helpers.satelite import ProcessSatellite
from helpers.parcels import ProcessParcels
from helpers.mask import ProcessMask
from helpers.dataset import CreateTensorflowDataset

list_of_cantons = ['AG']
base_path = "/home/tfuser/project/Satelite/data/"
cell_size = 2500
threshold = 0.1
time_start: datetime = datetime(2023, 6, 1)
time_end: datetime = datetime(2023, 7, 31)
target_resolution = 10
border_width = 0.1
train = 0.8
test = 0.1
val = 0.1

def create_grid(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    print("Processing canton: ", canton)
    CreateGrid(data_path=path_gpkg, cell_size=cell_size,
                      non_essential_cells=threshold)
    print("Done processing canton: ", canton)

def create_satelite(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    grid_path = f"{base_path}/grid/{canton}_essential_grid.gpkg"
    grid_length = len(list(gpd.read_file(grid_path).iterfeatures()))
    for index in tqdm(range(1, grid_length + 1)):   
        process = ProcessSatellite(path_gpkg, time_start, time_end,
                                   target_resolution, index)
        process.create_satellite_mapper()
        process.select_min_coverage_scene()

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

if __name__ == "__main__":
    for canton in list_of_cantons:
        #create_grid(canton)
        #create_satelite(canton)
        #create_parcels(canton)
        #create_mask(canton)
        create_tensorflow_dataset(canton)
