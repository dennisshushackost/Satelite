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

from helpers.cantons import Cantons
from helpers.mask import ProcessMask
from helpers.satelite import ProcessSatellite

list_of_cantons = ['AG']
base_path = "/home/tfuser/project/Satelite/data"
cell_size = 2500
threshold = 0.1
time_start: datetime = datetime(2023, 6, 1)
time_end: datetime = datetime(2023, 7, 31)
target_resolution = 10
border_width = 0.1


def process_canton(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    print("Processing canton: ", canton)
    cantons = Cantons(data_path=path_gpkg, cell_size=cell_size,
                      threshold=threshold)
    cantons.create_grid()
    cantons.process_and_save_grid()
    cantons.remove_non_significant_geodataframes()
    print("Done processing canton: ", canton)


def process_satelite(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    path_parcels = f"{base_path}/parcels"
    parcels_files = list(Path(path_parcels).glob(f"{canton}_parcel_*.gpkg"))
    parcels_files.sort()
    for parcel_file in parcels_files:
        print("Processing parcel: ", parcel_file.stem)
        parcel_index = int(re.findall(r'\d+', parcel_file.stem)[0])
        process = ProcessSatellite(path_gpkg, time_start, time_end,
                                   target_resolution, parcel_index)
        process.create_satellite_mapper()
        process.select_min_coverage_scene()


def create_mask(canton: str):
    path_gpkg = f"{base_path}/{canton}.gpkg"
    path_images = f"{base_path}/satellite"
    satelite_images = list(Path(path_images).glob(f"{canton}_parcel_*.tif"))
    satelite_images.sort()
    for satellite_image in satelite_images:
        print("Creating mask for image mask: ", satellite_image.stem)
        parcel_index = int(re.findall(r'\d+', satellite_image.stem)[0])
        process = ProcessMask(path_gpkg, parcel_index)
        process.create_border_mask(border_width=border_width)


if __name__ == "__main__":
    for canton in list_of_cantons:
        # process_canton(canton)
        process_satelite(canton)
        create_mask(canton)
