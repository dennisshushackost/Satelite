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
from helpers.resample import Resample

# Initialize logging
log_filename = 'processing.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

list_of_cantons = ['AG', 'AI', 'BE', 'BL', 'FR', 'GE', 'GL', 'GR', 'JU', 'LU', 'NE', 'SG', 'SH', 'SO', 'SZ', 'TG', 'TI', 'UR', 'VS', 'ZG', 'ZH']
base_path = "/workspaces/Satelite/data/cantons/"
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
    try:
        path_gpkg = f"{base_path}/{canton}.gpkg"
        logging.info(f"Processing canton {canton}: Creating grid")
        CreateGrid(data_path=path_gpkg, cell_size=cell_size, non_essential_cells=threshold)
        logging.info(f"Done processing canton {canton}: Grid created")
    except Exception as e:
        logging.error(f"Error processing grid for canton {canton}: {e}")

def create_satellite(canton: str):
    try:
        logging.info(f"Processing canton {canton}: Creating satellite images")
        path_gpkg = f"{base_path}/{canton}.gpkg"
        path_gpkg = Path(path_gpkg)
        grid_path = str(path_gpkg.parent.parent / "grid" / f"{canton}_essential_grid.gpkg")
        grid_length = len(list(gpd.read_file(grid_path).iterfeatures()))
        
        def process_index(index):
            try:
                process = ProcessSatellite(path_gpkg, time_start, time_end,
                                           target_resolution, index, upscale=False, target_size=target_size)
                process.create_satellite_mapper()
                process.select_min_coverage_scene()
                logging.info(f"Processed satellite index {index} for canton {canton}")
            except Exception as e:
                logging.error(f"Error processing satellite index {index} for canton {canton}: {e}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            list(tqdm(executor.map(process_index, range(1, grid_length + 1)), total=grid_length))
        logging.info(f"Done processing canton {canton}: Satellite images created")
    except Exception as e:
        logging.error(f"Error processing satellite images for canton {canton}: {e}")

def process_image(image_path: str, upscale_factor: int):
    try:
        resampler = Resample(image_path, upscale_factor, target_size)
        resampler.resample_image()
        resampler.crop_or_pad_image()
        logging.info(f"Processed image {image_path} with upscale factor {upscale_factor}")
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
    
def create_upsampled_satellite(canton: str, upscale_factor: int = 2):
    try:
        logging.info(f"Processing canton {canton}: Creating upsampled satellite images")
        path_gpkg = f"{base_path}/{canton}.gpkg"
        path_gpkg = Path(path_gpkg)
        satellite_path = str(path_gpkg.parent.parent / "satellite")
        satellite_images = list(Path(satellite_path).glob(f"{canton}_parcel_*.tif"))
        logging.info(f"Found {len(satellite_images)} satellite images for canton {canton}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_image, str(image), upscale_factor) for image in satellite_images]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()
        logging.info(f"Done processing canton {canton}: Upsampled satellite images created")
    except Exception as e:
        logging.error(f"Error processing upsampled satellite images for canton {canton}: {e}")
    
def create_parcels(canton: str, scaled):
    try:
        path_gpkg = f"{base_path}/{canton}.gpkg"
        logging.info(f"Processing canton {canton}: Creating parcels (scaled={scaled})")
        ProcessParcels(data_path=path_gpkg, scaled=scaled)
        logging.info(f"Done processing canton {canton}: Parcels created (scaled={scaled})")
    except Exception as e:
        logging.error(f"Error creating parcels for canton {canton}: {e}")

def create_mask(canton: str, scaled=False):
    try:
        path_gpkg = f"{base_path}/{canton}.gpkg"
        if not scaled:
            path_images = f"{str(Path(base_path).parent)}/satellite"
        else:
            path_images = f"{str(Path(base_path).parent)}/satellite_upscaled"
            
        satelite_images = list(Path(path_images).glob(f"{canton}_parcel_*.tif"))
        logging.info(f"Processing canton {canton}: Creating masks (scaled={scaled})")
        logging.info(f"Found {len(satelite_images)} satellite images to process for masks (scaled={scaled})")
        satelite_images.sort()
        for satellite_image in tqdm(satelite_images, desc=f"Creating masks for {canton} (scaled={scaled})"):
            try:
                parcel_index = int(re.findall(r'\d+', satellite_image.stem)[0])
                process = ProcessMask(path_gpkg, parcel_index, resampled=scaled)
                process.create_border_mask(border_width=border_width)
                logging.info(f"Processed mask for image {satellite_image} (scaled={scaled})")
            except Exception as e:
                logging.error(f"Error creating mask for image {satellite_image}: {e}")
        logging.info(f"Done processing canton {canton}: Masks created (scaled={scaled})")
    except Exception as e:
        logging.error(f"Error processing masks for canton {canton}: {e}")

def create_tensorflow_dataset(canton: str):
    try:
        path_gpkg = f"{base_path}/{canton}.gpkg"
        logging.info(f"Processing canton {canton}: Creating TensorFlow dataset")
        CreateTensorflowDataset(data_path=path_gpkg, train=train, test=test, val=val)
        logging.info(f"Done processing canton {canton}: TensorFlow dataset created")
    except Exception as e:
        logging.error(f"Error creating TensorFlow dataset for canton {canton}: {e}")

def process_canton(canton: str):
    logging.info(f"Starting processing for canton {canton}")
    create_grid(canton)
    create_satellite(canton)  # This will block until all satellite tasks are finished
    create_upsampled_satellite(canton)
    create_parcels(canton, scaled=False)  # Create parcels with original satellite images
    create_parcels(canton, scaled=True)   # Create parcels with upscaled satellite images
    create_mask(canton, scaled=False)     # Create masks with original satellite images
    time.sleep(5)
    create_mask(canton, scaled=True)      # Create masks with upscaled satellite images
    # create_tensorflow_dataset(canton)
    logging.info(f"Finished processing for canton {canton}")

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(process_canton, canton) for canton in list_of_cantons]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()
