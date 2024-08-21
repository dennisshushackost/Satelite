from helpers.extract import ExtractPolygons
from helpers.eval2 import ParcelEvaluator
from helpers.images import SatelliteImageProcessor


"""
This will extract the polygons from the predicted images and save them as a geopandas dataframe
"""

# Please adjust these parameters as needed:
upscaled = True
experiment_name = "unet_upscale_True_augmentation_False"
model_name = "unet"
json_name = 'combined_file_mapping.json' # Path to the json file 
dataset_path = '/workspaces/Satelite/data/dataset_upscaled_True'
satellite_images_path = '/workspaces/Satelite/data/satellite'
parcel_path = "/workspaces/Satelite/data/parcels"
combined = True


#extractor = ExtractPolygons(upscaled, experiment_name, model_name, dataset_path, json_name, satellite_images_path, combined=combined)
#extractor.run()

"""
This creates the necessary PNG files from the TIF files
"""
#processor = SatelliteImageProcessor(satellite_images_path, extractor.output_dir)
#processor.process_images()

"""
This will do the evaluation of the predicted polygons:
"""
evaluator = ParcelEvaluator(parcel_path, f"/workspaces/Satelite/data/experiments/{experiment_name}/predictions", upscaled)
evaluator.analyze_parcels()
