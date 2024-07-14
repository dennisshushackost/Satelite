from helpers.extract import ExtractPolygons
from helpers.eval import ParcelEvaluator
from helpers.images import SatelliteImageProcessor


"""
This will extract the polygons from the predicted images and save them as a geopandas dataframe
"""
upscale = False
experiment_name = "resunet_experiment_augmentation_False"
model_name = "resunet"
json_name = 'test_file_mapping.json' # Path to the json file 
dataset_path = '/workspaces/Satelite/data/dataset_upscaled_False'
satellite_images_path = '/workspaces/Satelite/data/satellite'
parcel_path = "/workspaces/Satelite/data/parcels"


extractor = ExtractPolygons(upscale, experiment_name, model_name, dataset_path, json_name, satellite_images_path)
extractor.run()

"""
This creates the necessary PNG files from the TIF files
"""
processor = SatelliteImageProcessor(satellite_images_path, extractor.output_dir)
processor.process_images()

"""
This will do the evaluation of the predicted polygons:
"""
evaluator = ParcelEvaluator(parcel_path, extractor.output_dir)
evaluator.analyze_parcels()

