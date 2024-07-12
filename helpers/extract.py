import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from scipy import ndimage
import tensorflow as tf
import numpy as np
import geopandas as gpd
import rasterio
from load import LoadandAugment
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon
from pathlib import Path
import json
import model
import modelup
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, upscale, experiment_name, model_name, dataset_path, json_name):
        self.upscale = upscale
        self.json_name = json_name
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.base_path = self.dataset_path.parent
        self.experiment_path = self.base_path / 'experiment'
        self.weights_path = self.experiment_path / f"{self.experiment_name}"
        self.output_dir = self.experiment_path / 'predictions'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load metadata of the test dataset: 
        metadata_path = self.dataset_path / self.json_name
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.test_data = self.load_test_data()
        
        # Determine input shape from the first batch of the dataset
        for images, _ in self.test_data.dataset.take(1):
            self.input_shape = images.shape[1:]
            break
        print(f"Input shape determined from test dataset: {self.input_shape}")
        self.model = self.load_model()
        self.all_polygons = []  # Attribute to store all polygons

    def load_model(self):
        if self.model_name == 'unet':
            if self.upscale:
                loaded_model = modelup.unet(self.input_shape)
            else:
                loaded_model = model.unet(self.input_shape)
        
        elif self.model_name == 'attunet':
            if self.upscale:
                loaded_model = modelup.attunet(self.input_shape)
            else:
                loaded_model = model.attunet(self.input_shape)
        elif self.model_name == 'resunet':
            if self.upscale:
                loaded_model = modelup.resunet(self.input_shape)
            else:
                loaded_model = model.resunet(self.input_shape)
        else:
            raise ValueError(f"Model {self.model_name} not implemented")
        
        loaded_model.load_weights(self.weights_path)
        loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model

    def load_test_data(self):
        test_path = str(self.dataset_path / "test")  # Convert Path to string
        return LoadandAugment(dataset_path=test_path, data_type="test", batch=5, 
                              augmentation=False)

    def evaluate(self):
        evaluation = self.model.evaluate(self.test_data.dataset)
        print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

    def predict_and_save(self):
        predictions = self.model.predict(self.test_data.dataset)
        
        for i, prediction in enumerate(predictions):
            if i < len(self.metadata):
                mask_path = Path(self.metadata[i]['mask'])
                output_tif = self.output_dir / mask_path.name
                with rasterio.open(mask_path) as src:
                    profile = src.profile.copy()
                # Create binary mask from prediction
                binary_mask = (prediction[:,:,0] > 0.5).astype(np.uint8)
                profile.update(count=1, dtype=rasterio.uint8)
                with rasterio.open(output_tif, 'w', **profile) as dst:
                    dst.write(binary_mask, 1)
                
                self.extract_and_save_polygons(output_tif)
            else:
                print(f"Warning: Metadata not found for prediction {i}")

    def extract_polygons(self, mask, transform):
        # Invert the mask so we're extracting black areas
        inverted_mask = ~mask.astype(bool)
        shapes = features.shapes(inverted_mask.astype('uint8'), transform=transform)
        polygons = [shape(geom) for geom, value in shapes if value == 1]
        return polygons

    def create_geodataframe(self, polygon_data, crs):
        gdf = gpd.GeoDataFrame(polygon_data, crs=crs)
        return gdf

    def plot_and_save_contours(self, mask, polygons, output_file):
        plt.figure(figsize=(20, 20))  # Increased figure size for better resolution
        plt.imshow(mask, cmap='gray')
        for polygon in polygons:
            x, y = polygon.exterior.xy
            plt.plot(x, y, color='red', linewidth=1)  # Reduced linewidth for clarity
        plt.axis('off')
        plt.tight_layout(pad=0)  # Removes padding
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Verify the image was saved
        if not os.path.exists(output_file):
            print(f"Warning: Failed to save contour image to {output_file}")
        else:
            print(f"Contour image saved to {output_file}")

    def extract_and_save_polygons(self, tif_file):
        with rasterio.open(tif_file) as src:
            image = src.read(1)
            mask = image > 0.5
            polygons = self.extract_polygons(mask, src.transform)
            crs = src.crs

            # Save contour image
            contour_image_file = self.output_dir / f"{tif_file.stem}_contours.png"
            self.plot_and_save_contours(mask, polygons, contour_image_file)

        # Create polygon data for this Auschnitt
        polygon_data = [{'id': i, 'geometry': polygon} for i, polygon in enumerate(polygons)]
        
        # Create and save GeoDataFrame for this Auschnitt
        gdf = self.create_geodataframe(polygon_data, crs)
        output_gpkg = self.output_dir / f"{tif_file.stem}.gpkg"
        gdf.to_file(output_gpkg, driver="GPKG")
        print(f"GeoPackage for {tif_file.stem} saved to {output_gpkg}")

        # Append polygons to the all_polygons list for the combined GeoDataFrame
        for polygon in polygons:
            self.all_polygons.append({
                'file_name': tif_file.stem,
                'geometry': polygon
            })

    def create_final_geodataframe(self):
        if not self.all_polygons:
            print("No polygons extracted. Check your prediction process.")
            return

        # Create GeoDataFrame with all polygons
        gdf = gpd.GeoDataFrame(self.all_polygons)
        
        # Set the CRS to the CRS of the first polygon (assuming all have the same CRS)
        with rasterio.open(Path(self.metadata[0]['mask'])) as src:
            gdf.set_crs(src.crs, inplace=True)

        # Save the final combined GeoDataFrame
        output_gpkg = self.output_dir / "prediction_combined.gpkg"
        gdf.to_file(output_gpkg, driver="GPKG")
        print(f"Combined GeoPackage saved to {output_gpkg}")

    def run(self):
        self.evaluate()
        self.predict_and_save()
        self.create_final_geodataframe()

if __name__ == "__main__":
    upscale = False
    json_name = 'test_file_mapping.json'
    experiment_name = 'resunet_experiment_up.h5'
    model_name = 'attunet'
    dataset_path = '/workspaces/Satelite/data/dataset_upscaled_False'
    evaluator = ModelEvaluator(upscale, experiment_name, model_name, dataset_path, json_name)
    evaluator.run()