import os
import numpy as np
import rasterio
from pathlib import Path
import json
import tensorflow as tf
from scipy import ndimage
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from helpers.load import LoadandAugment
import helpers.model
import helpers.modelup
import pandas as pd

class ExtractPolygons:
    def __init__(self, upscale, experiment_name, model_name, dataset_path, json_name, satellite_images_path):
        self.upscale = upscale
        self.json_name = json_name
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.base_path = self.dataset_path.parent
        self.experiment_path = self.base_path / 'experiments' / self.experiment_name
        self.weights_path = self.experiment_path / f"{self.experiment_name}.h5"
        self.output_dir = self.experiment_path / 'predictions'
        self.output_dir.mkdir(exist_ok=True)
        self.satellite_images_path = Path(satellite_images_path)
        self.satellite_reference_path = self.output_dir / 'satellite_reference.json'
        
        metadata_path = self.dataset_path / self.json_name
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.test_data = self.load_test_data()
        
        for images, _ in self.test_data.dataset.take(1):
            self.input_shape = images.shape[1:]
            break
        print(f"Input shape determined from test dataset: {self.input_shape}")
        self.model = self.load_model()
        self.all_polygons = []


    def load_model(self):
        if self.model_name == 'unet':
            loaded_model = helpers.modelup.unet(self.input_shape) if self.upscale else helpers.model.unet(self.input_shape)
        elif self.model_name == 'attunet':
            loaded_model = helpers.modelup.attunet(self.input_shape) if self.upscale else helpers.model.attunet(self.input_shape)
        elif self.model_name == 'resunet':
            loaded_model = helpers.modelup.resunet(self.input_shape) if self.upscale else helpers.model.resunet(self.input_shape)
        else:
            raise ValueError(f"Model {self.model_name} not implemented")
        
        loaded_model.load_weights(self.weights_path)
        loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return loaded_model

    def load_test_data(self):
        test_path = str(self.dataset_path / "test")
        return LoadandAugment(dataset_path=test_path, data_type="test", batch=5, augmentation=False)

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
                binary_mask = (prediction[:,:,0] > 0.5).astype(np.uint8)
                profile.update(count=1, dtype=rasterio.uint8)
                with rasterio.open(output_tif, 'w', **profile) as dst:
                    dst.write(binary_mask, 1)
                
                self.extract_and_save_polygons(output_tif)
            else:
                print(f"Warning: Metadata not found for prediction {i}")

    def extract_polygons(self, mask, transform):
        inverted_mask = ~mask.astype(bool)
        shapes = features.shapes(inverted_mask.astype('uint8'), transform=transform)
        polygons = [shape(geom) for geom, value in shapes if value == 1]
        return polygons

    def create_geodataframe(self, polygon_data, crs):
        gdf = gpd.GeoDataFrame(polygon_data, crs=crs)
        return gdf

    def plot_and_save_contours(self, mask, polygons, output_file):
        plt.figure(figsize=(20, 20))
        plt.imshow(mask, cmap='gray')
        for polygon in polygons:
            x, y = polygon.exterior.xy
            plt.plot(x, y, color='red', linewidth=1)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        if not os.path.exists(output_file):
            print(f"Warning: Failed to save contour image to {output_file}")
        else:
            print(f"Contour image saved to {output_file}")

    def process_polygons(self, gdf):
        if any(gdf.geometry.type == 'MultiPolygon'):
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        
        gdf['area'] = gdf.geometry.area
        gdf = gdf[gdf['area'] >= 5000]
        gdf = gdf.drop(columns=['area'])
        
        return gdf

    def extract_and_save_polygons(self, tif_file):
        with rasterio.open(tif_file) as src:
            image = src.read(1)
            mask = image > 0.5
            polygons = self.extract_polygons(mask, src.transform)
            crs = src.crs

            contour_image_file = self.output_dir / f"{tif_file.stem}_contours.png"
            self.plot_and_save_contours(mask, polygons, contour_image_file)

        polygon_data = [{'id': i, 'geometry': polygon} for i, polygon in enumerate(polygons)]
        
        gdf = self.create_geodataframe(polygon_data, crs)
        
        output_gpkg = self.output_dir / f"{tif_file.stem}.gpkg"
        gdf.to_file(output_gpkg, driver="GPKG")
        print(f"Processed GeoPackage for {tif_file.stem} saved to {output_gpkg}")

        for _, row in gdf.iterrows():
            self.all_polygons.append({
                'file_name': tif_file.stem,
                'geometry': row['geometry']
            })

    def create_final_geodataframe(self):
        if not self.all_polygons:
            print("No polygons extracted. Check your prediction process.")
            return

        gdf = gpd.GeoDataFrame(self.all_polygons)
        
        with rasterio.open(Path(self.metadata[0]['mask'])) as src:
            gdf.set_crs(src.crs, inplace=True)

        gdf = self.process_polygons(gdf)

        output_gpkg = self.output_dir / "prediction_combined.gpkg"
        gdf.to_file(output_gpkg, driver="GPKG")
        print(f"Processed combined GeoPackage saved to {output_gpkg}")

    def create_satellite_reference(self):
        print("Creating reference for satellite images...")
        satellite_files = list(self.satellite_images_path.glob('*.tif'))
        
        if not satellite_files:
            raise FileNotFoundError(f"No .tif files found in {self.satellite_images_path}")

        metadata = []

        for file in satellite_files:
            with rasterio.open(file) as src:
                bounds = src.bounds
                parcel_name = file.stem
                canton = parcel_name.split('_')[0]

                metadata.append({
                    'parcel_name': parcel_name,
                    'canton': canton,
                    'file_path': str(file),
                    'x_min': bounds.left,
                    'y_min': bounds.bottom,
                    'x_max': bounds.right,
                    'y_max': bounds.top,
                    'width': src.width,
                    'height': src.height,
                    'crs': src.crs.to_string()
                })

        with open(self.satellite_reference_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Satellite reference created at {self.satellite_reference_path}")

    def run(self):
        self.evaluate()
        self.predict_and_save()
        self.create_final_geodataframe()
        self.create_satellite_reference()

