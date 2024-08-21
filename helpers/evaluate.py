import geopandas as gpd
import pandas as pd
from pathlib import Path
import csv
import warnings
import itertools
import glob
import re 

warnings.filterwarnings("ignore")

class ParcelEvaluator:
    def __init__(self, original_dir, predicted_dir):
        self.original_dir = Path(original_dir)
        self.predicted_dir = Path(predicted_dir)
        self.output_dir = self.create_folder()

    def create_folder(self):
        output_dir = self.predicted_dir.parent / 'evaluation'
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def load_gdf(self, path):
        try:
            gdf = gpd.read_file(path)
            if gdf.empty:
                print(f"Warning: Empty GeoDataFrame loaded from {path}")
            return gdf
        except Exception as e:
            print(f"Error loading GeoDataFrame from {path}: {e}")
            return None

    def explode_multipolygons(self, gdf):
        if any(gdf.geometry.type == 'MultiPolygon'):
            return gdf.explode(index_parts=False).reset_index(drop=True)
        return gdf

    def calculate_overlap(self, original_polygon, predicted_polygons):
        if predicted_polygons.empty:
            return 0, original_polygon.area
        
        intersection = gpd.overlay(gpd.GeoDataFrame(geometry=[original_polygon]), 
                                   predicted_polygons, how='intersection')
        true_positive = intersection.area.sum()
        false_negative = original_polygon.area - true_positive
        return true_positive, false_negative

    def identify_overpredictions(self):
        original_combined = self.original_gdf.unary_union
        predicted_combined = self.predicted_gdf.unary_union
        overpredicted_area = predicted_combined.difference(original_combined)
        overpredicted_gdf = gpd.GeoDataFrame(geometry=[overpredicted_area], crs=self.original_gdf.crs)
        overpredicted_gdf = self.explode_multipolygons(overpredicted_gdf)
        overpredicted_gdf = overpredicted_gdf[overpredicted_gdf.geometry.area > 5000]
        return overpredicted_gdf

    def analyze_parcels(self):
        statistics = []
        predicted_files = glob.glob(str(self.predicted_dir / f'*_CH_parcel_*.gpkg'))
        print(f"Found {len(predicted_files)} predicted files")

        canton_names = set(re.match(r"([A-Z]{2})_CH_parcel_\d+.gpkg", Path(f).name).group(1) for f in predicted_files)
        print(f"Found {len(canton_names)} cantons: {', '.join(canton_names)}")
        
        all_analysis_gdf = gpd.GeoDataFrame()
        all_lowrecall_gdf = gpd.GeoDataFrame()
        all_overprediction_gdf = gpd.GeoDataFrame()
        all_original_gdf = gpd.GeoDataFrame()

        for canton_name in canton_names:
            print(f"Processing canton: {canton_name}")
            canton_analysis_gdf = gpd.GeoDataFrame()
            canton_overprediction_gdf = gpd.GeoDataFrame()
            canton_lowiou_gdf = gpd.GeoDataFrame()
            
            canton_predicted_files = [f for f in predicted_files if f.__contains__(f"{canton_name}_CH_parcel_")]
            print(f"Found {len(canton_predicted_files)} files for canton {canton_name}")

            for predicted_file in canton_predicted_files:
                filename = Path(predicted_file).name
                original_file = self.original_dir / filename

                if original_file.exists():
                    try:
                        print(f"Processing file: {filename}")
                        self.original_gdf = self.load_gdf(original_file)
                        self.predicted_gdf = self.load_gdf(predicted_file)

                        if self.original_gdf is None or self.predicted_gdf is None or self.original_gdf.empty or self.predicted_gdf.empty:
                            print(f"Skipping {filename} due to loading error or empty GeoDataFrame")
                            continue

                        if self.original_gdf.crs is None:
                            self.original_gdf.set_crs(epsg=32632, inplace=True)
                        if self.original_gdf.crs != self.predicted_gdf.crs:
                            self.predicted_gdf = self.predicted_gdf.to_crs(self.original_gdf.crs)

                        self.original_gdf['canton'] = canton_name
                        self.original_gdf['excerpt'] = Path(predicted_file).stem
                        all_original_gdf = pd.concat([all_original_gdf, self.original_gdf], ignore_index=True)

                        analysis_gdf = self.explode_multipolygons(self.original_gdf)
                        analysis_gdf = analysis_gdf[analysis_gdf.geometry.area > 5000]
                        
                        analysis_gdf['canton'] = canton_name
                        analysis_gdf['excerpt'] = Path(predicted_file).stem

                        overpredicted_gdf = self.identify_overpredictions()

                        overpredicted_gdf['excerpt'] = Path(predicted_file).stem

                        true_positives = []
                        false_negatives = []
                        adjusted_ious = []
                        for idx, original_row in analysis_gdf.iterrows():
                            original_polygon = original_row.geometry
                            intersecting_predictions = self.predicted_gdf[self.predicted_gdf.intersects(original_polygon)]
                            true_positive, false_negative = self.calculate_overlap(original_polygon, intersecting_predictions)
                            true_positives.append(true_positive)
                            false_negatives.append(false_negative)
                            adjusted_iou = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                            adjusted_ious.append(adjusted_iou)

                        analysis_gdf['true_positive'] = true_positives
                        analysis_gdf['false_negative'] = false_negatives
                        analysis_gdf['recall'] = adjusted_ious

                        analysis_gdf['low_recall'] = analysis_gdf['recall'] <= 0.7

                        low_iou_gdf = analysis_gdf[analysis_gdf['low_recall']]

                        if not overpredicted_gdf.empty:
                            analysis_gdf['overpredicted'] = analysis_gdf.geometry.intersects(overpredicted_gdf.unary_union)
                        else:
                            analysis_gdf['overpredicted'] = False

                        if not overpredicted_gdf.empty:
                            overpredicted_gdf['recall'] = None
                            overpredicted_gdf['overpredicted'] = True
                            overpredicted_gdf['low_recall'] = False
                            overpredicted_gdf['true_positive'] = None
                            overpredicted_gdf['false_negative'] = None
                            overpredicted_gdf['canton'] = canton_name
                            overpredicted_gdf['excerpt'] = Path(predicted_file).stem
                            analysis_gdf = pd.concat([analysis_gdf, overpredicted_gdf], ignore_index=True)

                        analysis_gdf.loc[analysis_gdf['recall'].notnull(), 'overpredicted'] = False

                        file_stats = {
                            'canton': canton_name,
                            'excerpt': Path(predicted_file).stem,
                            'area': self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum(),
                            'overpredicted': overpredicted_gdf['geometry'].area.sum() if not overpredicted_gdf.empty else 0,
                            'low_recall': low_iou_gdf['geometry'].area.sum(),
                            'total_error': ((overpredicted_gdf['geometry'].area.sum() if not overpredicted_gdf.empty else 0) + low_iou_gdf['geometry'].area.sum()) / self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum(),
                            'overprediction_error': overpredicted_gdf['geometry'].area.sum() / self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum(),
                            'recall_error': low_iou_gdf['geometry'].area.sum() / self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum()
                        }
                        statistics.append(file_stats)

                        if not analysis_gdf.empty:
                            canton_analysis_gdf = pd.concat([canton_analysis_gdf, analysis_gdf], ignore_index=True)
                        if not overpredicted_gdf.empty:
                            canton_overprediction_gdf = pd.concat([canton_overprediction_gdf, overpredicted_gdf], ignore_index=True)
                        if not low_iou_gdf.empty:
                            canton_lowiou_gdf = pd.concat([canton_lowiou_gdf, low_iou_gdf], ignore_index=True)

                        auschnitt_output_dir = self.output_dir / canton_name
                        auschnitt_output_dir.mkdir(exist_ok=True)
                        (auschnitt_output_dir / 'analysis').mkdir(exist_ok=True)
                        (auschnitt_output_dir / 'lowrecall').mkdir(exist_ok=True)
                        (auschnitt_output_dir / 'overprediction').mkdir(exist_ok=True)

                        auschnitt_name = Path(predicted_file).stem
                        if not analysis_gdf.empty:
                            analysis_gdf.to_file(f"{auschnitt_output_dir}/analysis/{auschnitt_name}_analysis.gpkg", driver="GPKG")
                        if not low_iou_gdf.empty:
                            low_iou_gdf.to_file(f"{auschnitt_output_dir}/lowrecall/{auschnitt_name}_lowrecall.gpkg", driver="GPKG")
                        if not overpredicted_gdf.empty:
                            overpredicted_gdf.to_file(f"{auschnitt_output_dir}/overprediction/{auschnitt_name}_overprediction.gpkg", driver="GPKG")

                        print(f"Successfully processed {filename}")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        continue

            print(f"Finished processing {canton_name}")

            if not canton_analysis_gdf.empty:
                all_analysis_gdf = pd.concat([all_analysis_gdf, canton_analysis_gdf], ignore_index=True)
            if not canton_lowiou_gdf.empty:
                all_lowrecall_gdf = pd.concat([all_lowrecall_gdf, canton_lowiou_gdf], ignore_index=True)
            if not canton_overprediction_gdf.empty:
                all_overprediction_gdf = pd.concat([all_overprediction_gdf, canton_overprediction_gdf], ignore_index=True)

        if not all_analysis_gdf.empty:
            all_analysis_gdf.set_geometry('geometry', inplace=True)
            all_analysis_gdf.to_file(f"{self.output_dir}/analysis.gpkg", driver="GPKG")
        if not all_lowrecall_gdf.empty:
            all_lowrecall_gdf.set_geometry('geometry', inplace=True)
            all_lowrecall_gdf.to_file(f"{self.output_dir}/lowrecall.gpkg", driver="GPKG")
        if not all_overprediction_gdf.empty:
            all_overprediction_gdf.set_geometry('geometry', inplace=True)
            all_overprediction_gdf.to_file(f"{self.output_dir}/overprediction.gpkg", driver="GPKG")
        if not all_original_gdf.empty:
            all_original_gdf.set_geometry('geometry', inplace=True)
            all_original_gdf.to_file(f"{self.output_dir}/all_original_parcels.gpkg", driver="GPKG")
            print(f"Saved all original parcels to {self.output_dir}/all_original_parcels.gpkg")

        print(f"Total statistics gathered: {len(statistics)}")

        if statistics:
            for stat in statistics:
                for key, value in stat.items():
                    if isinstance(value, (int, float)):
                        stat[key] = round(value, 4)

            with open(f"{self.output_dir}/statistics.csv", 'w', newline='') as csvfile:
                fieldnames = list(statistics[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for stat in statistics:
                    writer.writerow(stat)

            overall_statistics = []
            statistics.sort(key=lambda x: x['canton'])
            for canton, canton_stats in itertools.groupby(statistics, key=lambda x: x['canton']):
                canton_stats = list(canton_stats)
                original_total_area = sum(stat['area'] for stat in canton_stats)
                overpredicted_area = sum(stat['overpredicted'] for stat in canton_stats)
                low_iou_area = sum(stat['low_recall'] for stat in canton_stats)
                avg_total_error = sum(stat['total_error'] for stat in canton_stats) / len(canton_stats)
                avg_overprediction_error = sum(stat['overprediction_error'] for stat in canton_stats) / len(canton_stats)
                avg_iou_error = sum(stat['recall_error'] for stat in canton_stats) / len(canton_stats)

                canton_statistics = {
                    'canton': canton,
                    'area': round(original_total_area, 4),
                    'overpredicted': round(overpredicted_area, 4),
                    'low_recall': round(low_iou_area, 4),
                    'average_total_error': round(avg_total_error, 4),
                    'average_overprediction_error': round(avg_overprediction_error, 4),
                    'average_recall_error': round(avg_iou_error, 4)
                }
                overall_statistics.append(canton_statistics)

            overall_original_area = sum(stat['area'] for stat in statistics)
            overall_overpredicted_area = sum(stat['overpredicted'] for stat in statistics)
            overall_low_iou_area = sum(stat['low_recall'] for stat in statistics)
            overall_total_error = sum(stat['total_error'] for stat in statistics) / len(statistics)
            overall_overprediction_error = sum(stat['overprediction_error'] for stat in statistics) / len(statistics)
            overall_iou_error = sum(stat['recall_error'] for stat in statistics) / len(statistics)
            
            overall_statistics.append({
                    'canton': 'ch',
                    'area': round(overall_original_area, 4),
                    'overpredicted': round(overall_overpredicted_area, 4),
                    'low_recall': round(overall_low_iou_area, 4),
                    'average_total_error': round(overall_total_error, 4),
                    'average_overprediction_error': round(overall_overprediction_error, 4),
                    'average_recall_error': round(overall_iou_error, 4)
                })

            with open(f"{self.output_dir}/overall_statistics.csv", 'w', newline='') as csvfile:
                fieldnames = ['canton', 'area', 'overpredicted', 'low_recall', 'average_total_error', 'average_overprediction_error', 'average_recall_error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for stat in overall_statistics:
                    writer.writerow(stat)
        else:
            print("No statistics generated. Check if there are matching files in the directories.")

# Usage example:
# evaluator = ParcelEvaluator('path/to/original/data', 'path/to/predicted/data')
# evaluator.analyze_parcels()