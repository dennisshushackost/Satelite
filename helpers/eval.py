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
    """
    A class to evaluate and analyze parcel predictions against original parcels.
    This class compares original parcels with predicted parcels, calculates metrics,
    and identifies overpredictions and low-performing predictions.
    
    For evaluating the performance of the semantic segmentation model in this scenario, with the particular concern about not penalizing the model for overpredictions due to the spatial resolution and connectivity issues, we should consider metrics that focus on the accuracy and quality of the predictions with respect to the true positives and the handling of false negatives. As such we will be using the following metrics:
    - True Positive: Correctly predicted area of the original parcel
    - False Negative: Area of the original parcel not covered by the predicted parcels

    We will thus use an adjusted IoU metric without considering false positives:
    Original IoU = True Positive / (True Positive + False Positive + False Negative)
    Adjusted IoU = True Positive / (True Positive + False Negative) = Recall = measures the ability of the model to find all the relevant cases (all original pixels).
    1. Spatial resolution issues: The segmentation images may not have enough detail to precisely delineate parcel boundaries.
    2. Connectivity problems: Predicted parcels may be erroneously connected due to imprecise boundary detection.
    
    The adjusted IoU (recall) focuses on how well the model identifies the actual parcel areas. A threshold of 0.7 is used to identify low IoU areas.
        - This means areas which the model has trouble identifying correctly.
    The overprediction error captures areas where the model predicts parcels that don't exist in the original data.
        - This is calculated as the area of overpredicted parcels divided by the total area of original parcels.
        - Shows if the model hallucinates parcels that don't exist / or should exist based on the spatelite images.
    The Total Error is the sum of the overprediction error and the IoU error.
    """

    def __init__(self, original_dir, predicted_dir):
        """
        Initialize the ParcelEvaluator with directories containing original and predicted parcel data.
        
        Parameters:
        - original_dir: Path to the directory containing original parcel data
        - predicted_dir: Path to the directory containing predicted parcel data
        - canton_name: Name of the canton being analyzed
        """
        self.original_dir = Path(original_dir)
        self.predicted_dir = Path(predicted_dir)
        self.output_dir = self.create_folder()

    def create_folder(self):
        """Create output folder for evaluation results."""
        output_dir = self.predicted_dir.parent / 'evaluation'
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def load_gdf(self, path):
        """
        Load GeoDataFrame from file.
        
        Parameters:
        - path: Path to the GeoPackage file
        
        Returns:
        - GeoDataFrame or None if there's an error
        """
        try:
            gdf = gpd.read_file(path)
            if gdf.empty:
                print(f"Warning: Empty GeoDataFrame loaded from {path}")
            return gdf
        except Exception as e:
            print(f"Error loading GeoDataFrame from {path}: {e}")
            return None

    def explode_multipolygons(self, gdf):
        """
        Explode MultiPolygons in a GeoDataFrame into individual Polygons.
        This is necessary to ensure each geometry is a single Polygon for accurate analysis.
        
        Parameters:
        - gdf: Input GeoDataFrame
        
        Returns:
        - GeoDataFrame with exploded MultiPolygons
        """
        if any(gdf.geometry.type == 'MultiPolygon'):
            return gdf.explode(index_parts=False).reset_index(drop=True)
        return gdf

    def calculate_overlap(self, original_polygon, predicted_polygons):
        """
        Calculate the overlap between an original polygon and predicted polygons.
        
        Parameters:
        - original_polygon: A single polygon representing the ground truth or original parcel.
        - predicted_polygons: Predicted parcel polygons that intersect with the original polygon.
        
        Returns:
        - true_positive: Area of intersection between original and predicted polygons
        - false_negative: Area of original polygon not covered by predicted polygons
        """
        if predicted_polygons.empty:
            return 0, original_polygon.area
        
        # Calculate the intersection between original and predicted polygons
        intersection = gpd.overlay(gpd.GeoDataFrame(geometry=[original_polygon]), 
                                   predicted_polygons, how='intersection')
        true_positive = intersection.area.sum()
        false_negative = original_polygon.area - true_positive
        return true_positive, false_negative

    def identify_overpredictions(self):
        """
        Identify overpredicted areas by comparing original and predicted parcels.
        This function finds areas that are predicted as parcels but don't exist in the original data.
        It returns only overpredicted areas larger than 5000 m².
        
        Returns:
        - GeoDataFrame containing overpredicted areas
        """
        # Use unary_union to combine all original and predicted polygons into a single polygon
        original_combined = self.original_gdf.unary_union
        predicted_combined = self.predicted_gdf.unary_union
        # Find areas that are in predicted_combined but not in original_combined
        overpredicted_area = predicted_combined.difference(original_combined)
        overpredicted_gdf = gpd.GeoDataFrame(geometry=[overpredicted_area], crs=self.original_gdf.crs)
        overpredicted_gdf = self.explode_multipolygons(overpredicted_gdf)
        # Filter out small overpredicted areas:
        overpredicted_gdf = overpredicted_gdf[overpredicted_gdf.geometry.area > 5000]
        return overpredicted_gdf

    
    def analyze_parcels(self):
        """
        Analyze parcels for all matching files in the directories.
        This function processes each pair of original and predicted parcel files,
        calculates various metrics, and saves the results.
        """
        statistics = []
        predicted_files = glob.glob(str(self.predicted_dir / f'*_CH_parcel_*.gpkg'))
        print(f"Found {len(predicted_files)} predicted files")

        # Extract canton names from predicted files
        canton_names = set(re.match(r"([A-Z]{2})_CH_parcel_\d+.gpkg", Path(f).name).group(1) for f in predicted_files)
        print(f"Found {len(canton_names)} cantons: {', '.join(canton_names)}")
        
        # Initialize combined GeoDataFrames
        all_analysis_gdf = gpd.GeoDataFrame()
        all_lowrecall_gdf = gpd.GeoDataFrame()
        all_overprediction_gdf = gpd.GeoDataFrame()
        all_original_gdf = gpd.GeoDataFrame()  # New GeoDataFrame for all original parcels

        for canton_name in canton_names:
            print(f"Processing canton: {canton_name}")
            # Initialize canton-wide GeoDataFrames
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
                        # Load original and predicted GeoDataFrames
                        self.original_gdf = self.load_gdf(original_file)
                        self.predicted_gdf = self.load_gdf(predicted_file)

                        if self.original_gdf is None or self.predicted_gdf is None or self.original_gdf.empty or self.predicted_gdf.empty:
                            print(f"Skipping {filename} due to loading error or empty GeoDataFrame")
                            continue

                        # Ensure CRS is set and matching
                        if self.original_gdf.crs is None:
                            self.original_gdf.set_crs(epsg=32632, inplace=True)
                        if self.original_gdf.crs != self.predicted_gdf.crs:
                            self.predicted_gdf = self.predicted_gdf.to_crs(self.original_gdf.crs)

                        # Add all original parcels to the combined GeoDataFrame
                        self.original_gdf['Canton'] = canton_name
                        self.original_gdf['Auschnitt'] = Path(predicted_file).stem
                        all_original_gdf = pd.concat([all_original_gdf, self.original_gdf], ignore_index=True)

                        # Create analysis GeoDataFrame, explode MultiPolygons, and filter small parcels
                        analysis_gdf = self.explode_multipolygons(self.original_gdf)
                        analysis_gdf = analysis_gdf[analysis_gdf.geometry.area > 5000]
                        
                        # Add canton name and Auschnitt name to the analysis GeoDataFrame
                        analysis_gdf['canton'] = canton_name
                        analysis_gdf['Auschnitt'] = Path(predicted_file).stem

                        # Identify overpredicted areas
                        overpredicted_gdf = self.identify_overpredictions()

                        # Add Auschnitt name to the overpredicted GeoDataFrame
                        overpredicted_gdf['Auschnitt'] = Path(predicted_file).stem

                        # Calculate True Positive, False Negative, and Adjusted IoU for each parcel
                        true_positives = []
                        false_negatives = []
                        adjusted_ious = []
                        for idx, original_row in analysis_gdf.iterrows():
                            original_polygon = original_row.geometry
                            # Find predicted parcels that intersect with the original polygon:
                            intersecting_predictions = self.predicted_gdf[self.predicted_gdf.intersects(original_polygon)]
                            true_positive, false_negative = self.calculate_overlap(original_polygon, intersecting_predictions)
                            true_positives.append(true_positive)
                            false_negatives.append(false_negative)
                            # Adjusted IoU calculation
                            adjusted_iou = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                            adjusted_ious.append(adjusted_iou)

                        # Add calculated metrics to analysis GeoDataFrame
                        analysis_gdf['True Positive (m²)'] = true_positives
                        analysis_gdf['False Negative (m²)'] = false_negatives
                        analysis_gdf['Recall'] = adjusted_ious

                        # Add Low IoU flag
                        analysis_gdf['Low Recall'] = analysis_gdf['Recall'] <= 0.7

                        # Create low_iou_gdf
                        low_iou_gdf = analysis_gdf[analysis_gdf['Low Recall']]

                        # Add overprediction flag to analysis GeoDataFrame
                        if not overpredicted_gdf.empty:
                            analysis_gdf['Overpredicted'] = analysis_gdf.geometry.intersects(overpredicted_gdf.unary_union)
                        else:
                            analysis_gdf['Overpredicted'] = False

                        # Add overpredicted areas to analysis GeoDataFrame
                        if not overpredicted_gdf.empty:
                            overpredicted_gdf['Recall'] = None
                            overpredicted_gdf['Overpredicted'] = True
                            overpredicted_gdf['Low Recall'] = False
                            overpredicted_gdf['True Positive (m²)'] = None
                            overpredicted_gdf['False Negative (m²)'] = None
                            overpredicted_gdf['Canton'] = canton_name
                            overpredicted_gdf['Auschnitt'] = Path(predicted_file).stem
                            analysis_gdf = pd.concat([analysis_gdf, overpredicted_gdf], ignore_index=True)

                        # Ensure a parcel is only overpredicted if it has NULL as Adjusted IoU
                        analysis_gdf.loc[analysis_gdf['Recall'].notnull(), 'Overpredicted'] = False

                        # Calculate statistics for this file
                        file_stats = {
                            'Canton': canton_name,
                            'Auschnitt': Path(predicted_file).stem,
                            'Area (m²)': self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum(),
                            'Overpredicted (m²)': overpredicted_gdf['geometry'].area.sum() if not overpredicted_gdf.empty else 0,
                            'Low Recall  (m²)': low_iou_gdf['geometry'].area.sum(),
                            'Total Error': ((overpredicted_gdf['geometry'].area.sum() if not overpredicted_gdf.empty else 0) + low_iou_gdf['geometry'].area.sum()) / self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum(),
                            'Overprediction Error': overpredicted_gdf['geometry'].area.sum() / self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum(),
                            'Recall Error': low_iou_gdf['geometry'].area.sum() / self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum()
                        }
                        statistics.append(file_stats)

                        # Save analysis results
                        if not analysis_gdf.empty:
                            canton_analysis_gdf = pd.concat([canton_analysis_gdf, analysis_gdf], ignore_index=True)
                        if not overpredicted_gdf.empty:
                            canton_overprediction_gdf = pd.concat([canton_overprediction_gdf, overpredicted_gdf], ignore_index=True)
                        if not low_iou_gdf.empty:
                            canton_lowiou_gdf = pd.concat([canton_lowiou_gdf, low_iou_gdf], ignore_index=True)

                        # Save individual Auschnitt GeoPackages
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

            # Append canton data to combined GeoDataFrames
            if not canton_analysis_gdf.empty:
                all_analysis_gdf = pd.concat([all_analysis_gdf, canton_analysis_gdf], ignore_index=True)
            if not canton_lowiou_gdf.empty:
                all_lowrecall_gdf = pd.concat([all_lowrecall_gdf, canton_lowiou_gdf], ignore_index=True)
            if not canton_overprediction_gdf.empty:
                all_overprediction_gdf = pd.concat([all_overprediction_gdf, canton_overprediction_gdf], ignore_index=True)

        # Save combined GeoDataFrames
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

        # Print summary of statistics
        print(f"Total statistics gathered: {len(statistics)}")

        # Save parcel statistics to a single CSV file
        if statistics:
            # Round numeric values in statistics to 2 decimal places
            for stat in statistics:
                for key, value in stat.items():
                    if isinstance(value, (int, float)):
                        stat[key] = round(value, 2)

            with open(f"{self.output_dir}/statistics.csv", 'w', newline='') as csvfile:
                fieldnames = list(statistics[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for stat in statistics:
                    writer.writerow(stat)

            # Calculate canton-wide statistics
            overall_statistics = []
            statistics.sort(key=lambda x: x['Canton'])
            for canton, canton_stats in itertools.groupby(statistics, key=lambda x: x['Canton']):
                canton_stats = list(canton_stats)
                original_total_area = sum(stat['Area (m²)'] for stat in canton_stats)
                overpredicted_area = sum(stat['Overpredicted (m²)'] for stat in canton_stats)
                low_iou_area = sum(stat['Low Recall  (m²)'] for stat in canton_stats)
                avg_total_error = sum(stat['Total Error'] for stat in canton_stats) / len(canton_stats)
                avg_overprediction_error = sum(stat['Overprediction Error'] for stat in canton_stats) / len(canton_stats)
                avg_iou_error = sum(stat['Recall Error'] for stat in canton_stats) / len(canton_stats)

                canton_statistics = {
                    'Canton': canton,
                    'Area (m²)': round(original_total_area, 2),
                    'Overpredicted (m²)': round(overpredicted_area, 2),
                    'Low Recall  (m²)': round(low_iou_area, 2),
                    'Average Total Error': round(avg_total_error, 2),
                    'Average Overprediction Error': round(avg_overprediction_error, 2),
                    'Average Recall Error': round(avg_iou_error, 2)
                }
                overall_statistics.append(canton_statistics)

            # Calculate overall statistics for all cantons
            overall_original_area = sum(stat['Area (m²)'] for stat in statistics)
            overall_overpredicted_area = sum(stat['Overpredicted (m²)'] for stat in statistics)
            overall_low_iou_area = sum(stat['Low Recall  (m²)'] for stat in statistics)
            overall_total_error = sum(stat['Total Error'] for stat in statistics) / len(statistics)
            overall_overprediction_error = sum(stat['Overprediction Error'] for stat in statistics) / len(statistics)
            overall_iou_error = sum(stat['Recall Error'] for stat in statistics) / len(statistics)

            overall_statistics.append({
                'Canton': 'CH',
                'Area (m²)': round(overall_original_area, 2),
                'Overpredicted (m²)': round(overall_overpredicted_area, 2),
                'Low Recall  (m²)': round(overall_low_iou_area, 2),
                'Average Total Error': round(overall_total_error, 2),
                'Average Overprediction Error': round(overall_overprediction_error, 2),
                'Average Recall Error': round(overall_iou_error, 2)
            })

            # Save overall statistics to a single CSV file
            with open(f"{self.output_dir}/overall_statistics.csv", 'w', newline='') as csvfile:
                fieldnames = ['Canton', 'Area (m²)', 'Overpredicted (m²)', 'Low Recall  (m²)', 'Average Total Error', 'Average Overprediction Error', 'Average Recall Error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for stat in overall_statistics:
                    writer.writerow(stat)
        else:
            print("No statistics generated. Check if there are matching files in the directories.")