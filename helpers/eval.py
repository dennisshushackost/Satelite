import geopandas as gpd
import pandas as pd
from pathlib import Path
import csv
import warnings
import glob

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

    def __init__(self, original_dir, predicted_dir, canton_name):
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
        self.canton_name = canton_name

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
        predicted_files = glob.glob(str(self.predicted_dir / f'{self.canton_name}_CH_parcel_*.gpkg'))

        # Initialize canton-wide GeoDataFrames
        canton_analysis_gdf = gpd.GeoDataFrame()
        canton_overprediction_gdf = gpd.GeoDataFrame()
        canton_lowiou_gdf = gpd.GeoDataFrame()

        for predicted_file in predicted_files:
            filename = Path(predicted_file).name
            original_file = self.original_dir / filename

            if original_file.exists():
                try:
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

                    # Create analysis GeoDataFrame, explode MultiPolygons, and filter small parcels
                    analysis_gdf = self.explode_multipolygons(self.original_gdf)
                    analysis_gdf = analysis_gdf[analysis_gdf.geometry.area > 5000]
                    
                    # Add canton name and Auschnitt name to the analysis GeoDataFrame
                    analysis_gdf['Canton'] = self.canton_name
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
                        # Normal IoU calculation for binary classification:
                        # iou = true_positive / (true_positive + false_positive + false_negative)
                        # Adjusted IoU calculation
                        adjusted_iou = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                        adjusted_ious.append(adjusted_iou)
                    
                    # Add calculated metrics to analysis GeoDataFrame
                    analysis_gdf['True Positive (m²)'] = true_positives
                    analysis_gdf['False Negative (m²)'] = false_negatives
                    analysis_gdf['Recall'] = adjusted_ious
                    
                    # Add Low IoU flag
                    analysis_gdf['Low Recall'] = analysis_gdf['Recall'] <= 0.7
                    
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
                        overpredicted_gdf['Canton'] = self.canton_name  # Add canton name to overpredicted GeoDataFrame
                        overpredicted_gdf['Auschnitt'] = Path(predicted_file).stem  # Add Auschnitt name to overpredicted GeoDataFrame
                        analysis_gdf = pd.concat([analysis_gdf, overpredicted_gdf], ignore_index=True)

                    # Ensure a parcel is only overpredicted if it has NULL as Adjusted IoU
                    analysis_gdf.loc[analysis_gdf['Recall'].notnull(), 'Overpredicted'] = False

                    # Prepare output filename
                    output_filename = Path(filename).stem
                    
                    # Save analysis results
                    analysis_gdf.to_file(f"{self.output_dir/output_filename}_analysis.gpkg", driver="GPKG")

                    # Save overprediction results
                    if not overpredicted_gdf.empty:
                        overpredicted_gdf.to_file(f"{self.output_dir/output_filename}_overprediction.gpkg", driver="GPKG")

                    # Save low IoU results
                    low_iou_gdf = analysis_gdf[analysis_gdf['Low Recall']]
                    low_iou_gdf.to_file(f"{self.output_dir/output_filename}_lowrecall.gpkg", driver="GPKG")

                    # Calculate statistics using the analysis_gdf
                    original_total_area = self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum()
                    overpredicted_area = analysis_gdf[analysis_gdf['Overpredicted']].geometry.area.sum()
                    low_iou_area = analysis_gdf[analysis_gdf['Low Recall']].geometry.area.sum()

                    total_error = (overpredicted_area + low_iou_area) / original_total_area if original_total_area > 0 else 0
                    overprediction_error = overpredicted_area / original_total_area if original_total_area > 0 else 0
                    iou_error = low_iou_area / original_total_area if original_total_area > 0 else 0

                    parcel_statistics = {
                        'Parcel Name': output_filename,
                        'Canton': self.canton_name,  # Add canton name to parcel statistics
                        'Auschnitt': Path(predicted_file).stem,  # Add Auschnitt name to parcel statistics
                        'Original Total Area (m²)': original_total_area,
                        'Overpredicted Area (m²)': overpredicted_area,
                        'Low Recall Area (m²)': low_iou_area,
                        'Total Error': total_error,
                        'Overprediction Error': overprediction_error,
                        'Recall Error': iou_error
                    }

                    statistics.append(parcel_statistics)

                    # Append to canton-wide GeoDataFrames
                    canton_analysis_gdf = pd.concat([canton_analysis_gdf, analysis_gdf], ignore_index=True)
                    canton_overprediction_gdf = pd.concat([canton_overprediction_gdf, overpredicted_gdf], ignore_index=True)
                    canton_lowiou_gdf = pd.concat([canton_lowiou_gdf, low_iou_gdf], ignore_index=True)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

        # Save canton-wide GeoDataFrames
        canton_analysis_gdf.to_file(f"{self.output_dir}/{self.canton_name}_analysis.gpkg", driver="GPKG")
        canton_overprediction_gdf.to_file(f"{self.output_dir}/{self.canton_name}_overprediction.gpkg", driver="GPKG")
        canton_lowiou_gdf.to_file(f"{self.output_dir}/{self.canton_name}_lowrecall.gpkg", driver="GPKG")

        # Calculate canton-wide statistics
        if statistics:
            canton_name = self.canton_name
            original_total_area = sum(stat['Original Total Area (m²)'] for stat in statistics)
            overpredicted_area = sum(stat['Overpredicted Area (m²)'] for stat in statistics)
            low_iou_area = sum(stat['Low Recall Area (m²)'] for stat in statistics)
            avg_total_error = sum(stat['Total Error'] for stat in statistics) / len(statistics)
            avg_overprediction_error = sum(stat['Overprediction Error'] for stat in statistics) / len(statistics)
            avg_iou_error = sum(stat['Recall Error'] for stat in statistics) / len(statistics)

            canton_statistics = {
                'Parcel Name': '',  # Empty string to create a blank row
                'Canton': '',  # Empty string to create a blank row
                'Auschnitt': '',  # Empty string to create a blank row
                'Original Total Area (m²)': '',
                'Overpredicted Area (m²)': '',
                'Low Recall Area (m²)': '',
                'Total Error': '',
                'Overprediction Error': '',
                'Recall Error': ''
            }
            statistics.append(canton_statistics)  # Add a blank row
            
            canton_statistics = {
            'Parcel Name': 'Canton Name',
            'Canton': 'Canton',
            'Auschnitt': 'Auschnitt',
            'Original Total Area (m²)': 'Total Area of Parcels',
            'Overpredicted Area (m²)': 'Overpredicted Area of Parcels',
            'Low Recall Area (m²)': 'Low Recall Area of Parcels',
            'Total Error': 'Total Average Error of Parcels',
            'Overprediction Error': 'Average Overprediction Error of Parcels',
            'Recall Error': 'Average Recall Error of Parcels'
        }
            statistics.append(canton_statistics)  # Add the new column names

            canton_statistics = {
                'Parcel Name': canton_name,
                'Canton': canton_name,
                'Auschnitt': 'All',
                'Original Total Area (m²)': original_total_area,
                'Overpredicted Area (m²)': overpredicted_area,
                'Low Recall Area (m²)': low_iou_area,
                'Total Error': avg_total_error,
                'Overprediction Error': avg_overprediction_error,
                'Recall Error': avg_iou_error
            }
            statistics.append(canton_statistics)  # Add the canton-wide statistics

            # Save all statistics to a single CSV file
            with open(f"{self.output_dir}/statistics.csv", 'w', newline='') as csvfile:
                fieldnames = statistics[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for stat in statistics:
                    writer.writerow(stat)
        else:
            print("No statistics generated. Check if there are matching files in the directories.")
                    
# Usage example
evaluator = ParcelEvaluator("/workspaces/Satelite/data/parcels",
                            "/workspaces/Satelite/data/experiment/predictions",
                            "ZH")  # Assuming "ZH" is the canton name
evaluator.analyze_parcels()