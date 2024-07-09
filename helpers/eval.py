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
        predicted_files = glob.glob(str(self.predicted_dir / f'{self.canton_name}_ZH_parcel_*.gpkg'))

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
                    
                    # Identify overpredicted areas
                    overpredicted_gdf = self.identify_overpredictions()
                    
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
                        adjusted_iou = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                        adjusted_ious.append(adjusted_iou)
                    
                    # Add calculated metrics to analysis GeoDataFrame
                    analysis_gdf['True Positive (m²)'] = true_positives
                    analysis_gdf['False Negative (m²)'] = false_negatives
                    analysis_gdf['Adjusted IoU'] = adjusted_ious
                    
                    # Add Low IoU flag
                    analysis_gdf['Low IoU'] = analysis_gdf['Adjusted IoU'] <= 0.7
                    
                    # Add overprediction flag to analysis GeoDataFrame
                    if not overpredicted_gdf.empty:
                        analysis_gdf['Overpredicted'] = analysis_gdf.geometry.intersects(overpredicted_gdf.unary_union)
                    else:
                        analysis_gdf['Overpredicted'] = False

                    # Add overpredicted areas to analysis GeoDataFrame
                    if not overpredicted_gdf.empty:
                        overpredicted_gdf['Adjusted IoU'] = None
                        overpredicted_gdf['Overpredicted'] = True
                        overpredicted_gdf['Low IoU'] = False
                        overpredicted_gdf['True Positive (m²)'] = None
                        overpredicted_gdf['False Negative (m²)'] = None
                        analysis_gdf = pd.concat([analysis_gdf, overpredicted_gdf], ignore_index=True)

                    # Ensure a parcel is only overpredicted if it has NULL as Adjusted IoU
                    analysis_gdf.loc[analysis_gdf['Adjusted IoU'].notnull(), 'Overpredicted'] = False

                    # Prepare output filename
                    output_filename = Path(filename).stem
                    
                    # Save analysis results
                    analysis_gdf.to_file(f"{self.output_dir/output_filename}_analysis.gpkg", driver="GPKG")

                    # Save overprediction results
                    if not overpredicted_gdf.empty:
                        overpredicted_gdf.to_file(f"{self.output_dir/output_filename}_overprediction.gpkg", driver="GPKG")

                    # Save low IoU results
                    low_iou_gdf = analysis_gdf[analysis_gdf['Low IoU']]
                    low_iou_gdf.to_file(f"{self.output_dir/output_filename}_lowiou.gpkg", driver="GPKG")

                    # Calculate statistics using the analysis_gdf
                    original_total_area = self.original_gdf[self.original_gdf.geometry.area > 5000].geometry.area.sum()
                    overpredicted_area = analysis_gdf[analysis_gdf['Overpredicted']].geometry.area.sum()
                    low_iou_area = analysis_gdf[analysis_gdf['Low IoU']].geometry.area.sum()

                    total_error = (overpredicted_area + low_iou_area) / original_total_area if original_total_area > 0 else 0
                    overprediction_error = overpredicted_area / original_total_area if original_total_area > 0 else 0
                    iou_error = low_iou_area / original_total_area if original_total_area > 0 else 0

                    parcel_statistics = {
                        'Parcel Name': output_filename,
                        'Original Total Area (m²)': original_total_area,
                        'Overpredicted Area (m²)': overpredicted_area,
                        'Low IoU Area (m²)': low_iou_area,
                        'Total Error': total_error,
                        'Overprediction Error': overprediction_error,
                        'IoU Error': iou_error
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
        canton_lowiou_gdf.to_file(f"{self.output_dir}/{self.canton_name}_lowiou.gpkg", driver="GPKG")

        # Calculate canton-wide statistics
        original_files = glob.glob(str(self.original_dir / f'{self.canton_name}_ZH_parcel_*.gpkg'))
        print(f"Found {len(original_files)} original parcel files for canton-wide analysis")

        if original_files:
            all_original_parcels = gpd.GeoDataFrame(pd.concat([self.load_gdf(file) for file in original_files], ignore_index=True))
            if not all_original_parcels.empty:
                canton_original_total_area = all_original_parcels[all_original_parcels.geometry.area > 5000].geometry.area.sum()
                canton_overpredicted_area = canton_analysis_gdf[canton_analysis_gdf['Overpredicted']].geometry.area.sum()
                canton_low_iou_area = canton_analysis_gdf[canton_analysis_gdf['Low IoU']].geometry.area.sum()

                canton_total_error = (canton_overpredicted_area + canton_low_iou_area) / canton_original_total_area if canton_original_total_area > 0 else 0
                canton_overprediction_error = canton_overpredicted_area / canton_original_total_area if canton_original_total_area > 0 else 0
                canton_iou_error = canton_low_iou_area / canton_original_total_area if canton_original_total_area > 0 else 0

                canton_statistics = {
                    'Parcel Name': self.canton_name,
                    'Original Total Area (m²)': canton_original_total_area,
                    'Overpredicted Area (m²)': canton_overpredicted_area,
                    'Low IoU Area (m²)': canton_low_iou_area,
                    'Total Error': canton_total_error,
                    'Overprediction Error': canton_overprediction_error,
                    'IoU Error': canton_iou_error
                }

                statistics.append(canton_statistics)
            else:
                print("Warning: All original parcels are empty after concatenation")
        else:
            print("Warning: No original parcel files found for canton-wide statistics calculation")

        # Save all statistics to a single CSV file
        if statistics:
            with open(f"{self.output_dir}/statistics.csv", 'w', newline='') as csvfile:
                fieldnames = statistics[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for stat in statistics:
                    writer.writerow(stat)
        else:
            print("No statistics generated. Check if there are matching files in the directories.")

# Usage example
evaluator = ParcelEvaluator("/workspaces/Satelite/data/parcels/",
                            "/workspaces/Satelite/data/experiment/predictions/",
                            "ZH")  # Assuming "ZH" is the canton name
evaluator.analyze_parcels()