import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap
from datetime import datetime, timedelta
import rasterio
import numpy as np
from PIL import Image
import os

# Initialize session state for map view and other variables
if 'map_view' not in st.session_state:
    st.session_state.map_view = {
        'center': [47.3769, 8.5417],
        'zoom': 9
    }
if 'show_overpredictions' not in st.session_state:
    st.session_state.show_overpredictions = False
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
if 'show_recall_values' not in st.session_state:
    st.session_state.show_recall_values = True
if 'recall_range' not in st.session_state:
    st.session_state.recall_range = (0.0, 1.0)
if 'selected_canton' not in st.session_state:
    st.session_state.selected_canton = 'CH'
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'selected_auschnitte' not in st.session_state:
    st.session_state.selected_auschnitte = []

# Load data
@st.cache_data
def load_data():
    gdf = gpd.read_file('/workspaces/Satelite/data/experiment/evaluation/analysis.gpkg')
    stats_df = pd.read_csv('/workspaces/Satelite/data/experiment/evaluation/statistics.csv')
    predictions_gdf = gpd.read_file('/workspaces/Satelite/data/experiment/predictions/prediction_combined.gpkg')
    original_data_gdf = gpd.read_file('/workspaces/Satelite/data/experiment/evaluation/all_original_parcels.gpkg')
    overall_stats_df = pd.read_csv('/workspaces/Satelite/data/experiment/evaluation/overall_statistics.csv')
    
    # Extract canton from file_name in predictions_gdf
    predictions_gdf['Canton'] = predictions_gdf['file_name'].apply(lambda x: x.split('_')[0])
    
    return gdf.to_crs(epsg=4326), stats_df, predictions_gdf.to_crs(epsg=4326), original_data_gdf.to_crs(epsg=4326), overall_stats_df

gdf, stats_df, predictions_gdf, original_data, overall_stats_df = load_data()

st.title('Field Parcel Segmentation Analysis')

# Sidebar for filters
st.sidebar.header('Filters')
new_recall_range = st.sidebar.slider('Recall Range', 0.0, 1.0, st.session_state.recall_range, 0.1)

# Check if the recall range has changed
if new_recall_range != st.session_state.recall_range:
    current_time = datetime.now()
    if current_time - st.session_state.last_update_time > timedelta(seconds=10):
        st.session_state.recall_range = new_recall_range
        st.session_state.last_update_time = current_time
        st.rerun()
    else:
        pass

new_show_overpredictions = st.sidebar.checkbox('Show Overpredictions', st.session_state.show_overpredictions, help="All Overpredictions (Predicted Parcels - Model Predictions), which are bigger than 5000m2)")
new_show_original_data = st.sidebar.checkbox('Show Original Data', False, help="Original Data without removing 5000m2 parcels")
new_show_predictions = st.sidebar.checkbox('Show Predictions', st.session_state.show_predictions, help="Predictions from the model")
new_show_recall_values = st.sidebar.checkbox('Show Recall Values', st.session_state.show_recall_values, help="Recall values of all parcels > 5000m2")

# Canton selection
cantons = ['CH'] + sorted(gdf['Canton'].unique().tolist())
new_selected_canton = st.selectbox('Select a Canton', cantons, index=cantons.index(st.session_state.selected_canton))

# Update session state
st.session_state.show_overpredictions = new_show_overpredictions
st.session_state.show_predictions = new_show_predictions
st.session_state.show_recall_values = new_show_recall_values
if new_selected_canton != st.session_state.selected_canton:
    st.session_state.selected_canton = new_selected_canton
    st.session_state.selected_auschnitte = []  # Reset selected Auschnitte when canton changes

# Filter data based on the selected canton and auschnitte
if st.session_state.selected_canton == 'CH':
    filtered_gdf = gdf
    filtered_stats_df = stats_df
    filtered_predictions_gdf = predictions_gdf
    filtered_original_data = original_data
else:
    filtered_gdf = gdf[gdf['Canton'] == st.session_state.selected_canton]
    filtered_stats_df = stats_df[stats_df['Canton'] == st.session_state.selected_canton]
    filtered_predictions_gdf = predictions_gdf[predictions_gdf['Canton'] == st.session_state.selected_canton]
    filtered_original_data = original_data[original_data['Canton'] == st.session_state.selected_canton]

if st.session_state.selected_auschnitte:
    filtered_gdf = filtered_gdf[filtered_gdf['Auschnitt'].isin(st.session_state.selected_auschnitte)]
    filtered_predictions_gdf = filtered_predictions_gdf[filtered_predictions_gdf['file_name'].isin(st.session_state.selected_auschnitte)]
    filtered_original_data = filtered_original_data[filtered_original_data['Auschnitt'].isin(st.session_state.selected_auschnitte)]

# Explicitly separate overpredicted parcels
overpredicted_gdf = filtered_gdf[filtered_gdf['Overpredicted'] == True].copy()
normal_gdf = filtered_gdf[filtered_gdf['Overpredicted'] != True].copy()

# Apply recall filter to both normal and overpredicted parcels
normal_gdf = normal_gdf[(normal_gdf['Recall'] >= st.session_state.recall_range[0]) & (normal_gdf['Recall'] <= st.session_state.recall_range[1])]
overpredicted_gdf = overpredicted_gdf[overpredicted_gdf['Recall'].isnull() | ((overpredicted_gdf['Recall'] >= st.session_state.recall_range[0]) & (overpredicted_gdf['Recall'] <= st.session_state.recall_range[1]))]

# Create map function
def create_map():
    m = folium.Map(location=st.session_state.map_view['center'], zoom_start=st.session_state.map_view['zoom'], tiles='CartoDB positron')
    
    # Create a color map
    colormap = LinearColormap(colors=['purple', 'blue', 'green', 'yellow'], vmin=0, vmax=1)
    
    def style_function(feature):
        recall = feature['properties']['Recall']
        if pd.isna(recall):
            return {'fillColor': 'red', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
        else:
            return {'fillColor': colormap(recall), 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}

    def highlight_function(feature):
        return {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    # Add satellite images for selected parcels
    for parcel in st.session_state.selected_auschnitte:
        satellite_image_path = f"/workspaces/Satelite/data/satellite/{parcel}.tif"
        png_path = f"/workspaces/Satelite/data/satellite/{parcel}.png"
        
        try:
            # Convert GeoTIFF to PNG if it doesn't exist
            if not os.path.isfile(png_path):
                with rasterio.open(satellite_image_path) as src:
                    # Read the image data
                    img = src.read()
                    bounds = src.bounds
                    
                    # Convert the image to RGB if it's not already
                    if img.shape[0] == 1:  # If it's a single-band image
                        img = np.tile(img, (3, 1, 1))
                    elif img.shape[0] == 4:  # If it's an RGBA image
                        img = img[:3, :, :]
                    
                    # Normalize and convert to 8-bit
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    # Convert to PIL Image and save as PNG
                    pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                    pil_img.save(png_path)
            else:
                # If PNG already exists, just read the bounds from the original GeoTIFF
                with rasterio.open(satellite_image_path) as src:
                    bounds = src.bounds

            # Add the PNG image to the map
            img = folium.raster_layers.ImageOverlay(
                name=f"Satellite Image - {parcel}",
                image=png_path,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.7,
                interactive=True,
                cross_origin=False,
                zindex=1,
            )
            
            folium.Popup(f"Satellite Image - {parcel}").add_to(img)
            img.add_to(m)
            
            st.write(f"Successfully added image for parcel {parcel}")
        except Exception as e:
            st.warning(f"Could not load satellite image for parcel {parcel}: {str(e)}")

    # Add original data if checkbox is checked
    if new_show_original_data:
        folium.GeoJson(
            filtered_original_data,
            style_function=lambda x: {'fillColor': 'gray', 'color': 'black', 'weight': 1, 'fillOpacity': 0.3},
            tooltip=folium.GeoJsonTooltip(
                fields=['nutzung', 'area', 'Canton'],
                aliases=['Nutzung', 'Area', 'Canton'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

    # Add predictions if checkbox is checked
    if st.session_state.show_predictions:
        folium.GeoJson(
            filtered_predictions_gdf,
            style_function=lambda x: {'fillColor': 'orange', 'color': 'black', 'weight': 1, 'fillOpacity': 0.5},
        ).add_to(m)

    # Add normal parcels if show_recall_values is checked
    if st.session_state.show_recall_values:
        folium.GeoJson(
            normal_gdf,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['nutzung', 'area', 'Canton', 'Recall', 'Overpredicted', 'Auschnitt'],
                aliases=['Nutzung', 'Area', 'Canton', 'Recall', 'Overpredicted', 'Auschnitt'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

    # Add overpredicted parcels if checkbox is checked
    if st.session_state.show_overpredictions:
        folium.GeoJson(
            overpredicted_gdf,
            style_function=lambda x: {'fillColor': 'red', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7},
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['nutzung', 'area', 'Canton', 'Recall', 'Overpredicted', 'Auschnitt'],
                aliases=['Nutzung', 'Area', 'Canton', 'Recall', 'Overpredicted', 'Auschnitt'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

    # Add color legend
    colormap.add_to(m)
    colormap.caption = 'Recall Score'
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Display the map
m = create_map()
folium_static(m, width=700, height=500)

# Display overall statistics
st.header('Overall Statistics')

# Filter overall statistics based on selected canton
if st.session_state.selected_canton == 'CH':
    filtered_overall_stats = overall_stats_df[overall_stats_df['Canton'] == 'CH']
else:
    filtered_overall_stats = overall_stats_df[overall_stats_df['Canton'] == st.session_state.selected_canton]

# Calculate average recall
filtered_gdf = gdf if st.session_state.selected_canton == 'CH' else gdf[gdf['Canton'] == st.session_state.selected_canton]
avg_recall = filtered_gdf['Recall'].mean()

for _, row in filtered_overall_stats.iterrows():
    st.subheader(f"Statistics for {row['Canton']}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Area of Parcels (m²)", f"{row['Area (m²)']:,.2f}")
        st.metric("Overpredicted Area (m²)", f"{row['Overpredicted (m²)']:,.2f}")
        st.metric("Low Recall Area (m²)", f"{row['Low Recall  (m²)']:,.2f}", 
                  help="The area of parcels with a Recall < 0.7")
        st.metric("Average Recall", f"{avg_recall:.4f}", 
                  help="Average Recall value over all parcels")
    with col2:
        st.metric("Average Overprediction Error", f"{row['Average Overprediction Error']:.4f}")
        st.metric("Average Recall Error", f"{row['Average Recall Error']:.4f}")
        st.metric("Average Total Error", f"{row['Average Total Error']:.4f}")

# Use smaller text for statistics
st.markdown("""
<style>
    .stMetric {
        font-size: 0.8rem;
    }
    .stMetric .st-emotion-cache-16v4zaw {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Display statistics table
st.header('Statistics by Parcel')

# Rename 'Auschnitt' to 'Parcel' in filtered_stats_df
filtered_stats_df = filtered_stats_df.rename(columns={'Auschnitt': 'Parcel'})

# Sort the filtered_stats_df by Total Error in descending order
filtered_stats_df = filtered_stats_df.sort_values('Total Error', ascending=False)

# Add a 'Select' column to the dataframe
filtered_stats_df['Select'] = filtered_stats_df['Parcel'].isin(st.session_state.selected_auschnitte)

# Reorder columns to put Select and Parcel at the beginning
cols = ['Select', 'Parcel', 'Canton'] + [col for col in filtered_stats_df.columns if col not in ['Select', 'Parcel', 'Canton']]
filtered_stats_df = filtered_stats_df[cols]

# Create a styled dataframe
# Create a styled dataframe
styled_df = filtered_stats_df.style.format({
    'Area (m²)': '{:.2f}',
    'Overpredicted (m²)': '{:.2f}',
    'Low Recall  (m²)': '{:.2f}',
    'Total Error': '{:.2f}',
    'Overprediction Error': '{:.2f}',
    'Recall Error': '{:.2f}'
})

# Apply background color to highlight selected rows
def highlight_selected(s):
    return ['background-color: #ADD8E6' if s.Select else '' for _ in s]

styled_df = styled_df.apply(highlight_selected, axis=1)

# Unselect All button
if st.session_state.selected_auschnitte:
    if st.button("Unselect All"):
        st.session_state.selected_auschnitte = []
        st.rerun()

# Display the dataframe
edited_df = st.data_editor(
    styled_df,
    hide_index=True,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Select",
            help="Select Parcel",
            default=False,
        )
    },
    disabled=["Parcel", "Canton", "Area (m²)", "Overpredicted (m²)", "Low Recall  (m²)", "Total Error", "Overprediction Error", "Recall Error"],
    key="edited_df"
)

# Update selected_auschnitte based on the checkboxes
st.session_state.selected_auschnitte = edited_df[edited_df['Select'] == True]['Parcel'].tolist()

# Display current filter state
st.write(f"Selected Canton: {st.session_state.selected_canton}")
st.write(f"Selected Parcels: {', '.join(st.session_state.selected_auschnitte) if st.session_state.selected_auschnitte else 'None'}")

# Rerun the app if selections have changed
if edited_df['Select'].tolist() != filtered_stats_df['Select'].tolist():
    st.rerun()