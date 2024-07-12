import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap

# Initialize session state for map view and other variables
if 'map_view' not in st.session_state:
    st.session_state.map_view = {
        'center': [47.3769, 8.5417],
        'zoom': 9
    }
if 'show_overpredictions' not in st.session_state:
    st.session_state.show_overpredictions = False
if 'recall_range' not in st.session_state:
    st.session_state.recall_range = (0.0, 1.0)
if 'selected_canton' not in st.session_state:
    st.session_state.selected_canton = 'CH'
if 'show_zh_ch_parcel_2' not in st.session_state:
    st.session_state.show_zh_ch_parcel_2 = False

# Load data
@st.cache_data
def load_data():
    gdf = gpd.read_file('/workspaces/Satelite/data/experiment/evaluation/ZH_analysis.gpkg')
    stats_df = pd.read_csv('/workspaces/Satelite/data/experiment/evaluation/statistics.csv')
    return gdf.to_crs(epsg=4326), stats_df

gdf, stats_df = load_data()

st.title('Field Parcel Segmentation Analysis')

# Sidebar for filters
st.sidebar.header('Filters')
new_recall_range = st.sidebar.slider('Recall Range', 0.0, 1.0, st.session_state.recall_range, 0.1)
new_show_overpredictions = st.sidebar.checkbox('Show Overpredictions', st.session_state.show_overpredictions)

# Canton selection
cantons = ['CH'] + sorted(gdf['Canton'].unique().tolist())
new_selected_canton = st.selectbox('Select a Canton', cantons, index=cantons.index(st.session_state.selected_canton))

# Update session state
st.session_state.recall_range = new_recall_range
st.session_state.show_overpredictions = new_show_overpredictions
st.session_state.selected_canton = new_selected_canton

# Filter data based on the selected canton
if st.session_state.selected_canton == 'CH':
    filtered_gdf = gdf
    filtered_stats_df = stats_df
else:
    filtered_gdf = gdf[gdf['Canton'] == st.session_state.selected_canton]
    filtered_stats_df = stats_df[stats_df['Canton'] == st.session_state.selected_canton]

# Filter for ZH_CH_parcel_2 if button is clicked
if st.session_state.show_zh_ch_parcel_2:
    filtered_gdf = filtered_gdf[filtered_gdf['Auschnitt'] == 'ZH_CH_parcel_2']

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

    # Add normal parcels
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
    
    return m

# Display the map
m = create_map()
folium_static(m, width=700, height=500)

# Button to show/hide ZH_CH_parcel_2
if st.button('Toggle ZH_CH_parcel_2'):
    st.session_state.show_zh_ch_parcel_2 = not st.session_state.show_zh_ch_parcel_2
    st.experimental_rerun()

# Display current filter state
st.write(f"Showing ZH_CH_parcel_2 only: {st.session_state.show_zh_ch_parcel_2}")