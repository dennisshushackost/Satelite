import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

# Load data python -m streamlit run web2.py
@st.cache_data
def load_data():
    gdf = gpd.read_file('/workspaces/Satelite/data/experiment/evaluation/ZH_analysis.gpkg')
    stats_df = pd.read_csv('/workspaces/Satelite/data/experiment/evaluation/statistics.csv')
    return gdf.to_crs(epsg=4326), stats_df

gdf, stats_df = load_data()

# Print data info for debugging
st.sidebar.write("Data Info:")
st.sidebar.write(f"Total parcels: {len(gdf)}")
st.sidebar.write(f"Overpredicted parcels: {gdf['Overpredicted'].sum()}")

st.title('Field Parcel Segmentation Analysis')

# Sidebar for filters
st.sidebar.header('Filters')
recall_range = st.sidebar.slider('Recall Range', 0.0, 1.0, (0.0, 1.0), 0.1)
show_overpredictions = st.sidebar.checkbox('Show Overpredictions')

# Canton selection
cantons = ['CH'] + sorted(gdf['Canton'].unique().tolist())
selected_canton = st.selectbox('Select a Canton', cantons)

# Filter data
if selected_canton == 'CH':
    filtered_gdf = gdf
    filtered_stats_df = stats_df
else:
    filtered_gdf = gdf[gdf['Canton'] == selected_canton]
    filtered_stats_df = stats_df[stats_df['Canton'] == selected_canton]

# Explicitly separate overpredicted parcels
overpredicted_gdf = filtered_gdf[filtered_gdf['Overpredicted'] == True].copy()
normal_gdf = filtered_gdf[filtered_gdf['Overpredicted'] != True].copy()

# Apply recall filter to both normal and overpredicted parcels
normal_gdf = normal_gdf[(normal_gdf['Recall'] >= recall_range[0]) & (normal_gdf['Recall'] <= recall_range[1])]
overpredicted_gdf = overpredicted_gdf[overpredicted_gdf['Recall'].isnull() | ((overpredicted_gdf['Recall'] >= recall_range[0]) & (overpredicted_gdf['Recall'] <= recall_range[1]))]

# Create map
fig = go.Figure()

# Hover template
hovertemplate = (
    "<b>Nutzung:</b> %{customdata[0]}<br>"
    "<b>Area:</b> %{customdata[1]:.2f}<br>"
    "<b>Canton:</b> %{customdata[2]}<br>"
    "<b>Recall:</b> %{customdata[3]}<br>"
    "<b>Overpredicted:</b> %{customdata[4]}"
)

# Add normal parcels
if not normal_gdf.empty:
    fig.add_trace(go.Choroplethmapbox(
        geojson=normal_gdf.__geo_interface__,
        locations=normal_gdf.index,
        z=normal_gdf['Recall'],
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        marker_opacity=0.7,
        marker_line_width=0,
        colorbar_title="Recall Score",
        customdata=normal_gdf[['nutzung', 'area', 'Canton', 'Recall', 'Overpredicted']],
        hovertemplate=hovertemplate
    ))

# Add overpredicted parcels
if show_overpredictions and not overpredicted_gdf.empty:
    fig.add_trace(go.Choroplethmapbox(
        geojson=overpredicted_gdf.__geo_interface__,
        locations=overpredicted_gdf.index,
        z=overpredicted_gdf['Recall'].isnull().astype(int),
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        marker_opacity=0.7,
        marker_line_width=0,
        customdata=overpredicted_gdf[['nutzung', 'area', 'Canton', 'Recall', 'Overpredicted']],
        hovertemplate=hovertemplate
    ))

if not filtered_gdf.empty:
    center_lat = filtered_gdf.geometry.centroid.y.mean()
    center_lon = filtered_gdf.geometry.centroid.x.mean()
else:
    center_lat, center_lon = 47.3769, 8.5417  # Default to Zurich coordinates

fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=9,
    mapbox_center={"lat": center_lat, "lon": center_lon},
    margin={"r":0,"t":0,"l":0,"b":0},
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# Display statistics table
st.subheader('Statistics')
filtered_stats_df = filtered_stats_df.sort_values(by='Total Error', ascending=False)
filtered_stats_df[['Original Total Area (m²)', 'Overpredicted Area (m²)', 'Low Recall Area (m²)', 'Total Error', 'Overprediction Error', 'Recall Error']] = filtered_stats_df[['Original Total Area (m²)', 'Overpredicted Area (m²)', 'Low Recall Area (m²)', 'Total Error', 'Overprediction Error', 'Recall Error']].round(2)
st.dataframe(filtered_stats_df[['Parcel Name', 'Canton', 'Auschnitt', 'Original Total Area (m²)', 'Overpredicted Area (m²)', 'Low Recall Area (m²)', 'Total Error', 'Overprediction Error', 'Recall Error']], height=650)