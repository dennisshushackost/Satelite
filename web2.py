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
else:
    filtered_gdf = gdf[gdf['Canton'] == selected_canton]

# Explicitly separate overpredicted parcels
overpredicted_gdf = filtered_gdf[filtered_gdf['Overpredicted'] == True].copy()
normal_gdf = filtered_gdf[filtered_gdf['Overpredicted'] != True].copy()

# Apply recall filter only to normal parcels
normal_gdf = normal_gdf[(normal_gdf['Recall'] >= recall_range[0]) & (normal_gdf['Recall'] <= recall_range[1])]

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

if show_overpredictions:
    # Show only overpredicted parcels
    if not overpredicted_gdf.empty:
        fig.add_trace(go.Choroplethmapbox(
            geojson=overpredicted_gdf.__geo_interface__,
            locations=overpredicted_gdf.index,
            z=overpredicted_gdf['Overpredicted'].astype(int),
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            marker_opacity=0.7,
            marker_line_width=0,
            customdata=overpredicted_gdf[['nutzung', 'area', 'Canton', 'Recall', 'Overpredicted']],
            hovertemplate=hovertemplate
        ))
    displayed_gdf = overpredicted_gdf
else:
    # Show normal parcels
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
    displayed_gdf = normal_gdf

if not displayed_gdf.empty:
    center_lat = displayed_gdf.geometry.centroid.y.mean()
    center_lon = displayed_gdf.geometry.centroid.x.mean()
else:
    center_lat, center_lon = 47.3769, 8.5417  # Default to Zurich coordinates

fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=9,
    mapbox_center={"lat": center_lat, "lon": center_lon},
    margin={"r":0,"t":0,"l":0,"b":0},
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Display some statistics
st.subheader('Statistics')
st.write(f"Total parcels displayed: {len(displayed_gdf)}")
if not show_overpredictions:
    st.write(f"Average Recall: {normal_gdf['Recall'].mean():.2f}")
st.write(f"Total overpredicted parcels: {len(overpredicted_gdf)}")

# Display parcel information on click
if st.checkbox('Show Parcel Information'):
    selected_parcel = st.selectbox('Select a parcel', displayed_gdf.index)
    if selected_parcel:
        parcel = displayed_gdf.loc[selected_parcel]
        st.write("Parcel Information:")
        for col in parcel.index:
            if col != 'geometry':
                st.write(f"{col}: {parcel[col]}")