import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px

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
recall_range = st.sidebar.slider('Recall Range', 0.0, 1.0, (0.0, 1.0), 0.1)
show_overpredictions = st.sidebar.checkbox('Show Overpredictions')

# Filter data
filtered_gdf = gdf[(gdf['Recall'] >= recall_range[0]) & (gdf['Recall'] <= recall_range[1])]
if show_overpredictions:
    filtered_gdf = filtered_gdf[filtered_gdf['Overpredicted']]

# Create map
fig = px.choropleth_mapbox(filtered_gdf,
                           geojson=filtered_gdf.geometry.__geo_interface__,
                           locations=filtered_gdf.index,
                           color='Recall',
                           color_continuous_scale="Viridis",
                           range_color=(0, 1),
                           mapbox_style="carto-positron",
                           zoom=9,
                           center={"lat": filtered_gdf.geometry.centroid.y.mean(), 
                                   "lon": filtered_gdf.geometry.centroid.x.mean()},
                           opacity=0.5,
                           labels={'Recall':'Recall Score'}
                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig, use_container_width=True)

# Display parcel information on click
selected_parcel = st.selectbox('Select a parcel', filtered_gdf.index)
if selected_parcel:
    parcel = filtered_gdf.loc[selected_parcel]
    st.write("Parcel Information:")
    for col in parcel.index:
        if col != 'geometry':
            st.write(f"{col}: {parcel[col]}")

# Display some statistics
st.subheader('Statistics')
st.write(f"Total parcels: {len(filtered_gdf)}")
st.write(f"Average Recall: {filtered_gdf['Recall'].mean():.2f}")
st.write(f"Overpredicted parcels: {filtered_gdf['Overpredicted'].sum()}")