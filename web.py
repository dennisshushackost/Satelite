import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import geopandas as gpd
import pandas as pd
import numpy as np
from dash_extensions.enrich import DashProxy, MultiplexerTransform
import dash_bootstrap_components as dbc

# Load data
gdf = gpd.read_file('/workspaces/Satelite/data/experiment/evaluation/ZH_analysis.gpkg')
stats_df = pd.read_csv('/workspaces/Satelite/data/experiment/evaluation/statistics.csv')

# Initialize the Dash app with Bootstrap dark theme
app = DashProxy(__name__, transforms=[MultiplexerTransform()], external_stylesheets=[dbc.themes.DARKLY])

# Define color scales
colorscale_recall = px.colors.sequential.Viridis
colorscale_error = px.colors.sequential.Reds

# Mapbox token (replace with your own)
mapbox_token = 'your_mapbox_token_here'

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Field Parcel Segmentation Analysis", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='switzerland-map', style={'height': '70vh'}),
            dbc.Row([
                dbc.Col(dcc.RangeSlider(id='recall-filter', min=0, max=1, step=0.1, value=[0, 1], marks={i/10: str(i/10) for i in range(11)}), width=8),
                dbc.Col(dbc.Checklist(id='overprediction-toggle', options=[{'label': 'Show Overpredictions', 'value': 'show'}], switch=True), width=4)
            ], className="mt-2"),
            dbc.Row([
                dbc.Col(dbc.Checklist(id='heatmap-toggle', options=[{'label': 'Show Heatmap', 'value': 'show'}], switch=True), width=4)
            ], className="mt-2")
        ], width=8),
        dbc.Col([
            dcc.Graph(id='error-chart'),
            html.Div(id='high-error-list', style={'maxHeight': '300px', 'overflowY': 'scroll'})
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='recall-histogram'), width=6),
        dbc.Col(dcc.Graph(id='error-by-region'), width=6)
    ], className="mt-4")
], fluid=True)

@app.callback(
    Output('switzerland-map', 'figure'),
    [Input('recall-filter', 'value'),
     Input('overprediction-toggle', 'value'),
     Input('heatmap-toggle', 'value')]
)
def update_map(recall_range, show_overpredictions, show_heatmap):
    filtered_gdf = gdf[(gdf['Recall'] >= recall_range[0]) & (gdf['Recall'] <= recall_range[1])]
    
    if show_overpredictions:
        filtered_gdf = filtered_gdf[filtered_gdf['Overpredicted']]
    
    if show_heatmap:
        fig = px.density_mapbox(filtered_gdf, lat=filtered_gdf.geometry.centroid.y, lon=filtered_gdf.geometry.centroid.x, z='Recall',
                                mapbox_style="dark", zoom=7, center={"lat": 46.8, "lon": 8.2},
                                opacity=0.5, color_continuous_scale=colorscale_recall)
    else:
        fig = px.choropleth_mapbox(filtered_gdf, geojson=filtered_gdf.geometry, locations=filtered_gdf.index, color='Recall',
                                   color_continuous_scale=colorscale_recall,
                                   mapbox_style="dark", zoom=7, center={"lat": 46.8, "lon": 8.2},
                                   opacity=0.5)

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, mapbox_accesstoken=mapbox_token)
    return fig

@app.callback(
    Output('error-chart', 'figure'),
    [Input('recall-filter', 'value')]
)
def update_error_chart(recall_range):
    filtered_df = stats_df[(stats_df['Recall'] >= recall_range[0]) & (stats_df['Recall'] <= recall_range[1])]
    fig = px.bar(filtered_df, x='Parcel Name', y=['Overprediction Error', 'Recall Error'], 
                 title='Error Rates by Parcel', barmode='stack')
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    Output('high-error-list', 'children'),
    [Input('recall-filter', 'value')]
)
def update_high_error_list(recall_range):
    filtered_df = stats_df[(stats_df['Recall'] >= recall_range[0]) & (stats_df['Recall'] <= recall_range[1])]
    sorted_df = filtered_df.sort_values('Total Error', ascending=False).head(10)
    
    return [dbc.ListGroup([
        dbc.ListGroupItem(f"{row['Parcel Name']}: {row['Total Error']:.2f}", 
                          id={'type': 'error-item', 'index': i},
                          action=True)
        for i, (_, row) in enumerate(sorted_df.iterrows())
    ])]

@app.callback(
    Output('recall-histogram', 'figure'),
    [Input('recall-filter', 'value')]
)
def update_recall_histogram(recall_range):
    filtered_gdf = gdf[(gdf['Recall'] >= recall_range[0]) & (gdf['Recall'] <= recall_range[1])]
    fig = px.histogram(filtered_gdf, x='Recall', nbins=50, title='Distribution of Recall Values')
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    Output('error-by-region', 'figure'),
    [Input('recall-filter', 'value')]
)
def update_error_by_region(recall_range):
    filtered_df = stats_df[(stats_df['Recall'] >= recall_range[0]) & (stats_df['Recall'] <= recall_range[1])]
    fig = px.scatter(filtered_df, x='Overprediction Error', y='Recall Error', 
                     color='Total Error', size='Original Total Area (m²)',
                     hover_name='Parcel Name', title='Error by Region')
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    Output('switzerland-map', 'figure'),
    [Input({'type': 'error-item', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('switzerland-map', 'figure'),
     State('recall-filter', 'value'),
     State('overprediction-toggle', 'value'),
     State('heatmap-toggle', 'value')]
)
def zoom_to_parcel(n_clicks, current_figure, recall_range, show_overpredictions, show_heatmap):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_figure
    
    clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
    clicked_index = json.loads(clicked_id)['index']
    
    parcel_name = stats_df.sort_values('Total Error', ascending=False).iloc[clicked_index]['Parcel Name']
    parcel_geom = gdf[gdf['Parcel Name'] == parcel_name].geometry.iloc[0]
    
    centroid = parcel_geom.centroid
    current_figure['layout']['mapbox']['center'] = {"lat": centroid.y, "lon": centroid.x}
    current_figure['layout']['mapbox']['zoom'] = 12
    
    return current_figure

if __name__ == '__main__':
    app.run_server(debug=True)