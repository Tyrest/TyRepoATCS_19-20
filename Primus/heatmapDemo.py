__version__ = '1.0'
__author__ = 'Tyler Ho'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# pandas imports the CSV as a Data Frame object
sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")

# takes out "Out of SF" police district
sfcrime = sfcrime[sfcrime['Police District' ] != 'Out of SF']

# takes only muggings from sfcrime dataset
muggings = sfcrime[ sfcrime['Incident Subcategory'] == 'Robbery - Street' ]

'''
Creates a heatmap with plotly
lat and lon are latitude and longitude for mapping
radius is to change how large points are on the heatmap
mapbox_style is the style of the map (I chose carto-positron for simplicity)
mapbox center and zoom are to determine the starting view of the map
margin sets margins around map so map fits to screen
'''
fig = go.Figure(go.Densitymapbox(lat=muggings.Latitude, lon=muggings.Longitude, radius=15))
fig.update_layout(mapbox_style="carto-positron", mapbox_center_lon=-122.431297, mapbox_center_lat=37.773972, mapbox_zoom=11)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
