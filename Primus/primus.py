__version__ = '1.0'
__author__ = 'Tyler Ho'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Combines two datasets
def combine(data1, csv, index1, index2, column):
	 temp = pd.crosstab(index=data1[index1],
						columns=data1[column])
	 temp2 = pd.read_csv(csv, index_col=index2)
	 return pd.concat( [temp2, temp], axis=1, sort=True)

# Applies a label to the title, xaxis, and yaxis and shows the plot
def finalizeGraph(title, xaxis, yaxis):
	plt.title(title)
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)
	plt.show()

# Creates a map color coded for each police district
def SFPDColor(data, title):
	district_colors = { 'Southern'   : "red",
			            'Northern'   : "orange",
			            'Mission'    : "black",
	                    'Central'    : "green",
	                    'Bayview'    : "deepskyblue",
	                    'Ingleside'  : "blue",
	                    'Taraval'    : "purple",
	                    'Tenderloin' : "yellow",
	                    'Richmond'   : "azure",
	                    'Park'       : "mediumseagreen",
	                    }
	for district in district_colors:
		plt.scatter(x=data[ data['Police District'] == district ]['Longitude'],
	            	y=data[ data['Police District'] == district ]['Latitude'],
	            	c=district_colors[district],
	            	alpha=0.5)
	plt.grid(True)
	finalizeGraph(title, "Longitude", "Latitude")

# Creates and returns a map with a heatmap of density of datapoints
def SFDensityHeatMap(data):
	fig = go.Figure(go.Densitymapbox(lat=data.Latitude, lon=data.Longitude, radius=30))
	fig.update_layout(mapbox_style="carto-positron",
					  mapbox_center_lon=-122.431297,
					  mapbox_center_lat=37.773972,
					  mapbox_zoom=11)
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	fig.show()

# Creates and shows a bar graph with one column divided by another
def valuePerDivi(data, value, divi, highlight, xaxis):
	data[value + '/' + divi] = data[value]/data[divi]
	data = data.sort_values(value + '/' + divi)
	data['color'] = 'b'
	data.loc[highlight, 'color'] = 'r'
	data.plot(kind='bar', y=value + '/' + divi, color=data['color'])
	finalizeGraph(value + " per " + divi + " in for 2018", xaxis, value + " per " + divi)

# Converts any nan values to 0 in a column
def nanToZero(data, column):
	for coords in data.iterrows():
		if data.isnull().loc[coords[0], column]:
			data.loc[coords[0], column] = 0
	return data

# pandas imports CSV as a Data Frame object
sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")

# removes Out of SF category from crimes data
sfcrime = sfcrime[ sfcrime['Police District'] != 'Out of SF' ]

# Creates two dataframes with only the data that I need for each crime
muggings = sfcrime[ sfcrime['Incident Subcategory'] == 'Robbery - Street' ]
liquor = sfcrime[ sfcrime['Incident Category'] == 'Liquor Laws' ]

# Bar Graph Top 20 Intersections for muggings
muggings['Intersection'].value_counts()[:20].plot(kind='bar')
finalizeGraph("Top 20 Intersections for Muggings in SF for 2018", "Intersection", "Muggings")

SFDensityHeatMap(muggings)

SFPDColor(liquor, "Liquor crimes by Police District in SF for 2018")

liquor = combine(liquor, "SF_Police_Districts.csv", 'Police District',
				 'PdDistrict', 'Incident Category')
liquor = nanToZero(liquor, 'Liquor Laws')

valuePerDivi(liquor, 'Liquor Laws', 'Population', 'Mission', 'Police District')
valuePerDivi(liquor, 'Liquor Laws', 'Land Mass', 'Mission', 'Police District')
