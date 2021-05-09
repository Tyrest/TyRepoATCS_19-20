""" This module introduces various ways of looking at and
	playing with data.
"""
__version__ = '0.2'
__author__ = 'Cesar Cesarotti'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pandas imports the CSV as a Data Frame object
airports = pd.read_csv("airports.csv")

# Very slick (although complicated) crosstab function!
#print(pd.crosstab(index=airports.Code,columns=airports.Year,values=airports.Cancelled,aggfunc=sum))


# Let's plot the top 10 with a bar chart:
# plt.figure()
# data=airports["Carriers.Total"].value_counts()[:10]
# data.plot(kind='bar')
# plt.show()
# Let's do some styling:
data.plot(kind='bar')
# plt.xlabel('Number of Carriers per Airport')
# plt.ylabel('Number of data rows')
# plt.title('10 Most frequent Number of Carriers per Airport')
# plt.text(5, 600, "Carrier: >= 1% US flights")
# plt.axis([40, 160, 0, 0.03]) # xmin, xmax, ymin, ymax
# plt.grid(True)
plt.show()

# What if we want the data for only one city?
denver = airports[airports["Code"] == 'DEN']
#print(denver[:5])

# Let's look at the amount of delays over time at Denver
den_delays=denver[['Delayed','Flights.Total','Year','Month','Month Name','Minutes Delayed.Weather','Minutes Delayed.Total']]
# print(den_delays[:5])
# den_delays.plot(x='Label',y='Delayed',kind='bar')
# plt.show()

# That's a little overwhelming, so let's zoom in a bit
# We could look at the data for one year:
# den_delays_2009 = den_delays[ den_delays['Year'] == 2009 ]
# den_delays_2009.plot(x='Month Name',y='Delayed',kind='bar')
# plt.ylabel('Number of Delayed Flights')
# plt.title('Number of Delayed Flights by Month for the year 2009')
# plt.text(8, 5000, "Denver")
# plt.show()

# Or we could look at the year totals by aggregating:
den_delays_annual=den_delays.groupby('Year').aggregate(sum)
# den_delays_annual.plot(y='Delayed',kind='bar')
# plt.ylabel('Number of Delayed Flights')
# plt.title('Number of Delayed Flights per year')
# plt.text(6, 50000, "Denver")
# plt.show()

# den_delays_annual.plot(y='Delayed',kind='line')
# plt.ylabel('Number of Delayed Flights')
# plt.title('Number of Delayed Flights per year')
# plt.text(6, 50000, "Denver")
# plt.show()

#den_delays_annual.plot(y='Delayed',kind='scatter')

# But what if the total number of flights is changing?
# 	Let's look at a percentage of flights delayed and create a new column:
den_delays_annual['Percent Delayed']=den_delays_annual['Delayed']/den_delays_annual['Flights.Total']
# den_delays_annual.plot(y='Percent Delayed',kind='bar')
# plt.show()


# Let's put those last two side by side to more clearly see the difference:
#	This uses straight pyplot commands, instead of the pandas versions
# f, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.bar(den_delays_annual.index,den_delays_annual['Delayed'])
# ax2.bar(den_delays_annual.index,den_delays_annual['Percent Delayed'])
# plt.show()

# Scatter plots can help us explore relationships between variables
# plt.scatter(airports['Minutes Delayed.Weather'],airports['Minutes Delayed.Total'],alpha=0.3)
# plt.xlabel('Minutes delayed due to Weather')
# plt.ylabel('Total Minutes Delayed')
# plt.show()

# We can use color to show more information
# plt.scatter(den_delays['Minutes Delayed.Weather'],den_delays['Minutes Delayed.Total'], c=den_delays["Month"], cmap=plt.get_cmap("jet"))
# plt.xlabel('Minutes delayed due to Weather')
# plt.ylabel('Total Minutes Delayed')
# plt.title('Denver Delays')
# plt.colorbar()
# plt.show()

# We can also have more than one data set on the figure.
#	Lets compare Denver and Vegas...
# vegas = airports[airports["Code"] == 'LAS']
# plot1 = plt.scatter(vegas['Minutes Delayed.Weather'],vegas['Minutes Delayed.Total'], c='b', label='Vegas')
# plot2 = plt.scatter(den_delays['Minutes Delayed.Weather'],den_delays['Minutes Delayed.Total'], c='r', label='Denver')
# plt.legend(handles=[plot1, plot2])
# plt.xlabel('Minutes delayed due to Weather')
# plt.ylabel('Total Minutes Delayed')
# plt.title('Vegas vs Denver Delays')
# plt.show()


# Let's compare Denver and Ohare delays by year...
ohare_delays_annual = airports[ airports['Code'] == 'ORD' ].groupby('Year').aggregate(sum)
ohare_delays_annual['Percent Delayed']=ohare_delays_annual['Delayed']/ohare_delays_annual['Flights.Total']

# plot1 = plt.scatter(x=ohare_delays_annual.index, y=ohare_delays_annual['Percent Delayed'], c='r', label='Ohare')
# plot2 = plt.scatter(x=den_delays_annual.index, y=den_delays_annual['Percent Delayed'], c='b',label='Denver')
# plt.legend(handles=[plot1, plot2])
# plt.xlabel('Year')
# plt.ylabel('Percent Delayed')
# plt.title('Ohare vs Denver Delays')
# plt.show()


# Now it's your turn!
# Create a plot for a city other than Denver that shows something other than flights Delayed.

SFO = airports[airports['Code'] == 'SFO']
# SFO_flights = SFO[['Flights.Total', 'Month', 'Year', 'Delayed']]
# SFO_flights_annual = SFO.groupby('Year').aggregate(sum)
# SFO_flights_annual.plot(y='Flights.Total', kind='line')
# plt.show()

# Create a scatter plot of some data not shown above.

plt.scatter(x=SFO['Flights.Total'], y=SFO['Delayed'], alpha=0.3)
plt.xlabel('Flights')
plt.ylabel('Delayed Flights')
plt.title('Delayed Flights vs Total Flights in SFO')
plt.show()

# Surprise me!

# f, graphs = plt.subplots(2, sharex=True)
# plt.show()

# Challenge:
# Create a plot with 2 or more graphs on it.


# Make sure you free up all the memory from the figures we've been drawing...
plt.close('all')
