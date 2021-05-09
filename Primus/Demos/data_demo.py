""" This module introduces various ways of looking at and
	playing with data.
"""
__version__ = '2.0'
__author__ = 'Cesar Cesarotti'

import numpy as np
import pandas as pd

# pandas imports the CSV as a Data Frame object
airports = pd.read_csv("airports.csv")

#print(airports.head())

# Let's get a summary of the data.
#airports.info()


#print(airports)

# you can splice data frames just like lists! To get the first five rows:
#print(airports[:5])

# you can get just one column by indexing that column:
#print(airports['Code'])

# Not surprisingsly, you can combine the two:
#print(airports['Code'][5:11])

# To get multiple columns, use a list of columns:
#print(airports[['Code','Delayed','Year']][5:11])

# The .value_counts() method gives a list of values
#	for the column sorted by frequency.
#print(airports["Year"].value_counts())
#print(airports["Label"].value_counts())
#print(airports["Code"].value_counts())

# print(airports["Year"].count())
# print(airports["# of Delays.Security"].sum())
# print("Max cancelled: " + str(airports["Cancelled"].max()))
# print("Min cancelled: " + str(airports["Cancelled"].min()))
# print("Mean cancelled: " + str(airports["Cancelled"].mean()))
# print("Median cancelled: " + str(airports["Cancelled"].median()))
# print(airports.groupby("Code")["Cancelled"].max())

# print(airports[airports["Cancelled"]==airports["Cancelled"].max()]) # ['Name']



# Which airport has the most total number of delays due to security?

print(airports[airports["# of Delays.Security"] == airports["# of Delays.Security"].max()][["Code","# of Delays.Security"]])

# Which airport has the most total number of delays due to weather?

print(airports[airports["# of Delays.Weather"] == airports["# of Delays.Weather"].max()][["Code","# of Delays.Weather"]])

# Is the airport with the most number of delays due to security
#  also the airport with the most minutes of delays due to security?

print(airports[airports["Delayed"] == airports["Delayed"].max()][["Code", "Delayed"]])
print(airports[airports["Delayed"] == airports["Delayed"].max()]["# of Delays.Security"])
print(airports[airports["Minutes Delayed.Security"] == airports["Minutes Delayed.Security"].max()]["Code"])

# Challenge:
# Which airport has the greatest percentage of flights delayed?

print(airports[airports["Delayed"]/airports["Flights.Total"] == (airports["Delayed"]/airports["Flights.Total"]).max()]["Code"])
