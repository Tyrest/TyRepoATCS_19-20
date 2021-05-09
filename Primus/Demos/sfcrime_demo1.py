""" This module shows some useful tricks with the sf crime dataset.
"""
__version__ = '2.0'
__author__ = 'Cesar Cesarotti'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pandas imports the CSV as a Data Frame object
sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")

sfcrime.info()

sfc_districts = sfcrime.groupby('Police District').aggregate('count')
#print(sfc_districts['Incident Category'])

crime_daydist = pd.crosstab(index=sfcrime["Incident Day of Week"],
                             columns=sfcrime["Police District"])
#print(crime_daydist)

crime_catdist = pd.crosstab(index=sfcrime["Incident Category"],
                            columns=sfcrime["Police District"])
#print(crime_catdist)

crime_distcat = pd.crosstab(index=sfcrime["Police District"],
                            columns=sfcrime["Incident Category"])

#print(crime_distcat)

#print(crime_distcat.describe())

#crime_distcat_stats = crime_distcat.describe()

#crime_distcat['Total'] = crime_distcat.apply(sum, axis=1)

#print(crime_distcat['Total'])
