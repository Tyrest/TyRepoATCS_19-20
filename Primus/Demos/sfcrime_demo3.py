""" This module shows some useful tricks with the sf crime dataset.
"""
__version__ = '0.2'
__author__ = 'Cesar Cesarotti'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pandas imports the CSV as a Data Frame object
sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")

sfcrime = sfcrime [ sfcrime['Police District'] != 'Out of SF' ]

crime_distcat = pd.crosstab(index=sfcrime["Police District"],
                            columns=sfcrime["Incident Category"])

crime_distcat['Total'] = crime_distcat.apply(sum, axis=1)

sfdistricts = pd.read_csv("SF_Police_Districts.csv", index_col='PdDistrict')

sfdistricts.info()

crime_distcat['Total'] = crime_distcat.apply(sum, axis=1)

sfcrime_districts = pd.concat( [crime_distcat, sfdistricts], axis=1)

print(sfcrime_districts['Population'].value_counts())

sfcrime_districts['per_capita']=sfcrime_districts['Total']/sfcrime_districts['Population']
sfcrime_districts['per_area']=sfcrime_districts['Total']/sfcrime_districts['Land Mass']
sfcrime_districts['Density']=sfcrime_districts['Population']/sfcrime_districts['Land Mass']
print(sfcrime_districts['per_capita'])
print(sfcrime_districts['per_area'])
print(sfcrime_districts['Density'])
