""" This module shows some neat graphs with the sf crime dataset.
"""
__version__ = '2.1'
__author__ = 'Cesar Cesarotti'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pandas imports the CSV as a Data Frame object
sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")

sfcrime.info()
print(sfcrime['Incident Category'].value_counts())
print(sfcrime['Resolution'].value_counts())
print(sfcrime['Police District'].value_counts())

burglary = sfcrime[ sfcrime['Incident Category'] == 'Burglary' ]
narcos = sfcrime[ sfcrime['Incident Category'] == 'Drug Offense' ]
arson = sfcrime[ sfcrime['Incident Category'] == 'Arson' ]


plt.scatter(x=burglary['Longitude'], y=burglary['Latitude'],alpha=0.3)
plt.title("Burglary Incidents by Location in SF for 2018")
plt.show()

taraval_burg = burglary[ burglary['Police District'] == 'Taraval' ]
plt.scatter(x=taraval_burg['Longitude'], y=taraval_burg['Latitude'],alpha=0.3)
plt.title("Burglary Incidents by Location in Taraval District of SF for 2018")
plt.show()

plt.scatter(x=narcos['Longitude'], y=narcos['Latitude'])
plt.title("Drug Offense Incidents by Location in SF for 2018")
plt.show()

plt.scatter(x=arson['Longitude'], y=arson['Latitude'], c='r')
plt.title("Arson Incidents by Location in SF for 2018")
plt.show()

def res_to_color (res):
	res_colors = { 'Open or Active' : 'red',
					'Cite or Arrest Adult' : 'green', 
					'Unfounded' : 'yellow',
					'Cite or Arrest Juvenile' : 'blue' }
	if res in res_colors:
		return res_colors[res]
	else:
		return 'grey'

district_markers = { 'Southern' : "o", # circle
					'Northern' : "s", # square
					'Mission' : "d",	# triangle_down
					'Central' : "x",	# x 
					'Bayview' : ">",	# triangle_right
					'Ingleside' : "h", #	hexagon1
					'Taraval' : "<",	# triangle_left
					'Tenderloin' : "+", # point
					'Richmond' : "p", # pentagon
					'Park' : "*", # star
					'Out of SF' : 'D' # diamond
					}
def dist_to_marker (district):
	return district_markers[district]

""" Other Marker Codes:
"^", # triangle_up
"1"	tri_down
"2"	tri_up
"3"	tri_left
"4"	tri_right
"P"	plus (filled)

"H"	hexagon2
"+"	plus
"X"	x (filled)
"D"	diamond
"d", # thin_diamond


"""

for district in district_markers:
	plt.scatter(x=narcos[ narcos['Police District'] == district ]['Longitude'],
	            y=narcos[ narcos['Police District'] == district ]['Latitude'],
	            c=narcos['Resolution'].apply(res_to_color),
	            marker=district_markers[district])
plt.title("Drug Offenses by Resolution and Location in SF for 2018")
plt.show()

for district in ['Central','Northern','Southern','Tenderloin']:
	plt.scatter(x=narcos[ narcos['Police District'] == district ]['Longitude'],
	            y=narcos[ narcos['Police District'] == district ]['Latitude'],
	            c=narcos['Resolution'].apply(res_to_color),
	            marker=district_markers[district])
plt.title("Drug Offenses by Resolution and Location in SF for 2018")
plt.show()


