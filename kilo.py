import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_reg = LinearRegression()
boston = pd.read_csv('boston_housing.csv')

boston_targets = boston['MEDIAN VALUE']

# Returns predictions for boston_targets based on inputted cols
def bostonPredict(cols):
    boston_inputs = boston[cols]
    lin_reg.fit(boston_inputs, boston_targets)
    return lin_reg.predict(boston_inputs)

def bostonOtherColumns():
    otherColumns = ['CRIME RATE', 'LARGE LOT', 'INDUSTRY', 'RIVER', 'NOX', 'PRIOR 1940', 'EMP DISTANCE', 'HWY ACCESS', 'PROP TAX RATE', 'STU TEACH RATIO', 'AFR AMER']
    boston_outputs_base = bostonPredict(['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2'])
    boston_mse_base = mean_squared_error(boston_targets, boston_outputs_base)
    print('Base: ' + str(boston_mse_base))
    for other in otherColumns:
        boston_outputs_other = bostonPredict(['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2', other])
        boston_mse_other = mean_squared_error(boston_targets, boston_outputs_other)
        percentImprovement = (boston_mse_base - boston_mse_other) / boston_mse_base
        if percentImprovement > 0.03:
            print(other + ': ' + str(boston_mse_other))

# 1
boston_outputs_1 = bostonPredict(['LOW STATUS'])

# 2
boston_outputs_2 = bostonPredict(['ROOMS'])

# 3
boston_outputs_3 = bostonPredict(['LOW STATUS', 'ROOMS'])

# 4
boston[ ['LOW STATUS^2'] ] = boston[ ['LOW STATUS'] ] ** 2
boston_outputs_4 = bostonPredict(['LOW STATUS', 'LOW STATUS^2'])

# 5
boston[ ['ROOMS^2'] ] = boston[ ['ROOMS'] ] ** 2
boston_outputs_5 = bostonPredict(['ROOMS', 'ROOMS^2'])

# 6
boston_outputs_6 = bostonPredict(['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2'])

# 7
boston[ ['LOWROOMS'] ] = boston[ ['LOW STATUS'] ]
boston[ ['LOWROOMS'] ].mul(boston[ ['ROOMS'] ], axis='columns', fill_value=0)
boston_outputs_7 = bostonPredict(['LOW STATUS', 'LOW STATUS^2', 'ROOMS', 'ROOMS^2', 'LOWROOMS'])

# 8
bostonOtherColumns()

# 9 (Optimized columns does not actually work)
optimizeColumns = ['LOW STATUS', 'ROOMS', 'INDUSTRY', 'NOX', 'PROP TAX RATE', 'STU TEACH RATIO']
columns = ['CRIME RATE', 'LARGE LOT', 'INDUSTRY', 'RIVER', 'NOX', 'ROOMS', 'PRIOR 1940', 'EMP DISTANCE', 'HWY ACCESS', 'PROP TAX RATE', 'STU TEACH RATIO', 'AFR AMER', 'LOW STATUS']
lots = []
for column in columns:
    for i in range(1, 641):
        boston[ [column + '^' + str(i)] ] = boston[ [column] ] ** (i / 256)
        lots.append(column + '^' + str(i))
    print(column)
boston_outputs_9 = bostonPredict(lots)
boston_mse_base = mean_squared_error(boston_targets, boston_outputs_9)
print('Total Error: ' + str(boston_mse_base * len(boston)))