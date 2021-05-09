import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

headers = ["Class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", 
           "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", 
           "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
           "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", 
           "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

mushrooms = pd.read_csv("mushrooms.csv", names=headers)
mushrooms["Class"] = mushrooms["Class"].apply(lambda x : 0 if x == 'e' else 1)
mushrooms_targets = mushrooms["Class"]
mushrooms_inputs = mushrooms.drop("Class", axis=1)

def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))

def logisticRegression(softmax_reg, train_inputs, train_targets, test_inputs, test_targets):
    print("\nLogistic Regression Classifier")
    softmax_reg.fit(train_inputs, train_targets)
    print("Mean test accuracy: ", softmax_reg.score(test_inputs, test_targets))
    softmax_outputs = softmax_reg.predict(test_inputs)
    print("Confusion Matrix:")
    print(confusion_matrix(test_targets, softmax_outputs))

def decisionTree(tree, train_inputs, train_targets, test_inputs, test_targets):
    print("\nDecision Tree Classifier")
    tree.fit(train_inputs, train_targets)
    print('Decision Tree fit with depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())
    print("Mean test accuracy: ", tree.score(test_inputs, test_targets))
    test_outputs = tree.predict(test_inputs)
    print("Confusion Matrix:")
    print(confusion_matrix(test_targets, test_outputs))

def kNearestNeighbors(knn, train_inputs, train_targets, test_inputs, test_targets):
    print("\nK Nearest Neighbors classifier")
    knn.fit(train_inputs, train_targets)
    print("Mean test accuracy:", knn.score(test_inputs, test_targets))
    test_outputs = knn.predict(test_inputs)
    print("Confusion Matrix:")
    print(confusion_matrix(test_targets, test_outputs))

def shuffle_and_test(targets, inputs):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)

    train_index, test_index = next(sss.split(inputs, targets))

    train_inputs = inputs.iloc[train_index]
    train_targets = targets.iloc[train_index]
    test_inputs = inputs.iloc[test_index]
    test_targets = targets.iloc[test_index]

    time_all_models(train_inputs, train_targets, test_inputs, test_targets)

def time_all_models(train_inputs, train_targets, test_inputs, test_targets):

    t0 = time.time()
    softmax_reg = LogisticRegression(solver='liblinear', random_state=7)
    logisticRegression(softmax_reg, train_inputs, train_targets, test_inputs, test_targets)
    print("Logistic Regression took: " + str(time.time() - t0) + " seconds")

    t0 = time.time()
    tree = DecisionTreeClassifier(random_state=7)
    decisionTree(tree, train_inputs, train_targets, test_inputs, test_targets)
    print("Decision Tree took: " + str(time.time() - t0) + " seconds")

    t0 = time.time()
    knn = KNeighborsClassifier()
    kNearestNeighbors(knn, train_inputs, train_targets, test_inputs, test_targets)
    print("K Nearest Neighbors took: " + str(time.time() - t0) + " seconds")

def encode(enc, inputs):
    enc.fit(inputs)
    inputs = enc.transform(inputs)
    return inputs

def columnToBinary(df, column):
    binarylength = len(bin(int(df[column].max()))) - 2
    df[column] = df[column].apply(lambda x : bin(int(x))[2:])
    df[column] = df[column].apply(lambda x : '0' * (binarylength - len(x)) + x)
    for power in range(1, binarylength + 1):
        df[column + '_p' + str(power)] = df[column].apply(lambda x: int(x[-(power - 1)]))
    df = df.drop(column, axis=1)
    return df

inputs = headers[1:]

print('\n==================================================================\n' +
      'One Hot Encoder')
one_hot = OneHotEncoder(sparse=False)
ohe_mushrooms_inputs = pd.DataFrame(data=encode(one_hot, mushrooms_inputs))
shuffle_and_test(mushrooms_targets, ohe_mushrooms_inputs)

print('\n==================================================================\n' +
      'Ordinal Encoder')
ordinal = OrdinalEncoder()
oe_mushrooms_inputs = pd.DataFrame(data=encode(ordinal, mushrooms_inputs), columns=inputs)
shuffle_and_test(mushrooms_targets, oe_mushrooms_inputs)

print('\n==================================================================\n' +
      'Binary Encoder')
be_inputs = oe_mushrooms_inputs

for column in inputs:
    be_inputs = columnToBinary(be_inputs, column)

shuffle_and_test(mushrooms_targets, be_inputs)