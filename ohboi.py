import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score

from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

def kNearestNeighbors(knn, train_inputs, train_targets, test_inputs, test_targets):
    print("\nK Nearest Neighbors classifier")
    knn.fit(train_inputs, train_targets)
    test_outputs = knn.predict(test_inputs)
    print("Mean test accuracy:", knn.score(test_inputs, test_targets))
    print("Confusion Matrix:")
    print(confusion_matrix(test_targets, test_outputs))

mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

def normalize(column):
    mnist_train[column] = mnist_train[column] - mnist_train[column].min()
    mnist_test[column] = mnist_test[column] - mnist_test[column].min()
    mnist_train[column] = mnist_train[column] / mnist_train[column].max()
    mnist_test[column] = mnist_test[column] / mnist_test[column].max()

def ink_used():
    mnist_train['Ink_Used'] = mnist_train[mnist_train.columns[1:785]].sum(axis=1)
    mnist_test['Ink_Used'] = mnist_test[mnist_test.columns[1:785]].sum(axis=1)
    normalize('Ink_Used')

def ink_top():
    mnist_train['Ink_Top'] = mnist_train[mnist_train.columns[1:337]].sum(axis=1)
    mnist_test['Ink_Top'] = mnist_test[mnist_test.columns[1:337]].sum(axis=1)
    normalize('Ink_Top')

def ink_bot():
    mnist_train['Ink_Bot'] = mnist_train[mnist_train.columns[337:785]].sum(axis=1)
    mnist_test['Ink_Bot'] = mnist_test[mnist_test.columns[337:785]].sum(axis=1)
    normalize('Ink_Bot')

def ink_topbot():
    mnist_train['Ink_TopBot'] = mnist_train['Ink_Top'] - mnist_train['Ink_Bot']
    mnist_test['Ink_TopBot'] = mnist_test['Ink_Top'] - mnist_test['Ink_Bot']
    normalize('Ink_TopBot')

def ink_left():
    left_side_columns = []
    for row in range(0, 28):
        for col in range(0, 14):
            left_side_columns.append(row * 28 + col)

    mnist_train['Ink_Left'] = mnist_train[mnist_train.columns[left_side_columns]].sum(axis=1)
    mnist_test['Ink_Left'] = mnist_test[mnist_test.columns[left_side_columns]].sum(axis=1)
    normalize('Ink_Left')

def ink_right():
    right_side_columns = []
    for row in range(0, 28):
        for col in range(14, 28):
            right_side_columns.append(row * 28 + col)
    
    mnist_train['Ink_Right'] = mnist_train[mnist_train.columns[right_side_columns]].sum(axis=1)
    mnist_test['Ink_Right'] = mnist_test[mnist_test.columns[right_side_columns]].sum(axis=1)
    normalize('Ink_Right')

def ink_lr():
    mnist_train['Ink_LR'] = mnist_train['Ink_Left'] - mnist_train['Ink_Right']
    mnist_test['Ink_LR'] = mnist_test['Ink_Left'] - mnist_test['Ink_Right']
    normalize('Ink_LR')

print("Normalizing to [0, 1]")
for i in range(1, 785):
    mnist_train[i] = mnist_train[i] / 255
    mnist_test[i] = mnist_test[i] / 255

print("Finding amount of Ink...")
ink_used()
print("Finding ink in top and bot")
ink_top()
ink_bot()
ink_topbot()
print("Finding ink in left and right")
ink_left()
ink_right()
ink_lr()

train_targets = mnist_train[0]
train_inputs = mnist_train[mnist_train.columns[1:]]

test_targets = mnist_test[0]
test_inputs = mnist_test[mnist_test.columns[1:]]

t0 = time.time()
# knn = KNeighborsClassifier(n_jobs=-1)
# kNearestNeighbors(knn, train_inputs, train_targets, test_inputs, test_targets)
print(str(time.time() - t0) + " seconds")