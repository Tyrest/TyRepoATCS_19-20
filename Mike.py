import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Helper functions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

# MNIST data set of handwritten digits
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

mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]

mnist_test_targets = mnist_test[0]
mnist_test_inputs = mnist_test[mnist_test.columns[1:]]

# Now let's shuffle the training set to reduce bias opportunities
smn_train_targets, smn_train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

# Softmax Regression or Multinomial Logistic Regression!
print("Training a Multinomial Logistic Regression classifier for ALL digits!")
softmax_reg = LogisticRegression(penalty="none", multi_class="multinomial", solver="saga", n_jobs=-1)
softmax_reg.fit(mnist_train_inputs, mnist_train_targets)
softmax_outputs = softmax_reg.predict(mnist_train_inputs)
print("Mean accuracy:")
print(softmax_reg.score(mnist_train_inputs, mnist_train_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets, softmax_outputs))