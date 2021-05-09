import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    test_outputs = knn.predict(test_inputs)
    print("Mean test accuracy:", knn.score(test_inputs, test_targets))
    print("Confusion Matrix:")
    print(confusion_matrix(test_targets, test_outputs))

# ==========================================================================================
# MNIST Number Classification

print('==================================================================\n' +
      'MNIST Number Classification')

mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]

test_targets = mnist_test[0]
test_inputs = mnist_test[mnist_test.columns[1:]]

train_targets, train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)
print("max_depth=16, min_samples_leaf=5, random_state=7, criterion='entropy'\n" +
      "Used these values because in testing, these gave the best results\n" +
      "Used a for-loop to loop through options for max_depth and min_samples_leaf and found where it peaked\n" +
      "Found that changing criterion to entropy yielded better results")
tree = DecisionTreeClassifier(max_depth=16, min_samples_leaf=5, random_state=7, criterion='entropy')

decisionTree(tree, train_inputs, train_targets, test_inputs, test_targets)

# ==========================================================================================
# Titanic Survived Classification

print('==================================================================\n' +
      '\nTitantic Survived Classification')

titanic_train = pd.read_csv("titanic_train2.csv")
titanic_test = pd.read_csv("titanic_test2.csv")

def alter_column(column, func):
    titanic_train[column] = titanic_train[column].apply(func)
    titanic_test[column] = titanic_test[column].apply(func)

# Column modifications
alter_column('Sex', lambda x : 1 if x == "female" else 0)   # Change gender into numbers

# Columns available: ['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Fare']
titanic_train_targets = titanic_train['Survived']
titanic_train_inputs = titanic_train[['Sex']]

test_targets = titanic_test['Survived']
test_inputs = titanic_test[['Sex']]

train_targets, train_inputs = shuffle(titanic_train_targets, titanic_train_inputs, random_state=42)

print("max_depth=1, random_state=7\n" +
      "Used these values because in testing, nothing changed when decreasing max depth or changing min_samples_leaf\n" +
      "Set to 1 purely to increase efficiency by that little amount")
tree = DecisionTreeClassifier(max_depth=1, random_state=7)
decisionTree(tree, train_inputs, train_targets, test_inputs, test_targets)

# ==========================================================================================
# Radio Noise Pulsar Classification

print('==================================================================\n' +
      'Radio Noise Pulsar Classification')

pulsar_train = pd.read_csv("pulsar_train.csv")
pulsar_test = pd.read_csv("pulsar_test.csv")

train_targets = pulsar_train['Pulsar']
train_inputs = pulsar_train.drop('Pulsar', axis=1)
test_targets = pulsar_test['Pulsar']
test_inputs = pulsar_test.drop('Pulsar',axis=1)

softmax_reg = LogisticRegression(solver='liblinear', random_state=7)
logisticRegression(softmax_reg, train_inputs, train_targets, test_inputs, test_targets)
print("solver='liblinear', random_state=7")

tree = DecisionTreeClassifier(max_depth=1, random_state=7)
decisionTree(tree, train_inputs, train_targets, test_inputs, test_targets)
print("max_depth=1, random_state=7")

knn = KNeighborsClassifier(n_neighbors=11)
kNearestNeighbors(knn, train_inputs, train_targets, test_inputs, test_targets)
print("n_neighbors=11")

# ==========================================================================================
# Breast Tissue Malignant Cancer Cells Classification

print('==================================================================\n' +
      'Breast Tissue Malignant Cancer Cells')

cancer_train = pd.read_csv("cancer_train.csv")
cancer_test = pd.read_csv("cancer_test.csv")

cancer_train['Class'] = cancer_train['Class'].apply(lambda x : 1 if x == 4 else 0)
cancer_test['Class'] = cancer_test['Class'].apply(lambda x : 1 if x == 4 else 0)
# cancer_train['Bare Nuclei'] = cancer_train['Bare Nuclei'].apply(lambda x : 1 if x == '?' else x)
# cancer_test['Bare Nuclei'] = cancer_test['Bare Nuclei'].apply(lambda x : 1 if x == '?' else x)

# Dropped Bare Nuclei for accuracy
train_targets = cancer_train['Class']
train_inputs = cancer_train.drop(['Class', 'Bare Nuclei'], axis=1)
test_targets = cancer_test['Class']
test_inputs = cancer_test.drop(['Class', 'Bare Nuclei'], axis=1)

softmax_reg = LogisticRegression(solver='liblinear', random_state=7)
logisticRegression(softmax_reg, train_inputs, train_targets, test_inputs, test_targets)
print("solver='liblinear', random_state=7")

tree = DecisionTreeClassifier(max_depth=3, random_state=7)
decisionTree(tree, train_inputs, train_targets, test_inputs, test_targets)
print("max_depth=3, random_state=7")

knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
kNearestNeighbors(knn, train_inputs, train_targets, test_inputs, test_targets)
print("n_neighbors=4")