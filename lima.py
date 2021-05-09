import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import Perceptron

iris_data = pd.read_csv("iris_data.csv")
titanic_train = pd.read_csv("titanic_train2.csv")
titanic_test = pd.read_csv("titanic_test2.csv")

print('Tyresty\'s assignment Lima')

def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))

def column_to_targets(col, target):
    if col == target:
        return 1
    else:
        return 0

def fit_line_iris(inputs, toClassify):
    iris_inputs = iris_data[inputs]
    iris_targets = iris_data['class'].apply(lambda x : column_to_targets(x, toClassify))

    percey = Perceptron(n_iter_no_change=50)
    percey.fit(iris_inputs, iris_targets)
    percey_outputs = percey.predict(iris_inputs)

    print('\nUsed ' + str(inputs) + ' to classify ' + toClassify)

    print_conf_matrix(iris_targets, percey_outputs)

# 1) use a Perceptron to classify setosa with the petal inputs

fit_line_iris(['petal width', 'petal length'], 'Iris-setosa')

# 2) use a Perceptron to classify virginica with the sepal inputs.

fit_line_iris(['sepal width', 'sepal length'], 'Iris-virginica')

# 3) use a Perceptron to classify virginica with the petal inputs.

fit_line_iris(['petal width', 'petal length'], 'Iris-virginica')

# 4) use a Perceptron to classify virginica with all four inputs.

fit_line_iris(['petal width', 'petal length', 'sepal width', 'sepal length'], 'Iris-virginica')

# 5) use a Perceptron to classify survivors of the Titanic.

def get_column_score(inputs):
    t_train_inputs = titanic_train[inputs]
    t_train_targets = titanic_train['Survived'].apply(lambda x : column_to_targets(x, 1))

    percey = Perceptron(n_iter_no_change=15)
    percey.fit(t_train_inputs, t_train_targets)
    return percey.score(t_train_inputs, t_train_targets)

def test_titanic(inputs):
    t_train_inputs = titanic_train[inputs]
    t_train_targets = titanic_train['Survived'].apply(lambda x : column_to_targets(x, 1))

    percey = Perceptron(n_iter_no_change=15)
    percey.fit(t_train_inputs, t_train_targets)
    percey_train_outputs = percey.predict(t_train_inputs)

    t_test_inputs = titanic_test[inputs]
    t_test_targets = titanic_test['Survived'].apply(lambda x : column_to_targets(x, 1))
    percey_test_outputs = percey.predict(t_test_inputs)

    print('Used ' + bestColumn + ' to predict survival')
    print('Training Data Confusion Matrix:')
    print_conf_matrix(t_train_targets, percey_train_outputs)
    print('Testing Data Confusion Matrix:')
    print_conf_matrix(t_test_targets, percey_test_outputs)

def alter_column(column, func):
    titanic_train[column] = titanic_train[column].apply(func)
    titanic_test[column] = titanic_test[column].apply(func)

print('\nTitantic survivor classification')

# Column modifications
alter_column('Name', lambda x : len(x))                     # Just for fun
alter_column('Sex', lambda x : 1 if x == "female" else 0)   # Change gender into numbers

print('Column Modifications: Names = length of names, Sex = changed female to 1 and male to 0')

# Determining best column (out of the numeric ones)
training_columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Fare']
bestColumn = ''
bestAccuracy = 0
for column in training_columns:
    accuracy = get_column_score([column])
    if accuracy > bestAccuracy:
        bestColumn = column
        bestAccuracy = accuracy

# Using best column to predict new data set
test_titanic([bestColumn])