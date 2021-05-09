import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

def process_cancer():
    cancer_train = cancer_preprocessing(pd.read_csv("cancer_train.csv"))
    cancer_test = cancer_preprocessing(pd.read_csv("cancer_test.csv"))

    return targets_and_inputs(cancer_train, cancer_test, "Class")

def cancer_preprocessing(cancer_df):
    cancer_df['Class'] = cancer_df['Class'].apply(lambda x : 1 if x == 4 else 0)
    cancer_df = cancer_df.drop('ID', axis=1)
    cancer_df = cancer_df[cancer_df['Bare Nuclei'] != '?']
    cancer_df['Bare Nuclei'] = cancer_df['Bare Nuclei'].apply(lambda x : int(x))
    return cancer_df

def process_pulsar():
    pulsar_train = pd.read_csv("pulsar_train.csv")
    pulsar_test = pd.read_csv("pulsar_test.csv")

    return targets_and_inputs(pulsar_train, pulsar_test, "Pulsar")

def targets_and_inputs(train, test, target):
    train_targets = train[target]
    train_inputs = standardize_dataframe(train.drop(target, axis=1))
    test_targets = test[target]
    test_inputs = standardize_dataframe(test.drop(target, axis=1))

    return train_inputs, train_targets, test_inputs, test_targets

def standardize_dataframe(df):
    scaler = StandardScaler()
    scaler.fit(df)
    return pd.DataFrame(data=scaler.transform(df))

def optimize_svc(param_grid, train_inputs, train_targets):
    # hi cesar! - sarah
    svc = SVC(kernel="poly")
    gs_svc = GridSearchCV(svc, param_grid, cv=3, scoring='recall', n_jobs=-1)
    gs_svc.fit(train_inputs, train_targets)
    print(gs_svc.best_params_)
    return gs_svc

def test_gs_svc(gs_svc, test_inputs, test_targets):
    test_outputs = gs_svc.predict(test_inputs)
    print("Mean test accuracy:", gs_svc.score(test_inputs, test_targets))
    print("Confusion Matrix:")
    print(confusion_matrix(test_targets, test_outputs))

cancer_train_inputs, cancer_train_targets, \
cancer_test_inputs, cancer_test_targets = process_cancer()

cancer_param_grid = [ { 'degree': [1, 2, 3, 4, 5],
                        'gamma': [.01, .1, 1, 5, 10],
                        'C': [.00001, .0001, .001, .01, .1, 1, 10, 100] } ]

cancer_gs_svc = optimize_svc(cancer_param_grid, cancer_train_inputs, cancer_train_targets)
test_gs_svc(cancer_gs_svc, cancer_test_inputs, cancer_test_targets)

pulsar_train_inputs, pulsar_train_targets, \
pulsar_test_inputs, pulsar_test_targets = process_pulsar()

# I'm not sure I did this right but I did optimize the params...
pulsar_param_grid = [ { 'degree': [3],
                        'gamma': [0.6],
                        'C': [1] } ]

pulsar_gs_svc = optimize_svc(pulsar_param_grid, pulsar_train_inputs, pulsar_train_targets)
test_gs_svc(pulsar_gs_svc, pulsar_test_inputs, pulsar_test_targets)