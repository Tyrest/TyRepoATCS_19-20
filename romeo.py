import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score

np.random.seed(94027)
pd.set_option("display.max_columns",None)

def parentPreprocessing(x):
    levels = ['some high school', 'high school', 'some college',
    'associate\'s degree', 'bachelor\'s degree', 'master\'s degree']

    return levels.index(x)

def studentsPreprocessing():
    # Import Data
    X = pd.read_csv('Data/StudentsPerformance.csv')

    # One Hot Encode Columns
    cols_to_encode = ['gender', 'race/ethnicity']

    enc = OneHotEncoder(sparse=False)
    encodeCols = enc.fit_transform(X[cols_to_encode])
    encColNames = ['Male', 'Female', 'R1', 'R2', 'R3', 'R4', 'R5']
    X = pd.concat([X, pd.DataFrame(encodeCols, columns=encColNames)], axis=1)
    # X = pd.concat([X, pd.DataFrame(encodeCols)], axis=1)
    X = X.drop(cols_to_encode, axis=1)

    # Encode Columns based on privelege
    X['parental level of education'] = X['parental level of education'].apply(parentPreprocessing)
    X['lunch'] = X['lunch'].apply(lambda x : 0 if x == 'free/reduced' else 1)
    X['test preparation course'] = X['test preparation course'].apply(lambda x : 0 if x == 'none' else 1)
    X['average score'] = (X['math score'] + X['reading score'] + X['writing score']) / 3

    return X

def exploreClustering(X):
    print("Using DBScan...")
    for num in range(5, 28, 1):
        DBS = DBSCAN(eps=num/10)
        cluster_labels = DBS.fit_predict(X)
        num_clusters = len(np.unique(cluster_labels)) - 1

        if num_clusters > 1:
            silhouette_avg = silhouette_score(X, cluster_labels)
            calhara = calinski_harabasz_score(X, cluster_labels)
            db = davies_bouldin_score(X, cluster_labels)
            print("For eps={0:2.2f}, {1:2d} clusters, with silhouette: {2:2.3f} calinski-harabasz: {3:2.3f} davies-bouldin: {4:2.3f}".format(
                num/10, num_clusters, silhouette_avg, calhara, db))
        else:
            print("For eps={0:2.2f}, only 1 cluster found.".format(num/10))

    print("\nUsing MiniBatchKMeans...")
    for num in range(2, 10, 1):
        MBKM = MiniBatchKMeans(n_clusters=num)
        cluster_labels = MBKM.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        calhara = calinski_harabasz_score(X, cluster_labels)
        db = davies_bouldin_score(X, cluster_labels)
        print("For{0:2d} clusters, with silhouette: {1:2.3f} calinski-harabasz: {2:2.3f} davies-bouldin: {3:2.3f}".format(
            num, silhouette_avg, calhara, db))

def exploreClustersDescribe(dataset, X, test_eps):
    algorithm = DBSCAN(eps=test_eps)
    Y = algorithm.fit_predict(X)

    print("Number of inputs: " + str(len(X)))
    total = 0
    clusters = []
    for foo in np.unique(Y):
        if foo != -1:
            clusters.append(dataset[Y==foo])
            print("Now collecting: ",foo,", num= ",len(clusters[foo]))
            total += len(clusters[foo])

    print("Total in clusters: ",total)


    for i in range(len(clusters)):
        print("\n\n--- Looking at cluster ",i,' ---\n')
        print("MODE: \n",clusters[i].mode())
        print("STATS: \n",clusters[i].describe())

students_dataset = studentsPreprocessing()
scaler = StandardScaler()
X = scaler.fit_transform(students_dataset)
exploreClustering(X)
exploreClustersDescribe(students_dataset, X, 2.2)
