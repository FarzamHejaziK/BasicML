from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

import numpy as np
from SVM import SVM

X, y = datasets.make_blobs(
n_samples=50, n_features=2, centers=3, cluster_std=1.05, random_state=40
)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=123
)

clf = SVM(learning_rate = 2, iter = 1000, reg = 0.001 )
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


print("SVM classification accuracy", accuracy(y_test, predictions))
def visualize_svm():


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    #ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()

visualize_svm()