from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from Kmeans import kmeans

X, y = datasets.make_blobs(
n_samples=5000, n_features=2, centers=12, cluster_std=0.3, random_state=40
)


km = kmeans(K = 15, max_iter = 100)
km.predict(X)

