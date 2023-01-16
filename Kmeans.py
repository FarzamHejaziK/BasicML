from matplotlib import pyplot as plt
import numpy as np

class kmeans:

    def __init__(self, K = 10, max_iter = 100) -> None:
        self.K = K
        self.max_iter = max_iter
        self.clusters = [[] for _ in range(K)]
        self.centroids = []
        self.no_change = False

    def predict(self,X):
        ## plot base 
        self.X = X
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        idx = np.random.choice(len(X), self.K, replace = False)
        self.centroids = self.X[idx]
        self.cluster_mapping = [0 for i in range(len(X))]
        for i in range(self.max_iter):
            if self.no_change: return
            self._update_clusters()
        
    def _update_clusters(self):
        self.clusters = [[] for _ in range(self.K)]
        self.no_change = True
        for i in range(len(self.X)):
            Distance = []
            for c in self.centroids:
                Distance.append(np.sum((self.X[i]-c)**2))
            new_cluster = np.argmin(Distance)
            self.clusters[new_cluster].append(self.X[i])
            if new_cluster != self.cluster_mapping[i]: 
                self.no_change = False
                self.cluster_mapping[i] = new_cluster
        if not self.no_change:
            self._update_centroids() 
        self.plot()
    
    def _update_centroids(self):
        for i in range(self.K):
            self.centroids[i] = np.mean(self.clusters[i], axis=0)
            
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for idx, clusters in enumerate(self.clusters):
            clusters = np.array(clusters)
            print(clusters.shape,clusters[:,0].shape,clusters[:,1].shape)
            ax.scatter(clusters[:,0], clusters[:,1], marker="o")#, c=[idx for _ in range(len(clusters[:,1]))])

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()









