import numpy as np
import numpy.linalg as lng

class MyKMeans:

    def __init__(self):
        self.cluster_means = {}
    
    def _closest_mean(self, x, cluster_means):
        min_means = np.argmin([lng.norm(x - mean) for mean in cluster_means])
        if type(min_means) is np.ndarray:
            return min_means[0]
        return min_means

    def _init_centroids(self, x, k):
        pass

    def fit(self, X, k=None, labels=None):
        cluster_means = X[np.random.permutation(X.shape[0])][:k]
        means_changed = True
        while means_changed:
            means_changed = False
            clusters = [[] for _ in range(k)]  
            for x in X:
                clusters[self._closest_mean(x, cluster_means)].append(x)

            for i, cl in enumerate(clusters):
                mean = np.mean(cl, axis=0)
                if not np.all(mean == cluster_means[i]):
                    cluster_means[i] = mean
                    means_changed = True

        if labels:
            self.cluster_means = {i: cluster_means[i] for i in range(k)}
        else:
            self.cluster_means = {labels[i]: cluster_means[i] for i in range(k)}                          
    
    def predict(self, X):
        preds = []
        class_labels = list(self.cluster_means.keys())
        for x in X:
            preds.append(class_labels[self._closest_mean(x, self.cluster_means.values())])
        return preds