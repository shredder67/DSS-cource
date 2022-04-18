import numpy as np
import numpy.linalg as lng

class MyKMeans:

    def __init__():
        pass  
    
    def _closest_mean(self, x, cluster_means):
        return np.argmin([lng.norm(x - mean) for mean in cluster_means])


    def fit(self, X, k=None, labels=None):
        clustsers = []
        cluster_means = [] # TODO: initialize means with seeds
        means_changed = True
        while means_changed:
            means_changed = False
            clusters = []
            for x in X:
                clusters[self._closest_mean(x, self.cluster_means)].append(x)

            for i, cl in enumerate(clusters):
                mean = np.mean(cl)
                if mean != cluster_means[i]:
                    cluster_means[i] = mean
                    means_changed = True

        if labels:
            self.cluster_means = {i: cluster_means[i] for i in range(k)}
        else:
            self.cluster_means = {labels[i]: cluster_means[i] for i in range(k)}                          
    
    def predict(self, X):
        preds = []
        class_labels = self.cluster_means.keys
        for x in X:
            preds.append(class_labels[self._closest_mean(x, self.cluster_means.values)])
        return preds