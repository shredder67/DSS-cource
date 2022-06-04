import numpy as np


def logit(x, w):
    return np.dot(x, w)


def sigmoid(h):
    return 1. / (1 + np.exp(-h))

# TODO:
# 1. Logloss for each class - DONE
# 2. Calculate the importance of features (w and b)


class MyLogisticRegression:

    def __init__(self):
        self.w  = None

    def fit(self, X, y, max_iter=100, lr=0.1):
        n, k = X.shape
        
        if self.w is None:
            self.w = np.ones(k + 1)
        
        X_train = np.concatenate((X, np.ones((n, 1))), axis=1)
        losses = []
        sep_losses = []
        for iter_num in range(max_iter):
            z = sigmoid(logit(X_train, self.w))
            grad = np.dot(X_train.T, (z - y)) / n
            self.w -= grad * lr
            sep_losses.append(self._loss(y, z))
            losses.append(sep_losses[-1][0] + sep_losses[-1][1])

        return losses, sep_losses

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((X, np.ones((n, 1))), axis=1)
        return sigmoid(logit(X_, self.w))

    def score(self, y_true, pred_probs):
        return self._loss(y_true, pred_probs)

    def get_metrics(self, y_true, y_pred):
        metrics = {}
        diff = y_pred - y_true
        metrics['Accuracy'] = 1.0 - (float(np.count_nonzero(diff)) / len(diff))
        
        return metrics

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w

    def _loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        l1 = - np.mean(y * np.log(p))
        l2 = - np.mean((1 - y) * np.log(1 - p))
        return l1, l2
    
