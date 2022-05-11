from scipy.special import logsumexp
import numpy as np

class GaussianDistribution:
    def __init__(self, feature):
        self.mean = feature.mean(axis=0)
        self.std = feature.std(axis=0)
    
    def logpdf(self, value):
        return -0.5 * np.log(2. * np.pi * self.std**2) - 0.5 * ((value - self.mean) / self.std)**2

    def pdf(self, value):
        return np.exp(self.logpdf(value))

class MyNaiveBayesClassifier:
    def fit(self, X, y, distributions=None):
        self.unique_labels = np.unique(y)

        distributions = distributions or [GaussianDistribution] * X.shape[1]
        self.label_likelyhood = {}

        # К каждому столбцу-признаку присваиваем распределение вероятности
        for label in self.unique_labels: 
            dist_for_col = []
            for feat_idx in range(X.shape[1]):
                feature_col = X[y == label, feat_idx]
                distr = distributions[feat_idx](feature_col)
                dist_for_col.append(distr)
            self.label_likelyhood[label] = dist_for_col

        # Эмпирические вероятности классов
        self.label_prior = {
            l: float((y == l).sum()) / y.shape[0] for l in self.unique_labels 
        }

    def predict_log_proba(self, X):
        class_log_probs = np.zeros((X.shape[0], len(self.unique_labels)))
        for label_idx, label in enumerate(self.unique_labels):
            for feat_idx in range(X.shape[1]):
                class_log_probs[:, label_idx] += self.label_likelyhood[label][feat_idx].logpdf(X[:, feat_idx])

        # Вычитаем логарифм вероятности x_i в целом (для этого суммируем все условные вероятности)
        for idx in range(X.shape[1]):
            class_log_probs -= logsumexp(class_log_probs, axis=1)[:, None]
        return class_log_probs

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)