import numpy as np
import numpy.linalg as lng

class MyLinearRegression:
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y):
        n, k = X.shape
        if self.fit_intercept:
            X = np.hstack((X, np.ones(n, 1)))
        
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y # Аналитическое решение методом псевдообратной матрицы
        return self
    
    def predict(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = X_train @ self.w

        return y_pred

    def get_weights(self):
        return self.w
    
    def score(self, y_true, y_pred):
        assert y_true.shape[0] == y_pred.shape[0]
        n = y_true.shape[0]
        m = len(self.w) - 1
        y_mean = y_true.mean()
        self.scores = dict()

        # TODO: calculate other metrics
        self.scores['MSE'] = np.square(y_true - y_pred).mean()
        self.scores['SD'] = self.scores['MSE'] ** 0.5
        self.scores['Q_R'] = np.sum(np.square(y_mean - y_pred))
        self.scores['Q_E'] = 
        self.scores['R_2'] = 
        self.scores['corr'] = 

        return self.scores

    def predict_score(self, X, y_true):
        return self.score(y_true, self.predict(X))

