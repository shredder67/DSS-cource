import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):
    EPS = 0.0005
    n = float(y.shape[0])
    p = np.sum(y, axis=0) / n # probability of each class 
    return - np.sum(np.log(p + EPS) * p)


def gini(y):
    n = float(y.shape[0])
    p = np.sum(y, axis=0) / n # probability of each class
    return 1 - np.sum(np.square(p))


def variance(y):
    return np.var(y)


def mad_median(y):
    return np.mean(abs(y - np.mean(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        self.predicted_value = None # this value makes sense only in leaves


class MyDecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug
 
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        greater_or_equal = X_subset[:, feature_index] >= threshold
        less = X_subset[:, feature_index] < threshold
        X_left, X_right = X_subset[less, :], X_subset[greater_or_equal, :]
        y_left, y_right = y_subset[less, :], y_subset[greater_or_equal, :]
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        greater_or_equal = X_subset[:, feature_index] >= threshold
        less = X_subset[:, feature_index] < threshold
        y_left, y_right = y_subset[less, :], y_subset[greater_or_equal, :]
        return y_left, y_right

    def make_split_only_x(self, feature_index, threshold, X_subset):
        greater_or_equal = X_subset[:, feature_index] >= threshold
        less = X_subset[:, feature_index] < threshold
        return X_subset[less, :], X_subset[greater_or_equal, :]

    def choose_best_split(self, X_subset, y_subset):
        # for each feature
        # calculate optimal threshold
        # select best feature, threshold pair based on selected criterion
        def calc_g(y, y_left, y_right): # add memo to not reculc every time
            L = float(y_left.shape[0])
            R = float(y_right.shape[0])
            return self.criterion(y) - (L / (L + R)) * self.criterion(y_left) - (R / (L + R)) * self.criterion(y_right) 

        threshold = None
        feature_idx = 0
        tr_G = self.criterion(y_subset)
        # Problem: this is wildly uneffective, but I can't find a correct approach for programming this
        for f_idx in range(X_subset.shape[1]):
            sorted_values = np.sort(X_subset[:, f_idx])
            for v in sorted_values[self.min_samples_split - 1 : -self.min_samples_split + 1]:
                y_left, y_right = self.make_split_only_y(f_idx, v, X_subset, y_subset)
                G = calc_g(y_subset, y_left, y_right)
                if G < tr_G:
                    threshold = v
                    feature_idx = f_idx
                    tr_G = G

        return feature_idx, threshold
    
    def make_tree(self, X_subset, y_subset, height):
        new_node = Node(None, None)
        self.depth = max(height + 1, self.depth)
        if height <= self.max_depth and y_subset.shape[0] > self.min_samples_split and np.count_nonzero((np.sum(y_subset, axis=0) != 0)) > 1:
            feature_index, threshold = self.choose_best_split(X_subset, y_subset)
            new_node.feature_index = feature_index
            new_node.value = threshold
            left_split, right_split = self.make_split(feature_index, threshold, X_subset, y_subset)
            new_node.left_child = self.make_tree(*left_split, height + 1)
            new_node.right_child = self.make_tree(*right_split, height + 1)
        else:
            if self.classification:
                    class_freqs = np.sum(y_subset, axis=0)
                    new_node.predicted_value = np.argmax(class_freqs) # idx of most probable class
                    new_node.proba = np.max(class_freqs) / float(y_subset.shape[0]) # probabilty of correct class
            else:
                new_node.predicted_value = np.mean(y_subset)
        return new_node
            
        
    def fit(self, X, y):
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y, 0)
        self.is_fitted
    
    def predict(self, X):
        y_predicted = np.ndarray((X.shape[0], 1))
        for i, x in enumerate(X):
            y_predicted[i] = self.make_prediction(x, self.root)
        return y_predicted

    def make_prediction(self, x, cur_node : Node):
        if cur_node.predicted_value is None:
            if x[cur_node.feature_index] < cur_node.value:
                return self.make_prediction(x, cur_node.left_child)
            else:
                return self.make_prediction(x, cur_node.right_child)
        return cur_node.predicted_value