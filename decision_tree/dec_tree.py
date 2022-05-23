from cmath import inf
from re import M
import numpy as np
from sklearn.base import BaseEstimator
from sympy import N


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    n = float(len(y))
    p = np.sum(y, axis=0) / n
    return 1 - np.sum(p * np.log(p + EPS))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    n = float(len(y))
    p = np.sum(y, axis=0) / n # probability of each class
    return 1 - np.sum(np.square(p))


def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    return np.var(y)


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
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
        self.predicted_value = None
        self.left_child = None
        self.right_child = None

    def __str__(self):
        s = ''
        s += f'feature_idx: {self.feature_index}\n'
        s += f'value: {self.value}\n'
        s += f'predicted_value: {self.predicted_value}\n'
        return s
        
        
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
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        less = X_subset[:, feature_index] < threshold
        X_left, X_right = X_subset[less, :], X_subset[~less, :]
        y_left, y_right = y_subset[less, :], y_subset[~less, :]
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """
        less = X_subset[:, feature_index] < threshold
        y_left, y_right = y_subset[less, :], y_subset[~less, :]
        return y_left, y_right  

    def calc_G(self, y, y_left, y_right):
        L = float(len(y_left))
        R = float(len(y_right))
        N = L + R
        return N * self.criterion(y) - L * self.criterion(y_left) - R * self.criterion(y_right)

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        D = X_subset.shape[1]
        min_loss = 0
        feature_idx = None
        optimal_thesh = None
        
        for j in range(D):
            for tresh in X_subset[:, j]:
                y_left, y_right = self.make_split_only_y(j, tresh, X_subset, y_subset)
                loss = self.calc_G(y_subset, y_left, y_right)
                if loss > min_loss:
                    feature_idx = j
                    optimal_thesh = tresh
                    min_loss = loss
        return feature_idx, optimal_thesh

    def check_stop_criterion(self, y_subset, cur_depth):
        """
        Checks if node should be turned into leaf. This can happend if
        1. self.max_depth has been reached
        2. self.min_samples split has been reached
        3. G(y) is small enough (in case of homogeneity G := 0)
        """
        should_stop = False
        should_stop |= cur_depth == self.max_depth
        should_stop |= len(y_subset) < self.min_samples_split
        should_stop |= self.criterion(y_subset) < 0.05
        return should_stop
    
    def make_tree(self, X_subset, y_subset, cur_depth):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        new_node = Node(None, None)
        if self.check_stop_criterion(y_subset, cur_depth): # This is a leaf-node
            if self.classification:
                y_subset = one_hot_decode(y_subset)
                labels, label_freqs = np.unique(y_subset[:, 0], return_counts=True)
                new_node.predicted_value = labels[np.argmax(label_freqs)]
                new_node.proba = label_freqs / float(len(y_subset))
            else:
                new_node.predicted_value = np.mean(y_subset[:, 0])
        else:
            feature_idx, thresh_value = self.choose_best_split(X_subset, y_subset)
            left_split, right_split = self.make_split(feature_idx, thresh_value, X_subset, y_subset)
            new_node.feature_index = feature_idx
            new_node.value = thresh_value
            new_node.left_child = self.make_tree(left_split[0], left_split[1], cur_depth + 1)
            new_node.right_child = self.make_tree(right_split[0], right_split[1], cur_depth + 1)
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        
        # One-hot encode labels for classification problem
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y, 0)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        y_predicted = np.ndarray((len(X), 1), dtype=np.int32)
        for i, x in enumerate(X):
            y_predicted[i] = self.make_prediction(x, self.root)
        
        return y_predicted

    def make_prediction(self, x, cur_node : Node, proba=False):
        if cur_node.left_child and x[cur_node.feature_index] < cur_node.value:
            return self.make_prediction(x, cur_node.left_child)
        if cur_node.right_child and x[cur_node.feature_index] >= cur_node.value:
            return self.make_prediction(x, cur_node.right_child)
        return cur_node.predicted_value if not proba else cur_node.proba
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = np.ndarray((len(X), self.n_classes))
        for i, x in enumerate(X):
            y_predicted_probs[i] = self.make_prediction(x, self.root, proba=True)
        
        return y_predicted_probs

    def get_str_nodes(self):
        queue = [self.root]
        res = ''
        while len(queue) > 0:
            cur = queue.pop(0)
            res += '\n--------\n' + str(cur) + '--------\n'
            if cur.left_child:
                queue.append(cur.left_child)
            if cur.right_child:
                queue.append(cur.right_child)
        return res