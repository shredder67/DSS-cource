import numpy as np
import pandas as pd
from decision_tree.decision_tree import entropy, gini, variance, mad_median, MyDecisionTree

def main():
    X = np.ones((4, 5), dtype=float) * np.arange(4)[:, None]
    y = np.arange(4)[:, None] + np.asarray([0.2, -0.3, 0.1, 0.4])[:, None]
    class_estimator = MyDecisionTree(max_depth=10, criterion_name='gini')

    (X_l, y_l), (X_r, y_r) = class_estimator.make_split(1, 1., X, y)

    assert np.array_equal(X[:1], X_l)
    assert np.array_equal(X[1:], X_r)
    assert np.array_equal(y[:1], y_l)
    assert np.array_equal(y[1:], y_r)



if __name__ == '__main__':
    main()