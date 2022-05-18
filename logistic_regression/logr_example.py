from logistic_regression import MyLogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# TODO:
# 1. Logloss for each class separately
# 2. OR for each weight (b1 and b0)

def main():
    # Two groups of points init
    x_0 = np.concatenate((np.random.rand(20, 1) * 10, np.zeros((20, 1))), axis=1)
    x_1  = np.concatenate((np.random.rand(20, 1) * 10 + 15, np.ones((20, 1))), axis=1)
    X_origin = np.random.permutation(np.concatenate((x_0, x_1), axis=0))
    X, y = X_origin.T[0], X_origin.T[1]
    X_train, X_test, y_train, y_test = train_test_split(X[:, np.newaxis], y, train_size=0.8)

    clf = MyLogisticRegression()
    loss_hist, sep_loss_hist = clf.fit(X_train, y_train, 50, 0.1)
    print("Classifier weights: ", clf.get_weights())

    # Logloss on test data
    test_sep_log_loss = clf.score(y_test, clf.predict_proba(X_test))
    print(f"Test Logloss: {test_sep_log_loss[0] + test_sep_log_loss[1]}")
    print(f"Test Logloss on class 1: {test_sep_log_loss[1]}")
    print(f"Test Logloss on class 2: {test_sep_log_loss[0]}")
    test_pred = clf.predict(X_test, threshold=0.5)
    print('\n'.join([f'{k}: {v}' for k,v in clf.get_metrics(y_test, test_pred).items()]))

    # Plot the results (scatter points and sigmoid function)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    xx = np.linspace(-10., 30., num=50)
    ax1.plot(xx, clf.predict_proba(xx[:, np.newaxis]))
    ax1.scatter(X_train, y_train)
    ax1.scatter(X_test, y_test, c='orange')
    ax1.grid()
    ax1.set(title='Logistic regression')

    it_num = len(loss_hist)
    xx = np.arange(0, it_num, 1)
    cl2_loss_hist, cl1_loss_hist = zip(*sep_loss_hist)
    ax2.plot(xx, cl1_loss_hist, c='orange', label='class 1 loss')
    ax2.plot(xx, cl2_loss_hist, c='gray', label='class 2 loss')
    ax2.plot(xx, loss_hist, linewidth=2.0)
    ax2.grid()
    ax2.set(title='Logloss curve')
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    main()