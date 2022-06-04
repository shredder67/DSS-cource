from logistic_regression import MyLogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# TODO:
# 1. Logloss for each class separately
# 2. OR for each weight (b1 and b0)

def main():
    # Two groups of points init
    num_of_points = 30
    p_std = 10
    class_delta = 40
    x_0 = np.concatenate((np.random.rand(num_of_points, 1) * p_std, np.zeros((num_of_points, 1))), axis=1)
    x_1  = np.concatenate((np.random.rand(num_of_points, 1) * p_std + class_delta, np.ones((num_of_points, 1))), axis=1)
    X_origin = np.random.permutation(np.concatenate((x_0, x_1), axis=0))
    X, y = X_origin[:, 0], X_origin[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X[:, np.newaxis], y, train_size=0.8)

    clf = MyLogisticRegression()
    loss_hist, sep_loss_hist = clf.fit(X_train, y_train, 1000, 0.05)
    print("Classifier weights: ", clf.get_weights())

    # Logloss on test data
    test_sep_log_loss = clf.score(y_test, clf.predict_proba(X_test))
    print(f"Test Logloss: {test_sep_log_loss[0] + test_sep_log_loss[1]}")
    print(f"Test Logloss on class 0: {test_sep_log_loss[1]}")
    print(f"Test Logloss on class 1: {test_sep_log_loss[0]}")
    test_pred = clf.predict(X_test, threshold=0.5)
    print('\n'.join([f'{k}: {v}' for k,v in clf.get_metrics(y_test, test_pred).items()]))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

    # Plot the results (scatter points and sigmoid function)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    dot_size = 15.

    xx = np.linspace(-p_std - 5., class_delta + p_std + 5., num=50)
    ax1.scatter(X, y, s=dot_size, c='green')
    ax1.set(title='Generated points')
    ax2.plot(xx, clf.predict_proba(xx[:, np.newaxis]))
    ax2.scatter(X_train, y_train, s=dot_size)
    ax2.scatter(X_test, y_test, s=dot_size, c='orange')
    ax2.set(title='Logistic regression')

    it_num = len(loss_hist)
    xx = np.arange(0, it_num, 1)
    cl2_loss_hist, cl1_loss_hist = zip(*sep_loss_hist)
    ax3.plot(xx, cl1_loss_hist, c='orange', label='class 1 loss')
    ax3.plot(xx, cl2_loss_hist, c='gray', label='class 2 loss')
    ax3.plot(xx, loss_hist, linewidth=2.0)
    ax3.set(title='Logloss curve')
    ax3.legend()

    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()


if __name__ == '__main__':
    main()