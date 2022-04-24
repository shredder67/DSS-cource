from logistic_regression.logistic_regression import MyLogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():
    # Инициализация двух групп точек
    x_0 = np.concatenate((np.random.rand(100, 1) * 100, np.zeros(shape=(100, 1))), axis=1)
    x_1  = np.concatenate((np.random.rand(100, 1) * 100 + 50, np.ones((100, 1))), axis=1)
    X = np.random.permutation(np.concatenate((x_0, x_1), axis=0))
    X, y = X.T[0], X.T[1]
    X_train, X_test, y_train, y_test = train_test_split(X[:, np.newaxis], y, train_size=0.85)

    clf = MyLogisticRegression()
    clf.fit(X_train, y_train, 30, 0.1)
    print("Classifier weights: ", clf.get_weights())

    # Logloss на тестовых данных
    print("Test Loss: ", clf.score(y_test, clf.predict_proba(X_test)))

    # Plot the results
    predictions = clf.predict_proba(X[:, np.newaxis])

    plt.show()


if __name__ == '__main__':
    main()