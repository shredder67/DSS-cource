from linear_regression import MyLinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():

    def f(x):
        return 7*x + 10

    obj_num = 25
    X = np.linspace(-10, 10, obj_num)
    X = np.random.permutation(X)
    y = f(X) + np.random.randn(obj_num) * 5

    prop = int(0.75 * X.shape[0])
    X_train, X_test, y_train, y_test = X[:prop], X[prop:], y[:prop], y[prop:]

    # Train model
    model = MyLinearRegression()
    model.fit(X_train[:, np.newaxis], y_train)

    # Test model
    y_pred = model.predict(X_test[:, np.newaxis])
    metrics = pd.Series(model.score(y_test, y_pred))
    w = pd.Series(model.get_weights())
    print(metrics)
    print(w)

    # Display results
    plt.plot(X, f(X), label='real', c='g')
    plt.plot(X, model.predict(X[:, np.newaxis]), label='predicted', c='r')
    plt.scatter(X_train, y_train, label='train', c='b')
    plt.scatter(X_test, y_test, label='test', c='orange')

    plt.title("Linear regression")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()