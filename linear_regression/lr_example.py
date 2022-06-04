from linear_regression import MyLinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():

    def f(x):
        return 7*x + 10

    obj_num = 5
    X = np.linspace(-10, 10, obj_num)
    #X = np.random.permutation(X)
    y = f(X) + np.random.randn(obj_num) * 3
    print(X, y)

    prop = int(1.0 * X.shape[0])
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    point_size = 10
    line_thickness = 0.8
    ax1.scatter(X, y, c='g', s=point_size)
    ax2.plot(X, f(X), label='real', c='g', linewidth=line_thickness)
    ax2.plot(X, model.predict(X[:, np.newaxis]), label='predicted', c='r',  linewidth=line_thickness)
    ax2.scatter(X_train, y_train, label='train', c='b', s=point_size)
    ax2.scatter(X_test, y_test, label='test', c='orange', s=point_size)

    ax1.set_title("Data")
    ax2.set_title("Linear regression")
    ax1.grid(alpha=0.2)
    ax2.grid(alpha=0.2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()