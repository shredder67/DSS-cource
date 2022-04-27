from k_means import MyKMeans, get_dist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def main():
    # Инициализация двух групп точек
    x_1 = np.random.rand(100, 2) * 100
    x_2  = np.random.rand(100, 2) * 100 + 50
    X = np.random.permutation(np.concatenate((x_1, x_2), axis=0))
    p = int(0.95 * X.shape[0])

    clf = MyKMeans()
    k = 3
    clf.fit(X, k, labels=['X_1', 'X_2'])
    cluster_means = clf.get_cluster_means()
    print('Calculated means:')
    print('\n'.join([f'{k}: {v}' for k, v in cluster_means.items()]))

    # Make predictions
    classes = clf.predict(X)

    # Format a .csv report for each data point
    point_records = []
    for i, point in enumerate(X):
        point_records.append((point, classes[i], get_dist(point, clf.cluster_means[classes[i]])))
    df = pd.DataFrame.from_records(point_records, columns=['Point', 'Cluster', 'Distance to Mean'])
    df.to_csv('kmeans.csv')

    # Plot the results
    xx, yy = zip(*X)
    x_means, y_means = zip(*cluster_means.values())
    sns.set_style("dark")
    sns.scatterplot(x=xx, y=yy, hue=classes, palette='tab10')
    plt.scatter(x_means, y_means, c='black', marker='D', linewidths=2.)
    plt.show()

if __name__ == '__main__':
    main()