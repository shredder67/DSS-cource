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
    print(clf.cluster_means)
    
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
    sns.scatterplot(x=xx, y=yy, hue=classes, palette='Set1')
    plt.show()

if __name__ == '__main__':
    main()