import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# print('Class labels:', np.unique(y))

# Splitting data into 70% training and 30% test data:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


# Standardizing the features:

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

# # K-nearest neighbors - a lazy learning algorithm
def knn_integral(n_neighbors, X_train_std, X_test_std, y_train, y_test, show_graph=False):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               p=2,
                               metric='minkowski')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    print('Number of neighbors: %d' % n_neighbors)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('-' * 60)

    if show_graph:
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        plot_decision_regions(X_combined_std, y_combined,
                              classifier=knn, test_idx=range(105, 150))

        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.tight_layout()

        plt.show()
    return n_neighbors, (y_test != y_pred).sum(), accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    result = {}
    # for state in range(1, 11):
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.3, random_state=1, stratify=y)
    for i in range(1, 16):
        index, mis_sample, acc_score = knn_integral(i, X_train_std, X_test_std, y_train, y_test)
        if index not in result:
            result[index] = {}
            result[index]['mis_sample'] = 0
            result[index]['acc_score'] = 0
        result[index]['mis_sample'] += mis_sample
        result[index]['acc_score'] += acc_score
    print(result)
    print("My name is Yijun Lou\n"
          "My NetId is: ylou4\n"
          "I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")