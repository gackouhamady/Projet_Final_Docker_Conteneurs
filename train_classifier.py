from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import numpy as np

import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, n_informative = 2, n_classes = 2, n_clusters_per_class = 1, class_sep = 1.0, n_samples = 10000):
        self.n_informative = n_informative
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.class_sep = class_sep
        self.n_samples = n_samples
        self.X, self.y = make_classification( n_classes=self.n_classes, \
                                            n_informative=self.n_informative, \
                                            n_clusters_per_class=self.n_clusters_per_class, \
                                            class_sep=self.class_sep, \
                                            n_samples=self.n_samples, \
                                            scale=100)

        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = self._generate_split()

    def _generate_split(self):
        X_train, X_rest, y_train, y_rest = train_test_split(self.X, self.y, test_size=0.3, random_state=1)
        X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=(1/3), random_state=1)

        np.savetxt("train.csv", np.hstack((X_train, y_train.reshape(-1, 1))), delimiter=",")
        np.savetxt("test.csv", np.hstack((X_test, y_test.reshape(-1, 1))), delimiter=",")
        np.savetxt("validation.csv", np.hstack((X_val, y_val.reshape(-1, 1))), delimiter=",")

        return X_train, y_train, X_test, y_test, X_val, y_val

    def get_train(self):
        return self.X_train, self.y_train
    def get_test(self):
        return self.X_test, self.y_test
    def get_validation(self):
        return self.X_val, self.y_val
    
    def reduce_dataset(self):
        pca = PCA(n_components=2, random_state=42)
        pca.fit(self.X_train)
        return pca.transform(self.X_train)
    
    def visualize(self):
        if self.X.shape[1] > 2:
            X = self.reduce_dataset()
        else:
            X = self.X_train
        plt.title("Dataset")
        plt.ylabel('X2')
        plt.xlabel('X1')
        c = ["red" if y == 1 else "blue" for y in self.y_train]
        plt.scatter(X[:,0], X[:,1], c=c)
        plt.show()

data = Dataset(n_informative=2, n_classes=2, n_clusters_per_class=1, class_sep=1.3)
data.visualize()