import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train)
        return self.model.predict(x_test)
