from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNClassifier:
    def __init__(self, n_neighbors=5, algorithm='auto', leaf_size=30, n_jobs=-1):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, n_jobs=n_jobs)

    def fit(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"KNN classifier accuracy: {accuracy}")
        return y_pred
