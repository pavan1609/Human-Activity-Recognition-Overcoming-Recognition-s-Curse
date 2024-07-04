from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

class Classifier_ENSEMBLE:
    def __init__(self, output_dir, input_shape, nb_classes, verbose=False, n_jobs=-1):
        self.output_dir = output_dir
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.verbose = verbose
        self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs)

    def fit(self, x_train, y_train, x_test, y_test):
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
        self.classifier.fit(x_train, y_train)
        y_pred = self.classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Ensemble classifier accuracy: {accuracy}")
        return y_pred

    def predict(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.classifier.predict(x)
