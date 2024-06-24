import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

class Classifier_RESNET:
    def __init__(self, output_dir, input_shape, nb_classes, nb_prototypes, classes, verbose=False, load_init_weights=True):
        self.output_dir = output_dir
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.nb_prototypes = nb_prototypes
        self.classes = classes
        self.verbose = verbose
        self.load_init_weights = load_init_weights
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train, epochs=10, verbose=self.verbose)
        y_pred = self.model.predict(x_test)
        return np.argmax(y_pred, axis=1)
