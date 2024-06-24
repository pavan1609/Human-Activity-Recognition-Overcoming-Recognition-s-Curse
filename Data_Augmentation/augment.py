import numpy as np
from dba import dba
from distances import dynamic_time_warping as dtw

def augment_train_set(x_train, y_train, classes, N, limit_N=True, weights_method_name='as', distance_algorithm='dtw'):
    weights = get_weights_average_selected(x_train, y_train, classes, weights_method_name, distance_algorithm)
    syn_x_train = []
    syn_y_train = []

    for c in classes:
        class_indices = np.where(y_train == c)[0]
        prototypes_count = min(len(class_indices), N)
        for i in range(prototypes_count):
            prototype = x_train[class_indices[i]]
            synthetic_sample = dba(weights, prototype, x_train[class_indices], y_train[class_indices], c, distance_algorithm)
            syn_x_train.append(synthetic_sample)
            syn_y_train.append(c)

    return np.array(syn_x_train), np.array(syn_y_train)

def get_weights_average_selected(x_train, y_train, classes, weights_method_name, distance_algorithm):
    if weights_method_name == 'as':
        weights = calculate_as_weights(x_train, y_train, classes, distance_algorithm)
    return weights

def calculate_as_weights(x_train, y_train, classes, distance_algorithm):
    weights = []
    for c in classes:
        class_indices = np.where(y_train == c)[0]
        for i in range(len(class_indices)):
            for j in range(i + 1, len(class_indices)):
                weight = dtw(x_train[class_indices[i]], x_train[class_indices[j]])
                weights.append(weight)
    return np.array(weights)
