import numpy as np
from numba import jit

@jit(nopython=True)
def dba_core(average_series, x_train, weights, path):
    new_average = np.zeros_like(average_series)
    total_weight = 0
    for i in range(len(x_train)):
        for a, b in path:
            new_average[a] += x_train[i][b] * weights[i]
            total_weight += weights[i]
    return new_average, total_weight

def dba(weights, prototype, x_train, y_train, class_label, distance_algorithm, n_iter=10):
    average_series = prototype
    class_indices = np.where(y_train == class_label)[0]
    x_train_class = x_train[class_indices]
    weights_class = weights[class_indices]
    
    for _ in range(n_iter):
        distances = [distance_algorithm(average_series, x) for x in x_train_class]
        paths = [d[1] for d in distances]
        new_average, total_weight = dba_core(average_series, x_train_class, weights_class, paths)
        average_series = new_average / total_weight
    return average_series
