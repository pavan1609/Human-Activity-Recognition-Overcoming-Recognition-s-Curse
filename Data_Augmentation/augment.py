import numpy as np
from dba import dba
from distances import dynamic_time_warping as dtw
from joblib import Parallel, delayed
from tqdm import tqdm

def augment_batch(batch_indices, x_train, y_train, c, weights, N, distance_algorithm):
    syn_x_batch = []
    syn_y_batch = []
    class_indices = np.where(y_train == c)[0]
    for i in batch_indices:
        if i < len(class_indices):
            prototype = x_train[class_indices[i]]
            synthetic_sample = dba(weights, prototype, x_train[class_indices], y_train[class_indices], c, distance_algorithm)
            syn_x_batch.append(synthetic_sample)
            syn_y_batch.append(c)
    return syn_x_batch, syn_y_batch

def augment_train_set(x_train, y_train, classes, N, batch_size=100, n_jobs=-1, weights_method_name='as', distance_algorithm='dtw'):
    weights = get_weights_average_selected(x_train, y_train, classes, weights_method_name, distance_algorithm)
    syn_x_train = []
    syn_y_train = []

    for c in classes:
        class_indices = np.where(y_train == c)[0]
        prototypes_count = min(len(class_indices), N)
        batch_indices = [range(i, min(i + batch_size, prototypes_count)) for i in range(0, prototypes_count, batch_size)]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(augment_batch)(indices, x_train, y_train, c, weights, N, distance_algorithm)
            for indices in tqdm(batch_indices, desc=f"Augmenting class {c}")
        )
        
        for batch_x, batch_y in results:
            syn_x_train.extend(batch_x)
            syn_y_train.extend(batch_y)

    return np.array(syn_x_train), np.array(syn_y_train)

def get_weights_average_selected(x_train, y_train, classes, weights_method_name, distance_algorithm):
    if weights_method_name == 'as':
        weights = calculate_as_weights(x_train, y_train, classes, distance_algorithm)
    return weights

def calculate_as_weights(x_train, y_train, classes, distance_algorithm):
    weights = []
    for c in classes:
        class_indices = np.where(y_train == c)[0]
        for i in tqdm(range(len(class_indices)), desc=f"Calculating weights for class {c}"):
            for j in range(i + 1, len(class_indices)):
                weight = dtw(x_train[class_indices[i]], x_train[class_indices[j]])
                weights.append(weight)
    return np.array(weights)
