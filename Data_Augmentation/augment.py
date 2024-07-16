import numpy as np
from dba import dba
from distances import dynamic_time_warping as dtw
from joblib import Parallel, delayed
from tqdm import tqdm
import os

def augment_batch(batch_indices, x_train, y_train, c, weights_dict, N, distance_algorithm):
    syn_x_batch = []
    syn_y_batch = []
    class_indices = np.where(y_train == c)[0]
    for i in batch_indices:
        if i < len(class_indices):
            prototype = x_train[class_indices[i]]
            synthetic_sample = dba(weights_dict[c], prototype, x_train[class_indices], y_train[class_indices], c, distance_algorithm)
            syn_x_batch.append(synthetic_sample)
            syn_y_batch.append(c)
    return syn_x_batch, syn_y_batch

def augment_train_set(x_train, y_train, classes, N, batch_size=100, n_jobs=-1, weights_method_name='as', distance_algorithm='dtw', save_dir='augmented_batches'):
    weights_file = get_weights_average_selected(x_train, y_train, classes, weights_method_name, distance_algorithm)
    weights_dict = np.load(weights_file, allow_pickle=True).item()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for c in classes:
        class_indices = np.where(y_train == c)[0]
        prototypes_count = min(len(class_indices), N)
        batch_indices = [range(i, min(i + batch_size, prototypes_count)) for i in range(0, prototypes_count, batch_size)]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(augment_batch)(indices, x_train, y_train, c, weights_dict, N, distance_algorithm)
            for indices in tqdm(batch_indices, desc=f"Augmenting class {c}")
        )
        
        syn_x_class = []
        syn_y_class = []
        for batch_x, batch_y in results:
            syn_x_class.extend(batch_x)
            syn_y_class.extend(batch_y)
        
        np.save(os.path.join(save_dir, f'augmented_x_class_{c}.npy'), np.array(syn_x_class))
        np.save(os.path.join(save_dir, f'augmented_y_class_{c}.npy'), np.array(syn_y_class))

    os.remove(weights_file)  # Clean up the temporary weights file

def get_weights_average_selected(x_train, y_train, classes, weights_method_name, distance_algorithm):
    if weights_method_name == 'as':
        weights_file = calculate_as_weights(x_train, y_train, classes, distance_algorithm)
    return weights_file

def calculate_as_weights(x_train, y_train, classes, distance_algorithm):
    weights_file = 'temp_weights.npy'
    weights_dict = {}
    
    for c in classes:
        class_indices = np.where(y_train == c)[0]
        class_weights = []
        for i in tqdm(range(len(class_indices)), desc=f"Calculating weights for class {c}"):
            for j in range(i + 1, len(class_indices)):
                weight = dtw(x_train[class_indices[i]], x_train[class_indices[j]])
                class_weights.append(weight)
        weights_dict[c] = class_weights
    
    np.save(weights_file, weights_dict, allow_pickle=True)
    return weights_file
