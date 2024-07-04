import numpy as np
import os

def read_all_datasets(data_path):
    datasets = {}
    for file in os.listdir(data_path):
        if file.endswith('.npy'):
            dataset_name = file.split('.')[0]
            dataset_path = os.path.join(data_path, file)
            datasets[dataset_name] = np.load(dataset_path, allow_pickle=True)
    return datasets
