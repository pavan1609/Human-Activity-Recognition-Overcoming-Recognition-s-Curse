import os
import numpy as np
import pandas as pd

def read_all_datasets(root_dir_dataset_archive, archive_name):
    datasets_dict = {}
    root_dir = os.path.join(root_dir_dataset_archive, archive_name)

    for dataset_name in os.listdir(root_dir):
        dataset_dir = os.path.join(root_dir, dataset_name)
        x_train = np.loadtxt(os.path.join(dataset_dir, f'{dataset_name}_TRAIN.tsv'), delimiter='\t')
        y_train = x_train[:, 0]
        x_train = x_train[:, 1:]
        x_test = np.loadtxt(os.path.join(dataset_dir, f'{dataset_name}_TEST.tsv'), delimiter='\t')
        y_test = x_test[:, 0]
        x_test = x_test[:, 1:]
        datasets_dict[dataset_name] = (x_train, y_train, x_test, y_test)

    return datasets_dict

def transform_labels(y_train, y_test):
    y_train = y_train - np.min(y_train)
    y_test = y_test - np.min(y_test)
    return y_train, y_test

def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    metrics = {'accuracy': accuracy}
    return pd.DataFrame([metrics])

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory
    else:
        return None

def plot_pairwise(root_dir, archive_dir, method1, method2):
    pass
