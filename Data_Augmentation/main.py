import numpy as np
import torch
from utils import read_all_datasets
from augment import augment_train_set
from ensemble import Classifier_ENSEMBLE
from knn import KNNClassifier
from resnet import Classifier_RESNET
import os

def load_augmented_data(save_dir='augmented_batches'):
    syn_x_train = []
    syn_y_train = []
    for file in os.listdir(save_dir):
        if file.startswith('augmented_x_class_'):
            syn_x_train.append(np.load(os.path.join(save_dir, file)))
        elif file.startswith('augmented_y_class_'):
            syn_y_train.append(np.load(os.path.join(save_dir, file)))
    
    return np.concatenate(syn_x_train), np.concatenate(syn_y_train)

def save_augmented_data(x_data, y_data, x_filename='augmented_x_train.npy', y_filename='augmented_y_train.npy'):
    np.save(x_filename, x_data)
    np.save(y_filename, y_data)
    print(f"Augmented data saved successfully to {x_filename} and {y_filename}.")

if __name__ == "__main__":
    # Load datasets
    data_path = 'D:/PAVAN/WEAR/wearchallenge_hasca2024/output/'
    datasets = read_all_datasets(data_path)

    x_train = datasets['segmented_data']
    y_train = datasets['segmented_labels']

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(x_train) * split_ratio)

    x_train, x_test = x_train[:split_index], x_train[split_index:]
    y_train, y_test = y_train[:split_index], y_train[split_index:]

    # Augment the training set
    classes = np.unique(y_train)
    N = 50  # Number of synthetic samples per class

    augment_train_set(x_train, y_train, classes, N)

    # Load augmented data
    syn_x_train, syn_y_train = load_augmented_data()

    # Combine original and synthetic data
    x_train_combined = np.concatenate((x_train, syn_x_train), axis=0)
    y_train_combined = np.concatenate((y_train, syn_y_train), axis=0)

    # Save augmented data
    save_augmented_data(x_train_combined, y_train_combined)

    # Train and evaluate Ensemble Classifier
    ensemble_clf = Classifier_ENSEMBLE(output_dir='output', input_shape=x_train_combined.shape[1:], nb_classes=len(classes))
    ensemble_clf.fit(x_train_combined, y_train_combined, x_test, y_test)

    # Train and evaluate KNN Classifier
    knn_clf = KNNClassifier(n_neighbors=5)
    knn_clf.fit(x_train_combined.reshape(x_train_combined.shape[0], -1), y_train_combined, 
                x_test.reshape(x_test.shape[0], -1), y_test)

    # Train and evaluate ResNet Classifier
    resnet_clf = Classifier_RESNET(input_shape=x_train_combined.shape[1:], num_classes=len(classes))
    resnet_clf.fit(x_train_combined, y_train_combined, x_test, y_test)
