from utils import read_all_datasets, calculate_metrics, transform_labels, create_directory, plot_pairwise
from constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES, UNIVARIATE_DATASET_NAMES as DATASET_NAMES, MAX_PROTOTYPES_PER_CLASS
from augment import augment_train_set
from resnet import Classifier_RESNET
from knn import KNNClassifier  # Assuming you have a KNN classifier in knn.py

import numpy as np

def augment_function(augment_algorithm_name, x_train, y_train, classes, N, limit_N=True):
    if augment_algorithm_name == 'as_dtw_dba_augment':
        return augment_train_set(x_train, y_train, classes, N, limit_N=limit_N, weights_method_name='as', distance_algorithm='dtw')

def read_data_from_dataset(datasets_dict, dataset_name):
    x_train, y_train, x_test, y_test = datasets_dict[dataset_name]
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    y_train, y_test = transform_labels(y_train, y_test)
    classes, classes_counts = np.unique(y_train, return_counts=True)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    max_prototypes = min(classes_counts.max() + 1, MAX_PROTOTYPES_PER_CLASS + 1)
    return x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes

root_dir_output = '/path/to/output/'
root_dir_dataset_archive = '/path/to/dataset_archive/'

# Assuming do_data_augmentation and do_ensemble are boolean variables
do_data_augmentation = True
do_ensemble = True

if do_ensemble:
    root_dir_output = root_dir_output + 'ensemble/'
else:
    root_dir_output = root_dir_output + 'resnet_augment/' if do_data_augmentation else root_dir_output + 'resnet/'

for archive_name in ARCHIVE_NAMES:
    datasets_dict = read_all_datasets(root_dir_dataset_archive, archive_name)
    for dataset_name in DATASET_NAMES:
        x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes = read_data_from_dataset(datasets_dict, dataset_name)
        output_dir = create_directory(root_dir_output + archive_name + '/' + dataset_name + '/')
        if output_dir is None:
            continue

        classifier = Classifier_RESNET(output_dir, x_train.shape[1:], nb_classes, max_prototypes, classes, verbose=True, load_init_weights=do_data_augmentation)
        if do_data_augmentation:
            syn_train_set, distance_algorithm = augment_function('as_dtw_dba_augment', x_train, y_train, classes, max_prototypes, limit_N=False)
            syn_x_train, syn_y_train = syn_train_set
            aug_x_train = np.array(x_train.tolist() + syn_x_train.tolist())
            aug_y_train = np.array(y_train.tolist() + syn_y_train.tolist())
            y_pred = classifier.fit(aug_x_train, aug_y_train, x_test, y_test)
        else:
            y_pred = classifier.fit(x_train, y_train, x_test, y_test)

        df_metrics = calculate_metrics(y_test, y_pred)
        df_metrics.to_csv(output_dir + 'df_metrics.csv', index=False)
        create_directory(output_dir + 'DONE')

if do_ensemble:
    from ensemble import Classifier_ENSEMBLE
    classifier_ensemble = Classifier_ENSEMBLE(output_dir, x_train.shape[1:], nb_classes, False)
    classifier_ensemble.fit(x_test, y_test)

plot_pairwise(root_dir_output, root_dir_dataset_archive, 'resnet', 'resnet_augment')
