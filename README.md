# Human Activity Recognition: Overcoming-the-curse-of-recognition-in-the-wild

## Credits

- **Author**: Marius Bock
- **Source Code**: https://github.com/mariusbock/wear
  
## Abstract
This repository contains an implementation of a Time Convolutional Network (TCN) model for analyzing inertial data, enhanced with Dynamic Time Warping (DTW) for segmentation and Dynamic Time Warping Barycenter Averaging (DBA) for data augmentation. This approach is designed for tasks such as human activity recognition, leveraging advanced time series techniques to improve model performance and robustness.

**Overview**

Inertial data analysis is critical in various domains, including healthcare, sports, and human-computer interaction. The combination of TCN, DTW segmentation, and DBA augmentation offers a powerful solution for time-series classification problems. TCNs efficiently capture temporal patterns in the data, while DTW segmentation ensures that activities are accurately segmented, and DBA augmentation enhances the training dataset by creating more representative samples.

**Features**

DTW Segmentation: The model processes inertial data using Dynamic Time Warping for segmentation, ensuring accurate alignment of time-series data points and better capturing the dynamics of activities.

DBA Augmentation: Dynamic Time Warping Barycenter Averaging is used to augment the dataset, creating more representative samples and improving model generalization.

TCN Architecture: The Temporal Convolutional Network (TCN) is used for its ability to model long-range dependencies in time-series data, making it ideal for activity recognition tasks.

Customizable Parameters: Users can adjust various parameters, including window size and overlap percentage for segmentation, to fine-tune the data preprocessing and model training steps.

Reproducibility: The repository includes utilities for setting random seeds, ensuring that results are reproducible across different runs and environments.

Training and Evaluation: The training script supports checkpoints, tensorboard logging, and configurable hyperparameters, facilitating robust model training and evaluation.

## Repository Structure

- **tcn.py:** Contains the definition of the TCN model architecture.
- **train.py:** Script for training the TCN model. It includes argument parsing for different hyperparameters and handles the training loop.
- **dtw.py:** Contains the algorithm to compute the euclidian distance by considering 2 dynamic time series values within threshold.
- **dba.py:** Algorithm which creates synthetic data to the segmented data, by averaging each features explicitly.
    
## Getting Started
**Prerequisites**

Python 3.6 or higher

PyTorch

Numpy

Pandas

dba

distances

joblib

tqdm

numba

sklearn

fastdtw

seaborn

argparse


## Download
The full dataset can be downloaded [here](https://bit.ly/wear_dataset)
- **Dataset by**: Marius Bock

The download folder is divided into 3 subdirectories
- **annotations (> 1MB)**: JSON-files containing annotations per-subject using the THUMOS14-style
- **processed (15GB)**: precomputed I3D, inertial and combined per-subject features
- **raw (130GB)**: Raw, per-subject video and inertial data
