import os
import numpy as np
import pandas as pd

# Define the folder path where your CSV files are located
data_folder = r'D:\PAVAN\WEAR\wearchallenge_hasca2024\wearchallenge_hasca2024'

# Define the label dictionary for replacing labels with numeric codes
label_dict = {
    'null': 0,
    'jogging': 1,
    'jogging (rotating arms)': 2,
    'jogging (skipping)': 3,
    'jogging (sidesteps)': 4,
    'jogging (butt-kicks)': 5,
    'stretching (triceps)': 6,
    'stretching (lunging)': 7,
    'stretching (shoulders)': 8,
    'stretching (hamstrings)': 9,
    'stretching (lumbar rotation)': 10,
    'push-ups': 11,
    'push-ups (complex)': 12,
    'sit-ups': 13,
    'sit-ups (complex)': 14,
    'burpees': 15,
    'lunges': 16,
    'lunges (complex)': 17,
    'bench-dips': 18
}

# Function to add synthetic timestamps
def add_timestamps(data, sampling_rate=50):
    num_samples = len(data)
    timestamps = np.arange(0, num_samples / sampling_rate, 1 / sampling_rate)
    return np.column_stack((timestamps, data))

# Function to load data from a CSV file
def load_data_from_file(file_path, sampling_rate=50):
    data = pd.read_csv(file_path).replace({"label": label_dict}).fillna(0).to_numpy()
    return add_timestamps(data, sampling_rate)

# Load data from all CSV files in the folder and store each file's data separately
def load_all_data(data_folder, sampling_rate=50):
    all_data = []
    for i in range(18):  # Assuming files are named sbj_0.csv to sbj_17.csv
        filename = f'sbj_{i}.csv'
        file_path = os.path.join(data_folder, filename)
        if os.path.exists(file_path):
            data_with_timestamps = load_data_from_file(file_path, sampling_rate)
            all_data.append(data_with_timestamps)
        else:
            print(f"Warning: File {file_path} not found.")
    return all_data

# Load data from all CSV files in the folder
all_data = load_all_data(data_folder)
for i, data in enumerate(all_data):
    print(f"Shape of data for file sbj_{i}.csv: {data.shape}")
