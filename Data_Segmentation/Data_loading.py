import os
import numpy as np
import pandas as pd

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

# Function to load data from all CSV files in the specified folder
def load_and_preprocess_data(data_folder):
    all_data = []
    sampling_rate = 50  # 50 Hz, adjust as necessary
    time_interval = 1 / sampling_rate

    pd.set_option('future.no_silent_downcasting', True)

    for i in range(18):  # Assuming files are named sbj_0.csv to sbj_17.csv
        filename = f'sbj_{i}.csv'
        file_path = os.path.join(data_folder, filename)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data = data.replace({"label": label_dict}).fillna(0)
            num_samples = data.shape[0]
            timestamps = np.arange(num_samples) * time_interval
            data.insert(0, 'timestamp', timestamps)  # Add synthetic timestamps
            all_data.append(data)
        else:
            print(f"Warning: File {file_path} not found.")
    
    concatenated_data = pd.concat(all_data, ignore_index=True)
    concatenated_data.to_csv('processed_data.csv', index=False)  # Save the processed data to a CSV file
    return concatenated_data.to_numpy()

# Load data from all CSV files in the folder
def load_data(data_folder):
    return load_and_preprocess_data(data_folder)
