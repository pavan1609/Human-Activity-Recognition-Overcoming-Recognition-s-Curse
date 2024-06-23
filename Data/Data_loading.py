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

# Function to load data from all CSV files in the specified folder
def load_data(data_folder):
    all_data = []
    for i in range(18):  # Assuming files are named sbj_0.csv to sbj_17.csv
        filename = f'sbj_{i}.csv'
        file_path = os.path.join(data_folder, filename)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path).replace({"label": label_dict}).fillna(0).to_numpy()
            all_data.append(data)
        else:
            print(f"Warning: File {file_path} not found.")
    return np.vstack(all_data)

# Load data from all CSV files in the folder
all_data = load_data(data_folder)
print(f"Shape of all_data: {all_data.shape}")
