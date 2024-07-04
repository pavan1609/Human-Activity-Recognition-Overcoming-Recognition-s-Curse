import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
output_dir = 'D:\\PAVAN\\WEAR\\verify'
os.makedirs(output_dir, exist_ok=True)

def load_segmented_data():
    data_path = 'D:\\PAVAN\\WEAR\\wearchallenge_hasca2024\\output\\segmented_data.npy'
    labels_path = 'D:\\PAVAN\\WEAR\\wearchallenge_hasca2024\\output\\segmented_labels.npy'
    
    if os.path.exists(data_path) and os.path.exists(labels_path):
        segmented_data = np.load(data_path)
        segmented_labels = np.load(labels_path)
        return segmented_data, segmented_labels
    else:
        raise FileNotFoundError("Segmented data files not found. Please ensure the segmentation process has completed successfully.")

# Load the segmented data and labels
segmented_data, segmented_labels = load_segmented_data()

# Verify the shape of the loaded data
print("Segmented Data Shape:", segmented_data.shape)
print("Segmented Labels Shape:", segmented_labels.shape)

# Function to handle FutureWarning by replacing and inferring objects
def read_and_replace(file_path, label_dict):
    data = pd.read_csv(file_path)
    data = data.replace({"label": label_dict}).fillna(0)
    data = data.infer_objects(copy=False)
    return data.to_numpy()

# Plot segments function
def plot_segments(output_x, output_y, n=5):
    for i in range(n):
        plt.figure(figsize=(12, 6))
        for j in range(output_x[i].shape[1]):
            plt.plot(output_x[i][:, j], label=f'Feature {j+1}')
        plt.title(f"Segment {i} - Label: {output_y[i]}")
        plt.xlabel('Time')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.savefig(f'{output_dir}/segment_{i}.png')
        plt.close()

# Check segment lengths function
def check_segment_lengths(output_x, window_size):
    segment_lengths = [len(segment) for segment in output_x]
    unique_lengths = np.unique(segment_lengths)
    print(f"Unique segment lengths: {unique_lengths}")
    inconsistent_lengths = [length for length in segment_lengths if length != window_size]
    print(f"Number of segments with inconsistent lengths: {len(inconsistent_lengths)}")
    
    plt.figure(figsize=(10, 5))
    plt.hist(segment_lengths, bins=range(min(segment_lengths), max(segment_lengths) + 1))
    plt.title('Histogram of Segment Lengths')
    plt.xlabel('Segment Length')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/segment_lengths_histogram.png')
    plt.close()

# Statistical analysis function
def statistical_analysis(output_x):
    for i, segment in enumerate(output_x[:5]):
        mean = np.mean(segment, axis=0)
        variance = np.var(segment, axis=0)
        print(f"Segment {i} - Mean: {mean}, Variance: {variance}")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=segment)
        plt.title(f'Segment {i} - Boxplot of Features')
        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.savefig(f'{output_dir}/boxplot_segment_{i}.png')
        plt.close()

# Manual inspection function
def manual_inspection(output_x, output_y, start_idx=0, end_idx=10):
    for i in range(start_idx, end_idx):
        print(f"Segment {i} - Label: {output_y[i]}")
        print("Data:")
        print(output_x[i])
        plt.figure(figsize=(12, 6))
        for j in range(output_x[i].shape[1]):
            plt.plot(output_x[i][:, j], label=f'Feature {j+1}')
        plt.title(f"Segment {i} - Label: {output_y[i]}")
        plt.xlabel('Time')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.savefig(f'{output_dir}/manual_inspection_segment_{i}.png')
        plt.close()

# Check outliers function
def check_outliers(output_x):
    for i, segment in enumerate(output_x[:5]):
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.mean(segment, axis=0)
            std_dev = np.std(segment, axis=0)
            z_scores = np.abs((segment - mean) / std_dev)
            z_scores[np.isnan(z_scores)] = 0  # Set NaN values to 0
            outliers = np.where(z_scores > 3)
            print(f"Segment {i} - Outliers found at positions: {outliers}")

# Check transition points function
def check_transition_points(output_x, output_y):
    transitions = np.where(np.diff(output_y) != 0)[0]
    print(f"Transition points: {transitions}")
    for idx in transitions[:5]:
        print(f"Transition at segment {idx} from {output_y[idx]} to {output_y[idx + 1]}")
        plt.figure(figsize=(12, 6))
        for j in range(output_x[idx].shape[1]):
            plt.plot(output_x[idx][:, j], label=f'Feature {j+1}')
        plt.title(f"Segment {idx} - Label: {output_y[idx]}")
        plt.xlabel('Time')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.savefig(f'{output_dir}/transition_segment_{idx}.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        for j in range(output_x[idx + 1].shape[1]):
            plt.plot(output_x[idx + 1][:, j], label=f'Feature {j+1}')
        plt.title(f"Segment {idx + 1} - Label: {output_y[idx + 1]}")
        plt.xlabel('Time')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.savefig(f'{output_dir}/transition_segment_{idx + 1}.png')
        plt.close()
