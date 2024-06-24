import numpy as np
import random
import collections

# Simulate the segmented_data and segmented_labels
# segmented_data = ...  # Your segmented data here
# segmented_labels = ...  # Your segmented labels here
# Load the segmented data and labels
segmented_data = np.load('D:\\PAVAN\\WEAR\\wearchallenge_hasca2024\\output\\segmented_data.npy')
segmented_labels = np.load('D:\\PAVAN\\WEAR\\wearchallenge_hasca2024\\output\\segmented_labels.npy')

# Verify the shape of the loaded data
print("Segmented Data Shape:", segmented_data.shape)
print("Segmented Labels Shape:", segmented_labels.shape)

# Check for NaNs or Infs in the segmented data
nan_infs = np.isnan(segmented_data).sum() + np.isinf(segmented_data).sum()
if nan_infs > 0:
    print(f"Found {nan_infs} NaNs or Infs in the segmented data.")
else:
    print("No NaNs or Infs found in the segmented data.")

# Print the distribution of the labels
label_distribution = collections.Counter(segmented_labels)
print("Label Distribution:", label_distribution)

# Print a few random segments for visual inspection
random_indices = random.sample(range(len(segmented_data)), 5)
for idx in random_indices:
    print(f"Segment {idx} - Label: {segmented_labels[idx]}")
    print("Data:")
    print(segmented_data[idx])
    print()

# Check temporal consistency within each segment
temporal_inconsistencies = 0
for i, segment in enumerate(segmented_data):
    if not np.all(np.diff(segment[:, 0]) > 0):
        temporal_inconsistencies += 1

print(f"Number of segments with temporal inconsistencies: {temporal_inconsistencies}")

# Function to check for temporal inconsistencies
def check_temporal_inconsistencies(data):
    num_segments_with_inconsistencies = 0
    inconsistent_segments = []

    for i, segment in enumerate(data):
        timestamps = segment[:, 0]
        if not np.all(np.diff(timestamps) > 0):  # Check if timestamps are not strictly increasing
            num_segments_with_inconsistencies += 1
            inconsistent_segments.append(i)

    return num_segments_with_inconsistencies, inconsistent_segments

# Load your data
# Assuming segmented_data and segmented_labels are already loaded as per your provided shapes

# Perform the temporal inconsistency check
num_inconsistent_segments, inconsistent_segments = check_temporal_inconsistencies(segmented_data)

print(f"Number of segments with temporal inconsistencies: {num_inconsistent_segments}")

# Debug: Print out some examples of inconsistent segments
for i in inconsistent_segments[:5]:  # Print first 5 examples
    print(f"Segment {i} with label {segmented_labels[i]}:")
    print(segmented_data[i])
