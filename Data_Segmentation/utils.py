import numpy as np
from scipy.spatial.distance import cdist
from fastdtw import fastdtw

def apply_dtw_segmentation(data, window_size, threshold):
    output_sbj = []
    output_x = []
    output_y = []

    for i in range(0, len(data) - window_size + 1, window_size):
        segment = data[i:i + window_size, :-2]
        subject_id = data[i, -2]
        label = data[i, -1]

        # If this is the first segment, appends it directly
        if len(output_x) == 0:
            output_sbj.append(subject_id)
            output_x.append(segment)
            output_y.append(label)
            continue

        # Compute DTW distance to the last segment
        dist, _ = fastdtw(output_x[-1], segment, dist=cdist)

        # If the distance is greater than the threshold, start a new segment
        if dist > threshold:
            output_sbj.append(subject_id)
            output_x.append(segment)
            output_y.append(label)
        else:
            # Otherwise, extend the last segment
            output_x[-1] = np.vstack((output_x[-1], segment))
            output_sbj[-1] = subject_id  # Assuming the subject is the same for continuous data
            output_y[-1] = label  # Assuming the label is the same for continuous data

    return np.array(output_sbj), output_x, output_y
