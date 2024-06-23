import numpy as np
from scipy.spatial.distance import cdist
from fastdtw import fastdtw

def dtw_distance(ts_a, ts_b):
    ts_a = np.atleast_2d(ts_a)
    ts_b = np.atleast_2d(ts_b)
    dist, _ = fastdtw(ts_a, ts_b, dist=lambda x, y: np.linalg.norm(x - y))
    print(f"Computed DTW distance: {dist}")
    return dist

def dtw_segmentation(data, window_size, threshold):
    segments = []
    start_idx = 0
    while start_idx < len(data):
        end_idx = start_idx + window_size
        if end_idx >= len(data):
            segments.append((start_idx, len(data)))
            break
        window_a = data[start_idx:end_idx]
        next_end_idx = end_idx + window_size
        if next_end_idx >= len(data):
            next_end_idx = len(data)
        window_b = data[end_idx:next_end_idx]
        dtw_dist = dtw_distance(window_a, window_b)
        if dtw_dist > threshold:
            segments.append((start_idx, end_idx))
            start_idx = end_idx
        else:
            start_idx += window_size
        print(f"Current segments: {segments}")
    return segments

def segment_data(data, window_size, threshold):
    segments = dtw_segmentation(data, window_size, threshold)
    segmented_data = []
    subject_ids = []
    labels = []
    for start, end in segments:
        segment = data[start:end]
        segmented_data.append(segment[:, :-2])  # Exclude the last two columns
        subject_ids.append(segment[0, -2])
        labels.append(segment[0, -1])
    print(f"Number of segments created: {len(segments)}")
    return subject_ids, segmented_data, labels  # Return lists instead of arrays
