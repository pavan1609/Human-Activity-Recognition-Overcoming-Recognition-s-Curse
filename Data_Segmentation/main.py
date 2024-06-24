import os
import time
import numpy as np
import pandas as pd
from Data_loading import load_all_data
from dtw_segmentations import segment_data
from verification import (
    plot_segments,
    check_segment_lengths,
    statistical_analysis,
    manual_inspection,
    check_outliers,
    check_transition_points
)

if __name__ == "__main__":
    data_folder = r'D:\PAVAN\WEAR\wearchallenge_hasca2024\wearchallenge_hasca2024'
    all_data = load_all_data(data_folder)

    # Parameters for segmentation
    window_size = 10  # Adjust as needed
    threshold = 5.0  # Adjust as needed

    segmented_subject_ids = []
    segmented_data_all = []
    segmented_labels_all = []

    # Process each file separately
    for i, data in enumerate(all_data):
        print(f"Processing file sbj_{i}.csv")
        start_time = time.time()
        subject_ids, segmented_data, segmented_labels = segment_data(data, window_size, threshold)
        end_time = time.time()
        print(f"Segmentation for sbj_{i}.csv completed in {end_time - start_time:.2f} seconds")

        segmented_subject_ids.extend(subject_ids)
        segmented_data_all.extend(segmented_data)
        segmented_labels_all.extend(segmented_labels)

    # Convert lists to numpy arrays
    segmented_subject_ids = np.array(segmented_subject_ids)
    segmented_data_all = np.array(segmented_data_all)
    segmented_labels_all = np.array(segmented_labels_all)

    print(f"Total number of segments: {len(segmented_data_all)}")

    # Save segmented data
    save_folder = r'D:\PAVAN\WEAR\wearchallenge_hasca2024\output'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    np.save(os.path.join(save_folder, 'segmented_data.npy'), segmented_data_all)
    np.save(os.path.join(save_folder, 'subject_ids.npy'), segmented_subject_ids)
    np.save(os.path.join(save_folder, 'segmented_labels.npy'), segmented_labels_all)

    print(f"Segmented data saved to {save_folder}")

    # Verification
    plot_segments(segmented_data_all, segmented_labels_all, n=5)
    check_segment_lengths(segmented_data_all, window_size)
    statistical_analysis(segmented_data_all)
    manual_inspection(segmented_data_all, segmented_labels_all, start_idx=0, end_idx=10)
    check_outliers(segmented_data_all)
    check_transition_points(segmented_data_all, segmented_labels_all)
