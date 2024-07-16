import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def segment_iou(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds):
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.iloc[sort_idx].reset_index(drop=True)

    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    for idx, this_pred in prediction.iterrows():
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               ground_truth[['t-start', 't-end']].values)
        
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, jdx] >= 0:
                    continue
                tp[tidx, idx] = 1
                lock_gt[tidx, jdx] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / npos

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = compute_average_precision(precision[tidx, :], recall[tidx, :])

    return ap

def compute_average_precision(precision, recall):
    """Compute average precision (VOC style)."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class InertialActivityDetectionEvaluator:
    def __init__(self, ground_truth: pd.DataFrame, tiou_thresholds: np.ndarray = np.linspace(0.1, 0.5, 5)):
        self.ground_truth = ground_truth
        self.tiou_thresholds = tiou_thresholds
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.id_to_activity = {v: k for k, v in self.activity_index.items()}

    def evaluate(self, predictions: pd.DataFrame) -> Dict:
        mAP = np.zeros(len(self.tiou_thresholds))
        results = {}

        # Temporal detection evaluation
        for activity, activity_id in self.activity_index.items():
            gt_activity = self.ground_truth[self.ground_truth['label'] == activity_id]
            pred_activity = predictions[predictions['label'] == activity_id]
            
            ap = compute_average_precision_detection(gt_activity, pred_activity, self.tiou_thresholds)
            mAP += ap
            results[activity] = ap

        mAP /= len(self.activity_index)
        results['mAP'] = mAP

        # Confusion matrix and classification metrics
        y_true = self.ground_truth['label']
        y_pred = predictions['label']

        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm

        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        for i, activity in self.id_to_activity.items():
            results[f'{activity}_precision'] = precision[i]
            results[f'{activity}_recall'] = recall[i]
            results[f'{activity}_f1'] = f1[i]

        results['macro_precision'] = np.mean(precision)
        results['macro_recall'] = np.mean(recall)
        results['macro_f1'] = np.mean(f1)

        return results
    
    def plot_confusion_matrix(self, results: Dict):
        cm = results['confusion_matrix']
        activities = list(self.activity_index.keys())

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=activities, yticklabels=activities)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def print_results(self, results: Dict):
        print(f"mAP: {results['mAP'].mean():.4f}")
        for i, tiou in enumerate(self.tiou_thresholds):
            print(f"mAP @ tIoU={tiou:.2f}: {results['mAP'][i]:.4f}")
        
        print("\nPer-activity metrics:")
        for activity in self.activity_index.keys():
            print(f"{activity}:")
            print(f"  Precision: {results[f'{activity}_precision']:.4f}")
            print(f"  Recall: {results[f'{activity}_recall']:.4f}")
            print(f"  F1-score: {results[f'{activity}_f1']:.4f}")
        
        print("\nOverall metrics:")
        print(f"Macro Precision: {results['macro_precision']:.4f}")
        print(f"Macro Recall: {results['macro_recall']:.4f}")
        print(f"Macro F1-score: {results['macro_f1']:.4f}")

def load_segmented_data(data_path):
    X = np.load(os.path.join(data_path, 'segmented_data.npy'))
    y = np.load(os.path.join(data_path, 'segmented_labels.npy'))
    subject_ids = np.load(os.path.join(data_path, 'subject_ids.npy'))
    
    # Split data into training and testing based on subjects
    unique_subjects = np.unique(subject_ids)
    test_subjects = unique_subjects[int(0.8 * len(unique_subjects)):]
    test_indices = np.isin(subject_ids, test_subjects)

    X_test, y_test = X[test_indices], y[test_indices]
    test_subject_ids = subject_ids[test_indices]

    return X_test, y_test, test_subject_ids

def prepare_dataframes(X, y, subject_ids):
    # Assuming each segment is 1 second long. Adjust if different.
    segment_duration = 1.0
    
    ground_truth = pd.DataFrame({
        't-start': np.arange(len(y)) * segment_duration,
        't-end': (np.arange(len(y)) + 1) * segment_duration,
        'label': y,
        'subject_id': subject_ids
    })
    
    # For this example, we'll use the true labels as predictions
    # In a real scenario, you would use your model's predictions here
    predictions = pd.DataFrame({
        't-start': np.arange(len(y)) * segment_duration,
        't-end': (np.arange(len(y)) + 1) * segment_duration,
        'label': y,
        'score': np.random.uniform(0.5, 1.0, len(y)),  # Random scores for demonstration
        'subject_id': subject_ids
    })
    
    return ground_truth, predictions

if __name__ == "__main__":
    data_path = 'D:/PAVAN/WEAR/wearchallenge_hasca2024/output'
    
    # Load and prepare data
    X_test, y_test, test_subject_ids = load_segmented_data(data_path)
    ground_truth, predictions = prepare_dataframes(X_test, y_test, test_subject_ids)
    
    # Initialize evaluator
    evaluator = InertialActivityDetectionEvaluator(ground_truth)

    # Perform evaluation
    results = evaluator.evaluate(predictions)

    # Print results
    evaluator.print_results(results)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(results)

    # Additional analysis: plot mAP vs tIoU threshold
    plt.figure(figsize=(10, 6))
    plt.plot(evaluator.tiou_thresholds, results['mAP'], marker='o')
    plt.title('mAP vs tIoU Threshold')
    plt.xlabel('tIoU Threshold')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.show()

    # Print per-class AP at different tIoU thresholds
    print("\nPer-class Average Precision at different tIoU thresholds:")
    for activity in evaluator.activity_index.keys():
        if activity != 'mAP':
            print(f"\n{activity}:")
            for i, tiou in enumerate(evaluator.tiou_thresholds):
                print(f"  AP @ tIoU={tiou:.2f}: {results[activity][i]:.4f}")
