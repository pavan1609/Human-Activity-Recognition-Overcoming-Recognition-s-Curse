import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tcn import TemporalConvNet
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from evaluation import InertialActivityDetectionEvaluator
from data_utils import convert_samples_to_segments
from os_utils import mkdir_if_missing

def load_data(data_path):
    X = np.load(os.path.join(data_path, 'segmented_data.npy'))
    y = np.load(os.path.join(data_path, 'segmented_labels.npy'))
    subject_ids = np.load(os.path.join(data_path, 'subject_ids.npy'))
    
    unique_subjects = np.unique(subject_ids)
    train_subjects = unique_subjects[:int(0.8 * len(unique_subjects))]
    test_subjects = unique_subjects[int(0.8 * len(unique_subjects)):]

    train_indices = np.isin(subject_ids, train_subjects)
    test_indices = np.isin(subject_ids, test_subjects)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    test_subject_ids = subject_ids[test_indices]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test, test_subject_ids

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_targets)

def prepare_dataframes(y, subject_ids, preds, scores, segment_duration=1.0):
    ground_truth = pd.DataFrame({
        't-start': np.arange(len(y)) * segment_duration,
        't-end': (np.arange(len(y)) + 1) * segment_duration,
        'label': y,
        'subject_id': subject_ids
    })
    
    predictions = pd.DataFrame({
        't-start': np.arange(len(preds)) * segment_duration,
        't-end': (np.arange(len(preds)) + 1) * segment_duration,
        'label': preds,
        'score': scores,
        'subject_id': subject_ids
    })
    
    return ground_truth, predictions

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = 'D:/PAVAN/WEAR/wearchallenge_hasca2024/output'
    X_train, y_train, X_test, y_test, test_subject_ids = load_data(data_path)
    
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = TemporalConvNet(num_inputs=input_size, num_channels=[32, 32, 32, 32], num_classes=num_classes, kernel_size=args.kernel_size, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare ground truth for evaluation
    ground_truth, _ = prepare_dataframes(y_test, test_subject_ids, y_test, np.ones_like(y_test))
    evaluator = InertialActivityDetectionEvaluator(ground_truth, tiou_thresholds=args.tiou_thresholds)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = evaluate_model(model, test_dataloader, criterion, device)

        with torch.no_grad():
            val_loss, val_preds, val_targets = evaluate_model(model, test_dataloader, criterion, device)
    
            # Prepare predictions for evaluation
            val_scores = model(X_test.to(device)).softmax(dim=1).max(dim=1)[0].cpu().numpy()
            _, predictions = prepare_dataframes(val_targets, test_subject_ids, val_preds, val_scores)

            # Evaluate
            results = evaluator.evaluate(predictions)

        print(f'Epoch: [{epoch+1}/{args.epochs}]')
        print(f'TRAINING: avg. loss {train_loss:.4f}')
        print(f'VALIDATION: avg. loss {val_loss:.4f}')
        evaluator.print_results(results)

        if (epoch + 1) % 10 == 0:
            evaluator.plot_confusion_matrix(results)

    # Save the final model
    torch.save(model.state_dict(), 'tcn_model_final.pth')

    # Final evaluation plots
    evaluator.plot_confusion_matrix(results)

    plt.figure(figsize=(10, 6))
    plt.plot(evaluator.tiou_thresholds, results['mAP'], marker='o')
    plt.title('mAP vs tIoU Threshold')
    plt.xlabel('tIoU Threshold')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.show()

    print("\nPer-class Average Precision at different tIoU thresholds:")
    for activity in evaluator.activity_index.keys():
        if activity != 'mAP':
            print(f"\n{activity}:")
            for i, tiou in enumerate(evaluator.tiou_thresholds):
                print(f"  AP @ tIoU={tiou:.2f}: {results[activity][i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--kernel_size', type=int, default=10)
    parser.add_argument('--tiou_thresholds', type=float, nargs='+', default=[0.1, 0.25, 0.5])
    parser.add_argument('--sampling_rate', type=int, default=100)
    args = parser.parse_args()
    
    main(args)
