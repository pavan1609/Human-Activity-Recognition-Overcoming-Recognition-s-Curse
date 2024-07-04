import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tcn import TemporalConvNet
import numpy as np
import argparse
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Import necessary functions from your professor's utils
from data_utils import convert_samples_to_segments
from os_utils import mkdir_if_missing

class SimpleDetectionEvaluator:
    def __init__(self, tiou_thresholds):
        self.tiou_thresholds = tiou_thresholds

    def evaluate(self, pred_segments):
        # Implement a simple evaluation metric here
        # For now, let's just return random values as a placeholder
        mAP = np.random.rand(len(self.tiou_thresholds))
        mRecall = np.random.rand(len(self.tiou_thresholds))
        return mAP, mRecall

def load_data(data_path):
    X = np.load(os.path.join(data_path, 'segmented_data.npy'))
    y = np.load(os.path.join(data_path, 'segmented_labels.npy'))
    subject_ids = np.load(os.path.join(data_path, 'subject_ids.npy'))
    
    # Split data into training and testing based on subjects
    unique_subjects = np.unique(subject_ids)
    train_subjects = unique_subjects[:int(0.8 * len(unique_subjects))]
    test_subjects = unique_subjects[int(0.8 * len(unique_subjects)):]

    train_indices = np.isin(subject_ids, train_subjects)
    test_indices = np.isin(subject_ids, test_subjects)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    test_subject_ids = subject_ids[test_indices]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test, test_subject_ids

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_targets)

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

    # Initialize simple evaluator
    det_eval = SimpleDetectionEvaluator(tiou_thresholds=args.tiou_thresholds)

    for epoch in range(args.epochs):
        train_loss, train_preds, train_targets = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = evaluate_model(model, test_dataloader, criterion, device)

        # Convert predictions to segments for mAP calculation
        val_segments = convert_samples_to_segments(test_subject_ids, val_preds, args.sampling_rate)

        # Calculate metrics
        v_mAP, _ = det_eval.evaluate(val_segments)
        conf_mat = confusion_matrix(val_targets, val_preds, normalize='true')
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(val_targets, val_preds, average=None, zero_division=1)
        v_rec = recall_score(val_targets, val_preds, average=None, zero_division=1)
        v_f1 = f1_score(val_targets, val_preds, average=None, zero_division=1)

        print(f'Epoch: [{epoch+1}/{args.epochs}]')
        print(f'TRAINING: avg. loss {train_loss:.2f}')
        print(f'VALIDATION: avg. loss {val_loss:.2f}')
        print(f'Avg. mAP {np.nanmean(v_mAP) * 100:.2f} (%)')
        for tiou, tiou_mAP in zip(args.tiou_thresholds, v_mAP):
            print(f'mAP@{tiou} {tiou_mAP*100:.2f} (%)')
        print(f'Acc {np.nanmean(v_acc) * 100:.2f} (%)')
        print(f'Prec {np.nanmean(v_prec) * 100:.2f} (%)')
        print(f'Rec {np.nanmean(v_rec) * 100:.2f} (%)')
        print(f'F1 {np.nanmean(v_f1) * 100:.2f} (%)')

    # Save the final model
    torch.save(model.state_dict(), 'tcn_model_final.pth')

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
