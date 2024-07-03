import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tcn import TemporalConvNet
import numpy as np
import argparse
import os

def load_data(data_path):
    X = np.load(os.path.join(data_path, 'segmented_data.npy'))
    y = np.load(os.path.join(data_path, 'segmented_labels.npy'))
    subject_ids = np.load(os.path.join(data_path, 'subject_ids.npy'))
    
    # Assuming you want to split data into training and testing based on subjects
    unique_subjects = np.unique(subject_ids)
    train_subjects = unique_subjects[:int(0.8 * len(unique_subjects))]
    test_subjects = unique_subjects[int(0.8 * len(unique_subjects)):]

    train_indices = np.isin(subject_ids, train_subjects)
    test_indices = np.isin(subject_ids, test_subjects)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Convert y_train and y_test to class indices (assuming they are not already)
    classes = np.unique(y)  # Get unique classes from y
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    y_train = np.vectorize(class_to_index.get)(y_train)  # Convert y_train to indices
    y_test = np.vectorize(class_to_index.get)(y_test)    # Convert y_test to indices

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test

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
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = 'D:/PAVAN/WEAR/wearchallenge_hasca2024/output'
    X_train, y_train, X_test, y_test = load_data(data_path)
    
    # Ensure target size matches model output
    input_size = X_train.shape[1]  # Assuming shape (N, C, L)
    num_classes = 19  # Number of classes in your dataset
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = TemporalConvNet(num_inputs=X_train.shape[1], num_channels=[32, 32, 32, 32], num_classes=num_classes, kernel_size=10, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'tcn_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
