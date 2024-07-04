import torch
import torch.nn as nn
import torch.optim as optim

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet1D, self).__init__()
        self.layer1 = BasicBlock(input_channels, 64, kernel_size=8, padding=4)
        self.layer2 = BasicBlock(64, 64, kernel_size=5, padding=2)
        self.layer3 = BasicBlock(64, 64, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Classifier_RESNET:
    def __init__(self, input_shape, num_classes):
        self.model = ResNet1D(input_shape[0], num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=64):
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            self.model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Validation Accuracy: {accuracy}%")
        
        return self.model

    def predict(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()
