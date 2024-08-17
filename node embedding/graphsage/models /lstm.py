import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

# Hyperparameters
input_dim = 64 * (3 + 1)  # Feature dimension (embedding size * (top_n + 1))
hidden_dim = 128
output_dim = 2  # Binary classification (up or down)
num_layers = 2
learning_rate = 0.001
batch_size = 64
epochs = 30

# Custom Dataset Class
class StockDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Load combined features and labels
combined_features = np.load('/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/integrated_features', allow_pickle=True).item()  # Assuming it's a dictionary with tickers as keys
labels = np.load('labels.npy')  # Replace with the path to your labels

# Prepare data
features = []
labels_list = []

for ticker, feature in combined_features.items():
    features.append(feature)
    labels_list.append(labels[ticker])

features = np.array(features)
labels_list = np.array(labels_list)

# Create Dataset and DataLoader
dataset = StockDataset(features, labels_list)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))  # LSTM expects 3D input (batch_size, seq_len, input_dim)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

# Initialize model, loss function, and optimizer
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation loop
model.eval()
val_predictions = []
val_targets = []
with torch.no_grad():
    for features, labels in val_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        val_predictions.extend(predicted.cpu().numpy())
        val_targets.extend(labels.cpu().numpy())

accuracy = accuracy_score(val_targets, val_predictions)
conf_matrix = confusion_matrix(val_targets, val_predictions)

print(f"Validation Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# To compare performance with and without related corporations' features, you could train the model again using only the target company's features and repeat the evaluation process.
