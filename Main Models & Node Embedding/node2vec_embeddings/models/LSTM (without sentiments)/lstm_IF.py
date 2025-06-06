import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Define the paths
features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/integrated_features'
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/labels.npy'

# Load labels
labels = np.load(labels_path, allow_pickle=True)

# Custom Dataset class for loading features and labels
class StockDataset(Dataset):
    def __init__(self, features_dir, labels, max_len=None):
        self.features_dir = features_dir
        self.labels = labels
        self.tickers = [f.replace('_integrated_features.npy', '') for f in os.listdir(features_dir) if f.endswith('.npy')]
        self.max_len = max_len if max_len else self.get_max_len()

    def get_max_len(self):
        max_len = 0
        for ticker in self.tickers:
            feature_path = os.path.join(self.features_dir, f"{ticker}_integrated_features.npy")
            features = np.load(feature_path)
            max_len = max(max_len, len(features))
        return max_len

    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):
        ticker = self.tickers[idx]
        feature_path = os.path.join(self.features_dir, f"{ticker}_integrated_features.npy")
        features = np.load(feature_path)
        padded_features = F.pad(torch.tensor(features, dtype=torch.float32), (0, self.max_len - len(features)))
        label = self.labels[idx]
        return padded_features, torch.tensor(label, dtype=torch.float32)

# Initialize the dataset and dataloader
dataset = StockDataset(features_dir, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the LSTM model
class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        out = self.sigmoid(out)
        return out

# Hyperparameters
input_dim = dataset[0][0].shape[0]  # The length of the feature vector
hidden_dim = 64
output_dim = 1
num_layers = 2
num_epochs = 50
learning_rate = 0.001

# Model, loss function, optimizer
model = StockPriceLSTM(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for features, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        output = model(features.unsqueeze(1))
        loss = criterion(output, label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# Save the model
torch.save(model.state_dict(), 'Integrated_features_stock_price_lstm.pth')
print("Model training completed and saved.")

# Evaluate the model
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for features, label in dataloader:
        output = model(features.unsqueeze(1))
        y_true.extend(label.numpy())
        y_pred.extend((output.squeeze().numpy() > 0.5).astype(int))  # Convert probabilities to binary predictions

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
