import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

# Define the paths
baseline_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/Technology_data'
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /LSTM(Without Sentiments)/labels.npy'

# Load labels
labels = np.load(labels_path, allow_pickle=True)

# Custom Dataset class for loading baseline features and labels
class BaselineStockDataset(Dataset):
    def __init__(self, features_dir, labels, max_seq_length=None):
        self.features_dir = features_dir
        self.labels = labels
        self.tickers = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
        if not self.tickers:
            raise ValueError("No CSV files found in the specified directory.")
        self.max_seq_length = max_seq_length or self.get_max_seq_length()

    def get_max_seq_length(self):
        max_len = 0
        for ticker in self.tickers:
            feature_path = os.path.join(self.features_dir, ticker, f"{ticker}_historic_data.csv")
            df = pd.read_csv(feature_path)
            max_len = max(max_len, len(df))
        return max_len

    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):
        ticker = self.tickers[idx]
        feature_path = os.path.join(self.features_dir, ticker, f"{ticker}_historic_data.csv")
        df = pd.read_csv(feature_path)
        
        # Select only the numeric columns needed for the model
        features = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
        
        # Pad sequences to the maximum sequence length
        if len(features) < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - len(features), features.shape[1]))
            features = np.vstack((features, padding))
        elif len(features) > self.max_seq_length:
            features = features[:self.max_seq_length]

        label = self.labels[idx]
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Initialize the dataset and dataloader for baseline features
baseline_dataset = BaselineStockDataset(baseline_features_dir, labels)
baseline_dataloader = DataLoader(baseline_dataset, batch_size=32, shuffle=True)

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
input_dim = baseline_dataset[0][0].shape[1]  # The number of columns in the feature vector
hidden_dim = 64
output_dim = 1
num_layers = 2
num_epochs = 50
learning_rate = 0.001

# Model, loss function, optimizer
baseline_model = StockPriceLSTM(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.BCELoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)

# Training loop for baseline features
baseline_model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for features, label in tqdm(baseline_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        output = baseline_model(features)
        loss = criterion(output, label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(baseline_dataloader)}")

# Save the baseline model
torch.save(baseline_model.state_dict(), 'baseline_stock_price_lstm.pth')
print("Baseline model training completed and saved.")
