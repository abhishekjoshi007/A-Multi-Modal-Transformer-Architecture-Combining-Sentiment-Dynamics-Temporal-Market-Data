import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Define the paths
features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/integrated_features'
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /LSTM(Without Sentiments)/labels.npy'

# Load labels
labels = np.load(labels_path, allow_pickle=True)

# Normalize labels if necessary
# Assuming labels are binary (0 or 1), otherwise, you might need to normalize/scale them accordingly
assert labels.max() <= 1 and labels.min() >= 0, "Labels must be binary (0 or 1) for BCELoss."

# Custom Dataset class for loading features and labels
class StockDataset(Dataset):
    def __init__(self, features_dir, labels):
        self.features_dir = features_dir
        self.labels = labels
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.tickers = [f.replace('_integrated_features.npy', '') for f in os.listdir(features_dir) if f.endswith('.npy')]
        
        # Normalize all features
        self.features = []
        for ticker in self.tickers:
            feature_path = os.path.join(self.features_dir, f"{ticker}_integrated_features.npy")
            features = np.load(feature_path)
            features = features.reshape(-1, 1)  # Reshape to 2D array
            normalized_features = self.scaler.fit_transform(features)
            self.features.append(normalized_features.flatten())  # Flatten back to 1D if needed
        
    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Initialize the dataset and dataloader
dataset = StockDataset(features_dir, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the LSTM model
class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.3):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])  # Take the output from the last time step and apply dropout
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Hyperparameters
input_dim = dataset[0][0].shape[0]  # Adjusted to the shape after normalization
hidden_dim = 128  # Increased hidden dimension for higher capacity
output_dim = 1
num_layers = 3  # Increased number of layers for more depth
num_epochs = 50
learning_rate = 0.0001  # Lowered learning rate

# Model, loss function, optimizer
model = StockPriceLSTM(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping parameters
patience = 5
best_loss = np.inf
epochs_no_improve = 0

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
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")
    
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_stock_price_lstm.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping due to no improvement.")
            break

# Load the best model if early stopping was triggered
model.load_state_dict(torch.load('best_stock_price_lstm.pth'))
print("Model training completed and saved.")
