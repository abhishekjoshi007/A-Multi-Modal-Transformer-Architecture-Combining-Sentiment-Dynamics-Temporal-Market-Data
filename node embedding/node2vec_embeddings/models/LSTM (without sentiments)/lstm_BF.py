import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Define the paths
baseline_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/Technology_data'  # Update with the correct path
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/labels.npy'

# Load labels
labels = np.load(labels_path, allow_pickle=True)

# Custom Dataset class for loading baseline features and labels
class BaselineStockDataset(Dataset):
    def __init__(self, features_dir, labels):
        self.features_dir = features_dir
        self.labels = labels
        self.tickers = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
        if not self.tickers:
            raise ValueError("No stock folders found in the specified directory.")
        
    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):
        ticker = self.tickers[idx]
        feature_path = os.path.join(self.features_dir, ticker, f"{ticker}_historic_data.csv")
        df = pd.read_csv(feature_path)
        
        # Select only the numeric columns needed for the model
        features = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
        
        # Optionally, pad sequences to match the length required for LSTM (same as what you did in integrated features)
        max_seq_len = 256
        if len(features) > max_seq_len:
            features = features[-max_seq_len:]
        else:
            padding = np.zeros((max_seq_len - len(features), features.shape[1]))
            features = np.vstack((padding, features))
        
        label = self.labels[idx]
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Initialize the dataset
baseline_dataset = BaselineStockDataset(baseline_features_dir, labels)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(baseline_dataset))
test_size = len(baseline_dataset) - train_size
train_dataset, test_dataset = random_split(baseline_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    for features, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        output = baseline_model(features)  # No need to unsqueeze, features is already 3D
        loss = criterion(output, label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Save the baseline model
torch.save(baseline_model.state_dict(), 'baseline_stock_price_lstm.pth')
print("Baseline model training completed and saved.")

# Evaluation function to compute accuracy, F1 score, precision, and recall
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            preds = outputs.round()  # Thresholding at 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Evaluate the model
evaluate_model(baseline_model, test_loader)
