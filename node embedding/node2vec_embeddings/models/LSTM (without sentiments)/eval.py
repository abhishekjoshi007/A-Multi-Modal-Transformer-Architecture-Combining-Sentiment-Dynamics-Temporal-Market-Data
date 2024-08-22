import os
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.utils.rnn as rnn_utils

# Define the LSTM model
class StockPriceLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StockPriceLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        out = self.sigmoid(out)
        return out

# Define the StockDataset class
class StockDataset(Dataset):
    def __init__(self, features_dir, labels, feature_dim):
        self.features_dir = features_dir
        self.labels = labels
        self.feature_dim = feature_dim
        self.tickers = [f.replace(f'_features.npy', '') for f in os.listdir(features_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):
        ticker = self.tickers[idx]
        feature_path = os.path.join(self.features_dir, f"{ticker}_features.npy")
        features = np.load(feature_path)
        
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch for {ticker}: expected {self.feature_dim}, got {features.shape[-1]}")

        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class StockDatasetBaseline(Dataset):
    def __init__(self, features_dir, labels):
        self.features_dir = features_dir
        self.labels = labels
        self.tickers = [f.replace(f'_features.npy', '') for f in os.listdir(features_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):
        ticker = self.tickers[idx]
        feature_path = os.path.join(self.features_dir, f"{ticker}_features.npy")
        features = np.load(feature_path)
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = rnn_utils.pad_sequence(features, batch_first=True)
    return features_padded, torch.stack(labels)


# Paths to the saved model weights
integrated_features_model_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/Integrated _features_stock_price_lstm.pth'
baseline_features_model_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/baseline_stock_price_lstm.pth'

# Load the trained models
integrated_input_dim = 256  # This matches the dimension used for integrated features
baseline_input_dim = 6  # This matches the dimension used for baseline features
hidden_dim = 64
output_dim = 1
num_layers = 2

integrated_features_model = StockPriceLSTM(integrated_input_dim, hidden_dim, output_dim, num_layers)
baseline_features_model = StockPriceLSTM(baseline_input_dim, hidden_dim, output_dim, num_layers)

integrated_features_model.load_state_dict(torch.load(integrated_features_model_path))
baseline_features_model.load_state_dict(torch.load(baseline_features_model_path))

integrated_features_model.eval()
baseline_features_model.eval()

# Assuming you have test datasets for both integrated and baseline features
integrated_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/integrated_features'
baseline_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/baseline_features'
labels = np.load('/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/labels.npy')

# Create Datasets and DataLoaders
integrated_test_dataset = StockDataset(integrated_features_dir, labels, integrated_input_dim)
baseline_test_dataset = StockDatasetBaseline(baseline_features_dir, labels)

integrated_test_loader = DataLoader(integrated_test_dataset, batch_size=32, shuffle=False)
baseline_test_loader = DataLoader(baseline_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


def evaluate_model(model, dataloader):
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")  # Debugging print
            
            # Ensure correct dimensions for LSTM input
            if len(features.shape) == 3:  # If features are already 3D, no need to unsqueeze
                input_data = features
            elif len(features.shape) == 2:  # If features are 2D, add a sequence length dimension
                input_data = features.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
            
            predictions = model(input_data)  # Pass the correctly shaped input to the model
            predictions = predictions.squeeze().round().cpu().numpy()
            
            labels = labels.cpu().numpy()
            
            all_labels.extend(labels)
            all_predictions.extend(predictions)
    
    print(f"All labels: {all_labels}")  # Debugging print
    print(f"All predictions: {all_predictions}")  # Debugging print
    
    if len(all_labels) == 0 or len(all_predictions) == 0:
        raise ValueError("No valid predictions were generated, check the input data and model.")
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy


# Evaluate both models using the actual data
accuracy_integrated = evaluate_model(integrated_features_model, integrated_test_loader)
accuracy_baseline = evaluate_model(baseline_features_model, baseline_test_loader)

print(f"Accuracy of LSTM with Integrated Features: {accuracy_integrated * 100:.2f}%")
print(f"Accuracy of LSTM with Baseline Features: {accuracy_baseline * 100:.2f}%")

# Compare the performance
if accuracy_integrated > accuracy_baseline:
    print("LSTM with Integrated Features performs better.")
else:
    print("LSTM with Baseline Features performs better.")
