import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

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

# Paths to the saved model weights
integrated_features_model_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /LSTM(Without Sentiments)/baseline_stock_price_lstm.pth'  # Adjust the path if necessary
baseline_features_model_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /LSTM(Without Sentiments)/baseline_stock_price_lstm.pth'  # Adjust the path if necessary

# Load the trained models
input_dim = 6  # You should set this based on your feature vector dimensions
hidden_dim = 64
output_dim = 1
num_layers = 2

integrated_features_model = StockPriceLSTM(input_dim, hidden_dim, output_dim, num_layers)
baseline_features_model = StockPriceLSTM(input_dim, hidden_dim, output_dim, num_layers)

integrated_features_model.load_state_dict(torch.load(integrated_features_model_path))
baseline_features_model.load_state_dict(torch.load(baseline_features_model_path))

integrated_features_model.eval()
baseline_features_model.eval()

# Prepare the test dataset (assuming you've already created it)
# For the purpose of this example, let's assume baseline_dataset is already split into training and test sets
# If you haven't split it yet, you'll need to split baseline_dataset into training and testing parts

# Initialize dataloaders for evaluation
# Replace with your actual test dataset and DataLoader
# test_loader = DataLoader(baseline_dataset, batch_size=32, shuffle=False)  # Replace with your test dataset

# Function to evaluate a model
def evaluate_model(model, dataloader):
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            predictions = model(features)
            predictions = predictions.squeeze().round().cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_labels.extend(labels)
            all_predictions.extend(predictions)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

# Evaluate both models
# Assuming you have test_loader ready
# accuracy_integrated = evaluate_model(integrated_features_model, test_loader)
# accuracy_baseline = evaluate_model(baseline_features_model, test_loader)

# For demonstration purposes, we'll print dummy accuracy values
accuracy_integrated = 0.75  # Replace with the result from evaluate_model
accuracy_baseline = 0.70  # Replace with the result from evaluate_model

print(f"Accuracy of LSTM with Integrated Features: {accuracy_integrated * 100:.2f}%")
print(f"Accuracy of LSTM with Baseline Features: {accuracy_baseline * 100:.2f}%")

# Compare the performance
if accuracy_integrated > accuracy_baseline:
    print("LSTM with Integrated Features performs better.")
else:
    print("LSTM with Baseline Features performs better.")
