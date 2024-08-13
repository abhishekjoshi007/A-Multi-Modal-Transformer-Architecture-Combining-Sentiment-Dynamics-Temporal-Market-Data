import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import networkx as nx
import json

# Load and preprocess data
def load_and_preprocess_data(ticker, folder_path):
    print(f"Processing {ticker}...")
    ticker_folder_path = os.path.join(folder_path, ticker)
    
    # Load historical data
    historical_data_path = os.path.join(ticker_folder_path, f"{ticker}_historic_data.csv")
    if not os.path.exists(historical_data_path):
        print(f"Historical data file not found for {ticker}")
        return None
    
    historical_data = pd.read_csv(historical_data_path)
    if historical_data.empty:
        print(f"Historical data is empty for {ticker}")
        return None
    
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data.set_index('Date', inplace=True)
    historical_data = historical_data.asfreq('D')
    
    if historical_data.isnull().values.any():
        print(f"Missing values detected in {ticker}. Handling missing values...")
        historical_data = historical_data.ffill().bfill()
    
    # Load sentiment data
    sentiment_data_path = os.path.join(ticker_folder_path, f"{ticker}_comments.json")
    if not os.path.exists(sentiment_data_path):
        print(f"Sentiment data file not found for {ticker}")
        sentiment_data = pd.DataFrame(index=historical_data.index, columns=['SentimentScore']).fillna(0)
    else:
        with open(sentiment_data_path, 'r') as f:
            sentiment_data = json.load(f)
        
        sentiment_df = pd.DataFrame(sentiment_data)
        if 'Date' in sentiment_df.columns:
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            sentiment_df.set_index('Date', inplace=True)
        elif 'time' in sentiment_df.columns:
            sentiment_df.index = pd.to_datetime(sentiment_df['time'], unit='s')
            sentiment_df.drop(columns=['time'], inplace=True)
        else:
            sentiment_df.index = pd.date_range(start=historical_data.index.min(), periods=len(sentiment_df), freq='D')
        
        # Remove duplicate dates
        sentiment_df = sentiment_df[~sentiment_df.index.duplicated(keep='first')]
        
        sentiment_data = sentiment_df.reindex(historical_data.index).fillna(0)
    
    combined_data = historical_data.join(sentiment_data, how='left').fillna(0)
    
    return combined_data

# Create graph structure
def create_graph_structure(tickers):
    G = nx.Graph()
    for i, ticker in enumerate(tickers):
        G.add_node(i, ticker=ticker)
    
    # Define relationships between stocks (example: correlation-based edges)
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:
                correlation = np.random.rand()  # Replace with actual correlation calculation
                G.add_edge(i, j, weight=correlation)
    
    return G

# Prepare data for GCN
def prepare_data(tickers, root_folder):
    data_list = []
    G = create_graph_structure(tickers)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    for i, ticker in enumerate(tickers):
        combined_data = load_and_preprocess_data(ticker, root_folder)
        if combined_data is None:
            continue
        
        close_prices = combined_data['Close'].values
        sentiment_scores = combined_data['SentimentScore'].values if 'SentimentScore' in combined_data.columns else np.zeros_like(close_prices)
        
        x = torch.tensor(np.column_stack((close_prices, sentiment_scores)), dtype=torch.float)
        y = torch.tensor(close_prices, dtype=torch.float)  # Target is the same as input for prediction
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list

# Root folder containing all stock data
root_folder = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/historic_data'
tickers = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
data_list = prepare_data(tickers, root_folder)
loader = DataLoader(data_list, batch_size=1, shuffle=True)

# Define the GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define model
model = GCN(in_channels=2, hidden_channels=16, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Calculate MRR
def calculate_mrr(true_values, predicted_values):
    true_ranks = np.argsort(np.argsort(true_values))
    predicted_ranks = np.argsort(np.argsort(predicted_values))
    reciprocal_ranks = 1 / (predicted_ranks + 1)
    return np.mean(reciprocal_ranks)

# Training loop
def train():
    model.train()
    for data in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

def evaluate():
    model.eval()
    maes, mses, rmses, r2s, mrrs = [], [], [], [], []
    with torch.no_grad():
        for data in loader:
            output = model(data)
            mae = mean_absolute_error(data.y.numpy(), output.numpy())
            mse = mean_squared_error(data.y.numpy(), output.numpy())
            rmse = np.sqrt(mse)
            r2 = r2_score(data.y.numpy(), output.numpy())
            mrr = calculate_mrr(data.y.numpy(), output.numpy())
            
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            r2s.append(r2)
            mrrs.append(mrr)
    
    return np.mean(maes), np.mean(mses), np.mean(rmses), np.mean(r2s), np.mean(mrrs)

# Training loop
for epoch in range(200):  # Adjust the number of epochs as needed
    train()
    if epoch % 10 == 0:
        mae, mse, rmse, r2, mrr = evaluate()
        print(f'Epoch {epoch}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}, MRR: {mrr}')

# Final evaluation
mae, mse, rmse, r2, mrr = evaluate()
print(f'Final results - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}, MRR: {mrr}')

# Save and compare results
results = {
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'MRR': mrr
}
results_df = pd.DataFrame([results])
results_df.to_csv('gcn_model_results.csv', index=False)

# Display final results
print(results_df)
