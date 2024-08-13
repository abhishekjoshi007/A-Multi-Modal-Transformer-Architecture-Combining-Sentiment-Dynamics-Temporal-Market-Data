import pandas as pd
import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import torch

# Directory where data is stored
data_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/Technology_data'
output_dir = 'output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to load historical data
def load_historical_data(ticker_folder, ticker):
    file_path = os.path.join(ticker_folder, f"{ticker}_historic_data.csv")
    if os.path.exists(file_path):
        historical_data = pd.read_csv(file_path)
        # Select relevant columns and ignore 'Market Cap'
        historical_data = historical_data[['Date', 'Ticker Name', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        return historical_data
    else:
        print(f"File not found for ticker: {ticker}")
        return None

# Function to load description data
def load_description_data(ticker_folder):
    file_path = os.path.join(ticker_folder, "description.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            description_data = json.load(file)
        return description_data['description']
    else:
        print(f"File not found in {ticker_folder}")
        return None

# Function to load holder data
def load_holder_data(ticker_folder):
    file_path = os.path.join(ticker_folder, "holder.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            holder_data = json.load(file)
        return holder_data
    else:
        print(f"File not found in {ticker_folder}")
        return None

# Function to calculate daily returns
def calculate_daily_returns(historical_data):
    historical_data['Daily Return'] = historical_data['Adj Close'].pct_change()
    return historical_data

# Function to calculate moving averages
def calculate_moving_averages(historical_data, window=5):
    historical_data[f'{window}-Day MA'] = historical_data['Adj Close'].rolling(window=window).mean()
    return historical_data

# Function to convert descriptions to BERT embeddings
def description_to_bert_embedding(description):
    inputs = tokenizer(description, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    # Get the mean pooling of the last hidden state
    embedding = torch.mean(outputs.last_hidden_state, 1).detach().numpy()
    return embedding

# Function to calculate total percentage of shares held by top N holders
def calculate_top_n_holders(holder_data, n=5):
    top_holders = sorted(holder_data, key=lambda x: float(x['% Out'].strip('%')), reverse=True)[:n]
    total_percentage = sum(float(holder['% Out'].strip('%')) for holder in top_holders)
    return total_percentage

# Function to normalize features
def normalize_features(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# Lists to store all features
all_historical_features = []
all_textual_features = []
all_holder_features = []

# Iterate over all tickers in the data directory
for ticker_folder in os.listdir(data_dir):
    full_path = os.path.join(data_dir, ticker_folder)
    if os.path.isdir(full_path):
        ticker = ticker_folder  # Assume the folder name is the ticker
        print(f"Processing ticker: {ticker}")
        
        # Load historical data
        historical_data = load_historical_data(full_path, ticker)
        if historical_data is not None:
            historical_data = calculate_daily_returns(historical_data)
            historical_data = calculate_moving_averages(historical_data)
            historical_features = historical_data[['Daily Return', '5-Day MA']].dropna().values
            all_historical_features.append(historical_features)
        
        # Load description data
        description_data = load_description_data(full_path)
        if description_data is not None:
            textual_features = description_to_bert_embedding(description_data)
            all_textual_features.append(textual_features)
        
        # Load holder data
        holder_data = load_holder_data(full_path)
        if holder_data is not None:
            holder_features = calculate_top_n_holders(holder_data)
            all_holder_features.append([holder_features])

# Concatenate all features for normalization
historical_features_concat = np.concatenate(all_historical_features, axis=0)
textual_features_concat = np.concatenate(all_textual_features, axis=0)
holder_features_concat = np.array(all_holder_features)

# Normalize the features
normalized_historical_features = normalize_features(historical_features_concat)
normalized_textual_features = normalize_features(textual_features_concat)
normalized_holder_features = normalize_features(holder_features_concat)

# Combine and save the features
all_combined_features = []

for i, ticker_folder in enumerate(os.listdir(data_dir)):
    full_path = os.path.join(data_dir, ticker_folder)
    if os.path.isdir(full_path):
        ticker = ticker_folder  # Assume the folder name is the ticker
        
        # Combine features: Historical, Textual, and Holder
        if i < len(normalized_historical_features) and i < len(normalized_textual_features) and i < len(normalized_holder_features):
            historical_features = normalized_historical_features[i]
            textual_features = normalized_textual_features[i]
            holder_features = normalized_holder_features[i]
            
            # Flatten the features if needed
            if len(historical_features.shape) > 1:
                historical_features = historical_features.flatten()
            if len(textual_features.shape) > 1:
                textual_features = textual_features.flatten()
            
            # Combine features
            combined_features = np.concatenate((historical_features, textual_features, holder_features))
            
            # Save the combined features
            all_combined_features.append(combined_features)
            output_file = os.path.join(output_dir, f"{ticker}_features.csv")
            np.savetxt(output_file, combined_features, delimiter=",")
            print(f"Saved features for ticker: {ticker}")

# Save all combined features in a single file for easy loading
all_combined_features = np.array(all_combined_features)
np.savetxt(os.path.join(output_dir, "all_combined_features.csv"), all_combined_features, delimiter=",")
print("All combined features have been saved.")
