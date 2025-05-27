import os
import pandas as pd
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import gc

# Directory where data is stored
ticker_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/Technology_data'
output_dir = 'output_graphs_2'
checkpoint_file = 'processed_dates.txt'

# Function to read data for each ticker
def read_data(ticker_folder, ticker):
    historic_data = pd.read_csv(os.path.join(ticker_folder, f"{ticker}_historic_data.csv"))
    with open(os.path.join(ticker_folder, "holder.json")) as f:
        holders_data = json.load(f)
    with open(os.path.join(ticker_folder, "description.json")) as f:
        description_data = json.load(f)
    return historic_data, holders_data, description_data

# Function to sanitize attributes for graph nodes
def sanitize_attributes(attributes):
    sanitized = {}
    for key, value in attributes.items():
        new_key = key.replace(" ", "_").replace("%", "percent").replace(".", "")
        if isinstance(value, (list, dict, np.ndarray)):
            sanitized[new_key] = json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
        else:
            sanitized[new_key] = value
    return sanitized

# Function to sanitize list of holders data
def sanitize_holders(holders_data):
    sanitized_holders = []
    for holder in holders_data:
        sanitized_holder = sanitize_attributes(holder)
        sanitized_holders.append(sanitized_holder)
    return json.dumps(sanitized_holders)

# Function to generate a graph with nodes and edges based on correlations
def generate_graph(daily_data, all_holders, all_descriptions, historical_data):
    G = nx.Graph()
    tickers = daily_data['Ticker Name'].unique()
    
    for ticker in tickers:
        if ticker in all_holders and ticker in all_descriptions:
            holders = sanitize_holders(all_holders[ticker])
            description = json.dumps(sanitize_attributes(all_descriptions[ticker]))
            features = daily_data[daily_data['Ticker Name'] == ticker][['Close']].values.flatten()
            normalized_features = StandardScaler().fit_transform(features.reshape(-1, 1)).flatten()
            G.add_node(ticker, holders=holders, description=description, features=json.dumps(normalized_features.tolist()))
    
    historical_subset = historical_data[historical_data['Ticker Name'].isin(tickers)]
    correlation_matrix = historical_subset.pivot_table(index='Date', columns='Ticker Name', values='Close').corr()
    
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ticker_i = tickers[i]
            ticker_j = tickers[j]
            correlation = correlation_matrix.loc[ticker_i, ticker_j]
            if not pd.isna(correlation):
                G.add_edge(ticker_i, ticker_j, weight=correlation)
    
    return G

# Function to write the graph to various formats
def write_graph_to_file(G, date, output_dir):
    date_dir = os.path.join(output_dir, date)
    os.makedirs(date_dir, exist_ok=True)

    gml_file_path = os.path.join(date_dir, f"{date}.gml")
    nx.write_gml(G, gml_file_path)

    graphml_file_path = os.path.join(date_dir, f"{date}.graphml")
    nx.write_graphml(G, graphml_file_path)

    png_file_path = os.path.join(date_dir, f"{date}.png")
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(png_file_path)
    plt.close()

# Load processed dates from checkpoint file
def load_processed_dates(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            return set(file.read().splitlines())
    return set()

# Save processed date to checkpoint file
def save_processed_date(checkpoint_file, date):
    with open(checkpoint_file, 'a') as file:
        file.write(f"{date}\n")

# Main script
all_holders = {}
all_descriptions = {}
combined_data = pd.DataFrame()

# Read data for each ticker
ticker_folders = [f for f in os.listdir(ticker_dir) if os.path.isdir(os.path.join(ticker_dir, f))]
for ticker in tqdm(ticker_folders, desc="Reading data for tickers"):
    ticker_folder_path = os.path.join(ticker_dir, ticker)
    try:
        historic_data, holders_data, description_data = read_data(ticker_folder_path, ticker)
        combined_data = pd.concat([combined_data, historic_data], ignore_index=True)
        all_holders[ticker] = holders_data
        all_descriptions[ticker] = description_data
    except FileNotFoundError as e:
        print(f"Error reading data for {ticker}: {e}")

# Load processed dates
processed_dates = load_processed_dates(checkpoint_file)

# Process data in chunks
dates = combined_data['Date'].unique()
chunk_size = 1
for i in range(0, len(dates), chunk_size):
    date_chunk = dates[i:i+chunk_size]
    for date in tqdm(date_chunk, desc=f"Processing chunk {i//chunk_size+1}"):
        if date in processed_dates:
            continue
        
        date_dir = os.path.join(output_dir, date)
        gml_file_path = os.path.join(date_dir, f"{date}.gml")
        graphml_file_path = os.path.join(date_dir, f"{date}.graphml")
        png_file_path = os.path.join(date_dir, f"{date}.png")
        
        if os.path.exists(gml_file_path) and os.path.exists(graphml_file_path) and os.path.exists(png_file_path):
            save_processed_date(checkpoint_file, date)
            continue
        
        daily_data = combined_data[combined_data['Date'] == date]
        daily_graph = generate_graph(daily_data, all_holders, all_descriptions, combined_data)
        write_graph_to_file(daily_graph, date, output_dir)
        save_processed_date(checkpoint_file, date)
        
        del daily_graph, daily_data
        gc.collect()

print("Graphs have been generated and written to the file.")
