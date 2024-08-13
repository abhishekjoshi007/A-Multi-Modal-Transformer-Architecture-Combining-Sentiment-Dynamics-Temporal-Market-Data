import os
import pandas as pd
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

# Directory where data is stored
ticker_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/Technology_data'
output_dir = 'output_graphs_2'

# Function to read data for each ticker
def read_data(ticker_folder, ticker):
    # Read historical data
    historic_data = pd.read_csv(os.path.join(ticker_folder, f"{ticker}_historic_data.csv"))
    # Read holders data
    with open(os.path.join(ticker_folder, "holder.json")) as f:
        holders_data = json.load(f)
    # Read description data
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

# Function to generate a graph with nodes and edges based on correlations
def generate_graph(daily_data, all_holders, all_descriptions, historical_data):
    G = nx.Graph()
    tickers = daily_data['Ticker Name'].unique()
    
    # Add nodes with attributes
    for ticker in tickers:
        if ticker in all_holders and ticker in all_descriptions:
            holders = sanitize_attributes(all_holders[ticker])
            description = sanitize_attributes(all_descriptions[ticker])
            G.add_node(ticker, holders=holders, description=description)
    
    # Calculate correlations and add edges
    historical_subset = historical_data[historical_data['Ticker Name'].isin(tickers)]
    correlation_matrix = historical_subset.pivot_table(index='Date', columns='Ticker Name', values='Close').corr()
    
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ticker_i = tickers[i]
            ticker_j = tickers[j]
            correlation = correlation_matrix.loc[ticker_i, ticker_j]
            if not pd.isna(correlation):  # Ensure correlation is a valid number
                G.add_edge(ticker_i, ticker_j, weight=correlation)
    
    return G

# Function to write the graph to various formats
def write_graph_to_file(G, date, output_dir):
    # Create directory for the date if it doesn't exist
    date_dir = os.path.join(output_dir, date)
    os.makedirs(date_dir, exist_ok=True)

    # Save the graph as a GML file
    gml_file_path = os.path.join(date_dir, f"{date}.gml")
    nx.write_gml(G, gml_file_path)

    # Save the graph as a GraphML file
    graphml_file_path = os.path.join(date_dir, f"{date}.graphml")
    nx.write_graphml(G, graphml_file_path)

    # Save the graph as a PNG image with labels
    png_file_path = os.path.join(date_dir, f"{date}.png")
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(png_file_path)
    plt.close()

    # Save the graph as a PNG image with nodes and edges only
    png_file_path_simple = os.path.join(date_dir, f"{date}_simple.png")
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.savefig(png_file_path_simple)
    plt.close()

    # Save the graph as a text representation
    text_file_path = os.path.join(date_dir, f"{date}.txt")
    with open(text_file_path, 'w') as f:
        f.write("Nodes:\n")
        for node, data in G.nodes(data=True):
            f.write(f"{node}: {data}\n")
        f.write("\nEdges:\n")
        for u, v, data in G.edges(data=True):
            f.write(f"{u} - {v}: {data}\n")

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

# Generate graphs for each unique date
for date in tqdm(combined_data['Date'].unique(), desc="Generating graphs for each date"):
    daily_data = combined_data[combined_data['Date'] == date]
    daily_graph = generate_graph(daily_data, all_holders, all_descriptions, combined_data)
    write_graph_to_file(daily_graph, date, output_dir)

print("Graphs have been generated and written to the file.")
