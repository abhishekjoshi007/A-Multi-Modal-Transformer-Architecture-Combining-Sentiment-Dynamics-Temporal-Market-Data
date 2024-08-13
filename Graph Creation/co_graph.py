import os
import json
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the path to the data
data_path = 'Technology_data'
output_path = 'correlation_graphs'  # Define the output path for the graphs

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Initialize the graph
G = nx.Graph()

# Load and process data
all_historic_data = []

for company_dir in os.listdir(data_path):
    company_path = os.path.join(data_path, company_dir)
    
    # Load historical data
    historic_data_path = os.path.join(company_path, f"{company_dir}_historic_data.csv")
    if not os.path.exists(historic_data_path):
        print(f"File not found: {historic_data_path}")
        continue
    
    historic_df = pd.read_csv(historic_data_path)
    if historic_df.empty:
        print(f"Historic DataFrame for {company_dir} is empty.")
        continue

    if historic_df.shape[0] < 50:  # Filter out DataFrames with fewer than 50 rows
        print(f"Historic DataFrame for {company_dir} has fewer than 50 rows. Skipping.")
        continue

    print(f"Loaded historic data for {company_dir} with shape {historic_df.shape}")
    print(f"Sample data from {company_dir}:")
    print(historic_df.head())

    industry = historic_df['Industry'].iloc[0]  # Extract industry from the first row
    market_cap = historic_df['Market Cap'].iloc[-1]  # Extract the latest market cap
    historic_df = historic_df[['Date', 'Close']].rename(columns={'Close': company_dir})
    all_historic_data.append(historic_df)
    
    # Load holder data
    holder_data_path = os.path.join(company_path, 'holder.json')
    if os.path.exists(holder_data_path):
        with open(holder_data_path, 'r') as file:
            holder_data = json.load(file)
        holder_ids = {holder['Holder'] for holder in holder_data}  # Extract holder IDs
    else:
        holder_ids = set()

    # Extract the latest financial metrics
    latest_data = historic_df.iloc[-1]
    company_info = {
        "Ticker": company_dir,
        "Sector": "Technology",  # Assuming all are in the Technology sector
        "Industry": industry,
        "Market Cap": market_cap,
        "PE Ratio": np.random.uniform(10, 30),  # Placeholder, replace with actual data if available
        "Dividend Yield": np.random.uniform(0, 5),  # Placeholder, replace with actual data if available
        "EPS": np.random.uniform(1, 10),  # Placeholder, replace with actual data if available
        "holders": ",".join(holder_ids)  # Convert set to comma-separated string
    }
    
    # Add node to the graph
    G.add_node(company_info["Ticker"], **company_info)

# Add edges based on industry and market cap range with weights
for company1, data1 in G.nodes(data=True):
    for company2, data2 in G.nodes(data=True):
        if company1 != company2 and data1['Industry'] == data2['Industry']:
            market_cap_diff = abs(data1['Market Cap'] - data2['Market Cap'])
            if market_cap_diff < 5000000:  # Example threshold for market cap range
                weight = 1 / (market_cap_diff + 1e-5)  # Adding a small value to avoid division by zero
                G.add_edge(company1, company2, weight=weight, type='industry')

# Add edges based on common holders with weights
for company1, data1 in G.nodes(data=True):
    holder_data1 = set(data1.get('holders', "").split(","))
    for company2, data2 in G.nodes(data=True):
        if company1 != company2:
            holder_data2 = set(data2.get('holders', "").split(","))
            common_holders = holder_data1.intersection(holder_data2)
            if len(common_holders) > 5:  # Increased threshold for common holders
                G.add_edge(company1, company2, weight=len(common_holders), type='holders')

# Merge all historical data into a single DataFrame
if all_historic_data:
    merged_historic_df = all_historic_data[0]
    for df in all_historic_data[1:]:
        merged_historic_df = pd.merge(merged_historic_df, df, on='Date', how='inner')

    if not merged_historic_df.empty:
        # Calculate daily returns
        merged_historic_df['Date'] = pd.to_datetime(merged_historic_df['Date'])
        merged_historic_df.set_index('Date', inplace=True)
        returns_df = merged_historic_df.pct_change().dropna()

        # Function to visualize and save the graph
        def visualize_and_save_graph(graph, title, folder_name, file_name):
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(graph, seed=42)  # Positions for all nodes
            nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='skyblue')
            nx.draw_networkx_edges(graph, pos, edge_color='b')
            nx.draw_networkx_labels(graph, pos, font_size=8, font_family='sans-serif')
            plt.title(title)
            plt.savefig(os.path.join(folder_name, file_name))
            plt.close()

        if not returns_df.empty:
            # Iterate over each day to create dynamic graphs
            for date in returns_df.index:
                daily_returns = returns_df.loc[date].dropna()
                if isinstance(daily_returns, pd.Series):
                    daily_returns = daily_returns.to_frame().T
                correlation_matrix = daily_returns.corr()
                daily_graph = nx.Graph()

                for company1 in correlation_matrix.index:
                    daily_graph.add_node(company1, **G.nodes[company1])

                # Define higher correlation threshold
                correlation_threshold = 0.9

                for company1 in correlation_matrix.index:
                    for company2 in correlation_matrix.index:
                        if company1 != company2 and correlation_matrix.loc[company1, company2] > correlation_threshold:
                            daily_graph.add_edge(company1, company2, weight=correlation_matrix.loc[company1, company2], type='correlation')

                # Create subfolder based on the date
                date_folder = os.path.join(output_path, date.strftime('%Y-%m-%d'))
                if not os.path.exists(date_folder):
                    os.makedirs(date_folder)

                # Save the daily graph
                graph_file_name = os.path.join(date_folder, f"correlation_graph_{date.strftime('%Y-%m-%d')}.graphml")
                nx.write_graphml(daily_graph, graph_file_name)

                # Visualize and save the graph
                visualize_and_save_graph(daily_graph, f"Correlation Graph for {date.strftime('%Y-%m-%d')}", date_folder, f"correlation_graph_{date.strftime('%Y-%m-%d')}.png")
else:
    print("No historical data available.")
