import os
import json
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the path to the data
data_path = 'Technology_data'

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

# Initialize separate graphs
G_industry = nx.Graph()
G_holders = nx.Graph()
G_correlation = nx.Graph()

# Copy nodes to separate graphs
for node, data in G.nodes(data=True):
    G_industry.add_node(node, **data)
    G_holders.add_node(node, **data)
    G_correlation.add_node(node, **data)

# Add edges based on industry and market cap range with weights
for company1, data1 in G.nodes(data=True):
    for company2, data2 in G.nodes(data=True):
        if company1 != company2 and data1['Industry'] == data2['Industry']:
            market_cap_diff = abs(data1['Market Cap'] - data2['Market Cap'])
            if market_cap_diff < 5000000:  # Example threshold for market cap range
                weight = 1 / (market_cap_diff + 1e-5)  # Adding a small value to avoid division by zero
                G.add_edge(company1, company2, weight=weight, type='industry')
                G_industry.add_edge(company1, company2, weight=weight, type='industry')

# Add edges based on common holders with weights
for company1, data1 in G.nodes(data=True):
    holder_data1 = set(data1.get('holders', "").split(","))
    for company2, data2 in G.nodes(data=True):
        if company1 != company2:
            holder_data2 = set(data2.get('holders', "").split(","))
            common_holders = holder_data1.intersection(holder_data2)
            if len(common_holders) > 5:  # Increased threshold for common holders
                G.add_edge(company1, company2, weight=len(common_holders), type='holders')
                G_holders.add_edge(company1, company2, weight=len(common_holders), type='holders')

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

        if not returns_df.empty:
            # Standardize the returns for better comparison
            scaler = StandardScaler()
            scaled_returns = pd.DataFrame(scaler.fit_transform(returns_df), columns=returns_df.columns, index=returns_df.index)

            # Calculate correlation matrix
            correlation_matrix = scaled_returns.corr()

            # Define higher correlation threshold
            correlation_threshold = 0.9

            # Add correlation-based edges
            for company1 in correlation_matrix.columns:
                for company2 in correlation_matrix.columns:
                    if company1 != company2 and correlation_matrix.loc[company1, company2] > correlation_threshold:
                        print(f"Adding correlation edge between {company1} and {company2}: {correlation_matrix.loc[company1, company2]}")  # Debugging
                        G.add_edge(company1, company2, weight=correlation_matrix.loc[company1, company2], type='correlation')
                        G_correlation.add_edge(company1, company2, weight=correlation_matrix.loc[company1, company2], type='correlation')

            # Save the graph
            os.makedirs("graphs", exist_ok=True)
            nx.write_graphml(G, "graphs/technology_companies_graph.graphml")
            nx.write_graphml(G_industry, "graphs/industry_graph.graphml")
            nx.write_graphml(G_holders, "graphs/holders_graph.graphml")
            nx.write_graphml(G_correlation, "graphs/correlation_graph.graphml")

            # Save textual representation of the graph
            with open("graphs/graph_representation.txt", "w") as f:
                f.write("Textual Representation of the Graph:\n")
                for node in G.nodes(data=True):
                    f.write(f"Company: {node[0]}, Attributes: {node[1]}\n")

                f.write("\nEdges:\n")
                for edge in G.edges(data=True):
                    f.write(f"Edge between {edge[0]} and {edge[1]}, Type: {edge[2]['type']}, Weight: {edge[2]['weight']}\n")

                # Verification
                f.write("\nVerification:\n")
                if G.has_edge('AAPL', 'GOOG'):
                    f.write("AAPL and GOOG are connected.\n")

                # Checking specific relationships
                f.write("\nChecking specific relationships:\n")
                if G.has_edge('AAPL', 'GOOG'):
                    edge_data = G.get_edge_data('AAPL', 'GOOG')
                    f.write(f"Edge between AAPL and GOOG, Type: {edge_data['type']}, Weight: {edge_data['weight']}\n")

                # Degree of each node
                f.write("\nDegree of each node:\n")
                degrees = dict(G.degree())
                f.write(f"{degrees}\n")

                # Clustering coefficient of each node
                f.write("\nClustering coefficient of each node:\n")
                clustering_coeffs = nx.clustering(G)
                f.write(f"{clustering_coeffs}\n")

                # Centrality of each node
                f.write("\nCentrality of each node:\n")
                centrality = nx.degree_centrality(G)
                f.write(f"{centrality}\n")

            # Visualize the graph
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'industry'], edge_color='g', label='Industry')
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'holders'], edge_color='r', label='Holders')
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'correlation'], edge_color='b', label='Correlation')
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            plt.title("Technology Companies Graph")
            plt.legend(loc="upper left")
            plt.savefig("graphs/technology_companies_graph.png")
            plt.show()

            # Visualize the industry graph
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G_industry, seed=42)
            nx.draw_networkx_nodes(G_industry, pos, node_size=500, node_color='skyblue')
            nx.draw_networkx_edges(G_industry, pos, edge_color='g', label='Industry')
            nx.draw_networkx_labels(G_industry, pos, font_size=8, font_family='sans-serif')
            plt.title("Industry-Based Graph")
            plt.legend(loc="upper left")
            plt.savefig("graphs/industry_graph.png")
            plt.show()

            # Visualize the holders graph
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G_holders, seed=42)
            nx.draw_networkx_nodes(G_holders, pos, node_size=500, node_color='skyblue')
            nx.draw_networkx_edges(G_holders, pos, edge_color='r', label='Holders')
            nx.draw_networkx_labels(G_holders, pos, font_size=8, font_family='sans-serif')
            plt.title("Holders-Based Graph")
            plt.legend(loc="upper left")
            plt.savefig("graphs/holders_graph.png")
            plt.show()

            # Visualize the correlation graph
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G_correlation, seed=42)
            nx.draw_networkx_nodes(G_correlation, pos, node_size=500, node_color='skyblue')
            nx.draw_networkx_edges(G_correlation, pos, edge_color='b', label='Correlation')
            nx.draw_networkx_labels(G_correlation, pos, font_size=8, font_family='sans-serif')
            plt.title("Correlation-Based Graph")
            plt.legend(loc="upper left")
            plt.savefig("graphs/correlation_graph.png")
            plt.show()
else:
    print("No historical data available.")

