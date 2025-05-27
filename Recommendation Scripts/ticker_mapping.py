import os
import networkx as nx
import pandas as pd

# Directory where the GraphML files are stored
graphml_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/grapghml'
output_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/ticker_mapping.csv'

def generate_ticker_mapping(graphml_dir, output_file):
    ticker_mapping = {}
    node_index = 0
    
    date_folders = [f for f in os.listdir(graphml_dir) if os.path.isdir(os.path.join(graphml_dir, f))]

    for date_folder in date_folders:
        date_folder_path = os.path.join(graphml_dir, date_folder)
        for graphml_file in os.listdir(date_folder_path):
            if graphml_file.endswith('.graphml'):
                graph_path = os.path.join(date_folder_path, graphml_file)
                G = nx.read_graphml(graph_path)
                for node in G.nodes():
                    if node not in ticker_mapping:
                        ticker_mapping[node] = node_index
                        node_index += 1

    # Convert the mapping dictionary to a DataFrame
    df = pd.DataFrame(list(ticker_mapping.items()), columns=['Ticker', 'Index'])
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f'Ticker mapping saved to {output_file}')

generate_ticker_mapping(graphml_dir, output_file)
