import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import from_networkx
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd

# Directory where the processed data is stored
graphml_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output_graphs_2'
features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output'
output_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output_pyg'
model_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/model'
model_path = os.path.join(model_dir, 'saved_model.pth')
output_csv_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/embeddings.csv'

# Ensure output and model directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Function to load features from CSV files
def load_features(ticker, features_dir):
    feature_file = os.path.join(features_dir, f"{ticker}_features.csv")
    if os.path.exists(feature_file):
        return np.loadtxt(feature_file, delimiter=",")
    else:
        return None

# Function to load a graph and add node features
def load_graph_with_features(graphml_file, features_dir, feature_dim=10):
    G = nx.read_graphml(graphml_file)
    tickers = G.nodes()

    for ticker in tickers:
        features = load_features(ticker, features_dir)
        if features is not None:
            if len(features) < feature_dim:
                padded_features = np.pad(features, (0, feature_dim - len(features)), 'constant')
            elif len(features) > feature_dim:
                padded_features = features[:feature_dim]
            else:
                padded_features = features
            G.nodes[ticker]['features'] = padded_features
        else:
            G.nodes[ticker]['features'] = np.zeros(feature_dim, dtype=np.float32)  # Set default feature vector of appropriate size
    
    return G

# Function to sanitize node attributes to ensure consistency
def sanitize_node_attributes(G, feature_dim=10):
    required_attributes = ['features', 'holders', 'description']
    for node in G.nodes():
        for attr in required_attributes:
            if attr not in G.nodes[node]:
                if attr == 'features':
                    G.nodes[node][attr] = np.zeros(feature_dim, dtype=np.float32)
                else:
                    G.nodes[node][attr] = ''
            elif attr == 'features' and not isinstance(G.nodes[node][attr], torch.Tensor):
                G.nodes[node][attr] = torch.tensor(G.nodes[node][attr], dtype=torch.float32)
    return G

# Function to check and print node attributes for debugging
def debug_node_attributes(G):
    for node, data in G.nodes(data=True):
        if 'features' not in data or 'holders' not in data or 'description' not in data:
            print(f"Node {node} is missing attributes. Data: {data}")

# Function to save the PyTorch Geometric data object
def save_pyg_data(data, output_dir, date):
    output_path = os.path.join(output_dir, f'{date}_pyg.pt')
    torch.save(data, output_path)

# Function to assign random labels for demonstration purposes
def assign_random_labels(data, num_classes=10):
    num_nodes = data.num_nodes
    data.y = torch.randint(0, num_classes, (num_nodes,))
    return data

# Main script to process each graph
def process_graphs(graphml_dir, features_dir, feature_dim=10):
    date_folders = [f for f in os.listdir(graphml_dir) if os.path.isdir(os.path.join(graphml_dir, f))]
    data_list = []

    for date_folder in tqdm(date_folders, desc="Processing graphs"):
        date_folder_path = os.path.join(graphml_dir, date_folder)
        for graphml_file in os.listdir(date_folder_path):
            if graphml_file.endswith('.graphml'):
                graph_path = os.path.join(date_folder_path, graphml_file)
                G = load_graph_with_features(graph_path, features_dir, feature_dim)

                # Sanitize node attributes to ensure consistency
                G = sanitize_node_attributes(G, feature_dim)

                # Debug node attributes
                debug_node_attributes(G)

                # Convert NetworkX graph to PyTorch Geometric data object
                try:
                    data = from_networkx(G, group_node_attrs=['features'], group_edge_attrs=['weight'])
                    
                    # Assign random labels for demonstration purposes
                    data = assign_random_labels(data)
                    
                    # Save the PyTorch Geometric data object
                    save_pyg_data(data, output_dir, date_folder)
                    data_list.append(data)
                    print(f"Processed and saved graph for date: {date_folder}")
                except ValueError as e:
                    print(f"Error processing graph for date {date_folder}: {e}")
                    continue

    return data_list

# Process the graphs and get the data list if not already processed
if not os.path.exists(os.path.join(output_dir, 'some_pyg_file.pt')):  # replace with an actual file check
    data_list = process_graphs(graphml_dir, features_dir)
else:
    data_list = []
    for file in os.listdir(output_dir):
        if file.endswith('_pyg.pt'):
            data = torch.load(os.path.join(output_dir, file))
            data_list.append(data)

# Define GraphSAGE Model
class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, aggr='mean'):
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.fc(x)
        return x

# Hyperparameters
input_dim = data_list[0].num_features  # Use number of features from the first graph as input dimension
hidden_dim = 64
output_dim = 10  # Example output dimension
num_layers = 3
aggr = 'mean'

# Create the model
model = GraphSAGEModel(input_dim, hidden_dim, output_dim, num_layers, aggr)

# Load the model if already trained
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
else:
    # Create data loaders
    batch_size = 32
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    # Training settings
    epochs = 100
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), model_path)

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    for batch in data_loader:
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return accuracy

# Create data loaders for evaluation
data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

# Evaluate the model
accuracy = evaluate(model, data_loader)
print(f'Accuracy: {accuracy:.4f}')

# Function to generate embeddings
def generate_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    node_indices = []
    with torch.no_grad():
        for batch in data_loader:
            out = model(batch.x, batch.edge_index)
            embeddings.append(out.cpu().numpy())
            node_indices.append(batch.batch.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    node_indices = np.concatenate(node_indices)
    return embeddings, node_indices

# Function to save embeddings to CSV
def save_embeddings_to_csv(embeddings, node_indices, file_path):
    df = pd.DataFrame(embeddings)
    df['node_index'] = node_indices
    df.to_csv(file_path, index=False)

# Check if embeddings are already saved
if not os.path.exists(output_csv_path):
    # Generate embeddings
    embeddings, node_indices = generate_embeddings(model, data_loader)
    # Save embeddings to CSV
    save_embeddings_to_csv(embeddings, node_indices, output_csv_path)
    print(f"Embeddings saved to {output_csv_path}")
else:
    print(f"Embeddings already saved to {output_csv_path}")
``