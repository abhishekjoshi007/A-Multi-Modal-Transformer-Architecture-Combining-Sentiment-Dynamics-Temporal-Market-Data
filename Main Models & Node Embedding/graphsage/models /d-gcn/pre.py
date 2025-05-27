import os
import torch
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
import torch.nn as nn

def load_graphsage_embeddings(file_path):
    """
    Load GraphSAGE embeddings from a CSV file.
    """
    print(f"Loading GraphSAGE embeddings from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Assuming the last column is node ID, and the remaining are embeddings
    embeddings = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float)
    
    print(f"Loaded GraphSAGE embeddings shape: {embeddings.shape}")
    return embeddings

def load_graphml(file_path):
    """
    Load a graph from a GraphML file and convert it to a PyTorch Geometric Data object.
    """
    print(f"Loading graph from: {file_path}")
    graph = nx.read_graphml(file_path)
    
    # Ensure all nodes have the same set of attributes
    all_node_attributes = set()
    for node, attributes in graph.nodes(data=True):
        all_node_attributes.update(attributes.keys())
    
    for node in graph.nodes():
        for attribute in all_node_attributes:
            if attribute not in graph.nodes[node]:
                graph.nodes[node][attribute] = 0  # or any default value you prefer

    data = from_networkx(graph)
    print(f"Loaded graph with {data.num_nodes} nodes and {data.num_edges} edges")
    return data

def prepare_graph_snapshots_with_embeddings(root_dir, embeddings_file_path):
    """
    Prepare a list of graph snapshots from the root directory containing graphml files,
    using precomputed GraphSAGE embeddings.
    """
    data_list = []
    date_subfolders = sorted(os.listdir(root_dir))

    # Remove hidden files like .DS_Store
    date_subfolders = [folder for folder in date_subfolders if not folder.startswith('.')]

    print(f"Date directories found: {date_subfolders}")

    # Load the GraphSAGE embeddings
    node_features = load_graphsage_embeddings(embeddings_file_path)

    for date_folder in date_subfolders:
        date_folder_path = os.path.join(root_dir, date_folder)
        preprocessed_path = os.path.join(date_folder_path, 'preprocessed_data.pt')

        # Check if preprocessed data already exists
        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed data for date {date_folder}")
            data = torch.load(preprocessed_path)
            data_list.append(data)
            continue

        if os.path.isdir(date_folder_path):
            graphml_file = None

            for file in os.listdir(date_folder_path):
                if file.endswith('.graphml'):
                    graphml_file = os.path.join(date_folder_path, file)

            if graphml_file:
                print(f"\nProcessing GraphML file for date {date_folder}: {graphml_file}")
                graph_data = load_graphml(graphml_file)

                # Attach GraphSAGE embeddings as node features
                if node_features.shape[0] != graph_data.num_nodes:
                    print(f"Mismatch in node count between graph and embeddings for date {date_folder}. Adjusting...")
                    node_features = align_node_features_and_labels(graph_data, node_features)

                graph_data.x = node_features

                # Add dummy labels if necessary
                if not hasattr(graph_data, 'y') or graph_data.y is None or len(graph_data.y) != graph_data.num_nodes:
                    graph_data.y = torch.randint(0, 10, (graph_data.num_nodes,))  # Example: Random integer labels (0 to 9)

                # Save preprocessed data
                torch.save(graph_data, preprocessed_path)
                data_list.append(graph_data)
                print(f"Added snapshot with {graph_data.num_nodes} nodes.")
            else:
                print(f"\nMissing GraphML file for date {date_folder}")
        else:
            print(f"Invalid directory found for: {date_folder}")

    print(f"Total snapshots prepared: {len(data_list)}")
    return data_list

def align_node_features_and_labels(data, node_features):
    """
    Ensure that the number of node features matches the number of nodes in the graph.
    Truncate or pad as necessary.
    """
    num_nodes = data.num_nodes
    
    # Truncate or pad node features
    if node_features.shape[0] > num_nodes:
        print(f"Truncating node features from {node_features.shape[0]} to {num_nodes}")
        node_features = node_features[:num_nodes]
    elif node_features.shape[0] < num_nodes:
        print(f"Padding node features from {node_features.shape[0]} to {num_nodes}")
        padding = torch.zeros((num_nodes - node_features.shape[0], node_features.shape[1]))
        node_features = torch.cat([node_features, padding], dim=0)
    
    return node_features

# Define D-GCN model with additional layers and batch normalization
class DGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(DGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def evaluate_model(model, data_list):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_list:
            out = model(data)
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Train D-GCN with GraphSAGE embeddings
def train_dgcn_with_graphsage(root_dir, embeddings_file_path, hidden_dim, output_dim, epochs=50, learning_rate=0.001, dropout=0.5):
    data_list = prepare_graph_snapshots_with_embeddings(root_dir, embeddings_file_path)

    input_dim = data_list[0].x.shape[1]

    model = DGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(data_list):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_list):.4f}')
    
    # Evaluate the model after training
    evaluate_model(model, data_list)

if __name__ == "__main__":
    root_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/grapghml'  # Adjust this path as needed
    embeddings_file_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/embeddings.csv'  # Adjust this path as needed
    hidden_dim = 64
    output_dim = 10  # Example number of classes or output dimensions
    epochs = 50
    learning_rate = 0.001
    dropout = 0.5

    train_dgcn_with_graphsage(root_dir, embeddings_file_path, hidden_dim, output_dim, epochs, learning_rate, dropout)
