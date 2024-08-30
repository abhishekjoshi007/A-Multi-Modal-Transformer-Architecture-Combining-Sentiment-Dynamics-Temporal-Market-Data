import os
import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # Add these imports
import torch.nn.functional as F

def convert_emb_to_csv(emb_file_path, csv_file_path):
    """
    Convert an .emb file to .csv format, skipping any non-numeric header lines.
    """
    print(f"Converting {emb_file_path} to {csv_file_path}")
    with open(emb_file_path, 'r') as emb_file:
        lines = emb_file.readlines()
    
    data = []
    for line in lines[1:]:  # Start reading from the second line to skip potential headers
        parts = line.strip().split()
        ticker_name = parts[0]  # The ticker name (e.g., 'ASTI')
        embedding = parts[1:]
        data.append([ticker_name] + embedding)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False, header=False)
    print(f"Converted {emb_file_path} to {csv_file_path}")

def load_emb_embeddings(file_path):
    """
    Load embeddings from a CSV file, where the first column is the ticker name.
    """
    print(f"Loading embeddings from: {file_path}")
    df = pd.read_csv(file_path, header=None)
    
    # Create a dictionary to map ticker names to numeric indices
    ticker_to_index = {ticker: idx for idx, ticker in enumerate(df[0])}
    num_nodes = df.shape[0]
    emb_size = df.shape[1] - 1  # The first column is ticker name
    embeddings = torch.zeros((num_nodes, emb_size), dtype=torch.float)

    for i, row in df.iterrows():
        ticker_name = row[0]
        node_idx = ticker_to_index[ticker_name]
        values = row[1:].astype(float).tolist()
        embeddings[node_idx] = torch.tensor(values, dtype=torch.float)

    print(f"Loaded embeddings shape: {embeddings.shape}")
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

def prepare_graph_snapshots(root_dir):
    """
    Prepare a list of graph snapshots from the root directory containing both embeddings and graphml files.
    If preprocessed data already exists, load it instead of reprocessing.
    """
    data_list = []
    date_subfolders = sorted(os.listdir(root_dir))

    # Remove hidden files like .DS_Store
    date_subfolders = [folder for folder in date_subfolders if not folder.startswith('.')]

    print(f"Date directories found: {date_subfolders}")

    for date_folder in date_subfolders:
        date_folder_path = os.path.join(root_dir, date_folder)
        preprocessed_path = os.path.join(date_folder_path, 'preprocessed_data.pt')

        # Check if preprocessed data already exists
        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed data for date {date_folder}")
            data = torch.load(preprocessed_path)
            
            # Ensure dummy labels exist
            if not hasattr(data, 'y') or data.y is None or len(data.y) != data.num_nodes:
                print(f"Adding correct dummy labels for preprocessed data of date {date_folder}")
                data.y = torch.randint(0, 10, (data.num_nodes,))  # Example: Random integer labels (0 to 9)
                torch.save(data, preprocessed_path)  # Update preprocessed data with labels
            
            data_list.append(data)
            continue

        if os.path.isdir(date_folder_path):
            emb_file = None
            graphml_file = None
            csv_file = None

            for file in os.listdir(date_folder_path):
                if file.endswith('.emb'):
                    emb_file = os.path.join(date_folder_path, file)
                    csv_file = os.path.join(date_folder_path, file.replace('.emb', '.csv'))
                elif file.endswith('.graphml'):
                    graphml_file = os.path.join(date_folder_path, file)

            if emb_file and graphml_file:
                if not os.path.exists(csv_file):
                    convert_emb_to_csv(emb_file, csv_file)

                print(f"\nProcessing files for date {date_folder}:")
                print(f"CSV Embedding file: {csv_file}")
                print(f"GraphML file: {graphml_file}")

                node_features = load_emb_embeddings(csv_file)
                graph_data = load_graphml(graphml_file)

                # Check if the number of nodes in the graph matches the embeddings
                if node_features.shape[0] != graph_data.num_nodes:
                    print(f"Mismatch in node count between graph and embeddings for date {date_folder}. Adjusting...")
                    node_features = align_node_features_and_labels(graph_data, node_features)

                graph_data.x = node_features

                # Add dummy labels
                graph_data.y = torch.randint(0, 10, (graph_data.num_nodes,))  # Example: Random integer labels (0 to 9)

                # Save preprocessed data
                torch.save(graph_data, preprocessed_path)
                data_list.append(graph_data)
                print(f"Added snapshot with {graph_data.num_nodes} nodes.")
            else:
                print(f"\nMissing files for date {date_folder}")
                if not emb_file:
                    print(" - Missing embedding file (.emb)")
                if not graphml_file:
                    print(" - Missing GraphML file (.graphml)")
        else:
            print(f"Invalid directory found for: {date_folder}")

    print(f"Total snapshots prepared: {len(data_list)}")
    return data_list

class DGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(DGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
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
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def train_dgcn(root_dir, hidden_dim, output_dim, epochs=50 , learning_rate=0.01, dropout=0.5):
    data_list = prepare_graph_snapshots(root_dir)  # Load the prepared snapshots

    # Dynamically determine input_dim from the first snapshot
    input_dim = data_list[0].x.shape[1]

    model = DGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(data_list):
            if data.y is None or len(data.y) != data.num_nodes:
                print(f"Snapshot {i+1} does not have valid 'y' attribute for labels. Skipping this snapshot.")
                continue

            optimizer.zero_grad()
            out = model(data)

            # Check shapes of output and labels
            if out.shape[0] != data.y.shape[0]:
                print(f"Mismatch in sizes for Snapshot {i+1}: output {out.shape[0]}, labels {data.y.shape[0]}")
                continue

            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_list):.4f}')
    
    # Evaluate the model after training
    evaluate_model(model, data_list)

if __name__ == "__main__":
    root_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/organized_embeddings'  # Adjust this path as needed
    hidden_dim = 64
    output_dim = 10  # Example number of classes or output dimensions
    epochs = 50
    learning_rate = 0.01
    dropout = 0.5

    train_dgcn(root_dir, hidden_dim, output_dim, epochs, learning_rate, dropout)