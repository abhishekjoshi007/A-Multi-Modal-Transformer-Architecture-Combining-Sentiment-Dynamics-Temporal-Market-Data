import os
import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
from torch.utils.data import random_split
from main import prepare_graph_snapshots

class DGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(DGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
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
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def train_dgcn(root_dir, hidden_dim, output_dim, epochs=50, learning_rate=0.01, dropout=0.5):
    data_list = prepare_graph_snapshots(root_dir)  # Load the prepared snapshots

    # Dynamically determine input_dim from the first snapshot
    input_dim = data_list[0].x.shape[1]

    # Split data into training and validation sets
    train_size = int(0.8 * len(data_list))
    val_size = len(data_list) - train_size
    train_data, val_data = random_split(data_list, [train_size, val_size])

    model = DGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, data in enumerate(train_data):
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

        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}')

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_data:
                out = model(data)
                if out.shape[0] != data.y.shape[0]:
                    continue
                val_loss += F.nll_loss(out, data.y).item()
        
        val_loss /= len(val_data)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}')

    # Evaluate the model after training
    evaluate_model(model, data_list)

if __name__ == "__main__":
    root_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/organized_embeddings'  # Adjust this path as needed
    hidden_dim = 64
    output_dim = 10  # Example number of classes or output dimensions
    epochs = 100  # Increased epochs for better training
    learning_rate = 0.001  # Adjusted learning rate
    dropout = 0.5

    train_dgcn(root_dir, hidden_dim, output_dim, epochs, learning_rate, dropout)
