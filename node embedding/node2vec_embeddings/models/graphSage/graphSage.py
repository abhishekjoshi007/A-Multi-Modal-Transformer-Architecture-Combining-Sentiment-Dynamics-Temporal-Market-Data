# Import Required Libraries
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Function to Load Node2Vec Embeddings for All Dates (Handling Non-Numerical Data)
def load_all_node2vec_embeddings(embeddings_folder):
    embeddings_dict = {}
    for date_folder in sorted(os.listdir(embeddings_folder)):
        embeddings_file = os.path.join(embeddings_folder, date_folder, f'{date_folder}_embeddings.emb')
        if os.path.isfile(embeddings_file):
            with open(embeddings_file, 'r') as f:
                embeddings = []
                for line in f:
                    parts = line.strip().split()
                    # Skip non-numerical data
                    if len(parts) > 1 and parts[0].replace('.', '', 1).isdigit():
                        vector = np.array(list(map(float, parts[1:])))
                        embeddings.append(vector)
                embeddings_dict[date_folder] = np.array(embeddings)
    return embeddings_dict

# Function to Load Node Features
def load_features(features_folder):
    features_dict = {}
    for feature_file in sorted(os.listdir(features_folder)):
        if feature_file.endswith('_integrated_features.npy'):
            date = feature_file.split('_')[0]
            features_path = os.path.join(features_folder, feature_file)
            features_dict[date] = np.load(features_path)
    return features_dict

# Function to Load Labels
def load_labels(labels_file):
    return np.load(labels_file)

# Function to Prepare PyTorch Geometric Data Objects
def prepare_data(embeddings_dict, features_dict, labels):
    all_data = []
    
    for date, embeddings in embeddings_dict.items():
        # Load features and labels if available
        features = features_dict.get(date, embeddings)  # Use embeddings as features if no additional features are available
        
        # Number of nodes
        num_nodes = embeddings.shape[0]
        
        # Create an edge index assuming a fully connected graph for example purposes
        edge_index = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it bidirectional

        # Assuming that labels are provided as a flat array corresponding to nodes across all dates
        data_labels = torch.tensor(labels[:num_nodes], dtype=torch.long)
        labels = labels[num_nodes:]  # Update labels to remove used ones
        
        # Create PyTorch Geometric Data object
        data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, y=data_labels)
        all_data.append(data)
    
    return all_data

# Define GraphSAGE Model with Layer Normalization and Dropout
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_channels)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.layer_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

# Function to Train GraphSAGE Model with Evaluation Metrics
def train_graphsage_supervised(train_data, val_data, model, optimizer, scheduler=None):
    def train(data, model, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def evaluate(data, model):
        model.eval()
        correct = 0
        y_pred = []
        y_true = []
        for data in val_data:
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    epochs = 200
    patience = 30
    best_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        for data in train_data:
            loss = train(data, model, optimizer)
            total_loss += loss
        if scheduler:
            scheduler.step()

        avg_loss = total_loss / len(train_data)
        if epoch % 10 == 0:
            accuracy = evaluate(val_data, model)
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model)  # Load the best model

# Paths to Your Folders
embeddings_folder = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/organized_embeddings'  # Replace with your actual path
features_folder = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/integrated_features'  # Replace with your actual path
labels_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/labels.npy'  # Replace with your actual path

# Load Data
embeddings_dict = load_all_node2vec_embeddings(embeddings_folder)
features_dict = load_features(features_folder)
labels = load_labels(labels_file)

# Prepare Data for Training
all_data = prepare_data(embeddings_dict, features_dict, labels)

# Split data into training and validation
train_data = all_data[:int(0.8 * len(all_data))]
val_data = all_data[int(0.8 * len(all_data)):]

# Assuming the same feature dimensionality for all dates
in_channels = all_data[0].x.shape[1]
hidden_channels = 128  # Increased hidden channels for better learning capacity
out_channels = 2  # Binary classification

# Initialize Model, Optimizer, and Scheduler
model = GraphSAGE(in_channels, hidden_channels, out_channels)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)  # Adjusted learning rate
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # Reduce LR by half every 50 epochs

# Training Loop with Scheduler
train_graphsage_supervised(train_data, val_data, model, optimizer, scheduler)

# Save the Best Model
torch.save(model.state_dict(), 'graphsage_best_model.pth')
print("Model saved as graphsage_best_model.pth")

# Load Test Data for Evaluation
# Uncomment and adjust if test data is available
# test_embeddings_dict = load_all_node2vec_embeddings(test_embeddings_folder)
# test_features_dict = load_features(test_features_folder)
# test_labels = load_labels(test_labels_file)
# test_data = prepare_data(test_embeddings_dict, test_features_dict, test_labels)

# Evaluate the Model on Test Data
def evaluate_model(data, model):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in data:
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}')

# Uncomment to evaluate
# evaluate_model(test_data, model)
