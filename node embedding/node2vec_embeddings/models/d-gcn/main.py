import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from prepare_data import prepare_graph_snapshots  # Import your data preparation function

class DGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_dgcn(root_dir, hidden_dim, output_dim, epochs=100):
    data_list = prepare_graph_snapshots(root_dir)  # Load the prepared snapshots

    # Dynamically determine input_dim from the first snapshot
    input_dim = data_list[0].x.shape[1]

    # Check for consistent input dimensions across all snapshots
    for i, data in enumerate(data_list):
        if data.x.shape[1] != input_dim:
            raise ValueError(f"Input dimension mismatch in snapshot {i}. Expected {input_dim}, got {data.x.shape[1]}.")

    model = DGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(data_list):
            optimizer.zero_grad()
            out = model(data)
            
            # Check if 'y' is present and not None
            if hasattr(data, 'y') and data.y is not None:
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}/{epochs}, Snapshot {i+1}/{len(data_list)}, Loss: {loss.item()}')
            else:
                print(f"Snapshot {i+1} does not have valid 'y' attribute for labels. Skipping this snapshot.")

if __name__ == "__main__":
    root_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/organized_embeddings'  # Adjust this path as needed
    
    # Load snapshots and dynamically determine input dimension
    data_list = prepare_graph_snapshots(root_dir)
    input_dim = data_list[0].x.shape[1]  # Example dimension, adjust based on your data
    hidden_dim = 64
    output_dim = 10  # Example number of classes or output dimensions

    # Train the model
    train_dgcn(root_dir, hidden_dim, output_dim)
