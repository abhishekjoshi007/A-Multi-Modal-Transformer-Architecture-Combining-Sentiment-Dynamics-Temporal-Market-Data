import networkx as nx
from node2vec import Node2Vec
import os
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define directories
input_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/output_graphs_2'
output_dir = 'node2vec_embeddings'

os.makedirs(output_dir, exist_ok=True)

# Process each date's graph
for date_folder in tqdm(os.listdir(input_dir), desc="Processing graphs"):
    date_path = os.path.join(input_dir, date_folder)
    if os.path.isdir(date_path):
        gml_file_path = os.path.join(date_path, f"{date_folder}.gml")
        embedding_file_path = os.path.join(output_dir, f"{date_folder}_embeddings.emb")
        visual_file_path = os.path.join(output_dir, f"{date_folder}_embeddings_visualization.png")
        
        # Check if the embedding file exists
        if os.path.exists(embedding_file_path):
            print(f"Embedding file for {date_folder} already exists. Generating visualization.")
            
            # Load the graph from GML file
            G = nx.read_gml(gml_file_path)
            
            # Load embeddings from the existing file
            embeddings = {}
            with open(embedding_file_path, 'r') as f:
                for line in f.readlines()[1:]:  # Skip the header line
                    parts = line.strip().split()
                    node = parts[0]
                    embedding = np.array(parts[1:], dtype=float)
                    embeddings[node] = embedding
            
            # Prepare data for visualization
            nodes = list(embeddings.keys())
            embedding_values = np.array([embeddings[node] for node in nodes])
            
            # Apply t-SNE for 2D visualization
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embedding_values)

            # Plot the 2D embeddings
            plt.figure(figsize=(10, 10))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6)
            for i, label in enumerate(nodes):
                plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], label, fontsize=9)
            plt.title(f"Node2Vec Embeddings for {date_folder}")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid(True)
            plt.savefig(visual_file_path)
            plt.close()

            print(f"Visualization saved for date {date_folder}")

print("Visualization generation completed.")
