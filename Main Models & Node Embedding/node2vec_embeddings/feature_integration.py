import os
import numpy as np
from tqdm import tqdm
import csv

# Paths
embeddings_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/organized_embeddings'  # Adjust this path to your directory
output_dir = 'integrated_features/'
ticker_mapping_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/ticker_mapping.csv'

# Parameters
top_n = 3

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load ticker mapping
ticker_mapping = {}
with open(ticker_mapping_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    for row in reader:
        ticker = row[0]
        idx = int(row[1])
        ticker_mapping[ticker] = idx

# Function to load Node2Vec embeddings from .emb file
def load_node2vec_embeddings(embedding_file):
    embeddings = {}
    with open(embedding_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip the header
            parts = line.strip().split()
            node = parts[0]
            embedding = np.array(parts[1:], dtype=float)
            embeddings[node] = embedding
    return embeddings

# Load all Node2Vec embeddings
all_embeddings = {}
for date_folder in tqdm(os.listdir(embeddings_dir), desc="Loading embeddings"):
    embedding_file_path = os.path.join(embeddings_dir, date_folder, f"{date_folder}_embeddings.emb")
    if os.path.exists(embedding_file_path):
        all_embeddings.update(load_node2vec_embeddings(embedding_file_path))

# Integrate features
for ticker, idx in tqdm(ticker_mapping.items(), desc="Integrating features"):
    if ticker in all_embeddings:
        related_embeddings = []

        # Get the closest embeddings based on some custom logic or assumption here,
        # As you don't have similarities.csv, you'll need to assume some closeness metric.
        # For simplicity, let's take the first 'top_n' embeddings as related, although this is arbitrary.
        sorted_tickers = sorted(all_embeddings.keys())[:top_n]

        for related_ticker in sorted_tickers:
            if related_ticker != ticker and related_ticker in all_embeddings:
                related_embeddings.append(all_embeddings[related_ticker])

        # Combine the target ticker's embeddings with those of the related tickers
        combined_feature = np.concatenate([all_embeddings[ticker]] + related_embeddings)

        # Save the combined features to an .npy file
        output_file = os.path.join(output_dir, f"{ticker}_integrated_features.npy")
        np.save(output_file, combined_feature)

print(f"Combined features have been saved to {output_dir}")
