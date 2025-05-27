import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# File paths
embedding_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output_pyg/node_embeddings_1.pt'
similarities_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output_pyg/similarities.csv'
ticker_mapping_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/ticker_mapping.csv'  # Adjust the path as per your directory structure
output_dir = 'integrated_features'  # Adjust the path as per your directory structure
os.makedirs(output_dir, exist_ok=True)

# Parameters
top_n = 3  # Number of related corporations to consider
max_index = 716  # Maximum valid index for stocks

# Load ticker mapping
ticker_mapping = pd.read_csv(ticker_mapping_file)
ticker_mapping = ticker_mapping.set_index('Index').to_dict()['Ticker']

# Load embeddings
embeddings = torch.load(embedding_file)
embeddings_np = embeddings.cpu().numpy()

# Load similarity scores
similarities = pd.read_csv(similarities_file)

# Create a dictionary to store combined features
combined_features = {}

# Process each target company with progress bar
for _, row in tqdm(similarities.iterrows(), desc="Integrating features", total=len(similarities)):
    try:
        stock1_idx = int(row['Stock 1'])
        stock2_idx = int(row['Stock 2'])
        score = float(row['Similarity Score'])

        # Skip indices that are greater than 716
        if stock1_idx > max_index or stock2_idx > max_index:
            continue

        # Get ticker names
        ticker1 = ticker_mapping.get(stock1_idx)
        ticker2 = ticker_mapping.get(stock2_idx)

        # Ensure that indices are valid for the embeddings array
        if 0 <= stock1_idx < len(embeddings_np) and 0 <= stock2_idx < len(embeddings_np):
            # Initialize feature vectors if not already done
            if ticker1 not in combined_features:
                combined_features[ticker1] = np.zeros_like(embeddings_np[stock1_idx])

            # Aggregate top-N features
            if ticker1 in combined_features:
                combined_features[ticker1] += embeddings_np[stock2_idx]

    except (ValueError, IndexError) as e:
        print(f"Skipping invalid index: {row} due to error: {e}")
        continue

# Normalize the combined features (optional)
for ticker, feature_vector in combined_features.items():
    combined_features[ticker] = feature_vector / np.linalg.norm(feature_vector)

# Save the integrated features for each company
for ticker, feature_vector in combined_features.items():
    output_file = os.path.join(output_dir, f'{ticker}_integrated_features.npy')
    np.save(output_file, feature_vector)

print("Feature integration completed and saved in the output directory.")
