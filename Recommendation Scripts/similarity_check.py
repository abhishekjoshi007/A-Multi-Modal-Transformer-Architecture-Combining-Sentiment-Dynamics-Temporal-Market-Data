import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Define file paths
embedding_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output_pyg/node_embeddings_1.pt'
output_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output_pyg/similarities.csv'

# Check if embeddings already exist
if os.path.exists(embedding_file):
    print(f"✔ Embeddings already saved at {embedding_file}. Skipping to similarity computation...")
    embeddings = torch.load(embedding_file)
else:
    print(f"Embeddings file {embedding_file} not found. Please ensure the embeddings have been generated.")
    exit()

# Convert to numpy for compatibility with sklearn's cosine_similarity
embeddings_np = embeddings.cpu().numpy()

# Ensure the number of embeddings matches or exceeds the number of stocks
num_stocks = 718
if embeddings_np.shape[0] < num_stocks:
    print(f"Error: Number of embeddings ({embeddings_np.shape[0]}) is less than the expected number of stocks ({num_stocks}).")
    exit()
elif embeddings_np.shape[0] > num_stocks:
    print(f"Warning: Number of embeddings ({embeddings_np.shape[0]}) exceeds the expected number of stocks ({num_stocks}).")

# Reduce the chunk size to handle memory issues
chunk_size = 25  # Further reduced chunk size
num_chunks = num_stocks // chunk_size + (1 if num_stocks % chunk_size != 0 else 0)

# Open output file
with open(output_file, 'w') as f_out:
    f_out.write("Stock 1,Stock 2,Similarity Score\n")

    for i in tqdm(range(num_chunks), desc="Computing similarities"):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, num_stocks)

        for j in range(i, num_chunks):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, num_stocks)

            # Compute cosine similarity between chunks
            similarities_chunk = cosine_similarity(embeddings_np[start_i:end_i], embeddings_np[start_j:end_j])

            # Write top similarities to file
            for idx_i in range(similarities_chunk.shape[0]):
                for idx_j in range(similarities_chunk.shape[1]):
                    stock1_idx = start_i + idx_i
                    stock2_idx = start_j + idx_j
                    
                    # Ensure indices are within the expected range
                    if stock1_idx >= num_stocks or stock2_idx >= num_stocks:
                        continue  # Skip pairs with indices out of range

                    if i == j and idx_i >= idx_j:
                        continue  # Skip self-comparisons and duplicates

                    score = similarities_chunk[idx_i, idx_j]
                    f_out.write(f"{stock1_idx},{stock2_idx},{score}\n")

print(f"Similarities have been computed and saved to {output_file}")
