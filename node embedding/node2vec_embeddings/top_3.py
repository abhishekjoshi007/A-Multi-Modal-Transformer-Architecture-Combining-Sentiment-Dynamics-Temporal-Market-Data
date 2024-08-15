import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Directories
embedding_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/organized_embeddings'  # Replace with your actual path
output_dir = 'top_similarities_output'  # Replace with your desired output path

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to compute top 3 similarities for a given embedding file
def compute_top_3_similarities(embedding_file):
    # Load embeddings
    embeddings = {}
    with open(embedding_file, 'r') as f:
        header = f.readline().strip()  # Read header
        for line in f:
            parts = line.strip().split()
            ticker = parts[0]
            embedding = np.array(parts[1:], dtype=float)
            embeddings[ticker] = embedding

    # Convert to numpy array for cosine similarity computation
    tickers = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[ticker] for ticker in tickers])

    if embedding_matrix.ndim != 2:
        print(f"Error: Expected a 2D array for embedding_matrix, but got {embedding_matrix.ndim}D array instead.")
        return None

    # Compute cosine similarity between all pairs
    similarities = cosine_similarity(embedding_matrix)

    # List to store the top similarities
    top_similarities = []

    # Identify top similarities across all pairs
    for i in range(similarities.shape[0]):
        for j in range(i + 1, similarities.shape[0]):  # Avoid self-comparison and duplicates
            score = similarities[i, j]
            top_similarities.append((tickers[i], tickers[j], score))

    # Sort by similarity score and keep only the top 3
    top_similarities = sorted(top_similarities, key=lambda x: x[2], reverse=True)[:3]

    return top_similarities

# Iterate over each date folder and compute top 3 similarities
for date_folder in tqdm(os.listdir(embedding_dir), desc="Processing dates"):
    date_path = os.path.join(embedding_dir, date_folder)
    if os.path.isdir(date_path):
        embedding_file_path = os.path.join(date_path, f"{date_folder}_embeddings.emb")
        output_file_path = os.path.join(output_dir, f"{date_folder}_top_3_similarities.csv")

        if not os.path.exists(embedding_file_path):
            print(f"Embedding file for {date_folder} not found. Skipping.")
            continue

        top_similarities = compute_top_3_similarities(embedding_file_path)

        if top_similarities:
            # Save top 3 similar stocks to a file
            with open(output_file_path, 'w') as f_out:
                f_out.write("Stock 1,Stock 2,Similarity Score\n")
                for stock1, stock2, score in top_similarities:
                    f_out.write(f"{stock1},{stock2},{score}\n")

            print(f"Top 3 similarities saved for date {date_folder}.")

print("Top 3 similarities computation completed for all dates.")
