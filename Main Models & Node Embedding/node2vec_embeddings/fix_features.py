import os
import numpy as np

# Define the path to the integrated features directory
integrated_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/integrated_features'

# Expected dimension of the feature vectors
expected_dim = 256

# Function to check and fix the dimensions of a feature vector
def check_and_fix_dimensions(file_path, expected_dim):
    features = np.load(file_path)
    current_dim = features.shape[0]
    
    if current_dim < expected_dim:
        # Pad with zeros
        padding = np.zeros(expected_dim - current_dim)
        fixed_features = np.concatenate((features, padding), axis=0)
        np.save(file_path, fixed_features)
        print(f"File {os.path.basename(file_path)} was padded from {current_dim} to {expected_dim} dimensions.")
    
    elif current_dim > expected_dim:
        # Truncate the extra dimensions
        fixed_features = features[:expected_dim]
        np.save(file_path, fixed_features)
        print(f"File {os.path.basename(file_path)} was truncated from {current_dim} to {expected_dim} dimensions.")
    
    else:
        print(f"File {os.path.basename(file_path)} already has the correct dimensions: {current_dim}.")

# Loop through all files in the directory and fix their dimensions if necessary
for file_name in os.listdir(integrated_features_dir):
    if file_name.endswith('_integrated_features.npy'):
        file_path = os.path.join(integrated_features_dir, file_name)
        check_and_fix_dimensions(file_path, expected_dim)

print("Dimension check and fix process completed.")
