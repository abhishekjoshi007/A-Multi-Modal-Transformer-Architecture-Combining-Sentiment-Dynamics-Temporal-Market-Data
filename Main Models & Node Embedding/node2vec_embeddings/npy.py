import numpy as np
import os

# Path to the directory containing the integrated features
integrated_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/integrated_features'

# List all files in the directory
files = os.listdir(integrated_features_dir)

# Load and inspect the contents of each .npy file
for file_name in files:
    if file_name.endswith('.npy'):
        file_path = os.path.join(integrated_features_dir, file_name)
        # Load the .npy file
        features = np.load(file_path)
        print(f"Contents of {file_name}:")
        print(features)
        print(f"Shape of {file_name}: {features.shape}")
        print("-" * 50)
