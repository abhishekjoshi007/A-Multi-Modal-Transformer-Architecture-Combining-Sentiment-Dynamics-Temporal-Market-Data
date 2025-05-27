import numpy as np

# Path to labels.npy
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /LSTM(Without Sentiments)/labels.npy'

# Load and print labels
labels = np.load(labels_path, allow_pickle=True)
print(f"Number of labels: {labels.shape[0]}")
print(labels)
