import numpy as np

# Load the labels from the .npy file
labels = np.load('labels.npy')

# Print the labels to check their content
print("Labels:", labels)
print("Total number of labels:", labels.shape[0])
