import os
import pandas as pd
import numpy as np

# Define the directory containing your CSV files and where to save .npy files
csv_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/Technology_data'
npy_save_dir = 'baseline_features'

# Ensure the save directory exists
os.makedirs(npy_save_dir, exist_ok=True)

# Loop over each directory in the csv_dir
for ticker_dir in os.listdir(csv_dir):
    ticker_path = os.path.join(csv_dir, ticker_dir)
    if os.path.isdir(ticker_path):
        # Load the CSV file into a pandas DataFrame
        csv_file_path = os.path.join(ticker_path, f"{ticker_dir}_historic_data.csv")
        df = pd.read_csv(csv_file_path)
        
        # Select the feature columns including Market Cap
        features = df[['Market Cap', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Adjusted to include Market Cap
        
        # Convert to numpy array
        features_np = features.to_numpy()
        
        # Save as .npy file
        npy_filename = os.path.join(npy_save_dir, f"{ticker_dir}_features.npy")
        np.save(npy_filename, features_np)

        print(f"Saved {npy_filename}")

print("Conversion completed.")
