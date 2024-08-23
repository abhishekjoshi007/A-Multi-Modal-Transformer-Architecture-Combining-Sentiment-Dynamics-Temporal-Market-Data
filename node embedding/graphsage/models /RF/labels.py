import os
import pandas as pd
import numpy as np

# Define the main directory where your data is stored
main_directory = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/Technology_data'  # Replace with the actual path
output_label_file = 'labels.npy'

# Initialize a list to store labels
labels = []
processed_tickers = []

# Function to determine label based on price movement
def get_label(close_prices):
    return int(close_prices[-1] > close_prices[0])  # 1 if the last close price is greater than the first, else 0

# Process each stock's historical data
for stock_folder in os.listdir(main_directory):
    stock_path = os.path.join(main_directory, stock_folder)
    
    if os.path.isdir(stock_path):
        # Load historical data
        historical_data_file = os.path.join(stock_path, f"{stock_folder}_historic_data.csv")
        if os.path.exists(historical_data_file):
            print(f"Processing {historical_data_file}")  # Debug statement
            df = pd.read_csv(historical_data_file)

            # Ensure the DataFrame is sorted by date
            df = df.sort_values(by='Date')

            # Get closing prices
            close_prices = df['Close'].values
            print(f"Close prices: {close_prices}")  # Debug statement

            if len(close_prices) > 1:
                # Generate label based on price movement
                label = get_label(close_prices)
                print(f"Generated label: {label}")  # Debug statement
                labels.append(label)
                processed_tickers.append(stock_folder)
            else:
                print(f"Insufficient data in {historical_data_file}")  # Debug statement
        else:
            print(f"File not found: {historical_data_file}")  # Debug statement

# Convert labels to numpy array and save to .npy file
labels = np.array(labels)
np.save(output_label_file, labels)

print(f"Labels have been saved to {output_label_file}")
print(f"Total labels generated: {labels.shape[0]}")

# Save the tickers processed to ensure they match with feature files
processed_tickers_file = 'processed_tickers.npy'
np.save(processed_tickers_file, np.array(processed_tickers))

print(f"Processed tickers have been saved to {processed_tickers_file}")
