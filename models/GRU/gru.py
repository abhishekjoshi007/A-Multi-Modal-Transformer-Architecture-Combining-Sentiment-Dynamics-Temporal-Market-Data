import os
import json
import pandas as pd
from datetime import datetime, timezone
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Function to convert Unix timestamp to date
def convert_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d')

# Function to create sequences for GRU
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 3]  # Close price is the target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Function to train GRU model
def train_gru_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(GRU(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

# Function to evaluate GRU model
def evaluate_model_gru(model, X_test, y_test, scaler):
    test_predict = model.predict(X_test)
    
    # Ensure the shape matches the scaler's expected input
    if test_predict.ndim == 1:
        test_predict = test_predict.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    
    # Apply inverse transformation
    test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], scaler.scale_.shape[0] - 1)))))
    y_test = scaler.inverse_transform(np.hstack((y_test, np.zeros((y_test.shape[0], scaler.scale_.shape[0] - 1)))))
    
    mae = mean_absolute_error(y_test[:, 0], test_predict[:, 0])
    rmse = np.sqrt(mean_squared_error(y_test[:, 0], test_predict[:, 0]))
    directional_accuracy = np.mean(np.sign(np.diff(y_test[:, 0])) == np.sign(np.diff(test_predict[:, 0])))
    return mae, rmse, directional_accuracy

# Function to process all stocks
def process_all_stocks(data_dir):
    results = []

    for stock_dir in os.listdir(data_dir):
        stock_path = os.path.join(data_dir, stock_dir)
        if os.path.isdir(stock_path):
            try:
                # Load historical price data
                price_data = pd.read_csv(os.path.join(stock_path, f"{stock_dir}_historic_data.csv"), parse_dates=['Date'])
                price_data.rename(columns={'Date': 'date'}, inplace=True)

                # Normalize features
                features = price_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_features = scaler.fit_transform(features)

                # Prepare sequences
                seq_length = 60
                X, y = create_sequences(scaled_features, seq_length)

                # Split the data into training and testing sets
                split = int(0.8 * len(X))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                # Train and evaluate GRU model
                print(f"Training and evaluating GRU model for {stock_dir}")
                model_gru = train_gru_model(X_train, y_train, X_test, y_test)
                gru_mae, gru_rmse, gru_dir_acc = evaluate_model_gru(model_gru, X_test, y_test, scaler)
                print(f"Evaluated GRU model for {stock_dir}")

                results.append({
                    'Stock': stock_dir,
                    'GRU_MAE': gru_mae,
                    'GRU_RMSE': gru_rmse,
                    'GRU_Dir_Acc': gru_dir_acc
                })

            except Exception as e:
                print(f"Error processing {stock_dir}: {e}")

    return results

# Directory containing all stock data
data_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/historic_data'

# Process all stocks and get combined results
results = process_all_stocks(data_dir)

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
# results_df.to_csv('gru_model_results.csv', index=False)

# Calculate average metrics
average_mae = results_df['GRU_MAE'].mean()
average_rmse = results_df['GRU_RMSE'].mean()
average_dir_acc = results_df['GRU_Dir_Acc'].mean()

# Add average metrics to the results DataFrame
average_metrics = pd.DataFrame([{
    'Stock': 'Average',
    'GRU_MAE': average_mae,
    'GRU_RMSE': average_rmse,
    'GRU_Dir_Acc': average_dir_acc
}])
results_df = pd.concat([results_df, average_metrics], ignore_index=True)

# Save the updated results to CSV
results_df.to_csv('gru_model_results_with_averages.csv', index=False)

print(results_df)
