import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product

def load_and_preprocess_data(ticker, folder_path):
    print(f"Processing {ticker}...")
    
    ticker_folder_path = os.path.join(folder_path, ticker)
    
    # Load historical data
    historical_data_path = os.path.join(ticker_folder_path, f"{ticker}_historic_data.csv")
    if not os.path.exists(historical_data_path):
        print(f"Historical data file not found for {ticker}")
        return None
    
    historical_data = pd.read_csv(historical_data_path)
    if historical_data.empty:
        print(f"Historical data is empty for {ticker}")
        return None
    
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data.set_index('Date', inplace=True)
    historical_data = historical_data.asfreq('D')  # Set the frequency to daily
    print(f"Loaded historical data for {ticker}")
    
    # Check for missing values and fill them
    if historical_data.isnull().values.any():
        print(f"Missing values detected in {ticker}. Handling missing values...")
        historical_data = historical_data.fillna(method='ffill').fillna(method='bfill')  # Forward fill, then backward fill
    
    # Drop non-numeric columns
    historical_data = historical_data.drop(columns=['Ticker Name', 'Sector'], errors='ignore')
    
    return historical_data

def train_arima_model(train_data, p, d, q):
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

def evaluate_model_arima(model, test_data):
    predictions = model.forecast(steps=len(test_data))
    mae = mean_absolute_error(test_data, predictions)
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_data, predictions)
    
    # Calculate MRR and IRR if needed
    mrr = calculate_mrr(test_data, predictions)
    irr = calculate_irr(test_data, predictions)
    
    return mae, mse, rmse, r2, mrr, irr

def calculate_mrr(true_values, predicted_values):
    return np.mean([1 / (rank + 1) for rank in np.argsort(np.argsort(predicted_values))])

def calculate_irr(true_values, predicted_values):
    true_returns = np.diff(true_values) / true_values[:-1]
    predicted_returns = np.diff(predicted_values) / predicted_values[:-1]
    irr = np.sum(predicted_returns) / np.sum(true_returns)
    return irr if np.sum(true_returns) != 0 else 0

def grid_search_arima(train_data, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p, d, q in product(p_values, d_values, q_values):
        try:
            model_fit = train_arima_model(train_data, p, d, q)
            predictions = model_fit.forecast(steps=len(train_data))
            mse = mean_squared_error(train_data, predictions)
            if mse < best_score:
                best_score, best_cfg = mse, (p, d, q)
            print(f'ARIMA{(p, d, q)} MSE={mse:.3f}')
        except:
            continue
    print(f'Best ARIMA{best_cfg} MSE={best_score:.3f}')
    return best_cfg

# Root folder containing all stock data
root_folder = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/historic_data'

# List all unique tickers based on the historical data files
tickers = [os.path.basename(folder) for folder in glob.glob(os.path.join(root_folder, '*')) if os.path.isdir(folder)]

results = []

p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

for ticker in tickers:
    try:
        # Load and preprocess data
        historical_data = load_and_preprocess_data(ticker, root_folder)
        if historical_data is None:
            continue
        
        # Use 'Close' price for ARIMA
        close_prices = historical_data['Close']
        
        # Split data into training and testing sets for ARIMA
        train_size = int(len(close_prices) * 0.8)
        train_data, test_data = close_prices[:train_size], close_prices[train_size:]
        
        # Find the best ARIMA parameters using grid search
        best_cfg = grid_search_arima(train_data, p_values, d_values, q_values)
        
        print(f"Training and evaluating ARIMA model for {ticker} with order {best_cfg}")
        
        # Train and evaluate ARIMA model with best parameters
        model_arima = train_arima_model(train_data, *best_cfg)
        arima_mae, arima_mse, arima_rmse, arima_r2, arima_mrr, arima_irr = evaluate_model_arima(model_arima, test_data)
        print(f"Evaluated ARIMA model for {ticker}")
        
        results.append({
            'Stock': ticker,
            'ARIMA_MAE': arima_mae,
            'ARIMA_MSE': arima_mse,
            'ARIMA_RMSE': arima_rmse,
            'ARIMA_R2': arima_r2,
            'ARIMA_MRR': arima_mrr,
            'ARIMA_IRR': arima_irr
        })
        print(f"Processed {ticker} successfully.")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('arima_model_results.csv', index=False)

print(results_df)

# Calculate cumulative results
cumulative_results = {
    'Average_ARIMA_MAE': results_df['ARIMA_MAE'].mean(),
    'Average_ARIMA_MSE': results_df['ARIMA_MSE'].mean(),
    'Average_ARIMA_RMSE': results_df['ARIMA_RMSE'].mean(),
    'Average_ARIMA_R2': results_df['ARIMA_R2'].mean(),
    'Average_ARIMA_MRR': results_df['ARIMA_MRR'].mean(),
    'Average_ARIMA_IRR': results_df['ARIMA_IRR'].mean()
}

print("\nCumulative Results:")
print(cumulative_results)

# Save cumulative results to CSV
cumulative_results_df = pd.DataFrame([cumulative_results])
cumulative_results_df.to_csv('arima_cumulative_model_results.csv', index=False)

print("\nCumulative Results:")
print(cumulative_results_df)
