import json
import os
import pandas as pd
from datetime import datetime, timezone
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Function to convert Unix timestamp to date
def convert_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d')

# Function to get sentiment score from text
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 1  # Bullish
    elif analysis.sentiment.polarity < 0:
        return -1  # Bearish
    else:
        return 0  # Neutral

# Function to process each comment and its replies
def process_comments(data):
    sentiment_data = []

    for item in data:
        date = convert_timestamp(item['written_at'])
        
        # Check if sentiment label exists in additional_data
        if 'additional_data' in item and 'labels' in item['additional_data'] and 'ids' in item['additional_data']['labels']:
            labels = item['additional_data']['labels']['ids']
            if 'BULLISH' in labels:
                sentiment = 1
            elif 'BEARISH' in labels:
                sentiment = -1
            else:
                sentiment = 0
        else:
            # Perform sentiment analysis on the comment text
            if 'content' in item:
                text = " ".join([content['text'] for content in item['content'] if 'text' in content])
                sentiment = get_sentiment(text)
            else:
                sentiment = 0
        
        sentiment_data.append((date, sentiment))
        
        # Process replies
        for reply in item['replies']:
            reply_date = convert_timestamp(reply['written_at'])
            
            # Check if sentiment label exists in additional_data for reply
            if 'additional_data' in reply and 'labels' in reply['additional_data'] and 'ids' in reply['additional_data']['labels']:
                reply_labels = reply['additional_data']['labels']['ids']
                if 'BULLISH' in reply_labels:
                    reply_sentiment = 1
                elif 'BEARISH' in reply_labels:
                    reply_sentiment = -1
                else:
                    reply_sentiment = 0
            else:
                # Perform sentiment analysis on the reply text
                if 'content' in reply:
                    reply_text = " ".join([content['text'] for content in reply['content'] if 'text' in content])
                    reply_sentiment = get_sentiment(reply_text)
                else:
                    reply_sentiment = 0
            
            sentiment_data.append((reply_date, reply_sentiment))

    return sentiment_data

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 3]  # Close price is the target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Main function to process all stocks
def process_all_stocks(data_dir):
    combined_results = pd.DataFrame()

    for stock_dir in os.listdir(data_dir):
        stock_path = os.path.join(data_dir, stock_dir)
        if os.path.isdir(stock_path):
            try:
                # Read JSON comments file
                comments_file = os.path.join(stock_path, f"{stock_dir}_comments.json")
                if os.path.exists(comments_file):
                    with open(comments_file, 'r') as file:
                        comments_data = json.load(file)

                    # Process the comments and replies
                    sentiment_data = process_comments(comments_data)

                    # Create a DataFrame from the sentiment data
                    df_sentiment = pd.DataFrame(sentiment_data, columns=['date', 'sentiment'])

                    # Convert date column to datetime
                    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])

                    # Aggregate sentiment scores by date
                    sentiment_agg = df_sentiment.groupby('date')['sentiment'].sum().reset_index()

                    # Load historical price data
                    price_file = os.path.join(stock_path, f"{stock_dir}_historic_data.csv")
                    if os.path.exists(price_file):
                        price_data = pd.read_csv(price_file, parse_dates=['Date'])
                        price_data.rename(columns={'Date': 'date'}, inplace=True)

                        # Merge sentiment data with historical price data
                        combined_data = pd.merge(price_data, sentiment_agg, on='date', how='left')
                        combined_data['sentiment'] = combined_data['sentiment'].fillna(0)

                        # Calculate the historical score as the average of the past 5 days' returns
                        combined_data['return'] = combined_data['Close'].pct_change()
                        combined_data['historical_score'] = combined_data['return'].rolling(window=5).mean()
                        combined_data['historical_score'] = combined_data['historical_score'].fillna(0)

                        # Add stock ticker to the combined data
                        combined_data['Ticker'] = stock_dir

                        # Append to the combined results DataFrame
                        combined_results = pd.concat([combined_results, combined_data], ignore_index=True)
                    else:
                        print(f"Error: Historical data file not found for {stock_dir}")
                else:
                    print(f"Error: Comments file not found for {stock_dir}")

            except Exception as e:
                print(f"Error processing {stock_dir}: {e}")

    return combined_results

# Directory containing all stock data
data_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/historic_data'

# Process all stocks and get combined results
combined_results = process_all_stocks(data_dir)

# Save the combined results to a CSV file
combined_results.to_csv('combined_data_all_stocks.csv', index=False)

# Prepare input sequences for LSTM
features = combined_results[['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment', 'historical_score']].values
target = combined_results['Close'].values

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Prepare sequences
seq_length = 60
X, y = create_sequences(scaled_features, seq_length)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100)

# Predict on test data
predicted_prices = model.predict(X_test)

# Inverse transform the predicted prices
predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predicted_prices.shape[0], 6)), predicted_prices), axis=1))[:, 6]

# Inverse transform the actual prices
actual_prices = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 6)), y_test.reshape(-1, 1)), axis=1))[:, 6]

# Calculate MAE, MSE, RMSE, R2, MRR, and IRR
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, predicted_prices)
mrr = np.mean([1 / (rank + 1) for rank in np.argsort(np.argsort(predicted_prices))])
irr = np.sum((predicted_prices[1:] - predicted_prices[:-1]) / predicted_prices[:-1]) / np.sum((actual_prices[1:] - actual_prices[:-1]) / actual_prices[:-1])

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print(f"MRR: {mrr}")
print(f"IRR: {irr}")

# Save the metrics to the CSV file
metrics_data = pd.DataFrame({
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2],
    'MRR': [mrr],
    'IRR': [irr]
})

# Save the combined results and metrics to a CSV file
combined_results.to_csv('combined_LSTM_.csv', index=False)
metrics_data.to_csv('combined_LSTM_metrics_data.csv', index=False)

# Save the model in the recommended Keras format
model.save('lstm_model.keras')
