from stocknews import StockNews

# List of stock tickers you want to fetch news for
stocks = ['AAPL', 'MSFT', 'NFLX']

# Initialize the StockNews object with your stock list and World Trading Data API key
# Make sure to replace 'MY_WORLD_TRADING_DATA_KEY' with your actual World Trading Data API key
sn = StockNews(stocks, wt_key='MY_WORLD_TRADING_DATA_KEY')

# Summarize the news data and load it into a DataFrame
df = sn.summarize()

# Display the summarized data
print(df)