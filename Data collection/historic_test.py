# This data is without industry segregation 
from datetime import datetime, timedelta
import pandas as pd
import os
import time

def construct_download_url(ticker, interval='daily'):
    def convert_to_seconds(date):
        datetime_value = datetime.strptime(date, '%Y-%m-%d')
        total_seconds = int(time.mktime(datetime_value.timetuple())) + 86400
        return total_seconds

    try:
        interval_reference = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}
        _interval = interval_reference.get(interval)
        if _interval is None:
            print('Interval code is incorrect')
            return None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        period1 = start_date.strftime('%Y-%m-%d')
        period2 = end_date.strftime('%Y-%m-%d')

        p1 = convert_to_seconds(period1)
        p2 = convert_to_seconds(period2)

        url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={p1}&period2={p2}&interval={_interval}&filter=history'
        return url
    except Exception as e:
        print(f"Error constructing URL for {ticker}: {e}")
        return None

def main():
    ticker_data = pd.read_csv('Complete-List-of-SP-500-Index-Constituents-Apr-3-2024_1.csv')  # Update this path
    tickers_and_sectors = ticker_data[['Ticker', 'Sector']]
    
    base_dir = 'historic_data'
    os.makedirs(base_dir, exist_ok=True)  # Ensure the base directory exists

    for index, row in tickers_and_sectors.iterrows():
        ticker = row['Ticker']
        sector = row['Sector']
        ticker_dir = os.path.join(base_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)  # Create a directory for each ticker

        query_url = construct_download_url(ticker, 'daily')
        if query_url is None:
            continue

        try:
            df = pd.read_csv(query_url)
            if df.empty:
                print(f"No data fetched for ticker: {ticker}")
                continue

            df['Ticker Name'] = ticker
            df['Sector'] = sector
            # Define the columns order and ensure it includes 'Date'
            columns_order = ['Date', 'Ticker Name', 'Sector', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = df[columns_order]

            # Save each ticker's data in its own directory
            ticker_csv_path = os.path.join(ticker_dir, f'{ticker}_historical_data.csv')
            df.to_csv(ticker_csv_path, index=False)
            print(f"Data for {ticker} saved to {ticker_csv_path}")

        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

if __name__ == "__main__":
    main()
