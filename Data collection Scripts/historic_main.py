import pandas as pd
import os
import time
from datetime import datetime, timedelta

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
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # Changed to 180 days for 6 months
        period1 = start_date.strftime('%Y-%m-%d')
        period2 = end_date.strftime('%Y-%m-%d')

        p1 = convert_to_seconds(period1)
        p2 = convert_to_seconds(period2)

        url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={p1}&period2={p2}&interval={_interval}&filter=history'
        return url
    except Exception as e:
        print(e)
        return

def main():
    # Load the tickers, sectors, industry, and market cap from the CSV file
    csv_path = 'Technology data.csv'  # Update this path if necessary
    ticker_data = pd.read_csv(csv_path)
    tickers_and_details = ticker_data[['Ticker', 'Sector', 'Industry', 'Market Cap']]

    # Create the base directory for historical data
    base_dir = 'Technology_historic_data'
    os.makedirs(base_dir, exist_ok=True)

    for index, row in tickers_and_details.iterrows():
        ticker = row['Ticker']
        sector = row['Sector']
        industry = row['Industry']
        market_cap = row['Market Cap']

        # Ensure the ticker is a string
        if not isinstance(ticker, str):
            ticker = str(ticker)

        ticker_dir = os.path.join(base_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        try:
            # Retrieve the historical data for the current ticker
            query_url = construct_download_url(ticker, 'daily')
            df = pd.read_csv(query_url)
            df.set_index('Date', inplace=True)

            # Add Ticker Name, Sector, Industry, and Market Cap columns
            df['Ticker Name'] = ticker
            df['Sector'] = sector
            df['Industry'] = industry
            df['Market Cap'] = market_cap

            # Reorder columns to match the desired format
            df = df[['Ticker Name', 'Sector', 'Industry', 'Market Cap', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            df.reset_index(inplace=True)

            # Save the data as a CSV file
            csv_filename = os.path.join(ticker_dir, f'{ticker}_historic_data.csv')
            df.to_csv(csv_filename, index=False)

            print(f"Data for {ticker} (Sector: {sector}, Industry: {industry}, Market Cap: {market_cap}) has been saved to {ticker_dir}.")
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

if __name__ == "__main__":
    main()
