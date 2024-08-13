import pandas as pd
from stockdex import Ticker
from datetime import datetime

# Initialize a Ticker object for Apple Inc.
ticker = Ticker('AAPL')

# Get price data and save to CSV
price_data = ticker.price(range='1y', dataGranularity='1d')
price_data_df = pd.DataFrame(price_data)
price_data_df.to_csv('price_data.csv', index=False)

# Get the current trading period and save to CSV
current_trading_period = ticker.current_trading_period
current_trading_period_df = pd.DataFrame([current_trading_period])
current_trading_period_df.to_csv('current_trading_period.csv', index=False)

# Fundamental data from Yahoo Finance API
income_statement = ticker.income_statement(frequency='quarterly')
income_statement_df = pd.DataFrame(income_statement)
income_statement_df.to_csv('income_statement.csv', index=False)

cash_flow = ticker.cash_flow(format='raw')
cash_flow_df = pd.DataFrame(cash_flow)
cash_flow_df.to_csv('cash_flow.csv', index=False)

balance_sheet = ticker.balance_sheet(period1=datetime(2022, 1, 1))
balance_sheet_df = pd.DataFrame(balance_sheet)
balance_sheet_df.to_csv('balance_sheet.csv', index=False)

financials = ticker.financials(period1=datetime(2022, 1, 1), period2=datetime.today())
financials_df = pd.DataFrame(financials)
financials_df.to_csv('financials.csv', index=False)

# Analysts and estimates
analysis = ticker.analysis
analysis_df = pd.DataFrame(analysis)
analysis_df.to_csv('analysis.csv', index=False)

# Options data
calls = ticker.calls
calls_df = pd.DataFrame(calls)
calls_df.to_csv('calls.csv', index=False)

puts = ticker.puts
puts_df = pd.DataFrame(puts)
puts_df.to_csv('puts.csv', index=False)

# Profile data
key_executives = ticker.key_executives
key_executives_df = pd.DataFrame(key_executives)
key_executives_df.to_csv('key_executives.csv', index=False)

# Shareholders data
major_holders = ticker.major_holders
major_holders_df = pd.DataFrame([major_holders])
major_holders_df.to_csv('major_holders.csv', index=False)

top_institutional_holders = ticker.top_institutional_holders
top_institutional_holders_df = pd.DataFrame(top_institutional_holders)
top_institutional_holders_df.to_csv('top_institutional_holders.csv', index=False)

top_mutual_fund_holders = ticker.top_mutual_fund_holders
top_mutual_fund_holders_df = pd.DataFrame(top_mutual_fund_holders)
top_mutual_fund_holders_df.to_csv('top_mutual_fund_holders.csv', index=False)
