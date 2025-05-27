# Ensure stockdex is installed: pip install stockdex

from stockdex import Ticker
from datetime import datetime

# Initialize a Ticker object for Apple Inc.
ticker = Ticker('AAPL')

# Get price data and print
price_data = ticker.price(range='1y', dataGranularity='1d')
print("Price Data:\n", price_data)

# Get the current trading period and print
current_trading_period = ticker.current_trading_period
print("Current Trading Period:\n", current_trading_period)

# Fundamental data from Yahoo Finance API
income_statement = ticker.income_statement(frequency='quarterly')
print("Income Statement:\n", income_statement)

cash_flow = ticker.cash_flow(format='raw')
print("Cash Flow:\n", cash_flow)

balance_sheet = ticker.balance_sheet(period1=datetime(2022, 1, 1))
print("Balance Sheet:\n", balance_sheet)

financials = ticker.financials(period1=datetime(2022, 1, 1), period2=datetime.today())
print("Financials:\n", financials)

# Fundamental data through web scraping
summary = ticker.summary
print("Summary:\n", summary)

income_stmt = ticker.income_stmt
print("Income Statement (Web):\n", income_stmt)

balance_sheet_web = ticker.balance_sheet_web
print("Balance Sheet (Web):\n", balance_sheet_web)

cashflow_web = ticker.cashflow_web
print("Cash Flow (Web):\n", cashflow_web)

# Analysts and estimates
analysis = ticker.analysis
print("Analysis:\n", analysis)

# Data about options
calls = ticker.calls
puts = ticker.puts
print("Calls:\n", calls)
print("Puts:\n", puts)

# Profile data
key_executives = ticker.key_executives
print("Key Executives:\n", key_executives)

description = ticker.description
print("Description:\n", description)

corporate_governance = ticker.corporate_governance
print("Corporate Governance:\n", corporate_governance)

# Data about shareholders
major_holders = ticker.major_holders
print("Major Holders:\n", major_holders)

top_institutional_holders = ticker.top_institutional_holders
print("Top Institutional Holders:\n", top_institutional_holders)

top_mutual_fund_holders = ticker.top_mutual_fund_holders
print("Top Mutual Fund Holders:\n", top_mutual_fund_holders)

# Main statistics
# statistics = ticker.statistics
# print("Statistics:\n", statistics)

# # Historical dividends data
# dividends = ticker.dividends
# print("Dividends:\n", dividends)
