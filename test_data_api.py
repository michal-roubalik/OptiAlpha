import yfinance as yf
import pandas as pd

def audit_yfinance_history(tickers):
    print(f"--- Auditing yfinance History for {tickers} ---")
    
    # Fetching 'max' period to see the absolute limit
    data = yf.download(tickers, period="max", interval="1d", group_by='ticker')
    
    for ticker in tickers:
        # Adjusted Close is the 'Gold Standard' for backtesting
        series = data[ticker]['Adj Close'].dropna()
        
        if not series.empty:
            print(f"\nAsset: {ticker}")
            print(f"  Oldest: {series.index[0].date()}")
            print(f"  Newest: {series.index[-1].date()}")
            print(f"  Total Days: {len(series)}")
        else:
            print(f"\nAsset: {ticker} - No data found.")

universe = ["SPY", "QQQ", "XLE", "XLF", "BND", "GLD"]
audit_yfinance_history(universe)