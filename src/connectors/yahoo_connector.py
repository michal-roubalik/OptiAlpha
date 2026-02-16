import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class YahooConnector:
    """
    Retrieves market data using yfinance. 
    Provides adjusted price history and ETF sector compositions.
    """
    
    def get_historical_returns(self, symbol: str, period: str = "5y") -> pd.Series:
        """
        Fetches historical adjusted close prices and calculates daily returns.
        'Auto_adjust=True' ensures dividends and splits are handled.
        """
        try:
            ticker = yf.Ticker(symbol)
            # Fetch history with auto_adjust to handle splits/dividends
            df = ticker.history(period=period, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"No price data found for {symbol}")
                return pd.Series()
            
            # Calculate simple daily returns
            returns = df['Close'].pct_change().dropna()
            returns.name = symbol
            return returns
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.Series()

    def get_etf_sector_weights(self, symbol: str) -> pd.DataFrame:
        """
        Fetches sector weightings for an ETF. 
        Useful for analyzing concentration risk without needing full holdings.
        """
        try:
            ticker = yf.Ticker(symbol)
            # Access the 'funds_data' object for sector info
            if hasattr(ticker, 'funds_data') and ticker.funds_data.sector_weightings:
                sectors = ticker.funds_data.sector_weightings
                df = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Weight'])
                df['Ticker'] = symbol
                return df
            else:
                logger.warning(f"Sector data unavailable for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching sector weights for {symbol}: {e}")
            return pd.DataFrame()