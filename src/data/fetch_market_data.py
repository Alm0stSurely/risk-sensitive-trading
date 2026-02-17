"""
Market data fetching module using yfinance.
Fetches historical and intraday data for ETFs and French stocks.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asset definitions
ETF_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "FEZ", "^FCHI"]

FRENCH_STOCK_TICKERS = [
    "MC.PA", "TTE.PA", "SAN.PA", "OR.PA", "AIR.PA", "SU.PA",
    "AI.PA", "BNP.PA", "CS.PA", "RMS.PA", "SAF.PA", "DSY.PA",
    "DG.PA", "SGO.PA", "KER.PA"
]

ALL_TICKERS = ETF_TICKERS + FRENCH_STOCK_TICKERS


def fetch_historical_data(
    tickers: Optional[List[str]] = None,
    period: str = "30d",
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical market data for specified tickers.
    
    Args:
        tickers: List of ticker symbols (default: ALL_TICKERS)
        period: Data period (default: "30d")
        interval: Data interval (default: "1d")
    
    Returns:
        Dict mapping ticker to DataFrame with OHLCV data
    """
    if tickers is None:
        tickers = ALL_TICKERS
    
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
                
            results[ticker] = hist
            logger.info(f"✓ {ticker}: {len(hist)} rows")
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            continue
    
    return results


def fetch_current_prices(
    tickers: Optional[List[str]] = None
) -> Dict[str, Optional[float]]:
    """
    Fetch current/latest prices for specified tickers.
    
    Args:
        tickers: List of ticker symbols (default: ALL_TICKERS)
    
    Returns:
        Dict mapping ticker to current price (or None if error)
    """
    if tickers is None:
        tickers = ALL_TICKERS
    
    results = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get the most recent data (1 day)
            hist = stock.history(period="1d", interval="1m")
            
            if hist.empty:
                # Try daily data if intraday fails
                hist = stock.history(period="5d")
            
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                results[ticker] = None
                continue
            
            current_price = hist['Close'].iloc[-1]
            results[ticker] = float(current_price)
            
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            results[ticker] = None
    
    return results


def fetch_ticker_info(ticker: str) -> Dict:
    """
    Fetch general information about a ticker.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        Dict with ticker info (name, sector, currency, etc.)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "N/A"),
            "market_cap": info.get("marketCap"),
            "country": info.get("country", "N/A"),
        }
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {e}")
        return {"name": ticker, "error": str(e)}


if __name__ == "__main__":
    # Quick test
    print("Testing fetch_market_data...")
    prices = fetch_current_prices(["SPY", "MC.PA"])
    print("Current prices:", prices)
    
    hist = fetch_historical_data(["SPY"], period="5d")
    print("\nHistorical SPY:")
    print(hist.get("SPY").tail())
