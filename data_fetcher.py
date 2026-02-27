"""
╔══════════════════════════════════════════════════════════════════╗
║  QUANT ENGINE — Data Fetcher                                    ║
║  Downloads & caches market data from Yahoo Finance (free)       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _cache_path(tickers: list, start: str, end: str) -> str:
    """Generate a unique cache file path for this data request."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{'_'.join(sorted(tickers))}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"data_{h}.parquet")


def fetch_prices(
    tickers: list,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    use_cache: bool = True,
) -> dict:
    """
    Download OHLCV data for a list of tickers.

    Returns:
        dict with keys:
            'close'  : pd.DataFrame (date × ticker)
            'open'   : pd.DataFrame
            'high'   : pd.DataFrame
            'low'    : pd.DataFrame
            'volume' : pd.DataFrame
            'returns': pd.DataFrame (daily log returns)
    """
    cache_file = _cache_path(tickers, start, end)

    if use_cache and os.path.exists(cache_file):
        print(f"  📦 Loading cached data from {cache_file}")
        close = pd.read_parquet(cache_file)
        # Reconstruct other fields from close only (cache is close-only for speed)
        returns = np.log(close / close.shift(1)).dropna()
        return {
            "close": close,
            "open": close,  # Simplified — full OHLCV below when fresh
            "high": close,
            "low": close,
            "volume": pd.DataFrame(np.nan, index=close.index, columns=close.columns),
            "returns": returns,
        }

    print(f"  📡 Downloading data for {len(tickers)} tickers: {start} → {end}")
    print(f"     Tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")

    # Download with yfinance
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance. Check tickers and dates.")

    # Handle single vs multi-ticker column structure
    if len(tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        opn = raw[["Open"]].rename(columns={"Open": tickers[0]})
        high = raw[["High"]].rename(columns={"High": tickers[0]})
        low = raw[["Low"]].rename(columns={"Low": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    else:
        close = raw["Close"]
        opn = raw["Open"]
        high = raw["High"]
        low = raw["Low"]
        volume = raw["Volume"]

    # Drop tickers with too much missing data (>20%)
    missing_pct = close.isnull().sum() / len(close)
    valid = missing_pct[missing_pct < 0.20].index.tolist()
    dropped = [t for t in tickers if t not in valid]
    if dropped:
        print(f"  ⚠️  Dropped tickers with >20% missing data: {dropped}")

    close = close[valid].ffill().bfill()
    opn = opn[valid].ffill().bfill()
    high = high[valid].ffill().bfill()
    low = low[valid].ffill().bfill()
    volume = volume[valid].ffill().bfill()

    # Cache close prices
    close.to_parquet(cache_file)
    print(f"  💾 Cached to {cache_file}")

    # Compute log returns
    returns = np.log(close / close.shift(1)).dropna()

    print(f"  ✅ Loaded {len(close)} days × {len(valid)} tickers")
    print(f"     Date range: {close.index[0].strftime('%Y-%m-%d')} → {close.index[-1].strftime('%Y-%m-%d')}")

    return {
        "close": close,
        "open": opn,
        "high": high,
        "low": low,
        "volume": volume,
        "returns": returns,
    }


def compute_indicators(data: dict) -> dict:
    """
    Pre-compute common technical indicators used across strategies.

    Adds to data dict:
        'sma_20', 'sma_50', 'sma_200': Simple moving averages
        'ema_12', 'ema_26': Exponential moving averages
        'rsi_14': Relative Strength Index
        'atr_14': Average True Range (approximated)
        'bbands_upper', 'bbands_lower': Bollinger Bands
        'volume_sma_20': Volume moving average
        'realized_vol_20': 20-day realized volatility
    """
    close = data["close"]
    high = data["high"]
    low = data["low"]

    # Moving averages
    data["sma_20"] = close.rolling(20).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["sma_200"] = close.rolling(200).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR approximation (using close as proxy when OHLC are same)
    tr = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        h = high[col]
        l = low[col]
        c_prev = close[col].shift(1)
        tr[col] = pd.concat([
            (h - l),
            (h - c_prev).abs(),
            (l - c_prev).abs(),
        ], axis=1).max(axis=1)
    data["atr_14"] = tr.rolling(14).mean()

    # Bollinger Bands
    std_20 = close.rolling(20).std()
    data["bbands_upper"] = data["sma_20"] + 2 * std_20
    data["bbands_lower"] = data["sma_20"] - 2 * std_20

    # Volume SMA
    data["volume_sma_20"] = data["volume"].rolling(20).mean()

    # Realized volatility (annualized)
    data["realized_vol_20"] = data["returns"].rolling(20).std() * np.sqrt(252)

    print(f"  📊 Computed technical indicators for {len(close.columns)} tickers")
    return data
