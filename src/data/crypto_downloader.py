"""
Crypto Data Downloader
──────────────────────
Downloads Crypto (BTC, etc.) historical data via the yfinance library.
Designed to support multiple crypto tickers easily by referencing config.settings.CRYPTO_TICKERS.

Usage:
    from src.data.crypto_downloader import download_all_crypto, sync_crypto
    download_all_crypto()   # full history download
    sync_crypto("bitcoin")  # incremental update for bitcoin
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import CRYPTO_RAW_DIR, CRYPTO_TICKERS

logger = logging.getLogger(__name__)


def _download_ticker(
    ticker: str,
    csv_path: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "max",
) -> pd.DataFrame:
    """
    Download data for a single ticker and save to CSV.
    """
    t = yf.Ticker(ticker)

    if start:
        logger.info("Downloading %s from %s to %s …", ticker, start, end or "today")
        df = t.history(start=start, end=end, auto_adjust=True)
    else:
        logger.info("Downloading %s full history (period=%s) …", ticker, period)
        df = t.history(period=period, auto_adjust=True)

    if df.empty:
        logger.warning("No data returned for %s", ticker)
        return df

    df.index = df.index.tz_localize(None)
    df = df.reset_index()

    # Keep only the columns we need
    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Ensure output directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s → %s (%d rows, %s to %s)",
                ticker, csv_path, len(df),
                df["Date"].iloc[0].date() if isinstance(df["Date"].iloc[0], datetime) or hasattr(df["Date"].iloc[0], "date") else df["Date"].iloc[0],
                df["Date"].iloc[-1].date() if isinstance(df["Date"].iloc[-1], datetime) or hasattr(df["Date"].iloc[-1], "date") else df["Date"].iloc[-1])
    return df


def download_crypto(crypto_id: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download daily OHLC data for a specific crypto ID."""
    if crypto_id not in CRYPTO_TICKERS:
        raise ValueError(f"Unknown crypto_id: {crypto_id}. Available: {list(CRYPTO_TICKERS.keys())}")
    
    ticker = CRYPTO_TICKERS[crypto_id]
    csv_path = CRYPTO_RAW_DIR / f"{crypto_id}.csv"
    return _download_ticker(ticker, csv_path, start=start, end=end)


def download_all_crypto() -> dict[str, Path]:
    """
    Download full history for all cryptos registered in config.
    Returns dict mapping name → CSV path.
    """
    results = {}
    for crypto_id in CRYPTO_TICKERS:
        results[crypto_id] = CRYPTO_RAW_DIR / f"{crypto_id}.csv"
        download_crypto(crypto_id)
    return results


def _get_last_date(csv_path: Path) -> Optional[pd.Timestamp]:
    """Read the last date from an existing CSV file."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if df.empty:
            return None
        return pd.Timestamp(df["Date"].max())
    except Exception as e:
        logger.warning("Could not read last date from %s: %s", csv_path, e)
        return None


def sync_crypto(crypto_id: str) -> dict:
    """
    Incremental sync: check the last date in the CSV of target crypto,
    download only new data from yfinance, and append.
    """
    if crypto_id not in CRYPTO_TICKERS:
        raise ValueError(f"Unknown crypto_id: {crypto_id}. Available: {list(CRYPTO_TICKERS.keys())}")

    ticker = CRYPTO_TICKERS[crypto_id]
    csv_path = CRYPTO_RAW_DIR / f"{crypto_id}.csv"
    last_date = _get_last_date(csv_path)

    if last_date is None:
        logger.info("No existing %s data found. Doing full download.", crypto_id)
        df_new = _download_ticker(ticker, csv_path)
        return {
            "status": "full_download",
            "rows_added": len(df_new),
            "last_date": str(df_new["Date"].iloc[-1].date()) if len(df_new) > 0 and hasattr(df_new["Date"].iloc[-1], "date") else str(df_new["Date"].iloc[-1]) if len(df_new) > 0 else None,
            "total_rows": len(df_new),
        }

    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    if fetch_start > today:
        logger.info("%s data is already up to date (last: %s).", crypto_id, last_date.date())
        existing = pd.read_csv(csv_path)
        return {
            "status": "up_to_date",
            "rows_added": 0,
            "last_date": str(last_date.date()),
            "total_rows": len(existing),
        }

    logger.info("Syncing %s from %s to %s …", crypto_id, fetch_start, today)
    t = yf.Ticker(ticker)
    df_new = t.history(start=fetch_start, end=today, auto_adjust=True)

    if df_new.empty:
        logger.info("%s is up to date (no new rows returned).", crypto_id)
        existing = pd.read_csv(csv_path)
        return {
            "status": "up_to_date",
            "rows_added": 0,
            "last_date": str(last_date.date()),
            "total_rows": len(existing),
        }

    df_new.index = df_new.index.tz_localize(None)
    df_new = df_new.reset_index()
    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df_new.columns]
    keep_cols = [c for c in keep_cols if c in df_new.columns]
    df_new = df_new[keep_cols]

    # Append to existing
    df_old = pd.read_csv(csv_path)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    df_combined.to_csv(csv_path, index=False)

    logger.info("Synced %s. Added %d rows. Total: %d rows.", crypto_id, len(df_combined) - len(df_old), len(df_combined))
    return {
        "status": "synced",
        "rows_added": len(df_combined) - len(df_old),
        "last_date": str(df_combined["Date"].iloc[-1]),
        "total_rows": len(df_combined),
    }


def sync_all_crypto() -> dict[str, dict]:
    """Sync all registered cryptocurrencies."""
    results = {}
    for crypto_id in CRYPTO_TICKERS:
        results[crypto_id] = sync_crypto(crypto_id)
    return results
