"""
Yahoo Finance Data Downloader
─────────────────────────────
Downloads Nifty 50 and India VIX historical data via the yfinance library.

Tickers:
  • ^NSEI      – Nifty 50 index (daily OHLC + Volume)
  • ^INDIAVIX  – India VIX index (daily OHLC)

Usage:
    from src.data.yfinance_downloader import download_all, sync_data
    download_all()   # full history download
    sync_data()      # incremental update from last available date
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# Yahoo Finance tickers for Indian indices
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

# CSV filenames (used by loader.py)
NIFTY_CSV = "nifty50.csv"
VIX_CSV = "indiavix.csv"


def _download_ticker(
    ticker: str,
    csv_path: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "max",
) -> pd.DataFrame:
    """
    Download data for a single ticker and save to CSV.

    If start is provided, downloads from that date; otherwise uses period='max'.
    Returns the downloaded DataFrame.
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

    # yfinance returns timezone-aware DatetimeIndex – strip tz for consistency
    df.index = df.index.tz_localize(None)
    df = df.reset_index()

    # Keep only the columns we need
    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols]

    # Ensure output directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s → %s (%d rows, %s to %s)",
                ticker, csv_path, len(df),
                df["Date"].iloc[0].date(), df["Date"].iloc[-1].date())
    return df


def download_nifty(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download Nifty 50 daily OHLC data."""
    csv_path = RAW_DATA_DIR / NIFTY_CSV
    return _download_ticker(NIFTY_TICKER, csv_path, start=start, end=end)


def download_vix(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download India VIX daily OHLC data."""
    csv_path = RAW_DATA_DIR / VIX_CSV
    return _download_ticker(VIX_TICKER, csv_path, start=start, end=end)


def download_all() -> dict[str, Path]:
    """
    Download full history for both Nifty 50 and India VIX.
    Returns dict mapping name → CSV path.
    """
    nifty_path = RAW_DATA_DIR / NIFTY_CSV
    vix_path = RAW_DATA_DIR / VIX_CSV

    download_nifty()
    download_vix()

    return {"nifty": nifty_path, "vix": vix_path}


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


def sync_data() -> dict:
    """
    Incremental sync: check the last date in each CSV, download only
    new data from yfinance, and append it.

    Returns a summary dict with status info.
    """
    result = {"nifty": {}, "vix": {}}

    for name, ticker, csv_file in [
        ("nifty", NIFTY_TICKER, NIFTY_CSV),
        ("vix", VIX_TICKER, VIX_CSV),
    ]:
        csv_path = RAW_DATA_DIR / csv_file
        last_date = _get_last_date(csv_path)

        if last_date is None:
            # No existing data — do a full download
            logger.info("No existing %s data found. Doing full download.", name)
            df_new = _download_ticker(ticker, csv_path)
            result[name] = {
                "status": "full_download",
                "rows_added": len(df_new),
                "last_date": str(df_new["Date"].iloc[-1].date()) if len(df_new) > 0 else None,
                "total_rows": len(df_new),
            }
            continue

        # Calculate the fetch start date (day after last available)
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")

        if fetch_start > today:
            logger.info("%s data is already up to date (last: %s).", name, last_date.date())
            existing = pd.read_csv(csv_path)
            result[name] = {
                "status": "up_to_date",
                "rows_added": 0,
                "last_date": str(last_date.date()),
                "total_rows": len(existing),
            }
            continue

        logger.info("Syncing %s from %s …", name, fetch_start)
        t = yf.Ticker(ticker)
        df_new = t.history(start=fetch_start, auto_adjust=True)

        if df_new.empty:
            logger.info("No new data available for %s since %s.", name, fetch_start)
            existing = pd.read_csv(csv_path)
            result[name] = {
                "status": "no_new_data",
                "rows_added": 0,
                "last_date": str(last_date.date()),
                "total_rows": len(existing),
            }
            continue

        # Clean new data
        df_new.index = df_new.index.tz_localize(None)
        df_new = df_new.reset_index()
        keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df_new.columns]
        df_new = df_new[keep_cols]

        # Load existing and append
        existing = pd.read_csv(csv_path, parse_dates=["Date"])
        combined = pd.concat([existing, df_new], ignore_index=True)
        # De-duplicate on date (keep last = freshest)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date").reset_index(drop=True)

        combined.to_csv(csv_path, index=False)
        rows_added = len(combined) - len(existing)
        logger.info("Synced %s: +%d rows (total: %d, last: %s)",
                     name, rows_added, len(combined),
                     combined["Date"].iloc[-1].date())

        result[name] = {
            "status": "synced",
            "rows_added": rows_added,
            "last_date": str(combined["Date"].iloc[-1].date()),
            "total_rows": len(combined),
        }

    return result


# ── CLI helper ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Downloading all data from Yahoo Finance …")
    paths = download_all()
    for name, p in paths.items():
        print(f"  {name:>8}: {p}")
    print("Done.")
