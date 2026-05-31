"""
Data Loader & Preprocessor
───────────────────────────
Reads raw CSV files (Nifty 50 index & India VIX), cleans them, merges
on date, and engineers features used by the simulator and predictor.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    NIFTY_CLOSE_COL,
    NIFTY_DATE_COL,
    NIFTY_HIGH_COL,
    NIFTY_LOW_COL,
    NIFTY_OPEN_COL,
    NIFTY_VOLUME_COL,
    NIFTY_CSV_FILENAME,
    VIX_CSV_FILENAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ROLLING_WINDOW_SIZES,
    VIX_CLOSE_COL,
    VIX_DATE_COL,
)

logger = logging.getLogger(__name__)


# ── Data Quality Validation ──────────────────────────────────────────────

def validate_data_quality(df: pd.DataFrame, price_col: str = "Close") -> dict:
    """
    Run data quality checks on a price DataFrame.

    Returns a dict with issues found and a boolean ``ok`` flag.
    Checks: missing values, duplicate dates, price outliers (>15% daily
    move), weekend/holiday gaps, stale prices, negative prices.
    """
    issues: list[str] = []
    warnings: list[str] = []
    rows = len(df)

    # 1. Missing / NaN values in key columns
    key_cols = [c for c in ["Open", "High", "Low", price_col, "Volume"] if c in df.columns]
    for col in key_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            pct = n_missing / rows * 100
            msg = f"{col}: {n_missing} missing values ({pct:.1f}%)"
            (issues if pct > 5 else warnings).append(msg)

    # 2. Duplicate dates
    if "Date" in df.columns:
        n_dup = int(df["Date"].duplicated().sum())
        if n_dup > 0:
            issues.append(f"{n_dup} duplicate dates found")

    # 3. Price outliers — daily moves > 15% (circuit limit for Nifty is ~10-20%)
    if price_col in df.columns:
        pct_change = df[price_col].pct_change().abs()
        outliers = pct_change[pct_change > 0.15].dropna()
        if len(outliers) > 0:
            dates = df.loc[outliers.index, "Date"].dt.strftime("%Y-%m-%d").tolist()[:5]
            warnings.append(
                f"{len(outliers)} daily moves > 15% detected (e.g. {', '.join(dates)})"
            )

    # 4. Negative or zero prices
    if price_col in df.columns:
        n_neg = int((df[price_col] <= 0).sum())
        if n_neg > 0:
            issues.append(f"{n_neg} negative/zero prices in {price_col}")

    # 5. Stale prices (same close for 5+ consecutive days)
    if price_col in df.columns:
        stale = (df[price_col].diff() == 0)
        stale_runs = stale.astype(int).groupby((~stale).cumsum()).sum()
        max_stale = int(stale_runs.max()) if len(stale_runs) > 0 else 0
        if max_stale >= 5:
            warnings.append(f"Price unchanged for {max_stale} consecutive days (stale data?)")

    # 6. Date gaps > 5 calendar days (holidays typically ≤ 4 days)
    if "Date" in df.columns:
        date_diff = df["Date"].diff().dt.days
        big_gaps = date_diff[date_diff > 5].dropna()
        if len(big_gaps) > 0:
            gap_dates = df.loc[big_gaps.index, "Date"].dt.strftime("%Y-%m-%d").tolist()[:5]
            warnings.append(
                f"{len(big_gaps)} date gaps > 5 calendar days (e.g. {', '.join(gap_dates)})"
            )

    # 7. Data freshness — warn if latest date is > 7 days old
    if "Date" in df.columns:
        latest = pd.Timestamp(df["Date"].iloc[-1])
        days_old = (pd.Timestamp.now() - latest).days
        if days_old > 7:
            warnings.append(f"Data is {days_old} days old (latest: {latest.date()})")

    ok = len(issues) == 0
    summary = {
        "ok": ok,
        "rows": rows,
        "issues": issues,
        "warnings": warnings,
        "date_range": (
            f"{df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}"
            if "Date" in df.columns and rows > 0 else "N/A"
        ),
    }

    if issues:
        logger.warning("Data quality issues: %s", issues)
    if warnings:
        logger.info("Data quality warnings: %s", warnings)

    return summary


# ── Helpers ──────────────────────────────────────────────────────────────

def _find_csv(directory: Path, pattern: str) -> Path:
    """Return the first CSV that matches *pattern* (case‑insensitive)."""
    for f in sorted(directory.rglob("*.csv")):
        if pattern.lower() in f.stem.lower():
            return f
    raise FileNotFoundError(
        f"No CSV matching '{pattern}' found under {directory}"
    )


def _standardise_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse the date column and sort chronologically."""
    df = df.copy()
    # Try ISO8601 first (most common), then mixed format as fallback
    try:
        df[col] = pd.to_datetime(df[col], format="ISO8601")
    except (ValueError, TypeError):
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True)
    df = df.sort_values(col).reset_index(drop=True)
    return df


# ── Loaders ──────────────────────────────────────────────────────────────

def load_nifty_raw(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw Nifty 50 index data from yfinance CSV.
    """
    if csv_path is None:
        csv_path = RAW_DATA_DIR / NIFTY_CSV_FILENAME
        if not csv_path.exists():
            # Fallback: try keyword search
            try:
                csv_path = _find_csv(RAW_DATA_DIR, "nifty")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not locate '{NIFTY_CSV_FILENAME}' under {RAW_DATA_DIR}. "
                    "Please download data first: python main.py download"
                )

    logger.info("Loading Nifty 50 index data from %s", csv_path)
    df = pd.read_csv(csv_path)
    # Coerce numeric OHLC columns
    for col in [NIFTY_OPEN_COL, NIFTY_HIGH_COL, NIFTY_LOW_COL, NIFTY_CLOSE_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if NIFTY_VOLUME_COL in df.columns:
        df[NIFTY_VOLUME_COL] = pd.to_numeric(df[NIFTY_VOLUME_COL], errors="coerce")
    df = _standardise_date(df, NIFTY_DATE_COL)
    logger.info("Nifty 50 index: %d rows, %s → %s", len(df), df[NIFTY_DATE_COL].iloc[0].date(), df[NIFTY_DATE_COL].iloc[-1].date())
    return df


def load_vix_raw(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw India VIX data from yfinance CSV."""
    if csv_path is None:
        csv_path = RAW_DATA_DIR / VIX_CSV_FILENAME
        if not csv_path.exists():
            try:
                csv_path = _find_csv(RAW_DATA_DIR, "vix")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not locate '{VIX_CSV_FILENAME}' under {RAW_DATA_DIR}. "
                    "Please download data first: python main.py download"
                )

    logger.info("Loading India VIX data from %s", csv_path)
    df = pd.read_csv(csv_path)
    # Coerce numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _standardise_date(df, VIX_DATE_COL)
    logger.info("India VIX: %d rows, %s → %s", len(df), df[VIX_DATE_COL].iloc[0].date(), df[VIX_DATE_COL].iloc[-1].date())
    return df


# ── Feature Engineering ──────────────────────────────────────────────────

def add_returns(df: pd.DataFrame, price_col: str = NIFTY_CLOSE_COL) -> pd.DataFrame:
    """Add daily log‑return and simple return columns."""
    df = df.copy()
    df["simple_return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_rolling_features(
    df: pd.DataFrame,
    price_col: str = NIFTY_CLOSE_COL,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean, std, and Bollinger‑style z‑score for each window.
    """
    df = df.copy()
    windows = windows or ROLLING_WINDOW_SIZES
    for w in windows:
        tag = f"_{w}d"
        df[f"sma{tag}"] = df[price_col].rolling(w).mean()
        df[f"std{tag}"] = df[price_col].rolling(w).std()
        df[f"zscore{tag}"] = (df[price_col] - df[f"sma{tag}"]) / df[f"std{tag}"]
        # Rolling return
        df[f"return{tag}"] = df[price_col].pct_change(w)
        # Rolling volatility (annualised)
        df[f"vol{tag}"] = df["log_return"].rolling(w).std() * np.sqrt(252)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, price_col: str = NIFTY_CLOSE_COL) -> pd.DataFrame:
    """Add the Relative Strength Index."""
    df = df.copy()
    delta = df[price_col].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df[f"rsi_{period}"] = 100 - 100 / (1 + rs)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = NIFTY_CLOSE_COL,
) -> pd.DataFrame:
    """Add MACD, signal line, and histogram."""
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Add the Average True Range."""
    df = df.copy()
    high = df[NIFTY_HIGH_COL]
    low = df[NIFTY_LOW_COL]
    close_prev = df[NIFTY_CLOSE_COL].shift(1)
    tr = pd.concat(
        [high - low, (high - close_prev).abs(), (low - close_prev).abs()],
        axis=1,
    ).max(axis=1)
    df[f"atr_{period}"] = tr.rolling(period).mean()
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = NIFTY_CLOSE_COL,
) -> pd.DataFrame:
    """
    Add Bollinger Bands columns (Investopedia: 20-day SMA ± 2 std devs).

    Source: John Bollinger, "Bollinger on Bollinger Bands" (2002).
    Standard setting: 20-period SMA, 2 standard deviations.
    """
    df = df.copy()
    df["bb_mid"] = df[price_col].rolling(window).mean()
    rolling_std = df[price_col].rolling(window).std()
    df["bb_upper"] = df["bb_mid"] + num_std * rolling_std
    df["bb_lower"] = df["bb_mid"] - num_std * rolling_std
    # Bandwidth: (upper - lower) / mid.  Measures squeeze vs expansion.
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    # %B: where price sits within the bands (0 = lower, 1 = upper)
    band_range = df["bb_upper"] - df["bb_lower"]
    df["bb_pctb"] = (df[price_col] - df["bb_lower"]) / band_range.replace(0, np.nan)
    return df


def add_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived VIX features if vix_close is present:
      - vix_sma_21 : 21-day SMA of VIX (smoothed trend)
      - vix_pctrank : 252-day rolling percentile rank (0-100)
    """
    df = df.copy()
    if "vix_close" not in df.columns:
        return df
    df["vix_sma_21"] = df["vix_close"].rolling(21).mean()
    df["vix_pctrank"] = (
        df["vix_close"]
        .rolling(252, min_periods=63)
        .apply(lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100, raw=False)
    )
    return df


# ── Merge & Build Master DataFrame ──────────────────────────────────────

def build_master_dataframe(
    nifty_path: Optional[Path] = None,
    vix_path: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Load Nifty + VIX, merge on date, add all features, and optionally
    persist the processed DataFrame to disk.
    """
    nifty = load_nifty_raw(nifty_path)
    nifty = add_returns(nifty)
    nifty = add_rolling_features(nifty)
    nifty = add_rsi(nifty)
    nifty = add_macd(nifty)
    nifty = add_bollinger_bands(nifty)  # 20-day, 2σ (Bollinger standard)

    if NIFTY_HIGH_COL in nifty.columns and NIFTY_LOW_COL in nifty.columns:
        nifty = add_atr(nifty)

    try:
        vix = load_vix_raw(vix_path)
        vix = vix.rename(columns={VIX_CLOSE_COL: "vix_close"})
        vix_cols = [VIX_DATE_COL, "vix_close"]
        if "Open" in vix.columns:
            vix = vix.rename(columns={"Open": "vix_open"})
            vix_cols.append("vix_open")
        if "High" in vix.columns:
            vix = vix.rename(columns={"High": "vix_high"})
            vix_cols.append("vix_high")
        if "Low" in vix.columns:
            vix = vix.rename(columns={"Low": "vix_low"})
            vix_cols.append("vix_low")

        vix = vix[vix_cols]
        master = pd.merge(nifty, vix, on="Date", how="left")
        logger.info("Merged VIX into master dataframe.")
        master = add_vix_features(master)  # VIX SMA, percentile rank
    except FileNotFoundError:
        logger.warning("VIX data not found – proceeding without it.")
        master = nifty

    master = master.dropna(subset=[NIFTY_CLOSE_COL]).reset_index(drop=True)

    # ── Event metadata tagging ───────────────────────────────────────────
    try:
        from src.data.events import tag_dataframe
        master = tag_dataframe(master, date_col="Date")
    except Exception as e:
        logger.warning("Could not tag events: %s", e)

    # ── Valuation data (PE/PB/DivYield) — optional, won't fail if absent ──
    try:
        from src.data.valuation_scraper import load_valuation_data
        val_df = load_valuation_data()
        if not val_df.empty:
            val_df = val_df.rename(columns={"PE": "nifty_pe", "PB": "nifty_pb", "DivYield": "nifty_divy"})
            val_df["Date"] = pd.to_datetime(val_df["Date"])
            master = pd.merge(master, val_df[["Date", "nifty_pe", "nifty_pb", "nifty_divy"]], on="Date", how="left")
            # Forward-fill so every trading day has a valuation figure
            master[["nifty_pe", "nifty_pb", "nifty_divy"]] = (
                master[["nifty_pe", "nifty_pb", "nifty_divy"]].ffill()
            )
            logger.info("Merged PE/PB valuation data (%d rows).", val_df.shape[0])
    except Exception as e:
        logger.debug("PE/PB data not merged (not yet downloaded?): %s", e)

    if save:
        out = PROCESSED_DATA_DIR / "master.parquet"
        master.to_parquet(out, index=False)
        logger.info("Saved master dataframe → %s (%d rows)", out, len(master))

    return master


# ── Quick load from cache ────────────────────────────────────────────────

def load_master(rebuild: bool = False, **kwargs) -> pd.DataFrame:
    """Load the processed master dataframe, rebuilding if needed."""
    cache = PROCESSED_DATA_DIR / "master.parquet"
    if cache.exists() and not rebuild:
        logger.info("Loading cached master dataframe from %s", cache)
        return pd.read_parquet(cache)
    return build_master_dataframe(**kwargs)
