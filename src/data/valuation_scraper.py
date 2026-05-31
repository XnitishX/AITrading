"""
Nifty Valuation Scraper
────────────────────────
Scrapes daily Nifty 50 PE Ratio, PB Ratio, and Dividend Yield from
nifty-pe-ratio.com (sourced from NSE). Persists data to a local CSV
and provides incremental sync functionality matching the yfinance
download / sync pattern used elsewhere in this project.

Data Source : https://nifty-pe-ratio.com/
Persistence : data/raw/nifty_pe_pb.csv
Columns     : Date, PE, PB, DivYield

Usage:
    from src.data.valuation_scraper import download_history, sync_data, get_valuation_context

    # First-time download (saves today's data to CSV)
    df = download_history()

    # Daily incremental sync (no-op if already up to date)
    df = sync_data()

    # Rich valuation insights for the Leverage page
    ctx = get_valuation_context()
"""

import logging
import re
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

SITE_URL = "https://nifty-pe-ratio.com/"
CSV_FILENAME = "nifty_pe_pb.csv"
CSV_PATH = settings.RAW_DATA_DIR / CSV_FILENAME
CSV_COLUMNS = ["Date", "PE", "PB", "DivYield"]

# ── Valuation zone thresholds (consolidated basis, post-April 2021) ─────────
# Based on NSE's 25+ year historical data analysis published on nifty-pe-ratio.com
# Standalone pre-2021 thresholds were ~20% higher; these reflect the
# consolidated earnings basis NSE switched to in April 2021.
PE_ZONE_THRESHOLDS = {
    "undervalued_max": 17.0,    # Below -1 SD: best historic buying opportunities
    "fair_max": 21.0,           # -1 SD to median+SD: normal range
    "overvalued_max": 25.0,     # Median to +1 SD: cautious territory
    # Above 25: extreme overvaluation / caution zone
}

PB_ZONE_THRESHOLDS = {
    "undervalued_max": 2.5,     # Below 2.5: book value supports market price
    "fair_max": 3.5,            # Normal range
    "overvalued_max": 4.5,      # Slightly stretched
    # Above 4.5: significantly overvalued on book basis
}

DIV_YIELD_ZONE = {
    "undervalued_min": 1.5,     # ≥1.5%: income signal of undervaluation
    "neutral_min": 1.0,         # 1.0–1.5%: neutral
    # <1.0%: overvalued (yield compressed by high prices)
}

# Approximate current 10-yr G-Sec risk-free rate (basis for ERP)
# Update this periodically or load from RBI data
RISK_FREE_RATE_PCT = 6.8

# Leverage cap recommendations by PE zone
LEVERAGE_BY_ZONE = {
    "Undervalued":         {"max_leverage": 3.0, "color": "green",  "emoji": "🟢"},
    "Fairly Valued":       {"max_leverage": 2.0, "color": "yellow", "emoji": "🟡"},
    "Slightly Overvalued": {"max_leverage": 1.5, "color": "orange", "emoji": "🟠"},
    "Overvalued":          {"max_leverage": 1.0, "color": "red",    "emoji": "🔴"},
}


# ── HTTP helpers ──────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Return a requests.Session with browser-like headers."""
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


# ── Scraping ──────────────────────────────────────────────────────────────

def scrape_current() -> dict:
    """
    Scrape today's Nifty PE, PB, and Dividend Yield from nifty-pe-ratio.com.

    The page displays a dashboard with three metric cards showing the latest
    values published by NSE. The HTML structure is:
        <p>PE (TTM)</p>  →  <p>20.87</p>
        <p>PB Ratio</p>  →  <p>3.28</p>
        <p>Div. Yield</p>→  <p>1.30%</p>

    Returns:
        dict with keys: Date (ISO str), PE (float), PB (float), DivYield (float).

    Raises:
        RuntimeError: If the page cannot be fetched or PE cannot be parsed.
    """
    s = _make_session()
    try:
        resp = s.get(SITE_URL, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch {SITE_URL}: {exc}") from exc

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # -- Parse "Updated: DD-Mon-YYYY" date --------------------------------
    date_match = re.search(r"Updated:\s*(\d{2}-\w{3}-\d{4})", html)
    if date_match:
        try:
            parsed_date = datetime.strptime(date_match.group(1), "%d-%b-%Y").date()
        except ValueError:
            parsed_date = date.today()
    else:
        parsed_date = date.today()

    # -- Parse metric values from the metrics card section ----------------
    def _value_after(label_text: str) -> Optional[float]:
        """
        Find a <p> whose FULL text exactly matches *label_text* (anchored,
        case-insensitive), then return the next sibling <p>'s numeric value.
        Using an anchored pattern avoids false matches on meta/title tags that
        contain the label as a substring.
        """
        pattern = re.compile(r"^\s*" + re.escape(label_text) + r"\s*$", re.IGNORECASE)
        label_tag = soup.find("p", string=pattern)
        if label_tag:
            value_tag = label_tag.find_next_sibling("p")
            if value_tag:
                raw = re.sub(r"[^\d.]", "", value_tag.get_text(strip=True))
                try:
                    return float(raw)
                except ValueError:
                    pass
        return None

    pe = _value_after("PE (TTM)")
    pb = _value_after("PB Ratio")
    dy = _value_after("Div. Yield")

    # -- Regex fallback on raw HTML: look for the value tag pattern -------
    # Pattern: label <p> immediately followed by value <p> in same div
    if pe is None:
        m = re.search(r"PE\s*\(TTM\)</p>\s*<p[^>]*>([0-9]+\.[0-9]+)", html, re.IGNORECASE)
        pe = float(m.group(1)) if m else None
    if pb is None:
        m = re.search(r"PB Ratio</p>\s*<p[^>]*>([0-9]+\.[0-9]+)", html, re.IGNORECASE)
        pb = float(m.group(1)) if m else None
    if dy is None:
        m = re.search(r"Div\.?\s*Yield</p>\s*<p[^>]*>([0-9]+\.[0-9]+)", html, re.IGNORECASE)
        dy = float(m.group(1)) if m else None

    if pe is None:
        raise RuntimeError(
            "Could not parse PE from nifty-pe-ratio.com. "
            "The site layout may have changed."
        )

    logger.info(
        "Scraped: Date=%s  PE=%.2f  PB=%.2f  DivYield=%.2f%%",
        parsed_date, pe, pb or 0.0, dy or 0.0,
    )
    return {
        "Date": parsed_date.isoformat(),
        "PE": round(pe, 2),
        "PB": round(pb, 2) if pb else None,
        "DivYield": round(dy, 2) if dy else None,
    }


# ── Persistence helpers ───────────────────────────────────────────────────

def _load_csv() -> pd.DataFrame:
    """Load the persisted CSV, or return an empty DataFrame with correct schema."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df[CSV_COLUMNS]
    return pd.DataFrame(columns=CSV_COLUMNS).astype(
        {"Date": "datetime64[ns]", "PE": "float64", "PB": "float64", "DivYield": "float64"}
    )


def _save_csv(df: pd.DataFrame) -> None:
    """Save DataFrame to the valuation CSV (creates parent dirs if needed)."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out[CSV_COLUMNS].to_csv(CSV_PATH, index=False)
    logger.info("Saved %d rows → %s", len(out), CSV_PATH)


def _upsert(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new_rows to existing, dedup by Date (keep latest), sort."""
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined[CSV_COLUMNS]


# ── Public API ────────────────────────────────────────────────────────────

def download_history() -> pd.DataFrame:
    """
    Initialise the valuation CSV with today's scraped data.

    Equivalent to yfinance_downloader.download_all() — call once to bootstrap.
    Future historical data accumulates through daily sync_data() calls.

    Returns:
        DataFrame with columns: Date, PE, PB, DivYield.
    """
    existing = _load_csv()
    current = scrape_current()
    new_row = pd.DataFrame([current])
    new_row["Date"] = pd.to_datetime(new_row["Date"])
    df = _upsert(existing, new_row)
    _save_csv(df)
    logger.info(
        "download_history complete: %d rows, %s → %s",
        len(df),
        df["Date"].iloc[0].date(),
        df["Date"].iloc[-1].date(),
    )
    return df


def sync_data() -> pd.DataFrame:
    """
    Incrementally sync PE/PB data (append today if not already present).

    Equivalent to yfinance_downloader.sync_data() — safe to call daily.

    Returns:
        Updated DataFrame.
    """
    existing = _load_csv()
    today = date.today()

    if not existing.empty:
        last_date = pd.Timestamp(existing["Date"].iloc[-1]).date()
        if last_date >= today:
            logger.info("PE/PB already up-to-date (last: %s).", last_date)
            return existing

    current = scrape_current()
    new_row = pd.DataFrame([current])
    new_row["Date"] = pd.to_datetime(new_row["Date"])
    df = _upsert(existing, new_row)
    _save_csv(df)
    logger.info("Synced PE/PB: %d rows, latest=%s", len(df), current["Date"])
    return df


def load_valuation_data() -> pd.DataFrame:
    """
    Load persisted valuation data (does not fetch from network).

    Returns:
        DataFrame or empty DataFrame if CSV not yet created.
    """
    return _load_csv()


# ── Valuation Context & Insights ─────────────────────────────────────────

def _pe_zone(pe: float) -> str:
    if pe < PE_ZONE_THRESHOLDS["undervalued_max"]:
        return "Undervalued"
    if pe < PE_ZONE_THRESHOLDS["fair_max"]:
        return "Fairly Valued"
    if pe < PE_ZONE_THRESHOLDS["overvalued_max"]:
        return "Slightly Overvalued"
    return "Overvalued"


def _pb_zone(pb: float) -> str:
    if pb < PB_ZONE_THRESHOLDS["undervalued_max"]:
        return "Undervalued"
    if pb < PB_ZONE_THRESHOLDS["fair_max"]:
        return "Fairly Valued"
    if pb < PB_ZONE_THRESHOLDS["overvalued_max"]:
        return "Slightly Overvalued"
    return "Overvalued"


def _dy_zone(dy: float) -> str:
    if dy >= DIV_YIELD_ZONE["undervalued_min"]:
        return "Undervalued"
    if dy >= DIV_YIELD_ZONE["neutral_min"]:
        return "Neutral"
    return "Overvalued"


def get_valuation_context(df: Optional[pd.DataFrame] = None) -> dict:
    """
    Build a rich valuation context dict for use on the Leverage page.

    Fetches current PE/PB from the website, computes zone classification,
    equity risk premium, PE percentile (when historical data exists), and
    recommended maximum leverage multiplier.

    Args:
        df: Pre-loaded valuation DataFrame. If None, loads from CSV and
            falls back to scraping if CSV is empty.

    Returns:
        Dictionary with the following keys:
            current        : {date, pe, pb, div_yield}
            zones          : {pe_zone, pe_zone_color, pe_zone_emoji,
                              pb_zone, dy_zone, overall_signal}
            equity_risk_premium : {earnings_yield_pct, risk_free_rate_pct,
                                   erp_pct, signal}
            pe_stats       : {percentile, mean, median, std, min, max}
                             (empty dict if <30 data points)
            leverage_rec   : {max_leverage, color, emoji, reason}
            pe_thresholds  : zone boundary reference values
            meta           : {data_rows, data_from, data_to}
    """
    if df is None:
        df = _load_csv()

    # Refresh current values
    try:
        current = scrape_current()
    except Exception as exc:
        logger.warning("Could not scrape current data (%s); using last known.", exc)
        if df.empty:
            raise RuntimeError("No valuation data available. Run: python main.py sync-valuation") from exc
        last = df.iloc[-1]
        current = {
            "Date": str(pd.Timestamp(last["Date"]).date()),
            "PE": float(last["PE"]),
            "PB": float(last.get("PB") or 0),
            "DivYield": float(last.get("DivYield") or 0),
        }

    pe = float(current["PE"])
    pb = float(current.get("PB") or 0)
    dy = float(current.get("DivYield") or 0)

    zone = _pe_zone(pe)
    lev_info = LEVERAGE_BY_ZONE[zone]

    # Equity Risk Premium
    earnings_yield = (1 / pe) * 100 if pe > 0 else 0
    erp = earnings_yield - RISK_FREE_RATE_PCT

    # Historical PE statistics (only if enough data)
    pe_stats: dict = {}
    if not df.empty and len(df) >= 30:
        series = df["PE"].dropna()
        pe_stats = {
            "percentile": round(float((series < pe).mean() * 100), 1),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        }

    # Overall signal: all three metrics pointing same direction?
    zone_signals = [_pe_zone(pe), _pb_zone(pb) if pb else None, _dy_zone(dy) if dy else None]
    unique = set(z for z in zone_signals if z)
    if unique == {"Undervalued"}:
        overall = "Strong Buy"
    elif unique <= {"Undervalued", "Fairly Valued"}:
        overall = "Moderate Buy"
    elif unique <= {"Fairly Valued"}:
        overall = "Neutral"
    elif unique <= {"Fairly Valued", "Slightly Overvalued"}:
        overall = "Cautious"
    else:
        overall = "Reduce Exposure"

    # Leverage reason string
    pe_desc = f"PE {pe:.2f} is in the '{zone}' zone"
    erp_desc = (
        f"Earnings yield {earnings_yield:.1f}% vs risk-free {RISK_FREE_RATE_PCT}% "
        f"→ ERP {erp:+.1f}%"
    )
    reason = f"{pe_desc}. {erp_desc}. {lev_info['emoji']} Suggested max leverage: {lev_info['max_leverage']}×."

    return {
        "current": {
            "date": current["Date"],
            "pe": pe,
            "pb": pb,
            "div_yield": dy,
        },
        "zones": {
            "pe_zone": zone,
            "pe_zone_color": lev_info["color"],
            "pe_zone_emoji": lev_info["emoji"],
            "pb_zone": _pb_zone(pb) if pb else "N/A",
            "dy_zone": _dy_zone(dy) if dy else "N/A",
            "overall_signal": overall,
        },
        "equity_risk_premium": {
            "earnings_yield_pct": round(earnings_yield, 2),
            "risk_free_rate_pct": RISK_FREE_RATE_PCT,
            "erp_pct": round(erp, 2),
            "signal": "positive" if erp > 2 else ("neutral" if erp > 0 else "negative"),
        },
        "pe_stats": pe_stats,
        "leverage_rec": {
            "max_leverage": lev_info["max_leverage"],
            "color": lev_info["color"],
            "emoji": lev_info["emoji"],
            "reason": reason,
        },
        "pe_thresholds": PE_ZONE_THRESHOLDS,
        "meta": {
            "data_rows": len(df),
            "data_from": str(df["Date"].iloc[0].date()) if not df.empty else None,
            "data_to": str(df["Date"].iloc[-1].date()) if not df.empty else None,
        },
    }
