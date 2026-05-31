"""
India CPI Inflation Data
─────────────────────────
Provides historical India CPI inflation rates (annual %) from the World Bank API.

Data source: World Bank – India CPI (FP.CPI.TOTL.ZG)
  https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL.ZG?format=json

Falls back to a flat 6 % if the API is unreachable or the cache is stale.

Usage
-----
  from src.data.inflation import get_inflation_rate, get_monthly_inflation_factor

  rate_2023 = get_inflation_rate(2023)        # → 5.65  (% p.a.)
  monthly   = get_monthly_inflation_factor(2023)  # → (1 + 0.0565) ** (1/12)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_INFLATION_PCT = 6.0  # fallback if API unavailable (%)
_CACHE_TTL_DAYS = 30          # refresh cache every 30 days
_WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL.ZG"
    "?format=json&mrv=30&per_page=50"
)

# Resolved at import-time relative to project root (config/settings.py approach)
try:
    from config.settings import RAW_DATA_DIR
    _CACHE_PATH = Path(RAW_DATA_DIR) / "india_cpi.json"
except Exception:
    _CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "india_cpi.json"

# Hardcoded fallback data so the module always works offline.
# Source: World Bank / RBI annual reports (public domain).
_FALLBACK_CPI: dict[int, float] = {
    2007: 6.37,
    2008: 8.35,
    2009: 10.88,
    2010: 11.99,
    2011: 8.86,
    2012: 9.31,
    2013: 9.92,
    2014: 6.37,
    2015: 4.91,
    2016: 4.94,
    2017: 2.49,
    2018: 4.86,
    2019: 3.73,
    2020: 6.62,
    2021: 5.13,
    2022: 6.70,
    2023: 5.65,
    2024: 4.85,
    2025: 4.50,   # preliminary estimate
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_cache() -> Optional[dict[int, float]]:
    """Load CPI data from cache file if it exists and is fresh enough."""
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH, "r") as f:
            payload = json.load(f)
        # Check freshness
        cached_at = payload.get("cached_at", 0)
        if (time.time() - cached_at) > _CACHE_TTL_DAYS * 86_400:
            return None
        return {int(k): float(v) for k, v in payload.get("data", {}).items()}
    except Exception as e:
        logger.debug("CPI cache read failed: %s", e)
        return None


def _save_cache(data: dict[int, float]) -> None:
    """Persist CPI data to cache file."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump({"cached_at": time.time(), "data": {str(k): v for k, v in data.items()}}, f)
    except Exception as e:
        logger.debug("CPI cache write failed: %s", e)


def _fetch_from_worldbank() -> Optional[dict[int, float]]:
    """Fetch India CPI data from World Bank API. Returns None on failure."""
    try:
        import requests
        resp = requests.get(_WORLD_BANK_URL, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        # World Bank returns [metadata, data_list]
        data_list = payload[1] if len(payload) > 1 else []
        result: dict[int, float] = {}
        for entry in data_list:
            year = int(entry.get("date", 0))
            value = entry.get("value")
            if value is not None and year > 0:
                result[year] = round(float(value), 4)
        if result:
            logger.info("Fetched %d years of India CPI from World Bank.", len(result))
            return result
    except Exception as e:
        logger.warning("World Bank CPI fetch failed: %s. Using fallback data.", e)
    return None


# ── Module-level cache ─────────────────────────────────────────────────────────

_cpi_data: Optional[dict[int, float]] = None


def _get_cpi_data() -> dict[int, float]:
    """Return CPI data, loading from cache/API/fallback as needed (lazy)."""
    global _cpi_data
    if _cpi_data is not None:
        return _cpi_data

    # 1. Try disk cache
    cached = _load_cache()
    if cached:
        _cpi_data = cached
        return _cpi_data

    # 2. Try World Bank API
    fetched = _fetch_from_worldbank()
    if fetched:
        _cpi_data = {**_FALLBACK_CPI, **fetched}  # fetched values override fallback
        _save_cache(_cpi_data)
        return _cpi_data

    # 3. Use hardcoded fallback
    logger.info("Using hardcoded India CPI fallback data.")
    _cpi_data = dict(_FALLBACK_CPI)
    return _cpi_data


# ── Public API ────────────────────────────────────────────────────────────────

def get_inflation_rate(year: int) -> float:
    """
    Return India CPI inflation rate (% per annum) for the given year.

    Args:
        year: Calendar year (e.g., 2022).

    Returns:
        Annual inflation rate as a percentage (e.g., 6.70 means 6.70 % p.a.).
        Falls back to ``_DEFAULT_INFLATION_PCT`` (6.0 %) if year not in data.
    """
    data = _get_cpi_data()
    return data.get(year, _DEFAULT_INFLATION_PCT)


def get_monthly_inflation_factor(year: int) -> float:
    """
    Return the monthly compounding factor corresponding to the annual CPI for
    the given year.  Multiply a basket price by this each month to inflate it.

    Args:
        year: Calendar year.

    Returns:
        Monthly growth factor, e.g. 1.00487 for 6 % annual inflation.
    """
    annual_pct = get_inflation_rate(year)
    return (1.0 + annual_pct / 100.0) ** (1.0 / 12.0)


def get_default_inflation_pct() -> float:
    """Return the module-level default inflation rate (6 % p.a.)."""
    return _DEFAULT_INFLATION_PCT


def get_all_cpi_data() -> dict[int, float]:
    """Return the full year → CPI% mapping (for display / debugging)."""
    return dict(_get_cpi_data())


def refresh_cache() -> dict[int, float]:
    """
    Force-refresh CPI data from World Bank API and update cache.

    Returns:
        Updated CPI data dict.
    """
    global _cpi_data
    _cpi_data = None  # clear in-memory cache
    _CACHE_PATH.unlink(missing_ok=True)  # clear disk cache
    return _get_cpi_data()
