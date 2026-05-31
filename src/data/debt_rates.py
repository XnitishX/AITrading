"""
India Debt Return Model
────────────────────────
Provides expected returns for three common Indian debt instruments, modelled
as a spread over the prevailing RBI repo rate.

Instrument spreads (approximate, post-expenses):
  • Liquid Fund      : repo + 0.30 %   (very liquid, overnight / T+1)
  • Short-term Debt  : repo + 1.50 %   (3-12 month accrual, some credit risk)
  • Bank FD          : repo + 0.50 %   (term-locked, no mark-to-market)

These spreads are conservative and in line with historical category averages
published by AMFI and RBI data (2010-2025).

Usage
-----
  from src.data.debt_rates import get_debt_return_pct, get_debt_return_series

  # Single lookup
  monthly_rate = get_debt_return_pct("2024-03-01", "liquid_fund") / 12

  # Series over a date range
  monthly_df = get_debt_return_series(dates_index, "short_term_debt")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Instrument type alias ─────────────────────────────────────────────────────

DebtInstrument = Literal["liquid_fund", "short_term_debt", "bank_fd"]

# ── Spreads over RBI repo rate (percentage points) ────────────────────────────

_INSTRUMENT_SPREADS: dict[str, float] = {
    "liquid_fund":    0.30,   # ~overnight/7-day, very safe, high liquidity
    "short_term_debt": 1.50,  # 3-12 month accrual funds, slightly higher duration
    "bank_fd":         0.50,  # term deposit, no NAV risk, locked
}

# Human-readable labels (for UI display)
INSTRUMENT_LABELS: dict[str, str] = {
    "liquid_fund":    "Liquid Fund",
    "short_term_debt": "Short-term Debt Fund",
    "bank_fd":         "Bank FD",
}

# ── Import repo rate lookup from leverage_sim (single source of truth) ────────
# We deliberately re-use the existing historical table rather than duplicating.

try:
    from src.simulator.leverage_sim import (
        get_repo_rate,
        get_repo_rate_series,
        _RBI_REPO_RATES,
        _RBI_REPO_PARSED,
    )
    _HAVE_LEVERAGE_SIM = True
except Exception as e:  # pragma: no cover
    logger.warning("Could not import from leverage_sim; using minimal fallback: %s", e)
    _HAVE_LEVERAGE_SIM = False

    def get_repo_rate(query_date) -> float:  # type: ignore[misc]
        return 6.50  # safe fallback

    def get_repo_rate_series(dates) -> pd.Series:  # type: ignore[misc]
        return pd.Series([6.50] * len(dates), index=dates)

    _RBI_REPO_RATES = [("2007-06-01", 6.50)]
    _RBI_REPO_PARSED = [(pd.Timestamp("2007-06-01"), 6.50)]


# ── Public API ────────────────────────────────────────────────────────────────

def get_debt_return_pct(
    query_date: "pd.Timestamp | str | datetime | date",
    instrument: DebtInstrument = "liquid_fund",
) -> float:
    """
    Return the expected annualised pre-tax return (% p.a.) for a debt instrument
    on the given date.

    Args:
        query_date: Date to look up (any parseable format).
        instrument:  One of ``"liquid_fund"``, ``"short_term_debt"``, ``"bank_fd"``.

    Returns:
        Annual pre-tax yield as a percentage, e.g. 7.30.
    """
    spread = _INSTRUMENT_SPREADS.get(instrument, 0.30)
    repo = get_repo_rate(query_date)
    return round(repo + spread, 4)


def get_debt_monthly_factor(
    query_date: "pd.Timestamp | str | datetime | date",
    instrument: DebtInstrument = "liquid_fund",
    tax_rate_pct: float = 30.0,
) -> float:
    """
    Return the post-tax monthly compounding factor for the debt instrument.

    Debt interest is taxed as ordinary income in India (post-2023 rules).

    Args:
        query_date:   Date for repo rate lookup.
        instrument:   Debt instrument type.
        tax_rate_pct: Income tax slab applied to interest (%, default 30 %).

    Returns:
        Monthly factor, e.g. 1.00389 for ~4.7 % post-tax p.a.
    """
    gross_annual_pct = get_debt_return_pct(query_date, instrument)
    net_annual_pct = gross_annual_pct * (1.0 - tax_rate_pct / 100.0)
    return (1.0 + net_annual_pct / 100.0) ** (1.0 / 12.0)


def get_debt_return_series(
    dates: pd.DatetimeIndex,
    instrument: DebtInstrument = "liquid_fund",
) -> pd.Series:
    """
    Return a Series of annualised gross debt returns (%) indexed by date.

    Args:
        dates:      DatetimeIndex of dates to compute returns for.
        instrument: Debt instrument type.

    Returns:
        pd.Series of annual return percentages aligned to *dates*.
    """
    spread = _INSTRUMENT_SPREADS.get(instrument, 0.30)
    repo_series = get_repo_rate_series(dates)
    return (repo_series + spread).rename(f"{instrument}_return_pct")


def get_historical_rates_df(instrument: DebtInstrument = "liquid_fund") -> pd.DataFrame:
    """
    Return a DataFrame with columns [date, repo_rate_pct, instrument_return_pct]
    for every RBI rate-change event.  Useful for charting in the UI.

    Args:
        instrument: Debt instrument type to add spread for.

    Returns:
        DataFrame with rate-change history.
    """
    spread = _INSTRUMENT_SPREADS.get(instrument, 0.30)
    rows = []
    for dt, repo in _RBI_REPO_PARSED:
        rows.append({
            "date": dt,
            "repo_rate_pct": repo,
            "instrument_return_pct": round(repo + spread, 4),
            "instrument": INSTRUMENT_LABELS.get(instrument, instrument),
        })
    return pd.DataFrame(rows)


def list_instruments() -> list[dict]:
    """Return metadata for all supported instruments (for UI dropdowns)."""
    return [
        {
            "id": k,
            "label": INSTRUMENT_LABELS[k],
            "spread_pct": _INSTRUMENT_SPREADS[k],
            "description": _get_instrument_description(k),
        }
        for k in _INSTRUMENT_SPREADS
    ]


def _get_instrument_description(instrument: str) -> str:
    descriptions = {
        "liquid_fund": (
            "Invests in overnight/7-day money market instruments. "
            "T+1 liquidity, no exit load. Closest to repo rate with minimal spread."
        ),
        "short_term_debt": (
            "3-12 month accrual bond fund. Higher yield with some interest-rate and "
            "credit risk. Exit load may apply."
        ),
        "bank_fd": (
            "Term fixed deposit with scheduled commercial bank. "
            "Capital guaranteed (up to DICGC limits), but locked for the term."
        ),
    }
    return descriptions.get(instrument, "")
