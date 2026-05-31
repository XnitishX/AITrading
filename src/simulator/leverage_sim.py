"""
Leveraged Nifty Futures Simulator
──────────────────────────────────
Simulates holding 100% Nifty exposure via futures contracts while:
  • Capital sits in a liquid fund earning near-repo-rate interest
  • Monthly covered calls sold on the futures position at a user-chosen OTM%
  • Quarterly 90-DTE 20% OTM puts purchased as crash insurance (LEAPS proxy)
  • Futures carry cost modelled from RBI repo rate minus Nifty dividend yield
  • Black-Scholes option pricing estimated from VIX → implied-vol regime buckets

Simulation is in "capital units" (not lots) so lot-size changes do not matter.

Key design decisions
--------------------
- leverage_ratio = total_futures_notional / own_capital
  e.g. 30 L capital, 60 L notional Nifty → leverage_ratio = 2.0
- All capital acts as margin; it also earns liquid-fund rate less a spread
- Covered calls apply ONLY to the futures position → cash leg (if any) is uncapped
  (in this model the user has no separate cash Nifty leg — everything is futures)
- Puts protect the leveraged notional, not raw capital
- VIX data gap post-2016 falls back to "normal" IV regime (17 % implied vol)
"""

from __future__ import annotations

import math
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# RBI REPO RATE HISTORICAL TABLE
# Source: RBI policy announcements 2007-2024 (public domain data)
# Format: (date_from_inclusive, rate_percent)
# ──────────────────────────────────────────────────────────────────────────────

_RBI_REPO_RATES: list[tuple[str, float]] = [
    ("2007-06-01", 7.75),
    ("2008-06-12", 8.00),
    ("2008-07-05", 8.50),
    ("2008-07-30", 9.00),
    ("2008-10-21", 8.00),
    ("2008-11-04", 7.50),
    ("2008-12-08", 6.50),
    ("2009-01-05", 5.50),
    ("2009-03-05", 5.00),
    ("2009-04-21", 4.75),
    ("2010-03-19", 5.00),
    ("2010-04-20", 5.25),
    ("2010-07-02", 5.50),
    ("2010-09-16", 6.00),
    ("2011-05-03", 6.75),  # series of hikes through 2011
    ("2011-07-26", 7.25),
    ("2011-10-25", 7.50),  # held
    ("2012-04-17", 8.00),  # held through 2012
    ("2013-05-03", 7.25),
    ("2013-09-20", 7.50),
    ("2014-01-28", 8.00),
    ("2014-06-03", 8.00),
    ("2015-01-15", 7.75),
    ("2015-03-04", 7.50),
    ("2015-09-29", 6.75),
    ("2016-04-05", 6.50),
    ("2017-08-02", 6.00),
    ("2019-02-07", 6.25),
    ("2019-04-04", 6.00),
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2024-02-08", 6.50),  # held through 2024
    ("2025-02-07", 6.25),  # Feb 2025 cut
    ("2025-04-09", 6.00),  # April 2025 cut
]

# Parse once at module import
_RBI_REPO_PARSED: list[tuple[pd.Timestamp, float]] = [
    (pd.Timestamp(d), r) for d, r in _RBI_REPO_RATES
]


def get_repo_rate(query_date: pd.Timestamp | str | datetime | date) -> float:
    """Return RBI repo rate (%) in effect on query_date using a step function."""
    ts = pd.Timestamp(query_date)
    rate = _RBI_REPO_PARSED[0][1]  # default to earliest
    for dt, r in _RBI_REPO_PARSED:
        if ts >= dt:
            rate = r
        else:
            break
    return rate


def get_repo_rate_series(dates: pd.DatetimeIndex) -> pd.Series:
    """Vectorised version: return a Series of daily repo rates for a DatetimeIndex."""
    rates = np.zeros(len(dates))
    # Sort dates (assume already sorted in ascending order)
    sorted_changes = _RBI_REPO_PARSED  # already sorted

    current_rate = sorted_changes[0][1]
    change_idx = 0
    n_changes = len(sorted_changes)

    for i, ts in enumerate(dates):
        # Advance pointer while the next change is still <= current date
        while change_idx < n_changes and ts >= sorted_changes[change_idx][0]:
            current_rate = sorted_changes[change_idx][1]
            change_idx += 1
        rates[i] = current_rate

    return pd.Series(rates, index=dates, name="repo_rate_pct")


# ──────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European call price.

    Args:
        spot:   Current underlying price.
        strike: Option strike price.
        T:      Time to expiry in years.
        r:      Risk-free rate (annualised, as fraction e.g. 0.065).
        sigma:  Implied volatility (annualised, as fraction e.g. 0.17).

    Returns:
        Call option price in same units as spot.
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return spot * _norm_cdf(d1) - strike * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(spot: float, strike: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European put price (via put-call parity).

    Args:
        spot:   Current underlying price.
        strike: Option strike price.
        T:      Time to expiry in years.
        r:      Risk-free rate (annualised, as fraction e.g. 0.065).
        sigma:  Implied volatility (annualised, as fraction e.g. 0.17).

    Returns:
        Put option price in same units as spot.
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, strike - spot)
    call = bs_call_price(spot, strike, T, r, sigma)
    # Put-call parity: P = C - S + K * e^(-rT)
    return call - spot + strike * math.exp(-r * T)


# ──────────────────────────────────────────────────────────────────────────────
# VIX → IMPLIED VOL REGIME TABLE
# ──────────────────────────────────────────────────────────────────────────────

VIX_IV_REGIMES: list[tuple[float, float, str]] = [
    (0.0,  13.0, 0.12),   # Low:     VIX < 13   → IV = 12%
    (13.0, 20.0, 0.17),   # Normal:  13 ≤ VIX < 20 → IV = 17%
    (20.0, 30.0, 0.24),   # Elevated: 20 ≤ VIX < 30 → IV = 24%
    (30.0, 45.0, 0.35),   # High:    30 ≤ VIX < 45 → IV = 35%
    (45.0, 999., 0.55),   # Spike:   VIX ≥ 45   → IV = 55%
]

VIX_IV_REGIME_LABELS = [
    {"vix_low": r[0], "vix_high": r[1], "iv_pct": r[2] * 100, "label": lbl}
    for r, lbl in zip(
        VIX_IV_REGIMES,
        ["Low (<13)", "Normal (13-20)", "Elevated (20-30)", "High (30-45)", "Spike (≥45)"],
    )
]


def vix_to_iv(vix: float) -> float:
    """
    Map India VIX to annualised implied volatility used for option pricing.

    Returns:
        Implied vol as a fraction (e.g., 0.17 for 17%).
    """
    if np.isnan(vix) or vix <= 0:
        return 0.17  # fallback: normal regime
    for vix_low, vix_high, iv in VIX_IV_REGIMES:
        if vix_low <= vix < vix_high:
            return iv
    return 0.55  # above highest bucket


def vix_to_iv_call(vix: float) -> float:
    """IV for covered call pricing — call skew discount (~10% below ATM IV)."""
    return vix_to_iv(vix) * 0.90


def vix_to_iv_put(vix: float) -> float:
    """IV for protective put pricing — put skew premium (~15% above ATM IV)."""
    return vix_to_iv(vix) * 1.15


# ──────────────────────────────────────────────────────────────────────────────
# TRANSACTION COST MODEL
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TransactionCostModel:
    """
    India-specific NSE F&O transaction cost model.

    All rate fields are fractions (e.g. 0.0001 = 0.01%).
    Defaults reflect typical retail broker costs on NSE.
    """
    futures_stt_pct: float = 0.0001         # 0.01% STT on futures sell-side
    options_sell_stt_pct: float = 0.0005    # 0.05% STT on option premium (sell side)
    brokerage_per_order: float = 20.0       # Flat ₹20 brokerage per order
    exchange_charge_pct: float = 0.0002     # Exchange + SEBI charges (~0.02%)
    gst_on_brokerage: float = 0.18          # 18% GST on brokerage

    def brokerage_with_gst(self) -> float:
        """Return brokerage including GST."""
        return self.brokerage_per_order * (1.0 + self.gst_on_brokerage)


# ──────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LeverageResult:
    """Full simulation result for one (leverage_ratio, call_otm_pct) combination."""

    # Parameters
    leverage_ratio: float
    call_otm_pct: float
    put_otm_pct: float
    initial_capital: float
    start_date: str
    end_date: str

    # Terminal metrics
    final_capital: float = 0.0
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Cost/income breakdown (as % of initial capital)
    total_carry_cost_pct: float = 0.0
    total_put_cost_pct: float = 0.0
    total_call_income_pct: float = 0.0
    total_liquid_fund_income_pct: float = 0.0
    total_futures_pnl_pct: float = 0.0

    # Event counters
    put_payout_events: int = 0
    call_cap_events: int = 0
    vix_data_coverage_pct: float = 0.0  # % of days with real VIX data

    # Risk events and cost additions
    margin_call_triggered: bool = False
    margin_call_date: Optional[str] = None
    total_transaction_cost_pct: float = 0.0

    # Time-series (populated by run())
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SweepSummaryRow:
    """Summary row for the parameter sweep grid (no equity curve)."""
    leverage_ratio: float
    call_otm_pct: float
    final_capital: float
    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_carry_cost_pct: float
    total_put_cost_pct: float
    total_call_income_pct: float
    total_liquid_fund_income_pct: float
    put_payout_events: int
    call_cap_events: int
    margin_call_triggered: bool = False
    total_transaction_cost_pct: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# LEVERAGE SIMULATOR
# ──────────────────────────────────────────────────────────────────────────────

class LeverageSimulator:
    """
    Simulates leveraged Nifty futures strategy with hedges.

    Strategy:
      - Hold 100% Nifty via futures (leverage_ratio × capital notional)
      - Capital earns liquid fund rate (repo − liquid_fund_spread)
      - Pay futures carry cost: (repo_rate − dividend_yield) / 252 per day on notional
      - Every call_tenor_days: sell covered call at spot × (1 + call_otm_pct)
      - Every put_tenor_days: buy 90-DTE protective put at spot × (1 - put_otm_pct)

    Args:
        df:                   Master dataframe (must have 'Close', 'simple_return',
                              'vix_close' columns, DatetimeIndex or 'Date' column).
        leverage_ratio:       Notional / capital (e.g. 2.0 = 2× leverage).
        call_otm_pct:         Covered call strike offset, e.g. 0.05 = 5% OTM.
                              0.0 = no covered calls.
        put_otm_pct:          Put strike offset below spot, default 0.20 (20% OTM).
                              0.0 = no protective puts.
        call_tenor_days:      Trading days between each call roll (default 21 ≈ 1 month).
        put_tenor_days:       Trading days between each put roll (default 63 ≈ 3 months).
        liquid_fund_spread:   Spread subtracted from repo to get liquid fund yield.
        dividend_yield:       Nifty 50 annualised dividend yield (default 1.3%).
        initial_capital:      Starting capital in INR.
        start_date:           Inclusive start date (string or None for full history).
        end_date:             Inclusive end date (string or None for full history).
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        df: pd.DataFrame,
        leverage_ratio: float = 2.0,
        call_otm_pct: float = 0.05,
        put_otm_pct: float = 0.20,
        call_tenor_days: int = 21,
        put_tenor_days: int = 63,
        liquid_fund_spread: float = 0.005,
        dividend_yield: float = 0.013,
        initial_capital: float = 3_000_000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        # ── Realism improvements ───────────────────────────────────────────────
        use_vol_skew: bool = True,
        transaction_costs: Optional[TransactionCostModel] = None,
        strike_round_increment: float = 50.0,
        margin_call_threshold_pct: float = 0.20,
    ):
        self.leverage_ratio = leverage_ratio
        self.call_otm_pct = call_otm_pct
        self.put_otm_pct = put_otm_pct
        self.call_tenor_days = call_tenor_days
        self.put_tenor_days = put_tenor_days
        self.liquid_fund_spread = liquid_fund_spread
        self.dividend_yield = dividend_yield
        self.initial_capital = initial_capital
        self.use_vol_skew = use_vol_skew
        self.transaction_costs = transaction_costs
        self.strike_round_increment = strike_round_increment
        self.margin_call_threshold_pct = margin_call_threshold_pct

        # Prepare data slice
        self._df = self._prepare_df(df, start_date, end_date)
        self._start_date = str(self._df.index[0].date()) if len(self._df) > 0 else (start_date or "")
        self._end_date = str(self._df.index[-1].date()) if len(self._df) > 0 else (end_date or "")
    @staticmethod
    def _round_to_nse_strike(price: float, increment: float = 50.0) -> float:
        """Round price to nearest valid NSE Nifty option strike increment."""
        return round(price / increment) * increment
    # ── Data Prep ─────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_df(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
        """Slice the dataframe and ensure proper index."""
        d = df.copy()

        # Normalise index
        if "Date" in d.columns and not isinstance(d.index, pd.DatetimeIndex):
            d = d.set_index("Date")
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index)

        # Require 'Close' column
        if "Close" not in d.columns:
            raise ValueError("DataFrame must have a 'Close' column")

        # Compute simple_return if absent
        if "simple_return" not in d.columns:
            d["simple_return"] = d["Close"].pct_change()

        # Slice by date
        if start_date:
            d = d[d.index >= pd.Timestamp(start_date)]
        if end_date:
            d = d[d.index <= pd.Timestamp(end_date)]

        d = d.sort_index().dropna(subset=["Close"])
        return d

    # ── Main Simulation ───────────────────────────────────────────────────

    def run(self) -> LeverageResult:
        """Execute the full simulation and return a LeverageResult."""
        df = self._df
        n = len(df)
        if n < 2:
            raise ValueError("Not enough data rows to simulate (need ≥ 2).")

        dates = df.index
        closes = df["Close"].values.astype(float)
        returns = df["simple_return"].values.astype(float)

        # VIX — may be missing for many dates
        if "vix_close" in df.columns:
            vix_vals = df["vix_close"].values.astype(float)
        else:
            vix_vals = np.full(n, np.nan)

        vix_valid = np.sum(~np.isnan(vix_vals))
        vix_coverage_pct = 100.0 * vix_valid / n

        # Daily repo rate series
        repo_rates = get_repo_rate_series(dates).values / 100.0  # as fraction

        # ── Daily accumulators ────────────────────────────────────────────
        equity = np.zeros(n)
        drawdown = np.zeros(n)
        daily_strategy_return = np.zeros(n)

        cum_futures_pnl = 0.0
        cum_carry_cost = 0.0
        cum_put_cost = 0.0
        cum_call_income = 0.0
        cum_liquid_fund_income = 0.0
        cum_put_payout = 0.0
        cum_call_cap_loss = 0.0  # gain foregone due to call cap (negative for equity)

        put_payout_events = 0
        call_cap_events = 0

        capital = self.initial_capital  # tracks running equity

        # ── Option tracking ───────────────────────────────────────────────
        # Covered call state
        call_active = False
        call_strike = 0.0
        call_entry_spot = 0.0
        call_premium_received = 0.0
        call_days_remaining = 0
        bars_since_call = self.call_tenor_days  # open first call immediately at bar 0

        # Protective put state
        put_active = False
        put_strike = 0.0
        put_entry_spot = 0.0
        put_premium_paid = 0.0
        put_days_remaining = 0
        bars_since_put = self.put_tenor_days  # open first put at bar 0

        peak_equity = self.initial_capital

        # Monthly tracking for breakdown
        monthly_records: list[dict] = []
        month_start_equity = self.initial_capital
        month_start_idx = 0

        # Margin call tracking and transaction cost accumulator
        margin_call_triggered = False
        margin_call_date_str: Optional[str] = None
        cum_transaction_cost = 0.0
        tc = self.transaction_costs  # shorthand; None = costs disabled

        for i in range(n):
            spot = closes[i]
            repo = repo_rates[i]
            vix = vix_vals[i]
            iv = vix_to_iv(vix)
            # Skew-adjusted IVs: Nifty puts carry a ~15% premium, calls ~10% below ATM
            call_iv = vix_to_iv_call(vix) if self.use_vol_skew else iv
            put_iv = vix_to_iv_put(vix) if self.use_vol_skew else iv
            r_frac = repo  # risk-free rate as fraction

            # Notional exposure
            notional = capital * self.leverage_ratio

            # 1. Liquid fund income on capital (capital earns repo − spread daily)
            lf_rate_daily = (repo - self.liquid_fund_spread) / self.TRADING_DAYS_PER_YEAR
            lf_income_today = capital * lf_rate_daily
            cum_liquid_fund_income += lf_income_today

            # 2. Futures MTM P&L
            daily_ret = returns[i] if not np.isnan(returns[i]) else 0.0
            futures_pnl_today = notional * daily_ret
            cum_futures_pnl += futures_pnl_today

            # 3. Carry cost drag (repo − dividend_yield per day on notional)
            carry_rate_daily = (repo - self.dividend_yield) / self.TRADING_DAYS_PER_YEAR
            carry_today = notional * carry_rate_daily
            cum_carry_cost += carry_today

            # ── Covered Call Lifecycle ────────────────────────────────────
            call_settle_today = 0.0
            if self.call_otm_pct > 0:
                bars_since_call += 1
                if call_active:
                    call_days_remaining -= 1

                # Open new call at the start of each tenor period
                if bars_since_call >= self.call_tenor_days:
                    # If there was an active call, settle it first
                    if call_active:
                        if spot > call_strike:
                            # Call expired in-the-money: we owe the buyer the gain above strike
                            cap_loss = (spot - call_strike) / call_entry_spot * (capital * self.leverage_ratio)
                            cum_call_cap_loss += cap_loss
                            call_cap_events += 1
                            call_settle_today = -cap_loss  # reduces today's P&L
                        # else: call expired worthless, we keep the premium (already credited)
                        # Settlement brokerage (charged on expiry whether ITM or OTM)
                        if tc is not None:
                            cum_transaction_cost += tc.brokerage_with_gst()
                        call_active = False

                    # Futures roll cost (sell expiring + buy next monthly contract)
                    if tc is not None:
                        cum_transaction_cost += (
                            tc.futures_stt_pct * notional
                            + 2.0 * tc.brokerage_with_gst()
                            + 2.0 * tc.exchange_charge_pct * notional
                        )

                    # Open new covered call — rounded to valid NSE strike, skew-adjusted IV
                    call_strike = self._round_to_nse_strike(
                        spot * (1.0 + self.call_otm_pct), self.strike_round_increment
                    )
                    call_entry_spot = spot
                    T_call = self.call_tenor_days / self.TRADING_DAYS_PER_YEAR
                    premium = bs_call_price(spot, call_strike, T_call, r_frac, call_iv)
                    # Premium as fraction of notional
                    premium_income = (premium / spot) * notional
                    cum_call_income += premium_income
                    # Transaction costs: STT on premium (sell), brokerage to open, exchange charge
                    if tc is not None:
                        cum_transaction_cost += (
                            tc.options_sell_stt_pct * premium_income
                            + tc.brokerage_with_gst()
                            + tc.exchange_charge_pct * premium_income
                        )
                    call_premium_received = premium_income
                    call_days_remaining = self.call_tenor_days
                    call_active = True
                    bars_since_call = 0

            # ── Protective Put Lifecycle ──────────────────────────────────
            put_settle_today = 0.0
            if self.put_otm_pct > 0:
                bars_since_put += 1
                if put_active:
                    put_days_remaining -= 1

                # Open new put at the start of each tenor period
                if bars_since_put >= self.put_tenor_days:
                    # If there was an active put, settle it first
                    if put_active:
                        if spot < put_strike:
                            # Put expired in-the-money
                            payout = (put_strike - spot) / put_entry_spot * notional
                            cum_put_payout += payout
                            put_settle_today = payout
                            put_payout_events += 1
                        # else: put expired worthless
                        # Settlement brokerage
                        if tc is not None:
                            cum_transaction_cost += tc.brokerage_with_gst()
                        put_active = False

                    # Open new put — rounded to valid NSE strike, skew-adjusted IV
                    put_strike = self._round_to_nse_strike(
                        spot * (1.0 - self.put_otm_pct), self.strike_round_increment
                    )
                    put_entry_spot = spot
                    T_put = self.put_tenor_days / self.TRADING_DAYS_PER_YEAR
                    premium = bs_put_price(spot, put_strike, T_put, r_frac, put_iv)
                    put_cost = (premium / spot) * notional
                    cum_put_cost += put_cost
                    # Transaction costs: brokerage + exchange charge (no STT on buy side)
                    if tc is not None:
                        cum_transaction_cost += (
                            tc.brokerage_with_gst()
                            + tc.exchange_charge_pct * put_cost
                        )
                    put_premium_paid = put_cost
                    put_days_remaining = self.put_tenor_days
                    put_active = True
                    bars_since_put = 0

            # 4. Update running equity
            # equity = initial_capital
            #        + cum_futures_pnl  (leveraged Nifty MTM)
            #        + cum_liquid_fund_income  (cash earning interest)
            #        - cum_carry_cost   (futures roll drag)
            #        + cum_call_income  (call premiums received)
            #        - cum_put_cost     (put premiums paid)
            #        + cum_put_payout   (put payouts when in-the-money)
            #        - cum_call_cap_loss (gain foregone on capped calls)
            equity[i] = (
                self.initial_capital
                + cum_futures_pnl
                + cum_liquid_fund_income
                - cum_carry_cost
                + cum_call_income
                - cum_put_cost
                + cum_put_payout
                - cum_call_cap_loss
                - cum_transaction_cost        # NSE F&O transaction drag
            )
            capital = equity[i]  # update capital for next bar's sizing

            # Prevent negative capital (avoids division-by-zero in sizing)
            if capital <= 0:
                capital = 1.0  # floor; drawdown curve will show ruin

            # Margin call zone: flag first time equity drops below threshold
            if not margin_call_triggered and equity[i] < self.initial_capital * self.margin_call_threshold_pct:
                margin_call_triggered = True
                margin_call_date_str = str(dates[i].date())
                logger.warning(
                    "Margin call zone entered on %s: equity=%.0f (%.1f%% of initial capital)",
                    margin_call_date_str, equity[i], equity[i] / self.initial_capital * 100,
                )

            # Track drawdown
            if equity[i] > peak_equity:
                peak_equity = equity[i]
            drawdown[i] = (equity[i] - peak_equity) / peak_equity * 100.0  # negative pct

            # Daily strategy return for Sharpe/Sortino
            if i > 0 and equity[i - 1] > 0:
                daily_strategy_return[i] = (equity[i] - equity[i - 1]) / equity[i - 1]

            # Monthly breakdown record (end of each month)
            if i > 0 and (dates[i].month != dates[i - 1].month or i == n - 1):
                mo_equity_start = equity[month_start_idx]
                mo_equity_end = equity[i]
                monthly_records.append({
                    "year": dates[i].year,
                    "month": dates[i].month,
                    "month_label": dates[i].strftime("%Y-%m"),
                    "equity_start": mo_equity_start,
                    "equity_end": mo_equity_end,
                    "month_return_pct": (mo_equity_end / mo_equity_start - 1) * 100 if mo_equity_start > 0 else 0.0,
                })
                month_start_idx = i

        # ── Compute summary metrics ───────────────────────────────────────
        final_capital = float(equity[-1])
        total_return_pct = (final_capital / self.initial_capital - 1) * 100.0

        # Annualised return
        years = n / self.TRADING_DAYS_PER_YEAR
        if years > 0 and final_capital > 0:
            annual_return_pct = ((final_capital / self.initial_capital) ** (1.0 / years) - 1) * 100.0
        else:
            annual_return_pct = 0.0

        max_drawdown_pct = float(np.min(drawdown))  # most negative

        # Sharpe ratio (annualised)
        valid_returns = daily_strategy_return[1:]
        daily_rf = repo_rates[1:] / self.TRADING_DAYS_PER_YEAR
        excess = valid_returns - daily_rf
        if len(excess) > 1 and np.std(excess) > 0:
            sharpe_ratio = float(np.mean(excess) / np.std(excess) * math.sqrt(self.TRADING_DAYS_PER_YEAR))
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside std only)
        downside = excess[excess < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            sortino_ratio = float(np.mean(excess) / np.std(downside) * math.sqrt(self.TRADING_DAYS_PER_YEAR))
        else:
            sortino_ratio = 0.0

        # Calmar ratio
        if max_drawdown_pct < 0:
            calmar_ratio = float(annual_return_pct / abs(max_drawdown_pct))
        else:
            calmar_ratio = 0.0

        ic = self.initial_capital

        # ── Build equity curve DataFrame ──────────────────────────────────
        # Benchmark: simple buy-and-hold Nifty (1× notional, no costs)
        bh_equity = self.initial_capital * (1 + df["simple_return"].fillna(0)).cumprod().values

        equity_curve = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "equity": equity.tolist(),
            "drawdown": drawdown.tolist(),
            "benchmark_equity": bh_equity.tolist(),
        })

        monthly_df = pd.DataFrame(monthly_records)

        result = LeverageResult(
            leverage_ratio=self.leverage_ratio,
            call_otm_pct=self.call_otm_pct,
            put_otm_pct=self.put_otm_pct,
            initial_capital=self.initial_capital,
            start_date=self._start_date,
            end_date=self._end_date,
            final_capital=final_capital,
            total_return_pct=round(total_return_pct, 2),
            annual_return_pct=round(annual_return_pct, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            sharpe_ratio=round(sharpe_ratio, 3),
            sortino_ratio=round(sortino_ratio, 3),
            calmar_ratio=round(calmar_ratio, 3),
            total_carry_cost_pct=round(cum_carry_cost / ic * 100, 2),
            total_put_cost_pct=round(cum_put_cost / ic * 100, 2),
            total_call_income_pct=round(cum_call_income / ic * 100, 2),
            total_liquid_fund_income_pct=round(cum_liquid_fund_income / ic * 100, 2),
            total_futures_pnl_pct=round(cum_futures_pnl / ic * 100, 2),
            put_payout_events=put_payout_events,
            call_cap_events=call_cap_events,
            vix_data_coverage_pct=round(vix_coverage_pct, 1),
            margin_call_triggered=margin_call_triggered,
            margin_call_date=margin_call_date_str,
            total_transaction_cost_pct=round(cum_transaction_cost / ic * 100, 2),
            equity_curve=equity_curve,
            monthly_breakdown=monthly_df,
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# SWEEP FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_worker(args: tuple) -> Optional[SweepSummaryRow]:
    """
    Module-level worker for ProcessPoolExecutor (must be picklable).
    Reconstructs DataFrame from serialised dict/index to cross process boundary.
    """
    (
        df_dict, df_index,
        lev, cc, put_otm_pct,
        lf_spread, div_yield, capital, start, end,
        use_vol_skew, enable_tc, strike_round, margin_threshold,
    ) = args
    try:
        df = pd.DataFrame(df_dict, index=pd.to_datetime(df_index, format="%Y-%m-%d"))
        tc = TransactionCostModel() if enable_tc else None
        sim = LeverageSimulator(
            df=df,
            leverage_ratio=lev,
            call_otm_pct=cc,
            put_otm_pct=put_otm_pct,
            liquid_fund_spread=lf_spread,
            dividend_yield=div_yield,
            initial_capital=capital,
            start_date=start,
            end_date=end,
            use_vol_skew=use_vol_skew,
            transaction_costs=tc,
            strike_round_increment=strike_round,
            margin_call_threshold_pct=margin_threshold,
        )
        r = sim.run()
        return SweepSummaryRow(
            leverage_ratio=lev,
            call_otm_pct=cc,
            final_capital=r.final_capital,
            total_return_pct=r.total_return_pct,
            annual_return_pct=r.annual_return_pct,
            max_drawdown_pct=r.max_drawdown_pct,
            sharpe_ratio=r.sharpe_ratio,
            sortino_ratio=r.sortino_ratio,
            calmar_ratio=r.calmar_ratio,
            total_carry_cost_pct=r.total_carry_cost_pct,
            total_put_cost_pct=r.total_put_cost_pct,
            total_call_income_pct=r.total_call_income_pct,
            total_liquid_fund_income_pct=r.total_liquid_fund_income_pct,
            put_payout_events=r.put_payout_events,
            call_cap_events=r.call_cap_events,
            margin_call_triggered=r.margin_call_triggered,
            total_transaction_cost_pct=r.total_transaction_cost_pct,
        )
    except Exception as exc:
        logger.error("Sweep worker error (lev=%.1f, cc=%.3f): %s", lev, cc, exc)
        return None


def run_leverage_sweep(
    df: pd.DataFrame,
    leverage_ratios: list[float],
    call_otm_pcts: list[float],
    put_otm_pct: float = 0.20,
    initial_capital: float = 3_000_000.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    liquid_fund_spread: float = 0.005,
    dividend_yield: float = 0.013,
    use_vol_skew: bool = True,
    enable_transaction_costs: bool = True,
    strike_round_increment: float = 50.0,
    margin_call_threshold_pct: float = 0.20,
) -> list[SweepSummaryRow]:
    """
    Run all combinations of leverage_ratio × call_otm_pct.

    Uses ProcessPoolExecutor for parallel execution (~4× speedup on multi-core).
    Falls back to sequential execution if multiprocessing is unavailable.

    Returns a flat list of SweepSummaryRow (no equity curves — suitable for heatmap).
    """
    # Normalise the DataFrame index to DatetimeIndex before serialising.
    # (LeverageSimulator._prepare_df does this per-worker; do it once here.)
    _df = df.copy()
    if "Date" in _df.columns and not isinstance(_df.index, pd.DatetimeIndex):
        _df = _df.set_index("Date")
    if not isinstance(_df.index, pd.DatetimeIndex):
        _df.index = pd.to_datetime(_df.index)

    # Serialise once; workers deserialise independently.
    df_dict = _df.to_dict(orient="list")
    df_index = _df.index.strftime("%Y-%m-%d").tolist()

    all_args = [
        (
            df_dict, df_index,
            lev, cc, put_otm_pct,
            liquid_fund_spread, dividend_yield, initial_capital, start_date, end_date,
            use_vol_skew, enable_transaction_costs, strike_round_increment, margin_call_threshold_pct,
        )
        for lev in leverage_ratios
        for cc in call_otm_pcts
    ]

    total = len(all_args)
    results: list[SweepSummaryRow] = []

    try:
        max_workers = min(4, os.cpu_count() or 2, total)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {executor.submit(_sweep_worker, args): args for args in all_args}
            done = 0
            for future in as_completed(futures_map):
                row = future.result(timeout=120)
                if row is not None:
                    results.append(row)
                done += 1
                logger.info("Sweep progress: %d/%d", done, total)
    except Exception as exc:
        logger.warning("Parallel sweep unavailable (%s); running sequentially", exc)
        results = []
        for i, args in enumerate(all_args, 1):
            row = _sweep_worker(args)
            if row is not None:
                results.append(row)
            logger.info("Sweep progress (sequential): %d/%d", i, total)

    return results


def run_full_simulation(
    df: pd.DataFrame,
    leverage_ratio: float = 2.0,
    call_otm_pct: float = 0.05,
    put_otm_pct: float = 0.20,
    initial_capital: float = 3_000_000.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    liquid_fund_spread: float = 0.005,
    dividend_yield: float = 0.013,
    use_vol_skew: bool = True,
    enable_transaction_costs: bool = True,
    strike_round_increment: float = 50.0,
    margin_call_threshold_pct: float = 0.20,
) -> LeverageResult:
    """Convenience wrapper to run a single simulation and return LeverageResult."""
    tc = TransactionCostModel() if enable_transaction_costs else None
    sim = LeverageSimulator(
        df=df,
        leverage_ratio=leverage_ratio,
        call_otm_pct=call_otm_pct,
        put_otm_pct=put_otm_pct,
        liquid_fund_spread=liquid_fund_spread,
        dividend_yield=dividend_yield,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        use_vol_skew=use_vol_skew,
        transaction_costs=tc,
        strike_round_increment=strike_round_increment,
        margin_call_threshold_pct=margin_call_threshold_pct,
    )
    return sim.run()
