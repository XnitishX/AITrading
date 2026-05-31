"""
Crypto Leverage Monte Carlo Simulator
──────────────────────────────────────
Simulates leveraged cryptocurrency trading using a block-bootstrap Monte Carlo engine.
Analyzes long-term survival probability, liquidation risk, and median returns.

Architecture is coin-agnostic, supporting any coin listed in CRYPTO_TICKERS.
"""

from __future__ import annotations

import math
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from config.settings import CRYPTO_RAW_DIR, CRYPTO_TICKERS

logger = logging.getLogger(__name__)

# ── Input / Output Models ─────────────────────────────────────────────────────

class CryptoLeverageParams(BaseModel):
    """Configuration parameters for the Crypto Leverage simulator."""
    crypto_id: str = Field("bitcoin", description="Identifier of the cryptocurrency (e.g. bitcoin)")
    leverage_ratio: float = Field(2.0, ge=1.0, le=10.0, description="Notional exposure relative to own capital")
    borrow_rate_pct: float = Field(10.0, ge=0.0, le=100.0, description="Annual borrow interest rate in percent")
    initial_capital: float = Field(1_000_000.0, gt=0, description="Starting capital")
    horizon_years: int = Field(10, ge=1, le=50, description="Number of years to simulate")
    n_simulations: int = Field(1000, ge=10, le=10000, description="Number of Monte Carlo paths")
    block_size_months: int = Field(6, ge=1, le=60, description="Block size in months for block-bootstrap")
    liquidation_threshold_pct: float = Field(0.0, ge=0.0, le=100.0, description="Liquidation trigger threshold (% of NAV; 0.0 means NAV <= 0)")


class CryptoMCResult(BaseModel):
    crypto_id: str
    leverage_ratio: float
    borrow_rate_pct: float
    horizon_years: int
    n_simulations: int
    block_size_months: int
    survival_by_year: list[dict[str, Any]]
    ruin_by_year: list[dict[str, Any]]
    portfolio_percentiles: list[dict[str, Any]]
    median_final_value: float
    mean_final_value: float
    liquidation_rate_pct: float
    median_cagr_pct: float
    yearly_stats: list[dict[str, Any]]


class CryptoSweepPoint(BaseModel):
    leverage_ratio: float
    survival_5yr_pct: float
    survival_10yr_pct: float
    survival_20yr_pct: float
    median_cagr_pct: float
    liquidation_rate_pct: float


# ── Core Simulator Logic ──────────────────────────────────────────────────────

def prepare_monthly_crypto_data(crypto_id: str) -> pd.DataFrame:
    """
    Loads daily crypto data from CRYPTO_RAW_DIR/{crypto_id}.csv,
    resamples to monthly, and calculates close-to-close returns & worst-case intra-month drops.
    """
    csv_path = CRYPTO_RAW_DIR / f"{crypto_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No data for '{crypto_id}' found at {csv_path}. "
            "Please run `python main.py download-crypto` first."
        )
    df_daily = pd.read_csv(csv_path, parse_dates=["Date"])
    df_daily = df_daily.sort_values("Date").reset_index(drop=True)
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily['YearMonth'] = df_daily['Date'].dt.to_period('M')
    
    # We aggregate group properties
    grouped = df_daily.groupby('YearMonth')
    
    monthly_rows = []
    prev_close = None
    
    for ym, group in grouped:
        if prev_close is None:
            prev_close = group['Open'].iloc[0]
            
        cur_close = group['Close'].iloc[-1]
        lows = group['Low'].min()
        highs = group['High'].max()
        
        monthly_return = (cur_close / prev_close) - 1.0
        intra_month_max_drop = (lows / prev_close) - 1.0
        intra_month_max_gain = (highs / prev_close) - 1.0
        
        monthly_rows.append({
            "year_month": str(ym),
            "monthly_return": monthly_return,
            "intra_month_max_drop": intra_month_max_drop,
            "intra_month_max_gain": intra_month_max_gain,
            "close": cur_close,
        })
        
        prev_close = cur_close
        
    return pd.DataFrame(monthly_rows)


def run_single_simulation(
    params: CryptoLeverageParams,
    monthly_returns: np.ndarray,
    intra_month_max_drops: np.ndarray,
) -> tuple[np.ndarray, bool, Optional[int]]:
    """
    Runs a single simulation path.
    Returns:
      - portfolio_values: np.ndarray of shape (total_months + 1,)
      - liquidated: bool
      - liquidation_month: Optional[int] (1-based index)
    """
    total_months = len(monthly_returns)
    portfolio_values = np.zeros(total_months + 1)
    portfolio_values[0] = params.initial_capital
    
    L = params.leverage_ratio
    borrow_rate_monthly = (params.borrow_rate_pct / 100.0) / 12.0
    liq_thresh_factor = params.liquidation_threshold_pct / 100.0
    
    liquidated = False
    liquidation_month = None
    
    for m in range(total_months):
        NAV_prev = portfolio_values[m]
        
        if liquidated:
            portfolio_values[m + 1] = 0.0
            continue
            
        r_drop = intra_month_max_drops[m]
        
        # Worst NAV during the month
        NAV_worst = NAV_prev * (1.0 + L * r_drop - (L - 1.0) * borrow_rate_monthly)
        
        if NAV_worst <= liq_thresh_factor * NAV_prev:
            liquidated = True
            liquidation_month = m + 1
            portfolio_values[m + 1] = 0.0
            continue
            
        # End of month final return
        r_actual = monthly_returns[m]
        NAV_end = NAV_prev * (1.0 + L * r_actual - (L - 1.0) * borrow_rate_monthly)
        
        if NAV_end <= liq_thresh_factor * NAV_prev:
            liquidated = True
            liquidation_month = m + 1
            portfolio_values[m + 1] = 0.0
        else:
            portfolio_values[m + 1] = NAV_end
            
    return portfolio_values, liquidated, liquidation_month


def run_monte_carlo_crypto(params: CryptoLeverageParams) -> CryptoMCResult:
    """
    Run block-bootstrap Monte Carlo simulations on crypto leverage.
    """
    df_monthly = prepare_monthly_crypto_data(params.crypto_id)
    hist_returns = df_monthly["monthly_return"].to_numpy()
    hist_drops = df_monthly["intra_month_max_drop"].to_numpy()
    
    total_months = params.horizon_years * 12
    n_hist = len(df_monthly)
    
    # Block bootstrap
    rng = np.random.default_rng(108) # seed for reproducibility
    blocks_needed = math.ceil(total_months / params.block_size_months)
    n_start_positions = max(1, n_hist - params.block_size_months + 1)
    
    # pre-sample bootstrap starting indices for block alignment
    all_starts = rng.integers(0, n_start_positions, size=(params.n_simulations, blocks_needed))
    
    all_portfolios = np.zeros((params.n_simulations, total_months + 1))
    all_liquidated = np.zeros(params.n_simulations, dtype=bool)
    all_liq_months = [None] * params.n_simulations
    
    all_rep_returns = np.zeros((params.n_simulations, total_months))
    all_rep_drops = np.zeros((params.n_simulations, total_months))
    
    for i in range(params.n_simulations):
        starts = all_starts[i]
        
        m_ret_parts = []
        m_drop_parts = []
        for s in starts:
            m_ret_parts.append(hist_returns[s: s + params.block_size_months])
            m_drop_parts.append(hist_drops[s: s + params.block_size_months])
            
        m_returns = np.concatenate(m_ret_parts)[:total_months]
        m_drops = np.concatenate(m_drop_parts)[:total_months]
        
        all_rep_returns[i] = m_returns
        all_rep_drops[i] = m_drops
        
        p_vals, liquidated, liq_month = run_single_simulation(params, m_returns, m_drops)
        all_portfolios[i] = p_vals
        all_liquidated[i] = liquidated
        all_liq_months[i] = liq_month
        
    # Percentiles per year over time
    portfolio_percentiles = []
    for y in range(params.horizon_years + 1):
        month_idx = y * 12
        vals = all_portfolios[:, month_idx]
        portfolio_percentiles.append({
            "year": y,
            "p5": float(np.percentile(vals, 5)),
            "p10": float(np.percentile(vals, 10)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
        })
        
    # Year-by-year survival rates
    survival_by_year = []
    for y in range(1, params.horizon_years + 1):
        month_idx = y * 12
        survived_count = 0
        for i in range(params.n_simulations):
            if not all_liquidated[i]:
                survived_count += 1
            elif all_liq_months[i] > month_idx:
                survived_count += 1
        survival_by_year.append({
            "year": y,
            "survival_rate_pct": float(survived_count / params.n_simulations * 100.0)
        })
        
    # Liquidation histogram by year
    ruin_by_year = []
    liq_years = []
    for m in all_liq_months:
        if m is not None:
            liq_years.append(math.ceil(m / 12))
            
    for y in range(1, params.horizon_years + 1):
        count = liq_years.count(y)
        ruin_by_year.append({
            "year": y,
            "liquidation_rate_pct": float(count / params.n_simulations * 100.0)
        })
        
    # CAGRs
    cagrs = []
    final_values = all_portfolios[:, -1]
    for i in range(params.n_simulations):
        f_val = final_values[i]
        if f_val <= 0:
            cagrs.append(-1.0)
        else:
            cagrs.append((f_val / params.initial_capital) ** (1.0 / params.horizon_years) - 1.0)
            
    median_cagr_pct = float(np.median(cagrs)) * 100.0
    liquidation_rate_pct = float(np.sum(all_liquidated) / params.n_simulations * 100.0)
    
    # Representative median path
    median_val = float(np.median(final_values))
    median_idx = int(np.argmin(np.abs(final_values - median_val)))
    rep_returns = all_rep_returns[median_idx]
    rep_drops = all_rep_drops[median_idx]
    rep_p_vals, rep_liq, rep_liq_m = run_single_simulation(params, rep_returns, rep_drops)
    
    yearly_stats = []
    for y in range(params.horizon_years + 1):
        idx = y * 12
        portfolio_val = float(rep_p_vals[idx])
        
        if y == 0:
            yr_return_pct = 0.0
            yr_annual_borrow_cost = 0.0
        else:
            prev_yr_val = float(rep_p_vals[idx - 12])
            if prev_yr_val <= 0:
                yr_return_pct = -100.0
                yr_annual_borrow_cost = 0.0
            else:
                yr_return_pct = float((portfolio_val / prev_yr_val) - 1.0) * 100.0
                yr_annual_borrow_cost = 0.0
                L = params.leverage_ratio
                borrow_rate_monthly = (params.borrow_rate_pct / 100.0) / 12.0
                for m_idx in range(idx - 12, idx):
                    if m_idx < len(rep_p_vals) - 1:
                        yr_annual_borrow_cost += (L - 1.0) * rep_p_vals[m_idx] * borrow_rate_monthly
                        
        yearly_stats.append({
            "year": y,
            "portfolio_value": portfolio_val,
            "yearly_return_pct": yr_return_pct,
            "borrow_cost_paid": yr_annual_borrow_cost,
            "liquidated": (rep_liq_m is not None and rep_liq_m <= idx) if y > 0 else False,
            "liquidation_month": rep_liq_m if (rep_liq_m is not None and rep_liq_m <= idx) else None,
        })
        
    return CryptoMCResult(
        crypto_id=params.crypto_id,
        leverage_ratio=params.leverage_ratio,
        borrow_rate_pct=params.borrow_rate_pct,
        horizon_years=params.horizon_years,
        n_simulations=params.n_simulations,
        block_size_months=params.block_size_months,
        survival_by_year=survival_by_year,
        ruin_by_year=ruin_by_year,
        portfolio_percentiles=portfolio_percentiles,
        median_final_value=median_val,
        mean_final_value=float(np.mean(final_values)),
        liquidation_rate_pct=liquidation_rate_pct,
        median_cagr_pct=median_cagr_pct,
        yearly_stats=yearly_stats,
    )


def run_one_sweep_point(args) -> CryptoSweepPoint:
    """Helper to run a single sweep point in a process pool."""
    params, leverage = args
    p = params.model_copy(update={"leverage_ratio": leverage})
    res = run_monte_carlo_crypto(p)
    
    s5 = 100.0
    s10 = 100.0
    s20 = 100.0
    for s in res.survival_by_year:
        if s["year"] == min(5, params.horizon_years):
            s5 = s["survival_rate_pct"]
        if s["year"] == min(10, params.horizon_years):
            s10 = s["survival_rate_pct"]
        if s["year"] == min(20, params.horizon_years):
            s20 = s["survival_rate_pct"]
            
    return CryptoSweepPoint(
        leverage_ratio=leverage,
        survival_5yr_pct=s5,
        survival_10yr_pct=s10,
        survival_20yr_pct=s20,
        median_cagr_pct=res.median_cagr_pct,
        liquidation_rate_pct=res.liquidation_rate_pct,
    )


def run_crypto_sweep(
    params_base: CryptoLeverageParams,
    leverage_ratios: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
) -> list[CryptoSweepPoint]:
    """Run sweep over different leverage ratios in parallel if available."""
    results = []
    tasks = [(params_base, lev) for lev in leverage_ratios]
    
    try:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_one_sweep_point, t) for t in tasks]
            for f in as_completed(futures):
                results.append(f.result())
    except Exception as e:
        logger.warning("Parallel crypto sweep failed, falling back to sequential: %s", e)
        for t in tasks:
            results.append(run_one_sweep_point(t))
            
    results.sort(key=lambda x: x.leverage_ratio)
    return results
