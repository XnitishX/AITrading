"""
Retirement Monte Carlo Simulator
──────────────────────────────────
Models a Bucket Strategy for retirement corpus management:

  • Equity Bucket  : Invested in Nifty 50 (actual historical returns via bootstrap)
  • Debt Bucket    : Liquid Fund / Short-term Debt / Bank FD (repo-rate model)
  • Withdrawals    : Monthly, inflation-adjusted (CPI-linked)
  • Tax            : LTCG 12.5 % above ₹1.25 L/FY on equity; debt interest taxed as income
  • Replenishment  : Scheduled (every N years) + Emergency threshold (< X months expenses)

Monte Carlo approach
--------------------
Uses 12-month *block bootstrap* (preserves autocorrelation and seasonal clustering)
on actual Nifty historical monthly returns so crash regimes are realistic.

Key classes / functions
-----------------------
  RetirementParams      — Input parameters (Pydantic)
  TaxEngine             — Per-simulation tax tracker
  SimPath               — Output of a single simulation run
  MCResult              — Aggregated output of N simulations
  run_single_simulation — Run one path
  run_monte_carlo       — Run N bootstrap paths
  run_optimization_sweep— Sweep equity% from 10–90 to find optimal allocation
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LTCG_EXEMPTION_INR = 125_000.0   # ₹1.25 L per FY — LTCG tax-free threshold
LTCG_RATE = 0.125                # 12.5 % on gains above exemption (post-2024 budget)
MIN_SIMULATIONS = 100
MAX_SIMULATIONS = 5_000
_BLOCK_SIZE_MONTHS = 12          # block bootstrap preserves annual seasonality


# ── Input Parameters ──────────────────────────────────────────────────────────

class RetirementParams(BaseModel):
    """All configurable parameters for the retirement simulator."""

    # Corpus & Allocation
    initial_corpus: float = Field(10_000_000.0, gt=0, description="Starting corpus (₹)")
    equity_pct: float = Field(60.0, ge=0.0, le=100.0, description="% of corpus in equity")

    # Withdrawals
    monthly_withdrawal: float = Field(50_000.0, gt=0, description="Initial monthly withdrawal (₹)")
    inflation_rate_pct: float = Field(6.0, ge=0.0, le=30.0, description="Annual CPI inflation rate (%)")

    # Debt instrument
    debt_instrument: str = Field("liquid_fund", description="liquid_fund | short_term_debt | bank_fd")
    future_debt_rate_override: Optional[float] = Field(
        None, ge=0.0, le=30.0,
        description="Override future debt rate (% p.a.). None = use repo-rate model."
    )

    # Replenishment strategy
    swp_replenish_years: int = Field(5, ge=1, le=20, description="Refill debt from equity every N years")
    emergency_months_threshold: int = Field(
        12, ge=1, le=120,
        description="Emergency replenish if debt bucket < this many months of current withdrawal"
    )
    replenish_mode: str = Field(
        "fill_years",
        description=(
            "How to size each scheduled replenishment: "
            "'fill_years' = fill debt to cover next N years of expenses (default); "
            "'pct_equity' = sell replenish_pct % of current equity bucket; "
            "'rebalance'  = top up debt until it equals replenish_pct % of total portfolio."
        ),
    )
    replenish_pct: float = Field(
        20.0, ge=1.0, le=100.0,
        description="Percentage used when replenish_mode is 'pct_equity' or 'rebalance'.",
    )

    @field_validator("replenish_mode")
    @classmethod
    def valid_replenish_mode(cls, v: str) -> str:
        allowed = {"fill_years", "pct_equity", "rebalance"}
        if v not in allowed:
            raise ValueError(f"replenish_mode must be one of {allowed}, got '{v}'")
        return v

    # ── FIRE mode: multi-phase accumulation before full retirement ─────────
    fire_mode: str = Field(
        "classic",
        description=(
            "FIRE strategy mode: "
            "'classic' = immediate full withdrawal (default); "
            "'coast' = no withdrawals for phase1_years, corpus grows freely, then full retirement; "
            "'barista' = reduced withdrawal (barista_monthly_withdrawal) for phase1_years "
            "           funded partly from corpus, rest from part-time income, then full retirement."
        ),
    )
    phase1_years: int = Field(
        0, ge=0, le=40,
        description="Duration of Phase 1 (pre-retirement phase) in years. 0 = immediate retirement (classic).",
    )
    barista_monthly_withdrawal: float = Field(
        20_000.0, ge=0.0,
        description="Monthly withdrawal from corpus during Barista FIRE Phase 1 (₹). "
                    "Remainder is assumed covered by part-time income.",
    )
    monthly_contribution: float = Field(
        0.0, ge=0.0,
        description="Monthly top-up added to equity bucket during Phase 1 (₹). "
                    "Inflates annually at inflation_rate_pct. Typical for coast/barista workers.",
    )
    phase1_equity_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Equity allocation (%) during Phase 1. None = same as equity_pct throughout. "
                    "At phase transition, portfolio is rebalanced to equity_pct (taxable event).",
    )

    @field_validator("fire_mode")
    @classmethod
    def valid_fire_mode(cls, v: str) -> str:
        allowed = {"classic", "coast", "barista"}
        if v not in allowed:
            raise ValueError(f"fire_mode must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def fire_phase1_consistency(self):
        if self.fire_mode != "classic":
            if self.phase1_years < 1:
                raise ValueError("phase1_years must be >= 1 when fire_mode is 'coast' or 'barista'")
            if self.phase1_years >= self.total_years:
                raise ValueError(
                    f"phase1_years ({self.phase1_years}) must be < total_years ({self.total_years}) "
                    "so there is at least 1 year of full retirement."
                )
        return self

    # Horizon
    total_years: int = Field(30, ge=1, le=50, description="Simulation horizon (years)")

    # Tax
    tax_enabled: bool = Field(True, description="Apply Indian LTCG + income tax rules")
    income_tax_bracket_pct: float = Field(30.0, ge=0.0, le=42.0, description="Income slab for debt tax (%)")

    # Simulation controls
    n_simulations: int = Field(1_000, ge=1, le=MAX_SIMULATIONS)
    random_seed: Optional[int] = Field(None, description="Fixed seed for reproducibility")

    # Historical scenario
    starting_year: Optional[int] = Field(
        None, ge=2000, le=2030,
        description="If set, yearly table shows actual Nifty returns from this calendar year."
    )

    # ── Leveraged position: replace debt/cash bucket with leveraged Nifty ─
    use_leverage: bool = Field(
        False,
        description=(
            "If True, replace the debt/cash bucket with a leveraged Nifty futures position. "
            "The debt allocation still acts as margin collateral but earns leveraged Nifty returns "
            "instead of the liquid-fund rate. Monthly cost = leverage_ratio × Nifty return "
            "minus (leverage_ratio − 1) × borrow_rate/12. Losses are capped at −100% (margin call). "
            "Preserves the current cash view when False (default)."
        ),
    )
    leverage_ratio: float = Field(
        2.0, ge=1.01, le=5.0,
        description=(
            "Total Nifty notional / own margin capital. "
            "2.0 = 2× leverage (₹1 of capital controls ₹2 of Nifty). "
            "Higher ratio amplifies both gains and losses. Only used when use_leverage=True."
        ),
    )
    leverage_borrow_rate_pct: float = Field(
        9.0, ge=0.0, le=30.0,
        description=(
            "Gross annual borrowing/carry cost for the leveraged position (% p.a.). "
            "Typically repo rate + 2–3% spread (e.g. 6.5% repo + 2.5% = 9%). "
            "Only used when use_leverage=True."
        ),
    )

    @field_validator("debt_instrument")
    @classmethod
    def valid_instrument(cls, v: str) -> str:
        allowed = {"liquid_fund", "short_term_debt", "bank_fd"}
        if v not in allowed:
            raise ValueError(f"debt_instrument must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def corpus_vs_withdrawal(self):
        # Basic sanity: monthly withdrawal should be < 5% of corpus / 12
        if self.monthly_withdrawal > self.initial_corpus * 0.10 / 12:
            logger.warning(
                "monthly_withdrawal (%.0f) > 10%% p.a. of corpus (%.0f) — "
                "high probability of ruin.",
                self.monthly_withdrawal,
                self.initial_corpus,
            )
        return self


# ── Tax Engine ────────────────────────────────────────────────────────────────

class TaxEngine:
    """
    Tracks per-simulation Indian tax liabilities.

    LTCG rules (post Aug 2024 budget):
      - 12.5 % on equity long-term gains above ₹1.25 L exemption per FY.
      - No indexation benefit.
      - Gains ≤ exemption remaining → zero tax.

    Debt interest:
      - Taxed at income slab rate (30 % in default bracket).
    """

    def __init__(self, income_tax_bracket_pct: float = 30.0, enabled: bool = True):
        self.enabled = enabled
        self.income_slab = income_tax_bracket_pct / 100.0
        self._fy_ltcg_gains_so_far = 0.0  # gains realised in current FY
        self._total_tax_paid = 0.0

    def new_fy(self) -> None:
        """Reset FY-specific counters (call at April 1 boundary)."""
        self._fy_ltcg_gains_so_far = 0.0

    def apply_equity_sale(self, gross_sale_value: float, cost_basis: float) -> float:
        """
        Calculate and deduct LTCG tax on an equity sale.

        Args:
            gross_sale_value: ₹ received from selling equity.
            cost_basis: Original cost of units sold (₹).

        Returns:
            Net proceeds after LTCG tax.
        """
        if not self.enabled or gross_sale_value <= cost_basis:
            return gross_sale_value

        raw_gain = gross_sale_value - cost_basis
        # Remaining exemption
        exemption_remaining = max(0.0, LTCG_EXEMPTION_INR - self._fy_ltcg_gains_so_far)
        taxable_gain = max(0.0, raw_gain - exemption_remaining)
        tax = taxable_gain * LTCG_RATE
        self._fy_ltcg_gains_so_far += raw_gain
        self._total_tax_paid += tax
        return gross_sale_value - tax

    def apply_debt_interest(self, gross_interest: float) -> float:
        """
        Deduct income tax on debt interest.

        Args:
            gross_interest: Gross interest earned in the period (₹).

        Returns:
            After-tax interest amount.
        """
        if not self.enabled:
            return gross_interest
        tax = gross_interest * self.income_slab
        self._total_tax_paid += tax
        return gross_interest - tax

    @property
    def total_tax_paid(self) -> float:
        return self._total_tax_paid


# ── Simulation Output ─────────────────────────────────────────────────────────

@dataclass
class SimPath:
    """Holds the full monthly time series for one simulation run."""
    months: int
    equity_values: np.ndarray          # equity bucket value each month
    debt_values: np.ndarray            # debt bucket value each month
    portfolio_values: np.ndarray       # total = equity + debt
    withdrawals: np.ndarray            # actual withdrawal each month
    tax_paid: np.ndarray               # cumulative tax paid each month
    replenish_months: list[int]        # month indices when scheduled replenishment fired
    emergency_months: list[int]        # month indices when emergency replenishment fired
    replenish_amounts: list[float]     # ₹ moved equity→debt in each scheduled replenishment
    emergency_amounts: list[float]     # ₹ moved equity→debt in each emergency replenishment
    ruin_month: Optional[int]          # month when portfolio hit zero (None if survived)
    total_tax_paid: float
    # Phase tracking (for Coast/Barista FIRE multi-phase support)
    contributions: np.ndarray = field(default_factory=lambda: np.array([]))  # monthly contributions
    phase_labels: list[str] = field(default_factory=list)  # "coast"|"barista"|"full_retirement"|"classic" per month
    corpus_at_phase2_start: float = 0.0   # portfolio value at the first month of Phase 2

    @property
    def survived(self) -> bool:
        return self.ruin_month is None


@dataclass
class MCResult:
    """Aggregated results from N Monte Carlo simulations."""
    params: RetirementParams
    n_simulations: int
    n_survived: int                       # paths that did not hit ruin

    # Percentile bands: list of dicts {year, p5, p10, p25, p50, p75, p90, p95}
    portfolio_percentiles: list[dict]

    # Survival probability at each year: list of {year, survival_pct}
    survival_by_year: list[dict]

    # Ruin year histogram: list of {year_bucket, count, pct_of_total}
    ruin_histogram: list[dict]

    # Summary scalars
    survival_20yr_pct: float
    survival_30yr_pct: float
    median_final_corpus: float
    mean_final_corpus: float
    safe_withdrawal_rate_pct: float   # max monthly_withdrawal / corpus that gives >=90% survival
    median_total_tax: float
    tax_drag_pct: float               # % difference in median corpus vs zero-tax scenario
    median_emergency_count: float     # avg emergency replenishments per simulation

    # Sequence-of-returns data: first 5 years' annualised return for each path
    sequence_risk_data: list[float]   # list of n_simulations values

    # Year-by-year breakdown from one representative path
    yearly_stats: list          # list of dicts, one per simulation year
    actual_start_year: Optional[int]  # None = bootstrap; int = historical calendar year
    median_ruin_year: Optional[float]  # Median year of depletion across all failed paths (None if all survived)

    # Median debt bucket value per year (for fan chart debt line)
    debt_medians: list[dict]   # list of {year, p50}

    # ── FIRE phase summary metrics (populated when fire_mode != "classic") ──
    corpus_at_phase2_start_median: float = 0.0   # median portfolio at start of Phase 2
    corpus_at_phase2_start_p10: float = 0.0      # P10 portfolio at Phase 2 start
    corpus_at_phase2_start_p90: float = 0.0      # P90 portfolio at Phase 2 start
    phase2_survival_pct: float = 0.0             # % paths surviving through all of Phase 2
    expected_phase2_starting_withdrawal: float = 0.0  # full FIRE target inflated to Phase 2 start (₹/mo)
    phase1_portfolio_cagr_pct: float = 0.0       # median path CAGR from T=0 to Phase 2 start
    net_phase1_contribution_total: float = 0.0   # total contributions added during Phase 1 (nominal ₹)


# ── Year-by-year table builder ────────────────────────────────────────────────

def _build_yearly_table(
    path: "SimPath",
    params: RetirementParams,
    start_calendar_year: Optional[int] = None,
    n_real_months: Optional[int] = None,
) -> list:
    """
    Convert a SimPath into a list of year-by-year summary dicts.

    Args:
        path:               Simulation output.
        params:             Parameters used to run the simulation.
        start_calendar_year: If provided, include a 'calendar_year' column.
        n_real_months:      If set, rows beyond this month are bootstrap-projected
                            (shown with is_projected=True so UI can flag them).

    Returns:
        List of dicts with annual portfolio, withdrawal, tax, and replenishment details.
    """
    rows = []
    for yr in range(1, params.total_years + 1):
        m_start = (yr - 1) * 12
        m_end = min(yr * 12, path.months)

        eq_start = float(path.equity_values[m_start])
        debt_start = float(path.debt_values[m_start])
        portfolio_start = float(path.portfolio_values[m_start])

        eq_end = float(path.equity_values[m_end])
        debt_end = float(path.debt_values[m_end])
        portfolio_end = float(path.portfolio_values[m_end])

        # Withdrawals: months (m_start+1) … m_end
        annual_wd = float(np.sum(path.withdrawals[m_start + 1: m_end + 1]))
        # Monthly withdrawal in this year = first non-zero withdrawal of the year
        yr_wds = path.withdrawals[m_start + 1: m_end + 1]
        monthly_wd_yr = float(yr_wds[yr_wds > 0][0]) if np.any(yr_wds > 0) else 0.0

        # Incremental tax paid this year
        tax_yr = float(path.tax_paid[m_end]) - float(path.tax_paid[m_start])

        # Replenishments within this year
        sched_in_yr = [m for m in path.replenish_months if m_start < m <= m_end]
        emerg_in_yr = [m for m in path.emergency_months if m_start < m <= m_end]

        # Replenishment amounts this year (₹ moved equity→debt)
        repl_amt_yr = sum(
            path.replenish_amounts[i]
            for i, m in enumerate(path.replenish_months)
            if m_start < m <= m_end
        )
        emrg_amt_yr = sum(
            path.emergency_amounts[i]
            for i, m in enumerate(path.emergency_months)
            if m_start < m <= m_end
        )
        replenishment_amount = round(repl_amt_yr + emrg_amt_yr)

        # Equity market return % (add back replenishment sold to get price-only return)
        # Without this adjustment, replenishment years look like equity crashes.
        equity_return_pct: Optional[float] = None
        if eq_start > 0 and portfolio_start > 0:
            # repl_amt_yr: for standard mode = equity sold (positive → add back).
            # For leverage harvest mode: negative (equity received) → subtracts (correct).
            # Both directions handled correctly by: adj = eq_end + repl_amt_yr + emrg_amt_yr
            # because leverage emergency amounts are now always positive (received into equity)
            # and leverage harvest replenish amounts are negative.
            adj_eq_end = eq_end + repl_amt_yr + emrg_amt_yr  # equity if no transfers had occurred
            equity_return_pct = round((adj_eq_end / eq_start - 1.0) * 100.0, 1)
            # Cap at ±200% to avoid extreme noise when equity is tiny
            equity_return_pct = max(-200.0, min(200.0, equity_return_pct))

        # Withdrawal as % of portfolio at start of year
        withdrawal_rate_pct: Optional[float] = None
        if portfolio_start > 0:
            withdrawal_rate_pct = round(annual_wd / portfolio_start * 100.0, 1)

        portfolio_change_pct: Optional[float] = None
        if portfolio_start > 0:
            portfolio_change_pct = round((portfolio_end / portfolio_start - 1.0) * 100.0, 1)

        ruined_this_yr = path.ruin_month is not None and path.ruin_month <= m_end

        # Phase label for this year
        phase_label = "Classic"
        if len(path.phase_labels) > m_start:
            lbl = path.phase_labels[m_start]
            if lbl == "coast":
                phase_label = "Coast"
            elif lbl == "barista":
                phase_label = "Barista"
            elif lbl == "full_retirement":
                phase_label = "Full Retirement"
            else:
                phase_label = "Classic"

        # Monthly contribution amount for this year
        contributions_yr = 0.0
        if len(path.contributions) > m_start:
            contributions_yr = float(np.sum(path.contributions[m_start + 1: m_end + 1]))

        net_annual_cashflow = contributions_yr - annual_wd

        row: dict = {
            "year": yr,
            "phase": phase_label,
            "equity_start": round(eq_start),
            "debt_start": round(debt_start),
            "portfolio_start": round(portfolio_start),
            "equity_end": round(eq_end),
            "debt_end": round(debt_end),
            "portfolio_end": round(portfolio_end),
            "annual_withdrawal": round(annual_wd),
            "monthly_withdrawal": round(monthly_wd_yr),
            "tax_paid_year": round(tax_yr),
            "replenishment_amount": replenishment_amount,
            "equity_return_pct": equity_return_pct,
            "withdrawal_rate_pct": withdrawal_rate_pct,
            "scheduled_replenish": len(sched_in_yr) > 0,
            "emergency_replenish": len(emerg_in_yr) > 0,
            "replenish_events": len(sched_in_yr) + len(emerg_in_yr),
            "portfolio_change_pct": portfolio_change_pct,
            "survived": not ruined_this_yr,
            "monthly_contribution_amt": round(contributions_yr),
            "net_annual_cashflow": round(net_annual_cashflow),
        }
        if start_calendar_year is not None:
            row["calendar_year"] = start_calendar_year + yr - 1

        # Mark rows that use bootstrap-projected returns (after real history ends)
        if n_real_months is not None:
            row["is_projected"] = m_start >= n_real_months
        else:
            row["is_projected"] = False

        rows.append(row)
        if ruined_this_yr:
            break  # portfolio depleted — stop building table

    return rows


# ── Historical scenario runner ────────────────────────────────────────────────

def run_historical_scenario(
    params: RetirementParams,
    nifty_df: pd.DataFrame,
) -> tuple:
    """
    Run one deterministic simulation using actual Nifty returns from
    `params.starting_year`, continuing with block-bootstrap once real data
    runs out.  Useful for "what if I retired in 2008/2022/2024?" analysis.

    When the selected starting year is recent (e.g. 2024), real data only
    covers a few months of the horizon.  Rather than padding with 0 % returns
    (which unfairly penalises equity), the remaining months are filled with
    a deterministic block-bootstrap draw from the *full* Nifty history so
    the projection is realistic while clearly marked as "simulated" in the UI.

    The debt rate is set to the RBI repo rate that was in effect at the chosen
    starting year (not today's rate), so 2008 correctly uses ~9 % repo, etc.

    Args:
        params:    Simulation parameters; starting_year must be set.
        nifty_df:  Daily Nifty data (Date, Close).

    Returns:
        Tuple (SimPath, actual_start_year, n_real_months).
        n_real_months: how many months come from true historical data.
    """
    from src.data.debt_rates import get_debt_return_pct

    df = nifty_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    monthly = df.set_index("Date").sort_index()["Close"].resample("ME").last().dropna()

    requested = params.starting_year or int(monthly.index[0].year)
    from_date = pd.Timestamp(f"{requested}-01-01")
    monthly_from = monthly[monthly.index >= from_date]

    if len(monthly_from) < 2:
        # Starting year is before data begins — use full history
        monthly_from = monthly
        actual_start = int(monthly.index[0].year)
    else:
        actual_start = requested

    # Historical monthly equity returns from the chosen start date
    hist_returns = monthly_from.pct_change().dropna().values.astype(float)
    # Full history for bootstrap continuation
    full_hist_returns = monthly.pct_change().dropna().values.astype(float)
    total_months = params.total_years * 12

    # Copy real historical returns first
    eq_returns = np.zeros(total_months)
    copy_len = min(len(hist_returns), total_months)
    eq_returns[:copy_len] = hist_returns[:copy_len]
    n_real_months = copy_len  # will be attached to SimPath via yearly table marker

    # ── Fill remaining months with block-bootstrap (NOT zeros) ──────────
    # Using 0% returns would make equity stagnant for years when starting_year
    # is recent — unrealistic and biased against equity.
    if copy_len < total_months:
        remaining = total_months - copy_len
        rng_fill = np.random.default_rng(42)  # deterministic seed for reproducibility
        n_full = len(full_hist_returns)
        block_size = _BLOCK_SIZE_MONTHS
        blocks_needed = math.ceil(remaining / block_size)
        n_start_pos = max(1, n_full - block_size + 1)
        starts = rng_fill.integers(0, n_start_pos, size=blocks_needed)
        fill = np.concatenate([
            full_hist_returns[s: s + block_size] for s in starts
        ])[:remaining]
        eq_returns[copy_len:] = fill

    # ── Debt rate: time-varying — historical rate for real months, today's rate for projected ──
    # Using a single blended constant rate is wrong: it means even the historical months
    # (e.g. 2008-2012) get a diluted rate instead of the true ~9% repo in effect then.
    # Solution: build a per-month debt factor array with two segments.
    hist_start_date = pd.Timestamp(f"{actual_start}-01-01")
    if params.future_debt_rate_override is not None:
        hist_debt_pct = params.future_debt_rate_override
        today_debt_pct = params.future_debt_rate_override
    else:
        hist_debt_pct = get_debt_return_pct(hist_start_date, params.debt_instrument)
        today_debt_pct = get_debt_return_pct(pd.Timestamp.today(), params.debt_instrument)

    # Per-month debt growth factor array
    # Months 0..copy_len-1  → historical repo rate at the starting year
    # Months copy_len..end  → today's forward repo rate (bootstrap continuation)
    hist_debt_mf = (1.0 + hist_debt_pct / 100.0) ** (1.0 / 12.0)
    today_debt_mf = (1.0 + today_debt_pct / 100.0) ** (1.0 / 12.0)
    debt_factors_seg = np.empty(total_months)
    debt_factors_seg[:copy_len] = hist_debt_mf   # real historical months
    debt_factors_seg[copy_len:] = today_debt_mf  # bootstrap-projected months

    infl_mf = (1.0 + params.inflation_rate_pct / 100.0) ** (1.0 / 12.0)

    path = run_single_simulation(
        params,
        eq_returns,
        debt_factors_seg,
        np.full(total_months, infl_mf),
        np.random.default_rng(42),
    )
    return path, actual_start, n_real_months


# ── FIRE phase label helper ───────────────────────────────────────────────────

def _phase_label(fire_mode: str, month: int, phase1_months: int) -> str:
    """Return the phase label string for a given month."""
    if fire_mode == "classic":
        return "classic"
    if month <= phase1_months:
        return fire_mode  # "coast" or "barista"
    return "full_retirement"


# ── Core simulation loop ──────────────────────────────────────────────────────

def run_single_simulation(
    params: RetirementParams,
    monthly_equity_returns: np.ndarray,  # array of monthly return fractions (e.g. 0.012)
    debt_monthly_factors: np.ndarray,    # array of monthly debt growth factors (e.g. 1.00487)
    inflation_monthly_factors: np.ndarray, # array of monthly inflation factors
    rng: np.random.Generator,
    _track_phases: bool = True,
) -> SimPath:
    """
    Run one Monte Carlo path using pre-sampled return sequences.

    Args:
        params:                    Simulation parameters.
        monthly_equity_returns:    Array of length ≥ total_years*12 of monthly equity returns.
        debt_monthly_factors:      Corresponding monthly debt growth factors.
        inflation_monthly_factors: Corresponding monthly inflation factors.
        rng:                       NumPy random generator (for any stochastic elements).
        _track_phases:             If False, skip per-month phase_labels tracking (saves memory/time
                                   when running thousands of MC paths where only portfolio values matter).

    Returns:
        SimPath with complete monthly time series.
    """
    total_months = params.total_years * 12

    # ── FIRE phase setup ──────────────────────────────────────────────────
    fire_mode = params.fire_mode
    phase1_months = params.phase1_years * 12 if fire_mode != "classic" else 0

    # ── Initial allocation ────────────────────────────────────────────────
    # During Phase 1, use phase1_equity_pct if specified
    init_eq_pct = params.phase1_equity_pct if (params.phase1_equity_pct is not None and fire_mode != "classic") else params.equity_pct
    equity_alloc = init_eq_pct / 100.0
    equity_val = params.initial_corpus * equity_alloc
    debt_val = params.initial_corpus * (1.0 - equity_alloc)

    # Cost basis for LTCG (weighted average)
    equity_cost_basis = equity_val

    # Tax engine
    tax = TaxEngine(
        income_tax_bracket_pct=params.income_tax_bracket_pct,
        enabled=params.tax_enabled,
    )

    # Monthly withdrawal (inflation-adjusted, starts at current value)
    current_monthly_wd = params.monthly_withdrawal        # full FIRE target — inflates always
    current_barista_wd = params.barista_monthly_withdrawal  # barista Phase 1 draw — also inflates
    current_contribution = params.monthly_contribution    # monthly top-up to equity — also inflates

    # Output arrays
    equity_arr = np.empty(total_months + 1)
    debt_arr = np.empty(total_months + 1)
    portfolio_arr = np.empty(total_months + 1)
    withdrawal_arr = np.zeros(total_months + 1)
    tax_arr = np.zeros(total_months + 1)
    contribution_arr = np.zeros(total_months + 1)
    # Phase labels — only allocated when needed for the yearly table (not in bulk MC)
    phase_label_arr: list[str] = ([""] * (total_months + 1)) if _track_phases else []

    equity_arr[0] = equity_val
    debt_arr[0] = debt_val
    portfolio_arr[0] = equity_val + debt_val
    if _track_phases:
        phase_label_arr[0] = _phase_label(fire_mode, 0, phase1_months)

    replenish_months: list[int] = []
    emergency_months: list[int] = []
    replenish_amounts: list[float] = []
    emergency_amounts: list[float] = []
    ruin_month: Optional[int] = None
    corpus_at_phase2_start: float = 0.0

    replenish_interval_months = params.swp_replenish_years * 12
    # Minimum months between emergency fires (prevents constant forced-sale in low-equity scenarios)
    emergency_cooldown = max(3, replenish_interval_months // 4)
    # Initialize to -cooldown so the first emergency can fire at month 1
    last_emergency_m: int = -emergency_cooldown

    for m in range(1, total_months + 1):
        # ── Month boundary: FY reset (April = month 4, 16, 28, ...) ─────────
        # Approximate: every 12 months from start reset FY LTCG counter
        if m % 12 == 1 and m > 1:
            tax.new_fy()

        # ── Apply inflation to monthly withdrawal (annually) ──────────────
        if m % 12 == 1 and m > 1:
            annual_inflation = inflation_monthly_factors[m - 1] ** 12  # reconstitute annual
            current_monthly_wd *= annual_inflation
            current_barista_wd *= annual_inflation
            current_contribution *= annual_inflation

        # ── Phase detection ───────────────────────────────────────────────
        in_phase1 = (fire_mode != "classic") and (m <= phase1_months)
        if _track_phases:
            phase_label_arr[m] = _phase_label(fire_mode, m, phase1_months)

        # ── Phase transition: rebalance equity allocation at Phase 2 start ─
        # This only fires once, at the exact month when Phase 1 ends.
        if m == phase1_months + 1 and params.phase1_equity_pct is not None and fire_mode != "classic":
            total_portfolio = equity_val + debt_val
            corpus_at_phase2_start = total_portfolio
            target_eq_val = total_portfolio * (params.equity_pct / 100.0)
            delta = target_eq_val - equity_val  # positive = buy equity; negative = sell equity

            if delta < -1.0:
                # Need to sell equity → move proceeds to debt (LTCG taxable)
                sell_amount = min(-delta, equity_val)
                cost_fraction = equity_cost_basis / equity_val if equity_val > 0 else 1.0
                cost_of_sold = sell_amount * cost_fraction
                proceeds = tax.apply_equity_sale(sell_amount, cost_of_sold)
                equity_val -= sell_amount
                equity_cost_basis = max(0.0, equity_cost_basis - cost_of_sold)
                debt_val += proceeds
                replenish_months.append(m)
                replenish_amounts.append(proceeds)
            elif delta > 1.0:
                # Need to buy equity → move cash from debt (not taxable)
                buy_amount = min(delta, debt_val)
                debt_val -= buy_amount
                equity_val += buy_amount
                equity_cost_basis += buy_amount  # new cost basis at current market price

        elif m == phase1_months + 1 and fire_mode != "classic" and params.phase1_equity_pct is None:
            # No allocation change — just record corpus at phase transition
            corpus_at_phase2_start = equity_val + debt_val

        # Record Phase 2 start corpus for classic mode (= initial corpus at m=1 effectively)
        if fire_mode == "classic" and m == 1:
            corpus_at_phase2_start = equity_val + debt_val

        # ── Monthly contribution (Phase 1 only) ───────────────────────────
        if in_phase1 and current_contribution > 0.0:
            contrib = current_contribution
            equity_val += contrib
            equity_cost_basis += contrib
            contribution_arr[m] = contrib

        # ── Equity bucket grows by this month's return ────────────────────
        r_idx = min(m - 1, len(monthly_equity_returns) - 1)
        eq_return = monthly_equity_returns[r_idx]
        prev_equity = equity_val
        equity_val = equity_val * (1.0 + eq_return)

        # Cost basis stays fixed at original investment value — only decreases on sale.
        # (Do NOT scale with unrealised appreciation: that would understate LTCG tax.)

        # ── Debt / Leveraged bucket earns return ──────────────────────────
        d_factor = debt_monthly_factors[r_idx]
        if params.use_leverage and debt_val > 0:
            # Leveraged Nifty futures: notional = leverage_ratio × margin_capital
            # Monthly P&L = leverage_ratio × nifty_return × margin
            #              − (leverage_ratio − 1) × monthly_borrow_rate × margin
            # = margin × (leverage_ratio × eq_return − (leverage_ratio − 1) × monthly_borrow_rate)
            # Loss is capped at −100% of capital (exchange closes position at margin call).
            monthly_borrow = params.leverage_borrow_rate_pct / 100.0 / 12.0
            leveraged_return = (
                params.leverage_ratio * eq_return
                - (params.leverage_ratio - 1.0) * monthly_borrow
            )
            # Hard cap: can't lose more than invested (margin call / automatic liquidation)
            leveraged_return = max(leveraged_return, -1.0)
            gross_pnl = debt_val * leveraged_return
            # Tax: futures P&L = non-speculative business income in India → taxed at slab rate
            # Positive months: pay income tax; negative months: loss passes through as-is.
            if gross_pnl > 0:
                net_pnl = tax.apply_debt_interest(gross_pnl)   # reuses income-tax at slab rate
            else:
                net_pnl = gross_pnl  # losses applied fully (no intra-sim carry-forward)
            debt_val = max(0.0, debt_val + net_pnl)
        else:
            gross_interest = debt_val * (d_factor - 1.0)
            net_interest = tax.apply_debt_interest(gross_interest)
            debt_val = debt_val + net_interest

        # ── Determine this month's actual withdrawal ──────────────────────
        if fire_mode == "coast" and in_phase1:
            this_month_wd = 0.0          # coast: no withdrawal during phase 1
        elif fire_mode == "barista" and in_phase1:
            this_month_wd = current_barista_wd  # barista: reduced withdrawal
        else:
            this_month_wd = current_monthly_wd  # classic / phase 2: full target

        # ── Deduct monthly withdrawal ──────────────────────────────────────
        # Both standard and leverage modes withdraw from debt/leveraged bucket first,
        # then sell equity to cover any shortfall. This bucket-strategy design preserves
        # equity during bear markets, allowing it to recover before being spent.
        wd_this_month = min(this_month_wd, debt_val + equity_val)
        withdrawal_arr[m] = wd_this_month

        if wd_this_month > 0:
            if debt_val >= wd_this_month:
                debt_val -= wd_this_month
            else:
                # Debt/leveraged bucket insufficient → sell equity to cover remainder.
                # We need `shortfall` in NET cash after LTCG tax.
                # Solve: sell enough equity so that (sell - ltcg_tax) = shortfall.
                shortfall = wd_this_month - debt_val
                debt_used = debt_val          # cash from debt bucket
                debt_val = 0.0
                if equity_val > 0:
                    cost_fraction = equity_cost_basis / equity_val
                    gain_fraction = 1.0 - cost_fraction
                    # Gross up: to net `shortfall` after paying ltcg_tax at effective rate.
                    # Approximation: sell_gross = shortfall / (1 - gain_fraction * 0.125)
                    # where 0.125 = LTCG rate. Capped so we don't oversell.
                    ltcg_approx_rate = 0.125 if params.tax_enabled else 0.0
                    denom = max(0.01, 1.0 - gain_fraction * ltcg_approx_rate)
                    sell_gross = min(shortfall / denom, equity_val)  # can't sell more than we have
                    cost_of_sold = sell_gross * cost_fraction
                    proceeds = tax.apply_equity_sale(sell_gross, cost_of_sold)
                    equity_val -= sell_gross
                    equity_cost_basis = max(0.0, equity_cost_basis - cost_of_sold)
                    # Actual cash delivered to retiree: debt + proceeds
                    wd_this_month = debt_used + proceeds
                    withdrawal_arr[m] = wd_this_month  # correct the already-set value
                else:
                    equity_val = 0.0
                    equity_cost_basis = 0.0
                    wd_this_month = debt_used
                    withdrawal_arr[m] = wd_this_month

        # ── Check for ruin ────────────────────────────────────────────────
        total = equity_val + debt_val
        if total <= 0.0:
            equity_arr[m] = 0.0
            debt_arr[m] = 0.0
            portfolio_arr[m] = 0.0
            tax_arr[m] = tax.total_tax_paid
            ruin_month = m
            # Pad remaining months with zeros
            equity_arr[m + 1:] = 0.0
            debt_arr[m + 1:] = 0.0
            portfolio_arr[m + 1:] = 0.0
            withdrawal_arr[m + 1:] = 0.0
            tax_arr[m + 1:] = tax.total_tax_paid
            contribution_arr[m + 1:] = 0.0
            break

        # ── Replenishment / Rebalancing ───────────────────────────────────
        # Skip replenishment entirely during Coast Phase 1 (no withdrawal → no depletion)
        skip_replenishment = (fire_mode == "coast" and in_phase1)

        if params.use_leverage:
            # Leverage mode: withdrawals come from leveraged bucket first (debt-first strategy).
            # The leveraged bucket is the "growth-and-spend" engine; equity is the long-term buffer.
            #
            # Replenishment strategy (one-way harvest only):
            #   Scheduled: if leveraged grew above target%, harvest profits → equity.
            #              Never top up leveraged from equity at crash prices.
            #   Emergency: skip — when leveraged bucket wipes out, the standard withdrawal
            #              fallback (equity) handles it naturally without forced bottom-buying.

            if not skip_replenishment and m > 0 and m % replenish_interval_months == 0:
                total_port = equity_val + debt_val
                if total_port > 0:
                    target_lev = total_port * (1.0 - params.equity_pct / 100.0)
                    delta = debt_val - target_lev  # +: levered grew above target
                    if delta > 100.0:
                        # Harvest excess from leveraged position → equity (profit-taking)
                        harvest = min(delta, debt_val)
                        debt_val -= harvest
                        equity_val += harvest
                        equity_cost_basis += harvest  # reinvested at current market price
                        replenish_months.append(m)
                        replenish_amounts.append(-harvest)  # negative = received into equity
            # No emergency replenishment for leverage mode — avoid forcing equity sales
            # at crash prices. Equity naturally funds withdrawals when leveraged bucket depletes.

        else:
            if not skip_replenishment and m > 0 and m % replenish_interval_months == 0:
                equity_val, debt_val, equity_cost_basis, r_amt = _do_replenishment(
                    equity_val, debt_val, equity_cost_basis,
                    this_month_wd, params.swp_replenish_years, tax,
                    inflation_rate_pct=params.inflation_rate_pct,
                    replenish_mode=params.replenish_mode,
                    replenish_pct=params.replenish_pct,
                )
                replenish_months.append(m)
                replenish_amounts.append(r_amt)

            # ── Replenishment: Emergency threshold ───────────────────────────
            # Use this_month_wd as threshold — NOT current_monthly_wd — so barista mode
            # doesn't trigger emergency at the full FIRE target threshold during Phase 1.
            elif not skip_replenishment:
                emergency_wd_threshold = this_month_wd * params.emergency_months_threshold
                if (this_month_wd > 0 and debt_val < emergency_wd_threshold and equity_val > 0
                        and (m - last_emergency_m) >= emergency_cooldown):
                    # Emergency always uses fill_years to ensure a proper safety-net top-up
                    equity_val, debt_val, equity_cost_basis, e_amt = _do_replenishment(
                        equity_val, debt_val, equity_cost_basis,
                        this_month_wd, params.swp_replenish_years, tax,
                        inflation_rate_pct=params.inflation_rate_pct,
                        replenish_mode="fill_years",
                        replenish_pct=params.replenish_pct,
                    )
                    emergency_months.append(m)
                    emergency_amounts.append(e_amt)
                    last_emergency_m = m

        # ── Record state ──────────────────────────────────────────────────
        equity_arr[m] = equity_val
        debt_arr[m] = debt_val
        portfolio_arr[m] = equity_val + debt_val
        tax_arr[m] = tax.total_tax_paid

    return SimPath(
        months=total_months,
        equity_values=equity_arr,
        debt_values=debt_arr,
        portfolio_values=portfolio_arr,
        withdrawals=withdrawal_arr,
        tax_paid=tax_arr,
        replenish_months=replenish_months,
        emergency_months=emergency_months,
        replenish_amounts=replenish_amounts,
        emergency_amounts=emergency_amounts,
        ruin_month=ruin_month,
        total_tax_paid=tax.total_tax_paid,
        contributions=contribution_arr,
        phase_labels=phase_label_arr,
        corpus_at_phase2_start=corpus_at_phase2_start,
    )


def _do_replenishment(
    equity_val: float,
    debt_val: float,
    equity_cost_basis: float,
    monthly_wd: float,
    replenish_years: int,
    tax: TaxEngine,
    inflation_rate_pct: float = 0.0,
    replenish_mode: str = "fill_years",
    replenish_pct: float = 20.0,
) -> tuple[float, float, float, float]:
    """
    Sell equity to replenish the debt bucket.

    Three modes (controlled by `replenish_mode`):

    fill_years (default):
        Sell enough equity so the debt bucket covers the next `replenish_years`
        years of inflation-adjusted expenses.  Geometric-series target:
            target = monthly_wd * 12 * [(1+r)^n - 1] / r

    pct_equity:
        Sell exactly `replenish_pct` % of the current equity bucket value
        and move the net proceeds to debt.

    rebalance:
        Top up the debt bucket until it equals `replenish_pct` % of the
        total portfolio (equity + debt).  Sells equity to reach that target.

    Args:
        equity_val:          Current equity bucket value (₹).
        debt_val:            Current debt bucket value (₹).
        equity_cost_basis:   Weighted average cost basis of equity (₹).
        monthly_wd:          Current inflation-adjusted monthly withdrawal (₹).
        replenish_years:     Years of expenses to target (used in fill_years mode).
        tax:                 Active TaxEngine instance.
        inflation_rate_pct:  Annual CPI inflation rate (%) — used in fill_years mode.
        replenish_mode:      One of 'fill_years', 'pct_equity', 'rebalance'.
        replenish_pct:       % used for pct_equity and rebalance modes.

    Returns:
        (new_equity_val, new_debt_val, new_equity_cost_basis, net_proceeds)
    """
    if equity_val <= 0:
        return equity_val, debt_val, equity_cost_basis, 0.0

    total_portfolio = equity_val + debt_val

    if replenish_mode == "pct_equity":
        # Sell a fixed fraction of equity regardless of debt level
        sell_amount = equity_val * (replenish_pct / 100.0)

    elif replenish_mode == "rebalance":
        # Target debt = replenish_pct % of total portfolio
        target_debt = total_portfolio * (replenish_pct / 100.0)
        needed = max(0.0, target_debt - debt_val)
        if needed <= 0:
            return equity_val, debt_val, equity_cost_basis, 0.0
        sell_amount = min(needed, equity_val)

    else:  # fill_years (default)
        r = inflation_rate_pct / 100.0
        annual_wd = monthly_wd * 12
        if r > 1e-9:
            target_debt = annual_wd * ((1.0 + r) ** replenish_years - 1.0) / r
        else:
            target_debt = annual_wd * replenish_years
        needed = max(0.0, target_debt - debt_val)
        if needed <= 0:
            return equity_val, debt_val, equity_cost_basis, 0.0
        sell_amount = min(needed, equity_val)

    if sell_amount <= 0:
        return equity_val, debt_val, equity_cost_basis, 0.0

    # LTCG tax on the sold portion
    cost_fraction = equity_cost_basis / equity_val
    cost_of_sold = sell_amount * cost_fraction
    net_proceeds = tax.apply_equity_sale(sell_amount, cost_of_sold)

    equity_val -= sell_amount
    equity_cost_basis = max(0.0, equity_cost_basis - cost_of_sold)
    debt_val += net_proceeds

    return equity_val, debt_val, equity_cost_basis, net_proceeds


# ── Monthly return series extraction ─────────────────────────────────────────

def extract_monthly_returns(nifty_df: pd.DataFrame) -> np.ndarray:
    """
    Convert daily Nifty OHLCV dataframe to array of monthly log returns.

    Args:
        nifty_df: DataFrame with at least ['Date', 'Close'] columns.

    Returns:
        1-D numpy array of monthly simple returns (fractions, not %).
    """
    df = nifty_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    monthly = df["Close"].resample("ME").last().dropna()
    returns = monthly.pct_change().dropna().values.astype(float)
    return returns


# ── Main Monte Carlo runner ───────────────────────────────────────────────────

def run_monte_carlo(
    params: RetirementParams,
    nifty_df: pd.DataFrame,
    compute_swr: bool = True,
    compute_tax_drag: bool = True,
) -> MCResult:
    """
    Run N Monte Carlo simulations of the retirement strategy.

    Args:
        params:            Simulation parameters.
        nifty_df:          Daily Nifty 50 data (needs Date, Close columns).
        compute_swr:       If True, run binary-search for Safe Withdrawal Rate (~3000 extra sims).
        compute_tax_drag:  If True, estimate tax drag vs no-tax scenario (~200 extra sims).

    Returns:
        MCResult with percentile bands, survival rates, and summary metrics.
    """
    import importlib
    from src.data.debt_rates import get_debt_return_pct
    from src.data.inflation import get_inflation_rate

    total_months = params.total_years * 12
    rng = np.random.default_rng(params.random_seed)

    # ── Extract historical monthly equity returns ─────────────────────────
    hist_returns = extract_monthly_returns(nifty_df)
    n_hist = len(hist_returns)
    logger.info(
        "Running %d Monte Carlo simulations over %d months (%d historical return observations).",
        params.n_simulations, total_months, n_hist
    )

    # ── Build representative debt/inflation monthly factor arrays ─────────
    # We use current (latest available) rate as a conservative forward estimate
    from src.simulator.leverage_sim import get_repo_rate
    latest_date = pd.Timestamp.today()
    debt_annual_pct = (
        params.future_debt_rate_override
        if params.future_debt_rate_override is not None
        else get_debt_return_pct(latest_date, params.debt_instrument)
    )
    debt_monthly_factor = (1.0 + debt_annual_pct / 100.0) ** (1.0 / 12.0)
    inflation_monthly_factor = (1.0 + params.inflation_rate_pct / 100.0) ** (1.0 / 12.0)

    # Create constant arrays (can be extended to time-varying if desired)
    debt_factors = np.full(total_months, debt_monthly_factor)
    inflation_factors = np.full(total_months, inflation_monthly_factor)

    # ── Pre-sample all bootstrap paths at once (vectorized) ──────────────
    # Generate all random start indices in one shot — avoids repeated rng calls
    # in the hot loop, which is the main overhead for large n_simulations.
    n_hist = len(hist_returns)
    blocks_needed = math.ceil(total_months / _BLOCK_SIZE_MONTHS)
    n_start_positions = max(1, n_hist - _BLOCK_SIZE_MONTHS + 1)
    # Shape: (n_sims, blocks_needed) — all random indices at once
    all_starts = rng.integers(0, n_start_positions, size=(params.n_simulations, blocks_needed))

    def _get_path(sim_idx: int) -> np.ndarray:
        """Build one return path from pre-sampled block starts."""
        starts = all_starts[sim_idx]
        path = np.concatenate([
            hist_returns[s: s + _BLOCK_SIZE_MONTHS] for s in starts
        ])
        return path[:total_months]

    # ── Run simulations in parallel using threads ─────────────────────────
    # Threads are effective here because numpy releases the GIL during
    # heavy array operations in run_single_simulation. Each thread gets its
    # own RNG seeded from a child seed so results are reproducible.
    all_portfolios = np.zeros((params.n_simulations, total_months + 1))
    all_debts      = np.zeros((params.n_simulations, total_months + 1))
    all_eq_returns_store = np.empty((params.n_simulations, total_months))
    all_corpus_at_phase2 = np.zeros(params.n_simulations)
    ruin_months: list[Optional[int]] = [None] * params.n_simulations
    emergency_counts: list[int] = [0] * params.n_simulations
    total_taxes: list[float] = [0.0] * params.n_simulations
    seq_risk: list[float] = []
    seq_risk_lock_data: list = [None] * params.n_simulations

    # Child seeds for per-thread reproducibility
    child_seeds = rng.integers(0, 2**31, size=params.n_simulations)

    def _run_one(i: int):
        eq_returns = _get_path(i)
        child_rng = np.random.default_rng(int(child_seeds[i]))
        path = run_single_simulation(params, eq_returns, debt_factors, inflation_factors, child_rng, _track_phases=False)
        all_portfolios[i] = path.portfolio_values
        all_debts[i] = path.debt_values
        all_corpus_at_phase2[i] = path.corpus_at_phase2_start
        ruin_months[i] = path.ruin_month
        emergency_counts[i] = len(path.emergency_months)
        total_taxes[i] = path.total_tax_paid
        all_eq_returns_store[i] = eq_returns
        # seq-risk data stored for later processing (no lock needed — write to own index)
        if len(path.equity_values) >= 61 and path.equity_values[0] > 0:
            ann = (path.equity_values[60] / path.equity_values[0]) ** (1.0 / 5.0) - 1.0
            seq_risk_lock_data[i] = float(ann * 100)
        else:
            seq_risk_lock_data[i] = None

    n_workers = min(8, params.n_simulations)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_one, i) for i in range(params.n_simulations)]
        for f in as_completed(futures):
            f.result()  # propagate any exceptions

    seq_risk = [v for v in seq_risk_lock_data if v is not None]
    n_survived = sum(1 for r in ruin_months if r is None)

    # ── Percentile bands ──────────────────────────────────────────────────
    percentile_levels = [5, 10, 25, 50, 75, 90, 95]
    portfolio_percentiles = []
    debt_medians: list[dict] = []
    for yr in range(params.total_years + 1):
        m_idx = yr * 12
        col = all_portfolios[:, m_idx]
        row: dict = {"year": yr}
        for p in percentile_levels:
            row[f"p{p}"] = round(float(np.percentile(col, p)), 0)
        portfolio_percentiles.append(row)
        debt_medians.append({"year": yr, "p50": round(float(np.median(all_debts[:, m_idx])), 0)})

    # ── Survival by year ──────────────────────────────────────────────────
    survival_by_year = []
    for yr in range(1, params.total_years + 1):
        m_idx = yr * 12
        survived_at_yr = int(np.sum(all_portfolios[:, m_idx] > 0))
        pct = round(survived_at_yr / params.n_simulations * 100, 1)
        survival_by_year.append({"year": yr, "survival_pct": pct})

    # ── Ruin histogram ────────────────────────────────────────────────────
    ruin_years = [
        math.ceil(r / 12) for r in ruin_months if r is not None
    ]
    ruin_histogram = []
    if ruin_years:
        for yr in range(1, params.total_years + 1):
            count = ruin_years.count(yr)
            ruin_histogram.append({
                "year": yr,
                "count": count,
                "pct_of_total": round(count / params.n_simulations * 100, 2),
            })

    # ── Safe Withdrawal Rate + Tax Drag (run concurrently) ───────────────
    swr_future = None
    drag_future = None
    median_final = float(np.median(all_portfolios[:, -1]))  # needed early for tax drag

    if compute_swr or compute_tax_drag:
        with ThreadPoolExecutor(max_workers=2) as pool2:
            if compute_swr:
                swr_future = pool2.submit(
                    _compute_swr, params, hist_returns, debt_factors, inflation_factors,
                    np.random.default_rng(params.random_seed + 1 if params.random_seed else 1)
                )
            if compute_tax_drag:
                drag_future = pool2.submit(
                    _estimate_tax_drag, params, hist_returns, debt_factors, inflation_factors,
                    np.random.default_rng(params.random_seed + 2 if params.random_seed else 2),
                    median_final
                )

    if swr_future is not None:
        swr_pct = swr_future.result()
    else:
        swr_pct = round(params.monthly_withdrawal * 12 / params.initial_corpus * 100.0, 2)

    if drag_future is not None:
        tax_drag = drag_future.result()
    else:
        tax_drag = 0.0

    # ── Summary scalars ───────────────────────────────────────────────────
    def _survival_at_year(yr: int) -> float:
        entry = next((s for s in survival_by_year if s["year"] == yr), None)
        return entry["survival_pct"] if entry else 0.0

    surv_20 = _survival_at_year(min(20, params.total_years))
    surv_30 = _survival_at_year(min(30, params.total_years))
    final_col = all_portfolios[:, -1]
    mean_final = float(np.mean(final_col))
    median_tax = float(np.median(total_taxes))
    median_emerg = float(np.median(emergency_counts))

    # ── Median ruin year ────────────────────────────────────────
    ruin_years_list = [math.ceil(r / 12) for r in ruin_months if r is not None]
    median_ruin_year: Optional[float] = float(np.median(ruin_years_list)) if ruin_years_list else None

    # ── Year-by-year table ────────────────────────────────────────
    # If starting_year is specified → use actual historical returns from that year.
    # Otherwise → run one deterministic path with seed=0 (representative scenario).
    yearly_stats: list = []
    actual_start_year: Optional[int] = None

    if params.starting_year is not None:
        hist_path, actual_start_year, n_real_months = run_historical_scenario(params, nifty_df)
        yearly_stats = _build_yearly_table(
            hist_path, params,
            start_calendar_year=actual_start_year,
            n_real_months=n_real_months,
        )
    else:
        # Use the simulation path whose final corpus is closest to the median.
        # This is far more representative than a fixed-seed arbitrary path because:
        #  - seed=0 might land in the top 20% or bottom 10% of outcomes by chance
        #  - The median path reflects the P50 experience from historical Nifty returns
        final_values = all_portfolios[:, -1]
        median_val = float(np.median(final_values))
        median_idx = int(np.argmin(np.abs(final_values - median_val)))
        rep_eq = all_eq_returns_store[median_idx]
        rep_path = run_single_simulation(
            params, rep_eq, debt_factors, inflation_factors, np.random.default_rng(0)
        )
        yearly_stats = _build_yearly_table(rep_path, params, start_calendar_year=None)

    # ── FIRE phase summary fields ─────────────────────────────────────────
    fire_summary: dict = {}
    if params.fire_mode != "classic" and params.phase1_years > 0:
        phase1_m = params.phase1_years * 12
        # Corpus at Phase 2 start
        p2_col = all_corpus_at_phase2
        fire_summary["corpus_at_phase2_start_median"] = float(np.median(p2_col))
        fire_summary["corpus_at_phase2_start_p10"] = float(np.percentile(p2_col, 10))
        fire_summary["corpus_at_phase2_start_p90"] = float(np.percentile(p2_col, 90))

        # Phase 2 survival: % paths where portfolio > 0 at end of total_years
        # (same as overall survival but the metric label is clearer for FIRE modes)
        fire_summary["phase2_survival_pct"] = float(
            np.sum(all_portfolios[:, -1] > 0) / params.n_simulations * 100.0
        )

        # Expected Phase 2 starting withdrawal: inflation-adjusted full FIRE target
        # at the end of phase1_years (deterministic — same for all paths)
        infl_annual = 1.0 + params.inflation_rate_pct / 100.0
        fire_summary["expected_phase2_starting_withdrawal"] = float(
            params.monthly_withdrawal * (infl_annual ** params.phase1_years)
        )

        # Phase 1 portfolio CAGR (median path)
        median_p2 = fire_summary["corpus_at_phase2_start_median"]
        if median_p2 > 0 and params.initial_corpus > 0 and params.phase1_years > 0:
            fire_summary["phase1_portfolio_cagr_pct"] = float(
                ((median_p2 / params.initial_corpus) ** (1.0 / params.phase1_years) - 1.0) * 100.0
            )
        else:
            fire_summary["phase1_portfolio_cagr_pct"] = 0.0

        # Total nominal contribution during Phase 1 (deterministic — all paths same contributions)
        # Compute: sum of current_contribution * (inflation)^yr for yr in range(phase1_years)
        total_contrib = 0.0
        if params.monthly_contribution > 0:
            infl_monthly = infl_annual ** (1.0 / 12.0)
            contrib = params.monthly_contribution
            for mo in range(1, phase1_m + 1):
                if mo % 12 == 1 and mo > 1:
                    contrib *= infl_annual
                total_contrib += contrib
        fire_summary["net_phase1_contribution_total"] = float(total_contrib)
    else:
        fire_summary["corpus_at_phase2_start_median"] = float(params.initial_corpus)
        fire_summary["corpus_at_phase2_start_p10"] = float(params.initial_corpus)
        fire_summary["corpus_at_phase2_start_p90"] = float(params.initial_corpus)
        fire_summary["phase2_survival_pct"] = float(n_survived / params.n_simulations * 100.0)
        fire_summary["expected_phase2_starting_withdrawal"] = float(params.monthly_withdrawal)
        fire_summary["phase1_portfolio_cagr_pct"] = 0.0
        fire_summary["net_phase1_contribution_total"] = 0.0

    return MCResult(
        params=params,
        n_simulations=params.n_simulations,
        n_survived=n_survived,
        portfolio_percentiles=portfolio_percentiles,
        survival_by_year=survival_by_year,
        ruin_histogram=ruin_histogram,
        survival_20yr_pct=surv_20,
        survival_30yr_pct=surv_30,
        median_final_corpus=median_final,
        mean_final_corpus=mean_final,
        safe_withdrawal_rate_pct=swr_pct,
        median_total_tax=median_tax,
        tax_drag_pct=tax_drag,
        median_emergency_count=median_emerg,
        sequence_risk_data=seq_risk,
        yearly_stats=yearly_stats,
        actual_start_year=actual_start_year,
        median_ruin_year=median_ruin_year,
        debt_medians=debt_medians,
        corpus_at_phase2_start_median=fire_summary["corpus_at_phase2_start_median"],
        corpus_at_phase2_start_p10=fire_summary["corpus_at_phase2_start_p10"],
        corpus_at_phase2_start_p90=fire_summary["corpus_at_phase2_start_p90"],
        phase2_survival_pct=fire_summary["phase2_survival_pct"],
        expected_phase2_starting_withdrawal=fire_summary["expected_phase2_starting_withdrawal"],
        phase1_portfolio_cagr_pct=fire_summary["phase1_portfolio_cagr_pct"],
        net_phase1_contribution_total=fire_summary["net_phase1_contribution_total"],
    )


def _compute_swr(
    base_params: RetirementParams,
    hist_returns: np.ndarray,
    debt_factors: np.ndarray,
    inflation_factors: np.ndarray,
    rng: np.random.Generator,
    target_survival: float = 90.0,
    n_sims_swr: int = 200,
) -> float:
    """
    Binary search for the Safe Withdrawal Rate (as % of initial corpus p.a.)
    that achieves at least `target_survival` % survival over `total_years`.

    Returns monthly withdrawal as % of initial corpus per annum.
    """
    def _survival_pct(monthly_wd: float) -> float:
        test_params = base_params.model_copy(update={"monthly_withdrawal": monthly_wd, "n_simulations": n_sims_swr})
        total_months = test_params.total_years * 12

        def _sample():
            n_hist = len(hist_returns)
            blocks = math.ceil(total_months / _BLOCK_SIZE_MONTHS)
            starts = rng.integers(0, max(1, n_hist - _BLOCK_SIZE_MONTHS + 1), size=blocks)
            path = np.concatenate([hist_returns[s:s + _BLOCK_SIZE_MONTHS] for s in starts])
            return path[:total_months]

        survived = 0
        for _ in range(n_sims_swr):
            path = run_single_simulation(test_params, _sample(), debt_factors, inflation_factors, rng, _track_phases=False)
            if path.survived:
                survived += 1
        return survived / n_sims_swr * 100.0

    # Binary search between 0.1% and 10% of corpus per year
    lo_wd = base_params.initial_corpus * 0.001 / 12
    hi_wd = base_params.initial_corpus * 0.10 / 12

    for _ in range(8):  # 8 iterations → ~0.4% accuracy, faster than 10
        mid_wd = (lo_wd + hi_wd) / 2.0
        surv = _survival_pct(mid_wd)
        if surv >= target_survival:
            lo_wd = mid_wd  # can afford higher withdrawal
        else:
            hi_wd = mid_wd  # must withdraw less

    swr_monthly = (lo_wd + hi_wd) / 2.0
    swr_annual_pct = swr_monthly * 12 / base_params.initial_corpus * 100.0
    return round(swr_annual_pct, 2)


def _estimate_tax_drag(
    params: RetirementParams,
    hist_returns: np.ndarray,
    debt_factors: np.ndarray,
    inflation_factors: np.ndarray,
    rng: np.random.Generator,
    median_with_tax: float,
    n_sims: int = 100,
) -> float:
    """Run quick no-tax simulation and compute % difference in median final corpus."""
    if not params.tax_enabled:
        return 0.0

    no_tax_params = params.model_copy(update={"tax_enabled": False, "n_simulations": n_sims})
    total_months = params.total_years * 12
    n_hist = len(hist_returns)

    portfolios = []
    for _ in range(n_sims):
        blocks = math.ceil(total_months / _BLOCK_SIZE_MONTHS)
        starts = rng.integers(0, max(1, n_hist - _BLOCK_SIZE_MONTHS + 1), size=blocks)
        eq_returns = np.concatenate([hist_returns[s:s + _BLOCK_SIZE_MONTHS] for s in starts])[:total_months]
        path = run_single_simulation(no_tax_params, eq_returns, debt_factors, inflation_factors, rng, _track_phases=False)
        portfolios.append(path.portfolio_values[-1])

    median_no_tax = float(np.median(portfolios))
    # If the taxed scenario is already ruined (median=0), report drag vs no-tax median.
    # But if both are 0 (high ruin in both), tax is not the marginal cause — return 0.
    if median_no_tax > 0 and median_with_tax >= 0:
        drag = (median_no_tax - median_with_tax) / median_no_tax * 100.0
        # Cap at 99%: >99% drag is practically meaningless and confusing in the UI
        return round(min(drag, 99.0), 1)
    return 0.0


# ── Optimization sweep ────────────────────────────────────────────────────────

@dataclass
class OptPoint:
    """Single equity% allocation optimization result."""
    equity_pct: float
    survival_30yr_pct: float
    median_final_corpus: float
    safe_withdrawal_rate_pct: float
    tax_drag_pct: float
    median_emergency_count: float


@dataclass
class OptResult:
    """Full optimization sweep output."""
    points: list[OptPoint]
    optimal_survival: OptPoint      # max survival probability
    optimal_median_corpus: OptPoint  # max median final corpus
    optimal_balanced: OptPoint       # best combined score


def run_optimization_sweep(
    base_params: RetirementParams,
    nifty_df: pd.DataFrame,
    equity_pct_range: Optional[list[float]] = None,
    n_sims_per_point: int = 500,
) -> OptResult:
    """
    Sweep equity% from 10 to 90 (step 5) to find the optimal asset allocation.

    Args:
        base_params:        Base simulation parameters (equity_pct will be overridden).
        nifty_df:           Nifty daily OHLCV data.
        equity_pct_range:   Custom equity% values to test. Default: 10, 15, ..., 90.
        n_sims_per_point:   Simulations per equity% value (fewer = faster).

    Returns:
        OptResult with all sweep points and three recommended allocations.
    """
    if equity_pct_range is None:
        equity_pct_range = list(range(10, 95, 5))  # 10, 15, ..., 90

    hist_returns = extract_monthly_returns(nifty_df)
    total_months = base_params.total_years * 12
    rng = np.random.default_rng(base_params.random_seed or 42)

    from src.data.debt_rates import get_debt_return_pct
    from src.simulator.leverage_sim import get_repo_rate

    latest_date = pd.Timestamp.today()
    debt_annual_pct = (
        base_params.future_debt_rate_override
        if base_params.future_debt_rate_override is not None
        else get_debt_return_pct(latest_date, base_params.debt_instrument)
    )
    debt_monthly_factor = (1.0 + debt_annual_pct / 100.0) ** (1.0 / 12.0)
    inflation_monthly_factor = (1.0 + base_params.inflation_rate_pct / 100.0) ** (1.0 / 12.0)
    debt_factors = np.full(total_months, debt_monthly_factor)
    inflation_factors = np.full(total_months, inflation_monthly_factor)

    n_hist = len(hist_returns)

    def _run_point(eq_pct: float) -> OptPoint:
        test_params = base_params.model_copy(update={
            "equity_pct": eq_pct,
            "n_simulations": n_sims_per_point,
        })
        portfolios = []
        ruin_months_list = []
        taxes = []
        emerg_counts = []

        for _ in range(n_sims_per_point):
            blocks = math.ceil(total_months / _BLOCK_SIZE_MONTHS)
            starts = rng.integers(0, max(1, n_hist - _BLOCK_SIZE_MONTHS + 1), size=blocks)
            eq_returns = np.concatenate([hist_returns[s:s + _BLOCK_SIZE_MONTHS] for s in starts])[:total_months]
            path = run_single_simulation(test_params, eq_returns, debt_factors, inflation_factors, rng)
            portfolios.append(path.portfolio_values[-1])
            ruin_months_list.append(path.ruin_month)
            taxes.append(path.total_tax_paid)
            emerg_counts.append(len(path.emergency_months))

        survival_30yr = sum(1 for r in ruin_months_list if r is None) / n_sims_per_point * 100.0
        median_corp = float(np.median(portfolios))

        # Quick SWR estimate (cheaper than full binary search)
        swr_pct = _compute_swr(
            test_params, hist_returns, debt_factors, inflation_factors, rng,
            n_sims_swr=150
        )

        # Tax drag (cheap estimate)
        tax_drag = _estimate_tax_drag(
            test_params, hist_returns, debt_factors, inflation_factors, rng, median_corp, n_sims=100
        )

        return OptPoint(
            equity_pct=eq_pct,
            survival_30yr_pct=round(survival_30yr, 1),
            median_final_corpus=round(median_corp, 0),
            safe_withdrawal_rate_pct=swr_pct,
            tax_drag_pct=tax_drag,
            median_emergency_count=round(float(np.median(emerg_counts)), 1),
        )

    logger.info("Running optimization sweep over %d equity%% values.", len(equity_pct_range))
    points = [_run_point(ep) for ep in equity_pct_range]

    # ── Find optima ───────────────────────────────────────────────────────
    opt_survival = max(points, key=lambda p: p.survival_30yr_pct)
    opt_median = max(points, key=lambda p: p.median_final_corpus)

    # Balanced: max(survival_pct * 0.6 + normalized_median * 0.4)
    max_surv = max(p.survival_30yr_pct for p in points) or 1
    max_med = max(p.median_final_corpus for p in points) or 1
    opt_balanced = max(
        points,
        key=lambda p: (p.survival_30yr_pct / max_surv) * 0.6
                      + (p.median_final_corpus / max_med) * 0.4
    )

    return OptResult(
        points=points,
        optimal_survival=opt_survival,
        optimal_median_corpus=opt_median,
        optimal_balanced=opt_balanced,
    )
