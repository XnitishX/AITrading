"""
Global configuration settings for the AITrading project.

Uses Pydantic for validation so bad config values are caught at import time.
All original module-level constants remain available for backward compatibility.
"""

import os
from pathlib import Path
from pydantic import BaseModel, field_validator, model_validator


# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CRYPTO_RAW_DIR = RAW_DATA_DIR / "crypto"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, CRYPTO_RAW_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Validated Configuration Model ─────────────────────────────────────────

class TradingConfig(BaseModel):
    """Validated trading configuration — raises on invalid values."""

    # Yahoo Finance tickers
    yf_nifty_ticker: str = "^NSEI"
    yf_vix_ticker: str = "^INDIAVIX"

    # CSV filenames
    nifty_csv_filename: str = "nifty50.csv"
    vix_csv_filename: str = "indiavix.csv"

    # Column names
    nifty_date_col: str = "Date"
    nifty_close_col: str = "Close"
    nifty_open_col: str = "Open"
    nifty_high_col: str = "High"
    nifty_low_col: str = "Low"
    nifty_volume_col: str = "Volume"
    vix_date_col: str = "Date"
    vix_close_col: str = "Close"

    # Prediction horizons
    prediction_horizons: dict[str, int] = {
        "next_day": 1,
        "next_week": 5,
        "next_month": 21,
    }

    # Backtesting defaults
    default_initial_capital: float = 1_000_000
    default_position_size: float = 0.95
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04

    # Statistical settings
    rolling_window_sizes: list[int] = [5, 10, 21, 63, 126, 252]
    probability_bins: int = 50
    confidence_levels: list[float] = [0.50, 0.75, 0.90, 0.95]

    # Logging
    log_level: str = os.environ.get("AITRADING_LOG_LEVEL", "INFO")

    @field_validator("default_initial_capital")
    @classmethod
    def capital_positive(cls, v):
        if v <= 0:
            raise ValueError(f"initial_capital must be > 0, got {v}")
        return v

    @field_validator("default_position_size")
    @classmethod
    def position_size_range(cls, v):
        if not 0 < v <= 1:
            raise ValueError(f"position_size must be in (0, 1], got {v}")
        return v

    @field_validator("default_stop_loss_pct", "default_take_profit_pct")
    @classmethod
    def pct_range(cls, v):
        if not 0 < v < 1:
            raise ValueError(f"stop/take pct must be in (0, 1), got {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, v):
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v}")
        return v.upper()

    @model_validator(mode="after")
    def stop_less_than_take(self):
        if self.default_stop_loss_pct >= self.default_take_profit_pct:
            raise ValueError(
                f"stop_loss ({self.default_stop_loss_pct}) must be < "
                f"take_profit ({self.default_take_profit_pct})"
            )
        return self


# Instantiate validated config once at import time
_config = TradingConfig()


# ── Backward-Compatible Module-Level Constants ────────────────────────────
# (kept so existing imports like `from config.settings import X` still work)

YF_NIFTY_TICKER = _config.yf_nifty_ticker
YF_VIX_TICKER = _config.yf_vix_ticker

NIFTY_CSV_FILENAME = _config.nifty_csv_filename
VIX_CSV_FILENAME = _config.vix_csv_filename

NIFTY_DATE_COL = _config.nifty_date_col
NIFTY_CLOSE_COL = _config.nifty_close_col
NIFTY_OPEN_COL = _config.nifty_open_col
NIFTY_HIGH_COL = _config.nifty_high_col
NIFTY_LOW_COL = _config.nifty_low_col
NIFTY_VOLUME_COL = _config.nifty_volume_col
VIX_DATE_COL = _config.vix_date_col
VIX_CLOSE_COL = _config.vix_close_col

PREDICTION_HORIZONS = _config.prediction_horizons

DEFAULT_INITIAL_CAPITAL = _config.default_initial_capital
DEFAULT_POSITION_SIZE = _config.default_position_size
DEFAULT_STOP_LOSS_PCT = _config.default_stop_loss_pct
DEFAULT_TAKE_PROFIT_PCT = _config.default_take_profit_pct

ROLLING_WINDOW_SIZES = _config.rolling_window_sizes
PROBABILITY_BINS = _config.probability_bins
CONFIDENCE_LEVELS = _config.confidence_levels

LOG_LEVEL = _config.log_level
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"

# ── Leverage Simulator Defaults ───────────────────────────────────────────────
LEVERAGE_DEFAULT_CAPITAL = 3_000_000        # 30 lakh
LEVERAGE_RATIOS_DEFAULT = [1.0, 1.5, 2.0, 2.5, 3.0]
LEVERAGE_CALL_OTM_PCTS_DEFAULT = [0.0, 0.025, 0.05, 0.075]
LEVERAGE_PUT_OTM_PCT_DEFAULT = 0.20         # 20% OTM quarterly rolling put
LEVERAGE_LIQUID_FUND_SPREAD = 0.005         # 50 bps below repo rate
LEVERAGE_DIVIDEND_YIELD = 0.013             # Nifty 50 historical average ~1.3%
LEVERAGE_USE_VOL_SKEW = True                # Apply Nifty put-skew premium on option pricing
LEVERAGE_ENABLE_TRANSACTION_COSTS = True    # Include NSE F&O STT, brokerage, exchange charges
LEVERAGE_STRIKE_ROUND_INCREMENT = 50.0      # Nearest valid NSE near-month strike (₹50)
LEVERAGE_MARGIN_CALL_THRESHOLD_PCT = 0.20   # Flag margin-call zone if equity < initial × this

# ── Retirement Monte Carlo Simulator Defaults ─────────────────────────────────
RETIREMENT_DEFAULT_CORPUS          = 10_000_000    # ₹1 crore starting corpus
RETIREMENT_DEFAULT_EQUITY_PCT      = 60.0          # 60% equity, 40% debt
RETIREMENT_DEFAULT_MONTHLY_WD      = 50_000        # ₹50,000 / month starting withdrawal
RETIREMENT_DEFAULT_HORIZON_YEARS   = 30            # 30-year simulation horizon
RETIREMENT_DEFAULT_N_SIMS          = 1_000         # Monte Carlo paths
RETIREMENT_DEFAULT_INFLATION_PCT   = 6.0           # India CPI default (%)
RETIREMENT_DEFAULT_DEBT_INSTRUMENT = "liquid_fund" # conservative default
RETIREMENT_DEFAULT_REPLENISH_YEARS = 5             # refill debt every 5 years
RETIREMENT_EMERGENCY_MONTHS_THRESH = 12            # emergency if debt < 12 months expenses
RETIREMENT_DEFAULT_TAX_BRACKET_PCT = 30.0          # income slab for debt interest tax
RETIREMENT_LTCG_EXEMPTION_INR      = 125_000       # ₹1.25 L LTCG annual exemption
RETIREMENT_LTCG_RATE               = 0.125         # 12.5 % LTCG tax rate (post Aug 2024)

# ── Crypto Leverage MC Simulator Defaults ─────────────────────────────────────
CRYPTO_TICKERS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "ripple": "XRP-USD",
    "cardano": "ADA-USD",
    "dogecoin": "DOGE-USD",
    "avalanche": "AVAX-USD",
    "polkadot": "DOT-USD",
    "tron": "TRX-USD",
    "chainlink": "LINK-USD",
    "litecoin": "LTC-USD",
    "bitcoin-cash": "BCH-USD",
}
CRYPTO_DEFAULT_BORROW_RATE_PCT = 10.0
CRYPTO_DEFAULT_HORIZON_YEARS   = 10
CRYPTO_DEFAULT_N_SIMS          = 1_000
CRYPTO_DEFAULT_INITIAL_CAPITAL = 1_000_000   # ₹10 Lakh / $10,000 baseline
CRYPTO_DEFAULT_BLOCK_SIZE_MONTHS = 6         # 6-month block bootstrap for crypto cycles
