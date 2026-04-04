"""
Backtesting Engine
──────────────────
A vectorised + event‑driven backtester for simple strategies on the
Nifty 50 index.  It tracks portfolio value, positions, PnL, and
common risk metrics (Sharpe, max drawdown, etc.).
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_POSITION_SIZE,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    NIFTY_CLOSE_COL,
)
from src.simulator.registry import register_strategy

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    direction: str  # "long" or "short"
    size: float  # number of units
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None  # "signal", "stop_loss", "take_profit", "trailing_stop", "max_holding"
    peak_price: Optional[float] = None  # best price since entry (for trailing stop)
    entry_bar: Optional[int] = None  # bar index at entry (for max holding period)
    exit_bar: Optional[int] = None  # bar index at exit (for duration calc)
    # MAE/MFE — Maximum Adverse / Maximum Favorable Excursion
    mae: Optional[float] = None   # worst unrealised P&L as % of entry price
    mfe: Optional[float] = None   # best unrealised P&L as % of entry price
    r_multiple: Optional[float] = None  # PnL / initial risk (entry→stop distance)

    def __repr__(self) -> str:
        pnl_str = f"₹{self.pnl:+,.2f}" if self.pnl is not None else "open"
        dur = f"{self.exit_bar - self.entry_bar}bars" if self.entry_bar is not None and self.exit_bar is not None else "?"
        return (
            f"Trade({self.direction} ₹{self.entry_price:,.2f}→"
            f"{'₹' + f'{self.exit_price:,.2f}' if self.exit_price else '?'} "
            f"PnL={pnl_str} {dur} {self.exit_reason or ''})"
        )


@dataclass
class BacktestResult:
    strategy_name: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_trade_pnl: float
    profit_factor: float
    max_consecutive_losses: int
    risk_reward_ratio: float  # avg winning trade / |avg losing trade|
    max_drawdown_duration: int  # longest underwater period in trading bars
    avg_trade_duration: float  # mean bars held per trade
    exposure_time_pct: float  # % of bars spent in a position
    benchmark_return_pct: float  # buy-and-hold Nifty annualized return
    alpha_pct: float  # strategy annual return minus benchmark annual return
    information_ratio: float  # alpha / tracking error (risk-adjusted alpha)
    # ── New risk metrics ──────────────────────────────────────────────
    omega_ratio: float = 0.0       # Omega ratio (gain/loss area above/below threshold)
    tail_ratio: float = 0.0       # 95th percentile / |5th percentile| of returns
    value_at_risk_95: float = 0.0  # 1-day 95% VaR as negative %
    cvar_95: float = 0.0          # Conditional VaR (Expected Shortfall) 95%
    avg_mae_pct: float = 0.0      # Average MAE across trades (%)
    avg_mfe_pct: float = 0.0      # Average MFE across trades (%)
    avg_r_multiple: float = 0.0   # Average R-multiple across trades
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)  # columns: Date, equity, drawdown, rolling_sharpe
    trades: list[Trade] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.strategy_name}: "
            f"return={self.total_return_pct:+.2f}% "
            f"sharpe={self.sharpe_ratio:.2f} "
            f"dd={self.max_drawdown_pct:.2f}% "
            f"trades={self.total_trades} "
            f"win={self.win_rate_pct:.1f}%)"
        )

    def summary(self) -> str:
        lines = [
            f"═══ Backtest: {self.strategy_name} ═══",
            f"  Period            : {self.equity_curve['Date'].iloc[0].date()} → "
            f"{self.equity_curve['Date'].iloc[-1].date()}",
            f"  Initial Capital   : ₹{self.initial_capital:,.0f}",
            f"  Final Capital     : ₹{self.final_capital:,.0f}",
            f"  Total Return      : {self.total_return_pct:+.2f}%",
            f"  Annual Return     : {self.annual_return_pct:+.2f}%",
            f"  Benchmark (B&H)   : {self.benchmark_return_pct:+.2f}%",
            f"  Alpha             : {self.alpha_pct:+.2f}%",
            f"  Information Ratio : {self.information_ratio:.2f}",
            f"  Sharpe Ratio      : {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio     : {self.sortino_ratio:.2f}",
            f"  Calmar Ratio      : {self.calmar_ratio:.2f}",
            f"  Omega Ratio       : {self.omega_ratio:.2f}",
            f"  Tail Ratio        : {self.tail_ratio:.2f}",
            f"  Max Drawdown      : {self.max_drawdown_pct:.2f}%",
            f"  VaR (95%)         : {self.value_at_risk_95:.2f}%",
            f"  CVaR (95%)        : {self.cvar_95:.2f}%",
            f"  Trades            : {self.total_trades}",
            f"  Win Rate          : {self.win_rate_pct:.1f}%",
            f"  Avg Trade PnL     : ₹{self.avg_trade_pnl:,.2f}",
            f"  Profit Factor     : {self.profit_factor:.2f}",
            f"  Risk/Reward       : {self.risk_reward_ratio:.2f}",
            f"  Avg R-Multiple    : {self.avg_r_multiple:.2f}",
            f"  Avg MAE           : {self.avg_mae_pct:.2f}%",
            f"  Avg MFE           : {self.avg_mfe_pct:.2f}%",
            f"  Max DD Duration   : {self.max_drawdown_duration} bars",
            f"  Avg Trade Duration: {self.avg_trade_duration:.1f} bars",
            f"  Max Consec Losses : {self.max_consecutive_losses}",
            f"  Exposure Time     : {self.exposure_time_pct:.1f}%",
        ]
        return "\n".join(lines)


# ── Strategy Protocol ────────────────────────────────────────────────────
# A strategy is a callable:
#   signal = strategy(row_index, row, history_df) -> int
# where signal ∈ {1 = buy, -1 = sell, 0 = hold}

SignalFunction = Callable[[int, pd.Series, pd.DataFrame], int]


# ── Built-in Strategies ──────────────────────────────────────────────────
#
# Six focused strategies using industry-standard parameters sourced from
# Investopedia / original authors.  Each factory returns a SignalFunction.
# ──────────────────────────────────────────────────────────────────────────


@register_strategy(
    "sma_crossover",
    description="Simple Moving Average Crossover (Golden/Death Cross)",
    param_schema={"fast_window": "Fast SMA period (default: 50, range: 5-100)", "slow_window": "Slow SMA period (default: 200, range: 20-500)"},
)
def sma_crossover_strategy(
    fast_window: int = 50,
    slow_window: int = 200,
) -> SignalFunction:
    """
    Golden Cross / Death Cross (Investopedia).
    Buy when the fast SMA crosses above the slow SMA; sell on the reverse.

    Standard setting: 50-day / 200-day.
    Source: https://www.investopedia.com/terms/g/goldencross.asp

    Precomputes SMA series once on first call for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < slow_window:
            return 0
        if "fast" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            prices = df[NIFTY_CLOSE_COL]
            _cache["fast"] = prices.rolling(fast_window).mean().values
            _cache["slow"] = prices.rolling(slow_window).mean().values

        fast_sma = _cache["fast"][idx]
        slow_sma = _cache["slow"][idx]
        prev_fast = _cache["fast"][idx - 1]
        prev_slow = _cache["slow"][idx - 1]

        if np.isnan(fast_sma) or np.isnan(slow_sma) or np.isnan(prev_fast) or np.isnan(prev_slow):
            return 0
        if prev_fast <= prev_slow and fast_sma > slow_sma:
            return 1  # golden cross → buy
        if prev_fast >= prev_slow and fast_sma < slow_sma:
            return -1  # death cross → sell
        return 0

    _strategy.__name__ = f"SMA_Crossover({fast_window},{slow_window})"
    return _strategy


@register_strategy(
    "rsi_mean_reversion",
    description="RSI Mean Reversion - buy oversold, sell overbought",
    param_schema={"oversold": "Buy threshold (default: 30, range: 10-45)", "overbought": "Sell threshold (default: 70, range: 55-90)"},
)
def rsi_mean_reversion_strategy(
    rsi_col: str = "rsi_14",
    oversold: float = 30.0,
    overbought: float = 70.0,
    rsi_period: int = 14,
) -> SignalFunction:
    """
    RSI Mean Reversion (J. Welles Wilder, 1978).
    Buy when RSI < oversold; sell when RSI > overbought.

    Standard: 14-period RSI, oversold = 30, overbought = 70.
    Source: https://www.investopedia.com/terms/r/rsi.asp

    Self-contained: precomputes RSI internally from Close prices.
    Falls back to the DataFrame's rsi_col if available (backward compat).
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < rsi_period + 1:
            return 0

        # Precompute RSI once on first call (or if df identity changes)
        if "rsi" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            if rsi_col in df.columns:
                _cache["rsi"] = df[rsi_col].values
            else:
                # Self-contained RSI computation from Close prices
                prices = df[NIFTY_CLOSE_COL]
                delta = prices.diff()
                gain = delta.clip(lower=0).rolling(rsi_period).mean()
                loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
                rs = gain / loss
                _cache["rsi"] = (100 - 100 / (1 + rs)).values

        rsi_val = _cache["rsi"][idx]
        if np.isnan(rsi_val):
            return 0
        if rsi_val < oversold:
            return 1
        if rsi_val > overbought:
            return -1
        return 0

    _strategy.__name__ = f"RSI_MeanReversion({oversold},{overbought})"
    return _strategy


@register_strategy(
    "macd_crossover",
    description="MACD Signal Line Crossover",
    param_schema={"fast_ema": "Fast EMA period (default: 12)", "slow_ema": "Slow EMA period (default: 26)", "signal_period": "Signal line period (default: 9)"},
)
def macd_crossover_strategy(
    fast_ema: int = 12,
    slow_ema: int = 26,
    signal_period: int = 9,
) -> SignalFunction:
    """
    MACD Crossover (Gerald Appel, 1970s).
    Buy when MACD line crosses above the signal line; sell on bearish crossover.

    Standard: 12/26/9 (fast EMA, slow EMA, signal EMA).
    Source: https://www.investopedia.com/terms/m/macd.asp

    Precomputes EMA/MACD/signal once on first call for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < slow_ema + signal_period:
            return 0
        if "macd_diff" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            prices = df[NIFTY_CLOSE_COL]
            ef = prices.ewm(span=fast_ema, adjust=False).mean()
            es = prices.ewm(span=slow_ema, adjust=False).mean()
            macd_line = ef - es
            sig_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            _cache["macd_diff"] = (macd_line - sig_line).values

        curr_diff = _cache["macd_diff"][idx]
        prev_diff = _cache["macd_diff"][idx - 1]

        if prev_diff <= 0 and curr_diff > 0:
            return 1  # bullish crossover
        if prev_diff >= 0 and curr_diff < 0:
            return -1  # bearish crossover
        return 0

    _strategy.__name__ = f"MACD_Crossover({fast_ema},{slow_ema},{signal_period})"
    return _strategy


@register_strategy(
    "bollinger_band",
    description="Bollinger Band Mean Reversion",
    param_schema={"window": "SMA period (default: 20)", "num_std": "Number of standard deviations (default: 2.0)"},
)
def bollinger_band_strategy(
    window: int = 20,
    num_std: float = 2.0,
) -> SignalFunction:
    """
    Bollinger Band Mean Reversion (John Bollinger, 1980s).
    Buy when price falls below the lower band; sell above the upper band.

    Standard: 20-day SMA ± 2 standard deviations.
    Source: https://www.investopedia.com/terms/b/bollingerbands.asp

    Precomputes bands once on first call for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < window:
            return 0
        if "upper" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            prices = df[NIFTY_CLOSE_COL]
            rmean = prices.rolling(window).mean()
            rstd = prices.rolling(window).std()
            _cache["upper"] = (rmean + num_std * rstd).values
            _cache["lower"] = (rmean - num_std * rstd).values
            _cache["prices"] = prices.values

        upper = _cache["upper"][idx]
        lower = _cache["lower"][idx]
        if np.isnan(upper) or np.isnan(lower):
            return 0
        price = _cache["prices"][idx]
        if price <= lower:
            return 1
        if price >= upper:
            return -1
        return 0

    _strategy.__name__ = f"Bollinger({window},{num_std}σ)"
    return _strategy


@register_strategy(
    "atr_breakout",
    description="ATR Breakout / Chandelier-style",
    param_schema={"sma_window": "SMA period (default: 20)", "atr_period": "ATR period (default: 14)", "atr_multiplier": "ATR multiplier (default: 1.5)"},
)
def atr_breakout_strategy(
    sma_window: int = 20,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
) -> SignalFunction:
    """
    ATR Breakout / Chandelier-style (J. Welles Wilder / Chuck LeBeau).
    Buy when price breaks above SMA + N×ATR; sell below SMA − N×ATR.

    Standard ATR period: 14 days.
    Source: https://www.investopedia.com/terms/a/atr.asp

    Precomputes SMA and ATR once on first call for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < max(sma_window, atr_period + 1):
            return 0
        if "sma" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            prices = df[NIFTY_CLOSE_COL]
            _cache["sma"] = prices.rolling(sma_window).mean().values
            _cache["prices"] = prices.values

            # Compute ATR vectorially
            atr_col = f"atr_{atr_period}"
            if atr_col in df.columns:
                _cache["atr"] = df[atr_col].values
            else:
                high = df["High"]
                low = df["Low"]
                close_prev = prices.shift(1)
                tr = pd.concat(
                    [high - low, (high - close_prev).abs(), (low - close_prev).abs()],
                    axis=1,
                ).max(axis=1)
                _cache["atr"] = tr.rolling(atr_period).mean().values

        sma = _cache["sma"][idx]
        atr = _cache["atr"][idx]
        if np.isnan(sma) or np.isnan(atr):
            return 0

        price = _cache["prices"][idx]
        if price > sma + atr_multiplier * atr:
            return 1
        if price < sma - atr_multiplier * atr:
            return -1
        return 0

    _strategy.__name__ = f"ATR_Breakout(sma={sma_window},atr={atr_period},×{atr_multiplier})"
    return _strategy


@register_strategy(
    "vix_regime",
    description="VIX Regime Filter - long in calm markets, flat in fearful",
    param_schema={"buy_below": "VIX threshold to go long (default: 15)", "sell_above": "VIX threshold to exit (default: 25)"},
)
def vix_regime_strategy(
    vix_col: str = "vix_close",
    buy_below: float = 15.0,
    sell_above: float = 25.0,
) -> SignalFunction:
    """
    VIX Regime Filter for India VIX with Hysteresis.
    Stay long in calm (low-VIX) markets; exit in fearful (high-VIX) markets.

    Hysteresis prevents whipsaw: once in "long" mode, only exit when VIX
    exceeds sell_above.  Once flat, only re-enter when VIX drops below
    buy_below.  The dead zone between the two thresholds produces no signal.

    India VIX historical mean ≈ 19.8.  Low < 15, High > 25 are
    commonly used thresholds for regime classification.

    Precomputes VIX array once for O(n) total cost.
    """
    _cache: dict = {}
    _state: dict = {"regime": "neutral", "df_id": None}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        # Reset state if DataFrame changed
        df_id = id(df)
        if _state["df_id"] != df_id:
            _state["regime"] = "neutral"
            _state["df_id"] = df_id
            _cache.clear()

        # Precompute VIX values once
        if "vix" not in _cache or len(_cache["vix"]) != len(df):
            if vix_col in df.columns:
                _cache["vix"] = df[vix_col].values
            else:
                _cache["vix"] = None

        if _cache["vix"] is None:
            return 0

        vix_val = _cache["vix"][idx]
        if np.isnan(vix_val):
            return 0

        # Hysteresis state machine
        if _state["regime"] == "neutral":
            if vix_val < buy_below:
                _state["regime"] = "long"
                return 1   # enter long — calm market
            elif vix_val > sell_above:
                _state["regime"] = "fearful"
                return 0   # stay flat — already fearful
        elif _state["regime"] == "long":
            if vix_val > sell_above:
                _state["regime"] = "fearful"
                return -1  # exit long — market turning fearful
        elif _state["regime"] == "fearful":
            if vix_val < buy_below:
                _state["regime"] = "long"
                return 1   # enter long — fear subsiding
        return 0

    _strategy.__name__ = f"VIX_Regime({buy_below},{sell_above})"
    return _strategy


@register_strategy(
    "ema_crossover",
    description="Exponential Moving Average Crossover",
    param_schema={"fast_span": "Fast EMA span (default: 12)", "slow_span": "Slow EMA span (default: 26)"},
)
def ema_crossover_strategy(
    fast_span: int = 12,
    slow_span: int = 26,
) -> SignalFunction:
    """
    EMA Crossover — Exponential Moving Average cross signals.
    Faster‑reacting than SMA crossover due to exponential weighting.
    Buy when the fast EMA crosses above the slow EMA; sell on the reverse.

    Common pairs: 9/21, 12/26, 20/50.
    Source: https://www.investopedia.com/terms/e/ema.asp

    Precomputes EMAs once on first call for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < slow_span + 1:
            return 0
        if "fast" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            prices = df[NIFTY_CLOSE_COL]
            _cache["fast"] = prices.ewm(span=fast_span, adjust=False).mean().values
            _cache["slow"] = prices.ewm(span=slow_span, adjust=False).mean().values

        if _cache["fast"][idx - 1] <= _cache["slow"][idx - 1] and _cache["fast"][idx] > _cache["slow"][idx]:
            return 1
        if _cache["fast"][idx - 1] >= _cache["slow"][idx - 1] and _cache["fast"][idx] < _cache["slow"][idx]:
            return -1
        return 0

    _strategy.__name__ = f"EMA_Crossover({fast_span},{slow_span})"
    return _strategy


@register_strategy(
    "stochastic_oscillator",
    description="Stochastic Oscillator %K/%D Crossover",
    param_schema={"k_period": "%K period (default: 14)", "d_period": "%D smoothing (default: 3)", "oversold": "Oversold level (default: 20)", "overbought": "Overbought level (default: 80)"},
)
def stochastic_oscillator_strategy(
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20.0,
    overbought: float = 80.0,
) -> SignalFunction:
    """
    Stochastic Oscillator (George Lane, 1950s).
    Buy when %K crosses above %D below the oversold level;
    sell when %K crosses below %D above the overbought level.

    Standard: %K = 14, %D = 3‑period SMA of %K.
    Source: https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Precomputes %K and %D series once on first call for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < k_period + d_period:
            return 0
        if "k" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            highs = df["High"].rolling(k_period).max()
            lows = df["Low"].rolling(k_period).min()
            denom = highs - lows
            denom = denom.replace(0, np.nan)
            k_series = (df[NIFTY_CLOSE_COL] - lows) / denom * 100
            k_series = k_series.fillna(50)  # neutral if flat range
            d_series = k_series.rolling(d_period).mean()
            _cache["k"] = k_series.values
            _cache["d"] = d_series.values

        k_val = _cache["k"][idx]
        d_val = _cache["d"][idx]
        prev_k = _cache["k"][idx - 1]
        prev_d = _cache["d"][idx - 1]

        if np.isnan(k_val) or np.isnan(d_val) or np.isnan(prev_k) or np.isnan(prev_d):
            return 0

        # Bullish: %K crosses above %D below oversold
        if prev_k <= prev_d and k_val > d_val and d_val < oversold:
            return 1
        # Bearish: %K crosses below %D above overbought
        if prev_k >= prev_d and k_val < d_val and d_val > overbought:
            return -1
        return 0

    _strategy.__name__ = f"Stochastic({k_period},{d_period},{oversold}/{overbought})"
    return _strategy


@register_strategy(
    "mean_reversion_zscore",
    description="Z-Score Mean Reversion",
    param_schema={"lookback": "Lookback period (default: 20)", "entry_z": "Entry z-score (default: -2.0)", "exit_z": "Exit z-score (default: 0.0)"},
)
def mean_reversion_zscore_strategy(
    lookback: int = 20,
    entry_z: float = -2.0,
    exit_z: float = 0.0,
) -> SignalFunction:
    """
    Z-Score Mean Reversion — buy when price falls N standard deviations
    below the rolling mean; sell when it reverts back to the mean.

    Uses the Bollinger z-score concept: z = (price − SMA) / σ.
    Entry long at z ≤ entry_z (deeply oversold); exit long at z ≥ exit_z.
    Entry short at z ≥ |entry_z| (deeply overbought); exit short at z ≤ exit_z.

    Source: https://www.investopedia.com/terms/m/meanreversion.asp

    Precomputes z-scores once on first call for O(n) total cost.
    State is reset when the strategy is reused on a different DataFrame.
    """
    # Track whether we conceptually have a position for signal logic
    _state: dict = {"pos": 0, "df_id": None}
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        # Reset state if the underlying DataFrame has changed (reuse safety)
        df_id = id(df)
        if _state["df_id"] != df_id:
            _state["pos"] = 0
            _state["df_id"] = df_id
            _cache.clear()

        if idx < lookback:
            return 0

        # Precompute z-scores once
        if "z" not in _cache or len(_cache["z"]) != len(df):
            prices = df[NIFTY_CLOSE_COL]
            sma = prices.rolling(lookback).mean()
            std = prices.rolling(lookback).std()
            _cache["z"] = ((prices - sma) / std.replace(0, np.nan)).values

        z = _cache["z"][idx]
        if np.isnan(z):
            return 0

        if _state["pos"] == 0:
            # No position: look for entry
            if z <= entry_z:
                _state["pos"] = 1
                return 1   # deeply undervalued → buy
            if z >= abs(entry_z):
                _state["pos"] = -1
                return -1  # deeply overvalued → sell short
        elif _state["pos"] == 1:
            # Long position: exit when z reverts to exit_z or above
            if z >= exit_z:
                _state["pos"] = 0
                return -1  # close long
        elif _state["pos"] == -1:
            # Short position: exit when z reverts to exit_z or below
            if z <= exit_z:
                _state["pos"] = 0
                return 1   # close short
        return 0

    _strategy.__name__ = f"MeanRev_Z({lookback},{entry_z},{exit_z})"
    return _strategy


@register_strategy(
    "macd_histogram",
    description="MACD Histogram Reversal",
    param_schema={"fast_ema": "Fast EMA period (default: 12)", "slow_ema": "Slow EMA period (default: 26)", "signal_period": "Signal line period (default: 9)"},
)
def macd_histogram_strategy(
    fast_ema: int = 12,
    slow_ema: int = 26,
    signal_period: int = 9,
) -> SignalFunction:
    """
    MACD Histogram Reversal (Thomas Aspray, 1986).
    Buy when the MACD histogram turns from negative to positive
    (bearish momentum waning, bullish reversal imminent).
    Sell on the reverse.

    This is a more aggressive variation of MACD that catches
    momentum shifts earlier than the standard signal-line crossover.
    Source: https://www.investopedia.com/terms/m/macdhistogram.asp

    Precomputes histogram once on first call for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        if idx < slow_ema + signal_period + 1:
            return 0
        if "hist" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            prices = df[NIFTY_CLOSE_COL]
            ema_f = prices.ewm(span=fast_ema, adjust=False).mean()
            ema_s = prices.ewm(span=slow_ema, adjust=False).mean()
            macd_line = ema_f - ema_s
            sig_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            _cache["hist"] = (macd_line - sig_line).values

        curr_h = _cache["hist"][idx]
        prev_h = _cache["hist"][idx - 1]

        if prev_h < 0 and curr_h >= 0:
            return 1   # histogram turns positive → bullish
        if prev_h > 0 and curr_h <= 0:
            return -1  # histogram turns negative → bearish
        return 0

    _strategy.__name__ = f"MACD_Hist({fast_ema},{slow_ema},{signal_period})"
    return _strategy


@register_strategy(
    "composite_sniper",
    description="Multi-factor Composite (trend + RSI + Bollinger + VIX)",
    param_schema={"sma_period": "Trend SMA (default: 50)", "rsi_period": "RSI period (default: 14)", "rsi_oversold": "RSI oversold (default: 35)", "rsi_overbought": "RSI overbought (default: 65)", "vix_calm_threshold": "VIX calm threshold (default: 20)", "bb_window": "Bollinger window (default: 20)", "bb_num_std": "Bollinger std devs (default: 2.0)"},
)
def composite_sniper_strategy(
    sma_period: int = 50,
    rsi_period: int = 14,
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0,
    vix_calm_threshold: float = 20.0,
    bb_window: int = 20,
    bb_num_std: float = 2.0,
) -> SignalFunction:
    """
    Multi-Factor Composite "Sniper" Strategy
    ──────────────────────────────────────────
    Generates high-conviction trades only when multiple independent signals
    align simultaneously. Designed for fewer but higher-quality entries.

    BUY when ALL of:
      1. Trend: Price above 50-day SMA (uptrend confirmed)
      2. Pullback: RSI < 35 (oversold within uptrend — dip buy)
      3. Value: Price near/below Bollinger lower band (statistically cheap)
      4. Regime: VIX < calm threshold OR VIX declining (fear subsiding)

    SELL when ANY of:
      1. RSI > 65 (overbought — take profit)
      2. Price above Bollinger upper band AND RSI > 60 (extended + overheated)

    This is a bread-and-butter trend-pullback strategy with VIX regime filter
    that avoids trading during panic markets and catches high-probability
    bounce plays in confirmed uptrends.

    Precomputes all indicators once for O(n) total cost.
    """
    _cache: dict = {}

    def _strategy(idx: int, row: pd.Series, df: pd.DataFrame) -> int:
        min_warmup = max(sma_period, bb_window, rsi_period) + 1
        if idx < min_warmup:
            return 0

        # Lazy precompute all indicators once
        if "sma" not in _cache or _cache.get("_df_id") != id(df):
            _cache.clear()
            _cache["_df_id"] = id(df)
            prices = df[NIFTY_CLOSE_COL]
            _cache["prices"] = prices.values

            # SMA for trend
            _cache["sma"] = prices.rolling(sma_period).mean().values

            # RSI
            delta = prices.diff()
            gain = delta.clip(lower=0).rolling(rsi_period).mean()
            loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
            rs = gain / loss
            _cache["rsi"] = (100 - 100 / (1 + rs)).values

            # Bollinger Bands
            bb_sma = prices.rolling(bb_window).mean()
            bb_std = prices.rolling(bb_window).std()
            _cache["bb_upper"] = (bb_sma + bb_num_std * bb_std).values
            _cache["bb_lower"] = (bb_sma - bb_num_std * bb_std).values

            # VIX (if available)
            if "vix_close" in df.columns:
                _cache["vix"] = df["vix_close"].values
                _cache["vix_sma"] = df["vix_close"].rolling(10).mean().values
            else:
                _cache["vix"] = None

        price = _cache["prices"][idx]
        sma_val = _cache["sma"][idx]
        rsi_val = _cache["rsi"][idx]
        bb_upper = _cache["bb_upper"][idx]
        bb_lower = _cache["bb_lower"][idx]

        # Handle NaN
        if np.isnan(sma_val) or np.isnan(rsi_val) or np.isnan(bb_lower):
            return 0

        # VIX regime check
        vix_ok = True
        if _cache["vix"] is not None:
            vix_val = _cache["vix"][idx]
            vix_sma = _cache["vix_sma"][idx]
            if not np.isnan(vix_val) and not np.isnan(vix_sma):
                # Calm market OR VIX declining from elevated levels
                vix_ok = vix_val < vix_calm_threshold or vix_val < vix_sma

        # BUY: Trend up + oversold pullback + near BB lower + VIX ok
        if (price > sma_val and           # uptrend
                rsi_val < rsi_oversold and     # oversold
                price <= bb_lower * 1.02 and   # within 2% of lower band
                vix_ok):                       # regime filter
            return 1

        # SELL: Overbought or extended
        if rsi_val > rsi_overbought:
            return -1
        if price > bb_upper and rsi_val > 60:
            return -1

        return 0

    _strategy.__name__ = (
        f"Sniper(sma={sma_period},rsi={rsi_oversold}/{rsi_overbought},"
        f"bb={bb_window}/{bb_num_std},vix<{vix_calm_threshold})"
    )
    return _strategy


# ── Backtesting Engine ───────────────────────────────────────────────────

class Backtester:
    """
    Event‑driven backtester that walks forward through a DataFrame
    and executes a strategy function on each bar.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: SignalFunction,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        position_size_pct: float = DEFAULT_POSITION_SIZE,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        trailing_stop_pct: float = 0.0,
        slippage_pct: float = 0.0001,  # 0.01% default slippage per side
        commission_pct: float = 0.0002,  # 0.02% round-trip (brokerage + STT)
        vol_target: float = 0.0,  # annualised vol target for position sizing (0 = disabled)
        cooldown_bars: int = 0,  # min bars between trade exit and next entry (0 = disabled)
        max_holding_bars: int = 0,  # max bars to hold a position (0 = unlimited)
        allow_short: bool = False,
        dd_scale_threshold: float = 0.0,  # drawdown % threshold to start reducing position (0 = disabled)
    ):
        self.df = df.copy().reset_index(drop=True)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.vol_target = vol_target
        self.cooldown_bars = cooldown_bars
        self.max_holding_bars = max_holding_bars
        self.allow_short = allow_short
        self.dd_scale_threshold = dd_scale_threshold

        self.cash = initial_capital
        self.position: Optional[Trade] = None  # single‑position model
        self.trades: list[Trade] = []
        self.equity_records: list[dict] = []
        self._bars_in_market: int = 0  # count bars where a position is open
        self._bars_since_exit: int = 999_999  # bars since last trade exit (large = allow first trade)
        self._current_idx: int = 0  # current bar index (for _close_position)
        self._peak_equity: float = initial_capital  # track equity high-water mark
        self._trade_low_price: float = 0.0   # track lowest price since entry (for MAE)
        self._trade_high_price: float = 0.0  # track highest price since entry (for MFE)

        # Precompute 21-day rolling annualised vol for vol-scaling
        if vol_target > 0 and NIFTY_CLOSE_COL in self.df.columns:
            log_ret = np.log(self.df[NIFTY_CLOSE_COL] / self.df[NIFTY_CLOSE_COL].shift(1))
            self._rolling_vol = (log_ret.rolling(21).std() * np.sqrt(252)).values
        else:
            self._rolling_vol = None

    # ── Core loop ────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        logger.info(
            "Running backtest: %s on %d bars",
            getattr(self.strategy, "__name__", "custom"),
            len(self.df),
        )
        # Daily risk-free rate for idle-cash interest (India 10Y G-Sec ~6.5%)
        rf_daily = 0.065 / 252

        # Pre-extract arrays to avoid pd.Series creation per row (3-5× faster)
        _prices = self.df[NIFTY_CLOSE_COL].values
        _dates = self.df["Date"].values
        n_bars = len(self.df)

        for idx in range(n_bars):
            price = float(_prices[idx])
            date = _dates[idx]
            self._current_idx = idx

            # -- Accrue interest on idle cash when flat ----------------
            if self.position is None:
                self.cash *= (1 + rf_daily)
                self._bars_since_exit += 1

            # -- Check stop‑loss / take‑profit on open position --------
            if self.position is not None:
                self._check_exit_conditions(idx, price, date)

            # -- Generate signal ---------------------------------------
            signal = self.strategy(idx, None, self.df)

            # -- Execute signal (with cooldown check) ------------------
            can_open = self._bars_since_exit >= self.cooldown_bars
            if signal == 1 and self.position is None and can_open:
                self._open_position(idx, date, price, "long")
            elif signal == -1 and self.position is not None and self.position.direction == "long":
                self._close_position(date, price, "signal")
            elif signal == -1 and self.position is None and self.allow_short and can_open:
                self._open_position(idx, date, price, "short")
            elif signal == 1 and self.position is not None and self.position.direction == "short":
                self._close_position(date, price, "signal")

            # -- Track exposure time ------------------------------------
            if self.position is not None:
                self._bars_in_market += 1

            # -- Record equity -----------------------------------------
            equity = self._current_equity(price)
            self.equity_records.append({"Date": date, "equity": equity})

        # Close any remaining position at last price
        if self.position is not None:
            self._close_position(_dates[-1], float(_prices[-1]), "end_of_data")

        return self._compile_results()

    # ── Position Management ──────────────────────────────────────────

    def _open_position(self, idx, date, price, direction):
        # Apply slippage: buy at a slightly higher price, sell at slightly lower
        if direction == "long":
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        # Volatility-scaled position sizing
        pct = self.position_size_pct
        if self._rolling_vol is not None and idx < len(self._rolling_vol):
            current_vol = self._rolling_vol[idx]
            if not np.isnan(current_vol) and current_vol > 0:
                vol_scalar = self.vol_target / current_vol
                pct = min(self.position_size_pct, pct * vol_scalar)
                pct = max(0.05, pct)  # floor at 5%

        # Drawdown-based position scaling: reduce size when in drawdown
        if self.dd_scale_threshold > 0:
            current_equity = self._current_equity(price)
            self._peak_equity = max(self._peak_equity, current_equity)
            dd_pct = (self._peak_equity - current_equity) / self._peak_equity
            if dd_pct > self.dd_scale_threshold:
                # Linear scale-down: at 2× threshold, position = 50% of normal
                scale = max(0.25, 1.0 - (dd_pct - self.dd_scale_threshold) / self.dd_scale_threshold)
                pct *= scale
                logger.debug("DD scaling: dd=%.2f%%, scale=%.2f", dd_pct * 100, scale)

        alloc = self.cash * pct
        # Deduct entry commission
        commission = alloc * self.commission_pct
        alloc_after = alloc - commission
        size = alloc_after / fill_price
        self.position = Trade(
            entry_date=date,
            entry_price=fill_price,
            direction=direction,
            size=size,
            peak_price=fill_price,
            entry_bar=idx,
        )
        # Initialise MAE/MFE tracking
        self._trade_high_price = fill_price
        self._trade_low_price = fill_price
        self.cash -= alloc
        logger.debug("OPEN %s %.2f units @ ₹%.2f (slip ₹%.2f, size=%.0f%%) on %s", direction, size, fill_price, fill_price - price, pct * 100, date)

    def _close_position(self, date, price, reason):
        t = self.position
        t.exit_bar = self._current_idx  # record exit bar for duration calc

        # Compute MAE/MFE before exit
        if t.direction == "long":
            t.mae = (self._trade_low_price - t.entry_price) / t.entry_price * 100
            t.mfe = (self._trade_high_price - t.entry_price) / t.entry_price * 100
        else:
            t.mae = (t.entry_price - self._trade_high_price) / t.entry_price * 100
            t.mfe = (t.entry_price - self._trade_low_price) / t.entry_price * 100

        # R-multiple: PnL / initial risk
        initial_risk = t.entry_price * self.stop_loss_pct
        if initial_risk > 0:
            price_pnl = (price - t.entry_price) if t.direction == "long" else (t.entry_price - price)
            t.r_multiple = price_pnl / initial_risk
        else:
            t.r_multiple = 0.0

        # Apply slippage on exit
        if t.direction == "long":
            fill_price = price * (1 - self.slippage_pct)
        else:
            fill_price = price * (1 + self.slippage_pct)

        if t.direction == "long":
            pnl = (fill_price - t.entry_price) * t.size
        else:
            pnl = (t.entry_price - fill_price) * t.size

        # Deduct exit commission
        exit_value = t.size * fill_price
        commission = exit_value * self.commission_pct
        pnl -= commission

        t.exit_date = date
        t.exit_price = fill_price
        t.pnl = pnl
        t.exit_reason = reason
        self.trades.append(t)
        self.cash += exit_value - commission  # return capital minus commission
        self.position = None
        self._bars_since_exit = 0  # reset cooldown counter
        logger.debug("CLOSE %s @ ₹%.2f  PnL=₹%.2f  reason=%s", t.direction, fill_price, pnl, reason)

    def _check_exit_conditions(self, idx, price, date):
        t = self.position

        # Track MAE / MFE high-water / low-water prices
        self._trade_high_price = max(self._trade_high_price, price)
        self._trade_low_price = min(self._trade_low_price, price)

        # Max holding period: force exit if position held too long
        if self.max_holding_bars > 0 and t.entry_bar is not None:
            bars_held = idx - t.entry_bar
            if bars_held >= self.max_holding_bars:
                self._close_position(date, price, "max_holding")
                return

        if t.direction == "long":
            # Update peak price for trailing stop
            if t.peak_price is None or price > t.peak_price:
                t.peak_price = price
            change = (price - t.entry_price) / t.entry_price
            if change <= -self.stop_loss_pct:
                self._close_position(date, price, "stop_loss")
            elif self.trailing_stop_pct > 0 and t.peak_price > t.entry_price:
                # Trailing stop: how far has price dropped from peak?
                pullback = (t.peak_price - price) / t.peak_price
                if pullback >= self.trailing_stop_pct:
                    self._close_position(date, price, "trailing_stop")
            elif self.take_profit_pct > 0 and change >= self.take_profit_pct:
                self._close_position(date, price, "take_profit")
        else:  # short
            # Update peak (trough) for trailing stop
            if t.peak_price is None or price < t.peak_price:
                t.peak_price = price
            change = (t.entry_price - price) / t.entry_price
            if change <= -self.stop_loss_pct:
                self._close_position(date, price, "stop_loss")
            elif self.trailing_stop_pct > 0 and t.peak_price < t.entry_price:
                pullback = (price - t.peak_price) / t.peak_price
                if pullback >= self.trailing_stop_pct:
                    self._close_position(date, price, "trailing_stop")
            elif self.take_profit_pct > 0 and change >= self.take_profit_pct:
                self._close_position(date, price, "take_profit")

    def _current_equity(self, current_price: float) -> float:
        eq = self.cash
        if self.position is not None:
            if self.position.direction == "long":
                eq += self.position.size * current_price
            else:
                eq += self.position.size * (2 * self.position.entry_price - current_price)
        return eq

    # ── Results Compilation ──────────────────────────────────────────

    def _compile_results(self) -> BacktestResult:
        eq = pd.DataFrame(self.equity_records)
        eq["peak"] = eq["equity"].cummax()
        eq["drawdown"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100

        # Rolling Sharpe (63-day ≈ 3 months) — key for identifying regime breakdowns
        rf_daily = 0.065 / 252
        daily_returns = eq["equity"].pct_change().dropna()
        _rolling_excess = daily_returns - rf_daily
        rolling_mean = _rolling_excess.rolling(63, min_periods=21).mean()
        rolling_std = _rolling_excess.rolling(63, min_periods=21).std()
        eq["rolling_sharpe"] = (rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)).fillna(0)

        final = eq["equity"].iloc[-1]
        total_ret = (final - self.initial_capital) / self.initial_capital * 100
        n_days = (eq["Date"].iloc[-1] - eq["Date"].iloc[0]).days
        annual_ret = ((final / self.initial_capital) ** (365.25 / max(n_days, 1)) - 1) * 100

        excess_returns = daily_returns - rf_daily
        sharpe = (
            float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))
            if excess_returns.std() > 0
            else 0.0
        )

        winners = [t for t in self.trades if t.pnl and t.pnl > 0]
        losers = [t for t in self.trades if t.pnl and t.pnl <= 0]
        total_trades = len(self.trades)
        win_rate = len(winners) / total_trades * 100 if total_trades else 0
        avg_pnl = sum(t.pnl for t in self.trades if t.pnl) / total_trades if total_trades else 0
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")  # all winners, no losers
        else:
            profit_factor = 0.0  # no trades or all flat

        # Sortino ratio: penalises downside vol only
        downside = excess_returns[excess_returns < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else 0.0
        sortino = (
            float(excess_returns.mean() / downside_std * np.sqrt(252))
            if downside_std > 0
            else 0.0
        )

        # Calmar ratio: annual return / max drawdown
        max_dd_abs = abs(eq["drawdown"].min())
        calmar = annual_ret / max_dd_abs if max_dd_abs > 0 else 0.0

        # ── New risk metrics ──────────────────────────────────────────
        # Omega ratio: sum of gains above threshold / sum of losses below threshold
        threshold = rf_daily
        gains_above = excess_returns[excess_returns > threshold].sum()
        losses_below = abs(excess_returns[excess_returns <= threshold].sum())
        omega = float(gains_above / losses_below) if losses_below > 0 else 0.0

        # Tail ratio: right tail / left tail of returns distribution
        if len(daily_returns) > 10:
            p95 = float(np.percentile(daily_returns, 95))
            p5 = float(np.percentile(daily_returns, 5))
            tail_ratio = abs(p95 / p5) if p5 != 0 else 0.0
        else:
            tail_ratio = 0.0

        # Value at Risk (95%) — worst expected 1-day loss (parametric)
        if len(daily_returns) > 10:
            var_95 = float(np.percentile(daily_returns, 5)) * 100  # 5th percentile as %
        else:
            var_95 = 0.0

        # Conditional VaR (Expected Shortfall): avg loss beyond VaR
        if len(daily_returns) > 10:
            threshold_val = np.percentile(daily_returns, 5)
            tail_losses = daily_returns[daily_returns <= threshold_val]
            cvar_95 = float(tail_losses.mean()) * 100 if len(tail_losses) > 0 else var_95
        else:
            cvar_95 = 0.0

        # Average MAE / MFE / R-multiple across trades
        mae_vals = [t.mae for t in self.trades if t.mae is not None]
        mfe_vals = [t.mfe for t in self.trades if t.mfe is not None]
        r_vals = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else 0.0
        avg_mfe = sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0.0
        avg_r = sum(r_vals) / len(r_vals) if r_vals else 0.0

        # Max drawdown duration: longest period underwater (bars below peak)
        underwater = (eq["equity"] < eq["peak"]).values
        if underwater.any():
            groups = (~underwater).cumsum()
            uw_series = pd.Series(underwater)
            max_dd_dur = int(uw_series.groupby(groups).sum().max())
        else:
            max_dd_dur = 0

        # Max consecutive losses
        max_consec_losses = 0
        current_streak = 0
        for t in self.trades:
            if t.pnl is not None and t.pnl <= 0:
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0

        # Risk-reward ratio: avg win / |avg loss|
        avg_win = (sum(t.pnl for t in winners) / len(winners)) if winners else 0.0
        avg_loss_abs = (abs(sum(t.pnl for t in losers)) / len(losers)) if losers else 0.0
        risk_reward = avg_win / avg_loss_abs if avg_loss_abs > 0 else (float('inf') if avg_win > 0 else 0.0)

        # Average trade duration (in bars)
        durations = [
            (t.exit_bar - t.entry_bar)
            for t in self.trades
            if t.entry_bar is not None and t.exit_bar is not None
        ]
        avg_trade_dur = sum(durations) / len(durations) if durations else 0.0

        # Exposure time: fraction of bars spent in a position
        total_bars = len(self.df)
        exposure_pct = (self._bars_in_market / total_bars * 100) if total_bars > 0 else 0.0

        # Buy-and-hold benchmark: annualised return of just holding Nifty
        first_price = self.df[NIFTY_CLOSE_COL].iloc[0]
        last_price = self.df[NIFTY_CLOSE_COL].iloc[-1]
        benchmark_total = (last_price - first_price) / first_price
        benchmark_annual = ((1 + benchmark_total) ** (365.25 / max(n_days, 1)) - 1) * 100
        alpha = annual_ret - benchmark_annual

        # Information Ratio: alpha / tracking error
        benchmark_equity = self.initial_capital * (self.df[NIFTY_CLOSE_COL].values / first_price)
        eq["benchmark_equity"] = benchmark_equity

        benchmark_daily = pd.Series(benchmark_equity).pct_change().dropna()
        min_len = min(len(daily_returns), len(benchmark_daily))
        active_diff = daily_returns.values[:min_len] - benchmark_daily.values[:min_len]
        tracking_error = float(np.std(active_diff)) * np.sqrt(252) if len(active_diff) > 1 else 0.0
        information_ratio = (alpha / 100) / tracking_error if tracking_error > 0 else 0.0

        strategy_name = getattr(self.strategy, "__name__", "custom")

        result = BacktestResult(
            strategy_name=strategy_name,
            initial_capital=self.initial_capital,
            final_capital=final,
            total_return_pct=total_ret,
            annual_return_pct=annual_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd_abs,
            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate_pct=win_rate,
            avg_trade_pnl=avg_pnl,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consec_losses,
            risk_reward_ratio=risk_reward,
            max_drawdown_duration=max_dd_dur,
            avg_trade_duration=avg_trade_dur,
            exposure_time_pct=exposure_pct,
            benchmark_return_pct=benchmark_annual,
            alpha_pct=alpha,
            information_ratio=information_ratio,
            omega_ratio=omega,
            tail_ratio=tail_ratio,
            value_at_risk_95=var_95,
            cvar_95=cvar_95,
            avg_mae_pct=avg_mae,
            avg_mfe_pct=avg_mfe,
            avg_r_multiple=avg_r,
            equity_curve=eq[["Date", "equity", "drawdown", "rolling_sharpe", "benchmark_equity"]],
            trades=self.trades,
        )
        logger.info("\n%s", result.summary())
        return result


# ── Walk-Forward Analysis ────────────────────────────────────────────────

def walk_forward_analysis(
    df: pd.DataFrame,
    strategy_fn_factory: Callable,
    strategy_params: dict,
    n_splits: int = 5,
    train_pct: float = 0.70,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> dict:
    """
    Walk-Forward Analysis: split data into n sequential folds, train on
    the first train_pct of each fold and test on the remainder.

    Returns summary metrics for in-sample vs out-of-sample performance
    to detect overfitting.
    """
    total_rows = len(df)
    fold_size = total_rows // n_splits

    results = []
    for i in range(n_splits):
        start = i * fold_size
        end = min(start + fold_size, total_rows)
        fold = df.iloc[start:end].copy().reset_index(drop=True)

        split_idx = int(len(fold) * train_pct)
        if split_idx < 100 or (len(fold) - split_idx) < 50:
            continue

        train_df = fold.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = fold.iloc[split_idx:].copy().reset_index(drop=True)

        strat = strategy_fn_factory(**strategy_params)

        # In-sample backtest
        bt_train = Backtester(
            train_df, strat,
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        train_result = bt_train.run()

        # Out-of-sample backtest (re-create strategy to reset state)
        strat_oos = strategy_fn_factory(**strategy_params)
        bt_test = Backtester(
            test_df, strat_oos,
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        test_result = bt_test.run()

        results.append({
            "fold": i + 1,
            "train_start": str(train_df["Date"].iloc[0].date()),
            "train_end": str(train_df["Date"].iloc[-1].date()),
            "test_start": str(test_df["Date"].iloc[0].date()),
            "test_end": str(test_df["Date"].iloc[-1].date()),
            "train_return_pct": round(train_result.total_return_pct, 2),
            "test_return_pct": round(test_result.total_return_pct, 2),
            "train_sharpe": round(train_result.sharpe_ratio, 2),
            "test_sharpe": round(test_result.sharpe_ratio, 2),
            "train_max_dd": round(train_result.max_drawdown_pct, 2),
            "test_max_dd": round(test_result.max_drawdown_pct, 2),
            "train_trades": train_result.total_trades,
            "test_trades": test_result.total_trades,
            "train_win_rate": round(train_result.win_rate_pct, 1),
            "test_win_rate": round(test_result.win_rate_pct, 1),
        })

    if not results:
        return {"error": "Not enough data for walk-forward splits."}

    avg_train_ret = sum(r["train_return_pct"] for r in results) / len(results)
    avg_test_ret = sum(r["test_return_pct"] for r in results) / len(results)
    avg_train_sharpe = sum(r["train_sharpe"] for r in results) / len(results)
    avg_test_sharpe = sum(r["test_sharpe"] for r in results) / len(results)

    overfit_ratio = (avg_train_ret - avg_test_ret) / max(abs(avg_train_ret), 0.01)

    return {
        "n_splits": len(results),
        "train_pct": train_pct,
        "folds": results,
        "avg_train_return_pct": round(avg_train_ret, 2),
        "avg_test_return_pct": round(avg_test_ret, 2),
        "avg_train_sharpe": round(avg_train_sharpe, 2),
        "avg_test_sharpe": round(avg_test_sharpe, 2),
        "overfit_ratio": round(overfit_ratio, 2),
        "verdict": (
            "Low risk of overfitting" if overfit_ratio < 0.3
            else "Moderate risk of overfitting" if overfit_ratio < 0.6
            else "High risk of overfitting — test returns significantly degrade"
        ),
    }


# ── Monte Carlo Trade Resampling ─────────────────────────────────────────

def monte_carlo_trade_resample(
    trades: list[Trade],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    n_simulations: int = 1000,
) -> dict:
    """
    Bootstrap resample completed trades to estimate the distribution
    of backtest metrics. Tells you how robust your results are — a wide
    confidence interval means the backtest is luck-dependent.

    Returns percentile distributions for return, max drawdown, and Sharpe.
    """
    if not trades or len(trades) < 5:
        return {"error": "Need at least 5 trades for Monte Carlo resampling."}

    pnls = np.array([t.pnl for t in trades if t.pnl is not None])
    n_trades = len(pnls)
    if n_trades < 5:
        return {"error": "Need at least 5 trades with PnL."}

    rng = np.random.default_rng(42)
    sim_returns = []
    sim_drawdowns = []
    sim_sharpes = []

    for _ in range(n_simulations):
        # Resample trades with replacement
        sampled = rng.choice(pnls, size=n_trades, replace=True)
        equity = initial_capital + np.cumsum(sampled)
        equity = np.insert(equity, 0, initial_capital)

        total_ret = (equity[-1] - initial_capital) / initial_capital * 100
        sim_returns.append(total_ret)

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        sim_drawdowns.append(float(dd.min()))

        daily_ret = np.diff(equity) / equity[:-1]
        if daily_ret.std() > 0:
            sim_sharpes.append(float(daily_ret.mean() / daily_ret.std() * np.sqrt(252 / max(1, n_trades / 252))))
        else:
            sim_sharpes.append(0.0)

    percentiles = [5, 25, 50, 75, 95]
    return {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "return_distribution": {
            f"p{p}": round(float(np.percentile(sim_returns, p)), 2)
            for p in percentiles
        },
        "drawdown_distribution": {
            f"p{p}": round(float(np.percentile(sim_drawdowns, p)), 2)
            for p in percentiles
        },
        "sharpe_distribution": {
            f"p{p}": round(float(np.percentile(sim_sharpes, p)), 2)
            for p in percentiles
        },
        "probability_of_profit": round(float(np.mean(np.array(sim_returns) > 0) * 100), 1),
        "probability_of_20pct_drawdown": round(float(np.mean(np.array(sim_drawdowns) < -20) * 100), 1),
    }


# ── Parameter Sensitivity Sweep ──────────────────────────────────────────

def parameter_sweep(
    df: pd.DataFrame,
    strategy_type: str,
    param_name: str,
    param_values: list,
    base_params: dict | None = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> list[dict]:
    """
    Vary one strategy parameter across a range and return metrics for each.
    Useful for finding robust parameter zones (flat, profitable plateaus).
    """
    from src.simulator.registry import get_strategy_fn

    base = dict(base_params or {})
    results = []

    for val in param_values:
        params = {**base, param_name: val}
        try:
            strat_fn = get_strategy_fn(strategy_type, params)
            bt = Backtester(
                df, strat_fn,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            result = bt.run()
            results.append({
                param_name: val,
                "total_return_pct": round(result.total_return_pct, 2),
                "annual_return_pct": round(result.annual_return_pct, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 2),
                "max_drawdown_pct": round(result.max_drawdown_pct, 2),
                "total_trades": result.total_trades,
                "win_rate_pct": round(result.win_rate_pct, 1),
                "profit_factor": round(result.profit_factor, 2),
            })
        except Exception as e:
            results.append({param_name: val, "error": str(e)})

    return results


# ── Strategy Correlation Matrix ──────────────────────────────────────────

def strategy_correlation_matrix(
    df: pd.DataFrame,
    strategy_configs: list[dict],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> dict:
    """
    Run multiple strategies and compute correlation matrix of their daily returns.

    Parameters
    ----------
    strategy_configs : list of dicts
        Each dict must have 'type' and 'params' keys.

    Returns
    -------
    dict with correlation_matrix, strategy_names, and pairwise_correlations.
    """
    from src.simulator.registry import get_strategy_fn

    returns_dict = {}
    for config in strategy_configs:
        stype = config["type"]
        params = config.get("params", {})
        try:
            strat_fn = get_strategy_fn(stype, params)
            bt = Backtester(
                df, strat_fn,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            result = bt.run()
            eq = result.equity_curve
            daily_ret = eq["equity"].pct_change().dropna()
            returns_dict[result.strategy_name] = daily_ret.values
        except Exception as e:
            logger.warning("Skipping %s: %s", stype, e)

    if len(returns_dict) < 2:
        return {"error": "Need at least 2 strategies for correlation analysis."}

    # Align lengths (all should be same since same data)
    min_len = min(len(v) for v in returns_dict.values())
    aligned = {k: v[:min_len] for k, v in returns_dict.items()}
    returns_df = pd.DataFrame(aligned)
    corr = returns_df.corr()

    # Extract pairwise correlations
    pairwise = []
    names = list(corr.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise.append({
                "strategy_a": names[i],
                "strategy_b": names[j],
                "correlation": round(float(corr.iloc[i, j]), 3),
            })

    return {
        "strategy_names": names,
        "correlation_matrix": {k: {k2: round(float(v), 3) for k2, v in row.items()} for k, row in corr.to_dict().items()},
        "pairwise_correlations": sorted(pairwise, key=lambda x: abs(x["correlation"]), reverse=True),
        "avg_correlation": round(float(np.mean([abs(p["correlation"]) for p in pairwise])), 3),
    }
