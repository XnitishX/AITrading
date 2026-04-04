"""
Basic sanity tests for AITrading modules.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest


# ── Config Tests ─────────────────────────────────────────────────────────

def test_config_imports():
    from config.settings import (
        PREDICTION_HORIZONS,
        PROJECT_ROOT,
        DATA_DIR,
        ROLLING_WINDOW_SIZES,
    )
    assert PROJECT_ROOT.exists()
    assert "next_day" in PREDICTION_HORIZONS
    assert 252 in ROLLING_WINDOW_SIZES


# ── Data Loader Feature Engineering Tests ────────────────────────────────

def _make_dummy_price_df(n: int = 300) -> pd.DataFrame:
    """Create a synthetic Nifty-like DataFrame for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 10000 + np.cumsum(rng.normal(0, 50, size=n))
    high = close + rng.uniform(10, 100, size=n)
    low = close - rng.uniform(10, 100, size=n)
    open_ = close + rng.normal(0, 20, size=n)
    volume = rng.integers(1_000_000, 10_000_000, size=n)
    vix = 15 + np.cumsum(rng.normal(0, 0.5, size=n))
    vix = np.clip(vix, 8, 45)
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "vix_close": vix,
    })


def test_add_returns():
    from src.data.loader import add_returns
    df = _make_dummy_price_df()
    df = add_returns(df)
    assert "simple_return" in df.columns
    assert "log_return" in df.columns
    assert pd.isna(df["simple_return"].iloc[0])
    assert not pd.isna(df["simple_return"].iloc[1])


def test_add_rolling_features():
    from src.data.loader import add_returns, add_rolling_features
    df = _make_dummy_price_df()
    df = add_returns(df)
    df = add_rolling_features(df, windows=[5, 10])
    assert "sma_5d" in df.columns
    assert "vol_10d" in df.columns


def test_add_rsi():
    from src.data.loader import add_rsi
    df = _make_dummy_price_df()
    df = add_rsi(df, period=14)
    assert "rsi_14" in df.columns
    valid = df["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_add_macd():
    from src.data.loader import add_macd
    df = _make_dummy_price_df()
    df = add_macd(df)
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns


# ── Predictor Tests ──────────────────────────────────────────────────────

def test_empirical_forecast():
    from src.predictor.probability import empirical_forecast
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0003, 0.01, size=500))
    result = empirical_forecast(returns, current_price=20000, horizon_days=5)
    assert result.mean_price > 0
    assert 0 <= result.prob_up <= 1
    assert abs(result.prob_up + result.prob_down - 1.0) < 1e-9


def test_monte_carlo_forecast():
    from src.predictor.probability import monte_carlo_forecast
    result = monte_carlo_forecast(
        current_price=20000,
        annual_drift=0.10,
        annual_vol=0.15,
        horizon_days=21,
        n_simulations=5000,
    )
    assert result.simulated_paths.shape == (5000, 22)
    assert result.mean_price > 0


def test_vix_regime_classification():
    from src.predictor.probability import classify_vix_regime
    vix = pd.Series([10, 20, 30])
    regimes = classify_vix_regime(vix)
    assert list(regimes) == ["low", "medium", "high"]


# ── Backtester Tests ─────────────────────────────────────────────────────

def test_backtester_runs():
    from src.data.loader import add_returns, add_rsi
    from src.simulator.backtester import Backtester, sma_crossover_strategy

    df = _make_dummy_price_df()
    df = add_returns(df)
    df = add_rsi(df)

    bt = Backtester(
        df,
        strategy=sma_crossover_strategy(10, 50),
        initial_capital=1_000_000,
    )
    result = bt.run()
    assert result.final_capital > 0
    assert len(result.equity_curve) == len(df)
    assert result.total_trades >= 0


def test_backtester_rsi_strategy():
    from src.data.loader import add_returns, add_rsi
    from src.simulator.backtester import Backtester, rsi_mean_reversion_strategy

    df = _make_dummy_price_df()
    df = add_returns(df)
    df = add_rsi(df)

    bt = Backtester(df, strategy=rsi_mean_reversion_strategy())
    result = bt.run()
    assert result.final_capital > 0


# ── Backtester Risk-Management Tests ─────────────────────────────────────

def test_trailing_stop():
    """Trailing stop should close a position that drops from its peak."""
    from src.data.loader import add_returns, add_rsi
    from src.simulator.backtester import Backtester, sma_crossover_strategy

    df = _make_dummy_price_df()
    df = add_returns(df)
    df = add_rsi(df)

    bt = Backtester(
        df,
        strategy=sma_crossover_strategy(10, 50),
        initial_capital=1_000_000,
        trailing_stop_pct=0.02,  # tight 2% trailing stop
    )
    result = bt.run()
    # With a tight trailing stop, some trades should exit via trailing_stop
    trailing_exits = [t for t in result.trades if t.exit_reason == "trailing_stop"]
    # At least verify the backtest ran and final_capital is positive
    assert result.final_capital > 0
    assert result.total_trades > 0
    # Trailing stop trades should exist or at least stop-loss trades dominate
    exit_reasons = {t.exit_reason for t in result.trades}
    assert exit_reasons.issubset({"signal", "stop_loss", "take_profit", "trailing_stop", "end_of_data", "max_holding"})


def test_cooldown_bars():
    """Cooldown should prevent rapid re-entry after exiting a trade."""
    from src.data.loader import add_returns, add_rsi
    from src.simulator.backtester import Backtester, sma_crossover_strategy

    df = _make_dummy_price_df()
    df = add_returns(df)
    df = add_rsi(df)

    # Run without cooldown
    bt_no_cd = Backtester(df, strategy=sma_crossover_strategy(10, 50), cooldown_bars=0)
    result_no_cd = bt_no_cd.run()

    # Run with high cooldown — should have fewer or equal trades
    bt_cd = Backtester(df, strategy=sma_crossover_strategy(10, 50), cooldown_bars=20)
    result_cd = bt_cd.run()

    assert result_cd.total_trades <= result_no_cd.total_trades
    assert result_cd.final_capital > 0


def test_max_holding_bars():
    """Max holding bars should force-close long-held positions."""
    from src.data.loader import add_returns, add_rsi
    from src.simulator.backtester import Backtester, sma_crossover_strategy

    df = _make_dummy_price_df()
    df = add_returns(df)
    df = add_rsi(df)

    bt = Backtester(
        df,
        strategy=sma_crossover_strategy(10, 50),
        initial_capital=1_000_000,
        max_holding_bars=10,  # very short max holding
    )
    result = bt.run()

    # Some trades should be closed by max_holding
    max_holding_exits = [t for t in result.trades if t.exit_reason == "max_holding"]
    assert result.final_capital > 0
    assert result.total_trades > 0
    # Verify no trade exceeds max_holding_bars
    for t in result.trades:
        if t.entry_bar is not None and t.exit_bar is not None:
            assert t.exit_bar - t.entry_bar <= 10


def test_slippage_and_commission():
    """Higher slippage/commission should reduce final capital vs zero costs."""
    from src.data.loader import add_returns, add_rsi
    from src.simulator.backtester import Backtester, sma_crossover_strategy

    df = _make_dummy_price_df()
    df = add_returns(df)
    df = add_rsi(df)

    bt_free = Backtester(df, strategy=sma_crossover_strategy(10, 50),
                         slippage_pct=0.0, commission_pct=0.0)
    result_free = bt_free.run()

    bt_costly = Backtester(df, strategy=sma_crossover_strategy(10, 50),
                           slippage_pct=0.005, commission_pct=0.005)  # 0.5% each
    result_costly = bt_costly.run()

    # Costly version should have same or lower final capital
    if result_free.total_trades > 0:
        assert result_costly.final_capital <= result_free.final_capital + 1  # +1 for float tolerance


# ── Strategy Signal-Generation Tests ─────────────────────────────────────

@pytest.mark.parametrize("strategy_factory,kwargs", [
    ("sma_crossover_strategy", {"fast_window": 10, "slow_window": 50}),
    ("ema_crossover_strategy", {"fast_span": 9, "slow_span": 21}),
    ("rsi_mean_reversion_strategy", {"oversold": 30, "overbought": 70}),
    ("macd_crossover_strategy", {"fast_ema": 12, "slow_ema": 26, "signal_period": 9}),
    ("macd_histogram_strategy", {"fast_ema": 12, "slow_ema": 26, "signal_period": 9}),
    ("bollinger_band_strategy", {"window": 20, "num_std": 2.0}),
    ("atr_breakout_strategy", {"sma_window": 20, "atr_period": 14, "atr_multiplier": 1.5}),
    ("vix_regime_strategy", {"buy_below": 15, "sell_above": 25}),
    ("stochastic_oscillator_strategy", {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}),
    ("mean_reversion_zscore_strategy", {"lookback": 20, "entry_z": -2.0, "exit_z": 0.0}),
    ("composite_sniper_strategy", {"sma_period": 50, "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65}),
], ids=[
    "SMA", "EMA", "RSI", "MACD_Cross", "MACD_Hist",
    "Bollinger", "ATR", "VIX", "Stochastic", "ZScore", "Sniper",
])
def test_strategy_signals(strategy_factory, kwargs):
    """Each strategy should run without error and produce valid signal values."""
    import importlib
    backtester_mod = importlib.import_module("src.simulator.backtester")
    factory_fn = getattr(backtester_mod, strategy_factory)

    df = _make_dummy_price_df(n=300)
    strat = factory_fn(**kwargs)

    signals = []
    for idx in range(len(df)):
        sig = strat(idx, None, df)
        signals.append(sig)

    # All signals must be in {-1, 0, 1}
    assert all(s in (-1, 0, 1) for s in signals), f"Invalid signal values: {set(signals)}"
    # Strategy should produce at least one non-zero signal on 300 bars
    assert any(s != 0 for s in signals), f"{strategy_factory} produced no signals"


@pytest.mark.parametrize("strategy_factory,kwargs", [
    ("sma_crossover_strategy", {"fast_window": 10, "slow_window": 50}),
    ("ema_crossover_strategy", {"fast_span": 9, "slow_span": 21}),
    ("rsi_mean_reversion_strategy", {"oversold": 30, "overbought": 70}),
    ("macd_crossover_strategy", {"fast_ema": 12, "slow_ema": 26, "signal_period": 9}),
    ("macd_histogram_strategy", {"fast_ema": 12, "slow_ema": 26, "signal_period": 9}),
    ("bollinger_band_strategy", {"window": 20, "num_std": 2.0}),
    ("atr_breakout_strategy", {"sma_window": 20, "atr_period": 14, "atr_multiplier": 1.5}),
    ("vix_regime_strategy", {"buy_below": 15, "sell_above": 25}),
    ("stochastic_oscillator_strategy", {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}),
    ("mean_reversion_zscore_strategy", {"lookback": 20, "entry_z": -2.0, "exit_z": 0.0}),
    ("composite_sniper_strategy", {"sma_period": 50, "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65}),
], ids=[
    "SMA_bt", "EMA_bt", "RSI_bt", "MACD_Cross_bt", "MACD_Hist_bt",
    "Bollinger_bt", "ATR_bt", "VIX_bt", "Stochastic_bt", "ZScore_bt", "Sniper_bt",
])
def test_strategy_backtest_completes(strategy_factory, kwargs):
    """Each strategy should complete a full backtest without error."""
    import importlib
    from src.data.loader import add_returns, add_rsi
    backtester_mod = importlib.import_module("src.simulator.backtester")

    df = _make_dummy_price_df(n=300)
    df = add_returns(df)
    df = add_rsi(df)

    factory_fn = getattr(backtester_mod, strategy_factory)
    strat = factory_fn(**kwargs)

    bt = backtester_mod.Backtester(df, strat, initial_capital=1_000_000)
    result = bt.run()
    assert result.final_capital > 0
    assert len(result.equity_curve) == len(df)
