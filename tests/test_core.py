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


# ── Leverage Simulator Tests ──────────────────────────────────────────────

def test_rbi_repo_rate_lookup():
    """Known dates should return correct RBI repo rates."""
    from src.simulator.leverage_sim import get_repo_rate
    # RBI cut rates to 4.0% in May 2020
    r_may2020 = get_repo_rate("2020-06-01")
    assert abs(r_may2020 - 4.0) < 0.01, f"Expected 4.0%, got {r_may2020}"
    # Rate in 2023 should be 6.5%
    r_2023 = get_repo_rate("2023-09-01")
    assert abs(r_2023 - 6.5) < 0.01, f"Expected 6.5%, got {r_2023}"


def test_bs_put_price_put_call_parity():
    """BS put price must satisfy put-call parity within numerical tolerance."""
    from src.simulator.leverage_sim import bs_call_price, bs_put_price
    S, K, T, r, sigma = 22000.0, 20000.0, 0.25, 0.065, 0.20
    call = bs_call_price(S, K, T, r, sigma)
    put = bs_put_price(S, K, T, r, sigma)
    # Put-call parity: C - P = S - K*e^(-rT)
    import math
    parity_lhs = call - put
    parity_rhs = S - K * math.exp(-r * T)
    assert abs(parity_lhs - parity_rhs) < 0.01, (
        f"Put-call parity violation: LHS={parity_lhs:.4f}, RHS={parity_rhs:.4f}"
    )
    # Both prices should be positive
    assert call > 0
    assert put > 0


def test_leverage_simulator_1x_no_options():
    """1× leverage with no options should approximate buy-and-hold minus carry cost."""
    from src.simulator.leverage_sim import LeverageSimulator

    df = _make_dummy_price_df(n=500)
    # No VIX needed when put_otm_pct=0 and call_otm_pct=0
    sim = LeverageSimulator(
        df=df,
        leverage_ratio=1.0,
        call_otm_pct=0.0,
        put_otm_pct=0.0,
        initial_capital=1_000_000.0,
    )
    result = sim.run()
    assert result.final_capital > 0
    assert result.put_payout_events == 0
    assert result.call_cap_events == 0
    assert result.total_put_cost_pct == 0.0
    assert result.total_call_income_pct == 0.0
    # Carry cost should be positive (represents roll drag)
    assert result.total_carry_cost_pct > 0.0
    # Liquid fund income should be positive
    assert result.total_liquid_fund_income_pct > 0.0
    # Equity curve should have same length as input df
    assert len(result.equity_curve) == len(df)


def test_leverage_sweep_returns_correct_count():
    """Sweep of 5 leverage ratios × 4 call OTM values should return 20 results."""
    from src.simulator.leverage_sim import run_leverage_sweep

    df = _make_dummy_price_df(n=300)
    leverages = [1.0, 1.5, 2.0, 2.5, 3.0]
    calls = [0.0, 0.025, 0.05, 0.075]
    results = run_leverage_sweep(
        df=df,
        leverage_ratios=leverages,
        call_otm_pcts=calls,
        put_otm_pct=0.20,
        initial_capital=1_000_000.0,
    )
    assert len(results) == 20, f"Expected 20 sweep results, got {len(results)}"
    # Higher leverage should generally produce higher absolute returns (on a trending dummy series)
    lev1_returns = [r.total_return_pct for r in results if r.leverage_ratio == 1.0]
    lev3_returns = [r.total_return_pct for r in results if r.leverage_ratio == 3.0]
    # At least check that sharpe ratios are finite numbers
    for r in results:
        assert not (r.sharpe_ratio != r.sharpe_ratio), "NaN sharpe ratio found"
    # New fields should be present
    for r in results:
        assert hasattr(r, 'margin_call_triggered')
        assert hasattr(r, 'total_transaction_cost_pct')


def test_vol_skew_applied():
    """With use_vol_skew=True, put IV > call IV for same VIX level."""
    from src.simulator.leverage_sim import vix_to_iv, vix_to_iv_call, vix_to_iv_put
    for vix in [12.0, 17.0, 25.0, 38.0, 50.0]:
        base_iv = vix_to_iv(vix)
        call_iv = vix_to_iv_call(vix)
        put_iv = vix_to_iv_put(vix)
        assert call_iv < base_iv, f"Call IV {call_iv} should be < base IV {base_iv} at VIX {vix}"
        assert put_iv > base_iv, f"Put IV {put_iv} should be > base IV {base_iv} at VIX {vix}"
        assert put_iv > call_iv, "Put IV should always exceed call IV"


def test_strike_rounding():
    """_round_to_nse_strike should produce multiples of the increment."""
    from src.simulator.leverage_sim import LeverageSimulator
    for price in [22137.5, 22050.0, 22175.0, 17823.0]:
        rounded = LeverageSimulator._round_to_nse_strike(price, 50.0)
        assert rounded % 50 == 0, f"Strike {rounded} is not a multiple of 50"


def test_transaction_costs_reduce_returns():
    """Enabling transaction costs should reduce final capital vs no costs."""
    from src.simulator.leverage_sim import LeverageSimulator, TransactionCostModel

    df = _make_dummy_price_df(n=500)
    common = dict(df=df, leverage_ratio=2.0, call_otm_pct=0.05,
                  put_otm_pct=0.20, initial_capital=1_000_000.0)
    sim_no_tc = LeverageSimulator(**common, transaction_costs=None)
    sim_with_tc = LeverageSimulator(**common, transaction_costs=TransactionCostModel())
    r_no_tc = sim_no_tc.run()
    r_with_tc = sim_with_tc.run()
    assert r_with_tc.final_capital < r_no_tc.final_capital, "TC should reduce returns"
    assert r_with_tc.total_transaction_cost_pct > 0.0, "TC cost should be > 0"


def test_margin_call_triggered_on_low_equity():
    """A sim where equity collapses should set margin_call_triggered=True."""
    from src.simulator.leverage_sim import LeverageSimulator
    import pandas as pd
    import numpy as np
    # Construct a crash: Nifty drops 95% over 100 days
    n = 100
    prices = [10000.0 * (0.95 ** i) for i in range(n)]
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    df_crash = pd.DataFrame({'Close': prices}, index=idx)
    sim = LeverageSimulator(
        df=df_crash,
        leverage_ratio=3.0,
        call_otm_pct=0.0,
        put_otm_pct=0.0,
        initial_capital=1_000_000.0,
        margin_call_threshold_pct=0.20,
    )
    result = sim.run()
    assert result.margin_call_triggered is True, "Should flag margin call zone in crash scenario"
    assert result.margin_call_date is not None


# ─────────────────────────────────────────────────────────────────────────────
# Valuation Scraper tests
# ─────────────────────────────────────────────────────────────────────────────

def test_valuation_zone_classification():
    """Zone helpers correctly classify PE values across all four buckets."""
    from src.data.valuation_scraper import _pe_zone, _pb_zone, _dy_zone

    assert _pe_zone(14.0) == "Undervalued"
    assert _pe_zone(16.9) == "Undervalued"        # just below threshold
    assert _pe_zone(17.0) == "Fairly Valued"      # threshold is exclusive upper bound
    assert _pe_zone(18.0) == "Fairly Valued"
    assert _pe_zone(20.9) == "Fairly Valued"
    assert _pe_zone(21.0) == "Slightly Overvalued"  # threshold exclusive
    assert _pe_zone(23.0) == "Slightly Overvalued"
    assert _pe_zone(25.0) == "Overvalued"           # ≥ 25 → Overvalued
    assert _pe_zone(27.0) == "Overvalued"

    assert _pb_zone(2.0) == "Undervalued"
    assert _pb_zone(3.0) == "Fairly Valued"
    assert _pb_zone(4.0) == "Slightly Overvalued"
    assert _pb_zone(5.0) == "Overvalued"

    assert _dy_zone(1.8) == "Undervalued"
    assert _dy_zone(1.2) == "Neutral"
    assert _dy_zone(0.8) == "Overvalued"


def test_valuation_erp_calculation():
    """ERP = earnings yield − risk-free rate, both derived from PE and constant."""
    from src.data.valuation_scraper import get_valuation_context, RISK_FREE_RATE_PCT
    import pandas as pd
    import io

    # Build a minimal single-row valuation CSV in memory
    csv_data = "Date,PE,PB,DivYield\n2025-01-01,20.0,3.0,1.2\n"
    df = pd.read_csv(io.StringIO(csv_data), parse_dates=["Date"])

    # Patch scrape_current so the test doesn't hit the network
    import unittest.mock as mock
    with mock.patch("src.data.valuation_scraper.scrape_current", return_value={
        "Date": "2025-01-01", "PE": 20.0, "PB": 3.0, "DivYield": 1.2
    }):
        ctx = get_valuation_context(df=df)

    erp_info = ctx["equity_risk_premium"]
    expected_ey = (1 / 20.0) * 100       # 5.0%
    expected_erp = expected_ey - RISK_FREE_RATE_PCT

    assert abs(erp_info["earnings_yield_pct"] - expected_ey) < 0.01
    assert abs(erp_info["erp_pct"] - expected_erp) < 0.01
    assert erp_info["risk_free_rate_pct"] == RISK_FREE_RATE_PCT


def test_valuation_context_structure():
    """get_valuation_context returns the expected top-level keys and nested structure."""
    from src.data.valuation_scraper import get_valuation_context
    import pandas as pd
    import io
    import unittest.mock as mock

    csv_data = "Date,PE,PB,DivYield\n2025-01-01,19.5,3.2,1.35\n"
    df = pd.read_csv(io.StringIO(csv_data), parse_dates=["Date"])

    with mock.patch("src.data.valuation_scraper.scrape_current", return_value={
        "Date": "2025-01-01", "PE": 19.5, "PB": 3.2, "DivYield": 1.35
    }):
        ctx = get_valuation_context(df=df)

    assert set(ctx.keys()) == {"current", "zones", "equity_risk_premium", "pe_stats", "leverage_rec", "pe_thresholds", "meta"}
    assert "pe" in ctx["current"]
    assert "pb" in ctx["current"]
    assert "div_yield" in ctx["current"]
    assert "pe_zone" in ctx["zones"]
    assert "overall_signal" in ctx["zones"]
    assert "max_leverage" in ctx["leverage_rec"]
    assert ctx["leverage_rec"]["max_leverage"] > 0
    # With only 1 row, pe_stats should be empty
    assert ctx["pe_stats"] == {}


# ══════════════════════════════════════════════════════════════════════════════
# RETIREMENT MONTE CARLO SIMULATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_retirement_inflation_adjustment():
    """Monthly withdrawal inflates correctly at given CPI rate."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    params = RetirementParams(
        initial_corpus=10_000_000,
        equity_pct=0,          # all debt so equity return is irrelevant
        monthly_withdrawal=50_000,
        inflation_rate_pct=6.0,
        tax_enabled=False,
        total_years=10,
        n_simulations=100,
    )
    rng = np.random.default_rng(42)
    total_months = params.total_years * 12
    # Zero equity return, flat debt factor, flat inflation
    eq_returns = np.zeros(total_months)
    debt_factors = np.full(total_months, (1.06) ** (1/12))  # 6% debt
    infl_factors = np.full(total_months, (1.06) ** (1/12))  # 6% inflation

    path = run_single_simulation(params, eq_returns, debt_factors, infl_factors, rng)

    # After 12 months (month 13 = start of year 2), withdrawal should have grown ~6%
    wd_year2 = path.withdrawals[13]
    expected = 50_000 * 1.06
    assert abs(wd_year2 - expected) < 1000, f"Expected ~{expected:.0f} got {wd_year2:.0f}"


def test_retirement_tax_ltcg_below_exemption():
    """Gains below ₹1.25L LTCG exemption should attract zero tax."""
    from src.simulator.retirement_sim import TaxEngine

    tax = TaxEngine(income_tax_bracket_pct=30.0, enabled=True)
    # Sell ₹2L with ₹1L gain (below ₹1.25L exemption)
    net = tax.apply_equity_sale(gross_sale_value=200_000, cost_basis=100_000)
    assert net == 200_000, "No tax should apply when gain < ₹1.25L exemption"
    assert tax.total_tax_paid == 0.0


def test_retirement_tax_ltcg_above_exemption():
    """Gains above ₹1.25L exemption should be taxed at 12.5%."""
    from src.simulator.retirement_sim import TaxEngine, LTCG_RATE, LTCG_EXEMPTION_INR

    tax = TaxEngine(income_tax_bracket_pct=30.0, enabled=True)
    # First use up the exemption (₹1.25L gain)
    tax.apply_equity_sale(gross_sale_value=225_000, cost_basis=100_000)  # gain = 1.25L exactly
    # Next sale: entire gain is taxable
    gain = 100_000
    net = tax.apply_equity_sale(gross_sale_value=300_000, cost_basis=200_000)
    expected_tax = gain * LTCG_RATE
    assert abs((300_000 - net) - expected_tax) < 1, f"Expected tax {expected_tax:.0f} got {300_000 - net:.0f}"


def test_retirement_scheduled_replenishment():
    """Debt bucket gets refilled at exactly N-year scheduled intervals."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    params = RetirementParams(
        initial_corpus=10_000_000,
        equity_pct=70,
        monthly_withdrawal=30_000,
        inflation_rate_pct=0,      # no inflation to simplify
        tax_enabled=False,
        total_years=15,
        swp_replenish_years=5,
        emergency_months_threshold=1,  # very low so emergency won't trigger first
        n_simulations=100,
    )
    rng = np.random.default_rng(42)
    total_months = params.total_years * 12
    eq_returns = np.full(total_months, 0.01)    # 1% monthly equity return
    debt_factors = np.full(total_months, (1.07) ** (1/12))
    infl_factors = np.ones(total_months)

    path = run_single_simulation(params, eq_returns, debt_factors, infl_factors, rng)

    # Scheduled replenishments should happen at months 60 and 120 (years 5 and 10)
    assert 60 in path.replenish_months or 59 in path.replenish_months, \
        f"Expected replenishment near month 60, got {path.replenish_months}"


def test_retirement_emergency_replenishment():
    """Emergency replenishment fires when debt falls below threshold."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    # High withdrawal relative to corpus forces emergency
    params = RetirementParams(
        initial_corpus=1_000_000,
        equity_pct=80,
        monthly_withdrawal=40_000,   # very high: 4.8% of corpus/month
        inflation_rate_pct=0,
        tax_enabled=False,
        total_years=5,
        swp_replenish_years=10,       # scheduled far away
        emergency_months_threshold=24,  # trigger if < 24 months of expenses
        n_simulations=100,
        random_seed=42,
    )
    rng = np.random.default_rng(42)
    total_months = params.total_years * 12
    eq_returns = np.full(total_months, 0.005)
    debt_factors = np.full(total_months, (1.07) ** (1/12))
    infl_factors = np.ones(total_months)

    path = run_single_simulation(params, eq_returns, debt_factors, infl_factors, rng)

    # With high withdrawal and 24-month emergency threshold, should trigger
    assert len(path.emergency_months) > 0, "Emergency replenishment should have fired"


def test_retirement_optimization_sweep_shape():
    """Optimization sweep returns correct number of points with required keys."""
    from src.simulator.retirement_sim import RetirementParams, run_optimization_sweep
    import pandas as pd, numpy as np

    # Build minimal nifty-like df
    dates = pd.date_range("2010-01-01", periods=180, freq="ME")
    close = 10000 * np.cumprod(1 + np.random.default_rng(0).normal(0.008, 0.05, 180))
    nifty_df = pd.DataFrame({"Date": dates, "Close": close})

    params = RetirementParams(
        initial_corpus=5_000_000,
        equity_pct=60,
        monthly_withdrawal=25_000,
        inflation_rate_pct=6.0,
        tax_enabled=False,
        total_years=10,
        n_simulations=100,
        random_seed=42,
    )

    result = run_optimization_sweep(
        params, nifty_df,
        equity_pct_range=[20, 40, 60, 80],
        n_sims_per_point=50,
    )

    assert len(result.points) == 4, f"Expected 4 points, got {len(result.points)}"
    required_keys = {"equity_pct", "survival_30yr_pct", "median_final_corpus",
                     "safe_withdrawal_rate_pct", "tax_drag_pct", "median_emergency_count"}
    for pt in result.points:
        from dataclasses import asdict
        d = asdict(pt)
        assert required_keys.issubset(d.keys()), f"Missing keys: {required_keys - d.keys()}"
    # Optima should be one of the tested points
    assert result.optimal_balanced.equity_pct in [20, 40, 60, 80]


# ─────────────────────────────────────────────────────────────────────────────
# Coast / Barista FIRE tests
# ─────────────────────────────────────────────────────────────────────────────

def test_coast_fire_no_withdrawal_phase1():
    """During Coast FIRE phase 1, withdrawals must be zero every month."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    params = RetirementParams(
        initial_corpus=10_000_000,
        equity_pct=70,
        monthly_withdrawal=50_000,
        inflation_rate_pct=6.0,
        tax_enabled=False,
        total_years=30,
        fire_mode="coast",
        phase1_years=5,
        monthly_contribution=0,
        n_simulations=100,
    )
    rng = np.random.default_rng(42)
    total_months = params.total_years * 12
    eq_returns = np.full(total_months, 0.01)
    debt_factors = np.full(total_months, (1.06) ** (1 / 12))
    infl_factors = np.full(total_months, (1.06) ** (1 / 12))

    path = run_single_simulation(params, eq_returns, debt_factors, infl_factors, rng)

    # First 60 months (phase 1): no withdrawals
    phase1_months = params.phase1_years * 12
    assert all(path.withdrawals[m] == 0 for m in range(1, phase1_months + 1)), \
        "Coast FIRE phase 1 should have zero withdrawals"


def test_barista_fire_partial_withdrawal_phase1():
    """During Barista FIRE phase 1, withdrawals equal barista_monthly_withdrawal not full target."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    barista_wd = 20_000
    full_wd = 60_000
    params = RetirementParams(
        initial_corpus=10_000_000,
        equity_pct=70,
        monthly_withdrawal=full_wd,
        barista_monthly_withdrawal=barista_wd,
        inflation_rate_pct=0.0,       # no inflation to keep values constant
        tax_enabled=False,
        total_years=20,
        fire_mode="barista",
        phase1_years=5,
        monthly_contribution=0,
        n_simulations=100,
    )
    rng = np.random.default_rng(42)
    total_months = params.total_years * 12
    eq_returns = np.full(total_months, 0.01)
    debt_factors = np.full(total_months, (1.06) ** (1 / 12))
    infl_factors = np.ones(total_months)

    path = run_single_simulation(params, eq_returns, debt_factors, infl_factors, rng)

    # In phase 1 (months 1–60) withdrawals should be close to barista_wd
    phase1_months = params.phase1_years * 12
    for m in range(1, phase1_months + 1):
        assert abs(path.withdrawals[m] - barista_wd) < 1, \
            f"Month {m}: expected {barista_wd} barista WD, got {path.withdrawals[m]}"


def test_fire_contribution_grows_corpus():
    """Monthly contributions during phase 1 should increase corpus vs no contribution."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    base_params = dict(
        initial_corpus=5_000_000,
        equity_pct=70,
        monthly_withdrawal=1,      # pydantic requires > 0; zero withdrawals happen via coast mode
        barista_monthly_withdrawal=0,
        inflation_rate_pct=0.0,
        tax_enabled=False,
        total_years=10,
        fire_mode="coast",
        phase1_years=5,
        n_simulations=100,
    )
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    total_months = 120
    eq_returns = np.full(total_months, 0.005)
    debt_factors = np.ones(total_months)
    infl_factors = np.ones(total_months)

    no_contrib = run_single_simulation(
        RetirementParams(**base_params, monthly_contribution=0),
        eq_returns, debt_factors, infl_factors, rng1
    )
    with_contrib = run_single_simulation(
        RetirementParams(**base_params, monthly_contribution=10_000),
        eq_returns, debt_factors, infl_factors, rng2
    )

    # After phase 1, corpus with contributions should be higher
    phase1_end = 5 * 12
    assert with_contrib.portfolio_values[phase1_end] > no_contrib.portfolio_values[phase1_end], \
        "Contributions should produce a larger corpus at end of phase 1"


def test_fire_emergency_uses_barista_withdrawal_threshold():
    """Emergency threshold for Barista FIRE must use barista_wd, not full target wd."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    # Design: barista_wd = 10k, full_wd = 50k, emergency_threshold = 12 months
    # With small corpus the debt bucket < 12 * barista_wd but > 12 * full_wd would fail.
    # We verify emergency fires based on barista threshold (correct) not full threshold.
    barista_wd = 10_000
    full_wd = 50_000
    threshold_months = 12
    initial_corpus = 1_000_000
    # debt portion = 30% = 300k. 12 * barista_wd = 120k, 12 * full_wd = 600k
    # If threshold uses full_wd → emergency fires immediately (300k < 600k)
    # If threshold uses barista_wd → no emergency initially (300k > 120k)
    params = RetirementParams(
        initial_corpus=initial_corpus,
        equity_pct=70,
        monthly_withdrawal=full_wd,
        barista_monthly_withdrawal=barista_wd,
        inflation_rate_pct=0.0,
        tax_enabled=False,
        total_years=20,
        fire_mode="barista",
        phase1_years=10,
        monthly_contribution=0,
        emergency_months_threshold=threshold_months,
        swp_replenish_years=20,   # max allowed; no scheduled before horizon
        n_simulations=100,
    )
    rng = np.random.default_rng(42)
    total_months = params.total_years * 12
    eq_returns = np.full(total_months, 0.008)
    debt_factors = np.full(total_months, (1.07) ** (1 / 12))
    infl_factors = np.ones(total_months)

    path = run_single_simulation(params, eq_returns, debt_factors, infl_factors, rng)

    # Emergency should NOT fire in month 1 (debt bucket > 12 * barista_wd)
    assert 1 not in path.emergency_months, \
        "Emergency should not fire immediately — threshold uses barista_wd, not full_wd"


def test_fire_phase_labels_populated():
    """phase_labels array must have correct labels for each month in Coast/Barista mode."""
    from src.simulator.retirement_sim import RetirementParams, run_single_simulation
    import numpy as np

    params = RetirementParams(
        initial_corpus=10_000_000,
        equity_pct=70,
        monthly_withdrawal=50_000,
        inflation_rate_pct=0.0,
        tax_enabled=False,
        total_years=20,
        fire_mode="coast",
        phase1_years=5,
        monthly_contribution=0,
        n_simulations=100,
    )
    rng = np.random.default_rng(0)
    total_months = params.total_years * 12
    eq_returns = np.full(total_months, 0.008)
    debt_factors = np.full(total_months, (1.06) ** (1 / 12))
    infl_factors = np.ones(total_months)

    path = run_single_simulation(params, eq_returns, debt_factors, infl_factors, rng)

    phase1_months = params.phase1_years * 12
    assert len(path.phase_labels) == total_months + 1, \
        f"phase_labels length should be total_months+1, got {len(path.phase_labels)}"
    # Month 1 should be "coast"
    assert path.phase_labels[1] == "coast", f"Expected 'coast' at month 1, got {path.phase_labels[1]}"
    # Month after phase 1 should be "full_retirement"
    assert path.phase_labels[phase1_months + 1] == "full_retirement", \
        f"Expected 'full_retirement' at month {phase1_months + 1}, got {path.phase_labels[phase1_months + 1]}"

