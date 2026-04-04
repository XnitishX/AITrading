"""
Probability Predictor
─────────────────────
Estimates the probability distribution of where Nifty will be at
different future horizons (next day, week, month) using:

  1. **Historical return distribution** – kernel density estimation on
     rolling windows of past returns, conditioned on the current VIX
     regime.
  2. **Monte Carlo simulation** – GBM paths calibrated to recent
     realised vol and optional VIX adjustment.
  3. **Regime‑aware percentile forecasts** – split history into
     low / medium / high VIX regimes and compute empirical quantiles.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import (
    CONFIDENCE_LEVELS,
    NIFTY_CLOSE_COL,
    PREDICTION_HORIZONS,
    PROBABILITY_BINS,
)

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Container for a single‑horizon probability forecast."""
    horizon_name: str
    horizon_days: int
    current_price: float
    mean_price: float
    median_price: float
    std_price: float
    percentiles: dict[float, float]  # confidence → price
    prob_up: float  # P(future > current)
    prob_down: float  # P(future < current)
    simulated_paths: Optional[np.ndarray] = None  # (n_sims, horizon+1)
    kde_x: Optional[np.ndarray] = None
    kde_y: Optional[np.ndarray] = None


@dataclass
class PredictionReport:
    """Collection of predictions across all horizons."""
    as_of_date: pd.Timestamp
    current_price: float
    current_vix: Optional[float]
    vix_regime: str  # "low", "medium", "high"
    predictions: dict[str, PredictionResult] = field(default_factory=dict)

    def summary_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising each horizon."""
        rows = []
        for name, pred in self.predictions.items():
            row = {
                "horizon": name,
                "days": pred.horizon_days,
                "current": pred.current_price,
                "mean": pred.mean_price,
                "median": pred.median_price,
                "std": pred.std_price,
                "prob_up_%": round(pred.prob_up * 100, 2),
                "prob_down_%": round(pred.prob_down * 100, 2),
            }
            for cl, val in pred.percentiles.items():
                row[f"P{int(cl*100)}"] = round(val, 2)
            rows.append(row)
        return pd.DataFrame(rows)


# ── VIX Regime Detection ─────────────────────────────────────────────────

def classify_vix_regime(
    vix_series: pd.Series,
    low_threshold: float = 15.0,
    high_threshold: float = 25.0,
) -> pd.Series:
    """Classify each day as low / medium / high VIX regime."""
    return pd.cut(
        vix_series,
        bins=[-np.inf, low_threshold, high_threshold, np.inf],
        labels=["low", "medium", "high"],
    )


def current_regime(vix_value: Optional[float]) -> str:
    if vix_value is None:
        return "unknown"
    if vix_value < 15:
        return "low"
    if vix_value < 25:
        return "medium"
    return "high"


# ── Historical Distribution Approach ─────────────────────────────────────

def empirical_forecast(
    returns: pd.Series,
    current_price: float,
    horizon_days: int,
    confidence_levels: list[float] | None = None,
) -> PredictionResult:
    """
    Use the empirical distribution of *horizon_days*‑period returns to
    estimate future prices.
    """
    confidence_levels = confidence_levels or CONFIDENCE_LEVELS
    period_returns = returns.rolling(horizon_days).sum().dropna()

    future_prices = current_price * np.exp(period_returns.values)

    mean_p = float(np.mean(future_prices))
    median_p = float(np.median(future_prices))
    std_p = float(np.std(future_prices))
    prob_up = float(np.mean(future_prices > current_price))
    prob_down = 1.0 - prob_up

    percentiles = {}
    for cl in confidence_levels:
        lo = (1 - cl) / 2
        hi = 1 - lo
        percentiles[cl] = (
            float(np.quantile(future_prices, lo)),
            float(np.quantile(future_prices, hi)),
        )

    # KDE for smooth density
    try:
        kde = stats.gaussian_kde(future_prices)
        x = np.linspace(future_prices.min(), future_prices.max(), PROBABILITY_BINS)
        y = kde(x)
    except Exception:
        x, y = None, None

    return PredictionResult(
        horizon_name="",
        horizon_days=horizon_days,
        current_price=current_price,
        mean_price=mean_p,
        median_price=median_p,
        std_price=std_p,
        percentiles=percentiles,
        prob_up=prob_up,
        prob_down=prob_down,
        kde_x=x,
        kde_y=y,
    )


# ── Monte Carlo GBM Simulation ──────────────────────────────────────────

def monte_carlo_forecast(
    current_price: float,
    annual_drift: float,
    annual_vol: float,
    horizon_days: int,
    n_simulations: int = 10_000,
    confidence_levels: list[float] | None = None,
) -> PredictionResult:
    """
    Simulate *n_simulations* GBM price paths and derive statistics.
    """
    confidence_levels = confidence_levels or CONFIDENCE_LEVELS
    dt = 1 / 252  # one trading day
    mu_dt = (annual_drift - 0.5 * annual_vol**2) * dt
    sigma_sqrt_dt = annual_vol * np.sqrt(dt)

    rng = np.random.default_rng()
    shocks = rng.normal(mu_dt, sigma_sqrt_dt, size=(n_simulations, horizon_days))
    log_paths = np.cumsum(shocks, axis=1)
    log_paths = np.hstack([np.zeros((n_simulations, 1)), log_paths])
    paths = current_price * np.exp(log_paths)

    terminal = paths[:, -1]
    mean_p = float(np.mean(terminal))
    median_p = float(np.median(terminal))
    std_p = float(np.std(terminal))
    prob_up = float(np.mean(terminal > current_price))

    percentiles = {}
    for cl in confidence_levels:
        lo = (1 - cl) / 2
        hi = 1 - lo
        percentiles[cl] = (
            float(np.quantile(terminal, lo)),
            float(np.quantile(terminal, hi)),
        )

    try:
        kde = stats.gaussian_kde(terminal)
        x = np.linspace(terminal.min(), terminal.max(), PROBABILITY_BINS)
        y = kde(x)
    except Exception:
        x, y = None, None

    return PredictionResult(
        horizon_name="",
        horizon_days=horizon_days,
        current_price=current_price,
        mean_price=mean_p,
        median_price=median_p,
        std_price=std_p,
        percentiles=percentiles,
        prob_up=prob_up,
        prob_down=1 - prob_up,
        simulated_paths=paths,
        kde_x=x,
        kde_y=y,
    )


# ── Regime‑Conditioned Forecast ─────────────────────────────────────────

def regime_conditioned_forecast(
    df: pd.DataFrame,
    current_price: float,
    vix_regime: str,
    horizon_days: int,
    confidence_levels: list[float] | None = None,
) -> PredictionResult:
    """
    Filter historical returns by the VIX regime matching *vix_regime*
    and produce an empirical forecast from that subset.
    """
    if "vix_regime" not in df.columns:
        if "vix_close" in df.columns:
            df = df.copy()
            df["vix_regime"] = classify_vix_regime(df["vix_close"])
        else:
            logger.warning("No VIX data — falling back to full‑sample forecast.")
            return empirical_forecast(
                df["log_return"], current_price, horizon_days, confidence_levels
            )

    mask = df["vix_regime"] == vix_regime
    subset = df.loc[mask, "log_return"].dropna()
    if len(subset) < horizon_days * 2:
        logger.warning(
            "Regime '%s' has only %d observations – using full sample.",
            vix_regime,
            len(subset),
        )
        subset = df["log_return"].dropna()

    return empirical_forecast(subset, current_price, horizon_days, confidence_levels)


# ── Main Prediction Pipeline ────────────────────────────────────────────

def predict(
    df: pd.DataFrame,
    horizons: dict[str, int] | None = None,
    method: str = "ensemble",
    n_simulations: int = 10_000,
) -> PredictionReport:
    """
    Generate a full prediction report.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with at least ``Close``, ``log_return``, and
        optionally ``vix_close``.
    horizons : dict, optional
        ``{ name: days }`` mapping.  Defaults to ``PREDICTION_HORIZONS``.
    method : str
        ``"empirical"``, ``"monte_carlo"``, ``"regime"``, or ``"ensemble"``
        (average of all three).
    n_simulations : int
        Number of Monte Carlo paths.

    Returns
    -------
    PredictionReport
    """
    horizons = horizons or PREDICTION_HORIZONS

    last_row = df.iloc[-1]
    current_price = float(last_row[NIFTY_CLOSE_COL])
    current_date = last_row["Date"]
    cur_vix = float(last_row["vix_close"]) if "vix_close" in df.columns else None
    regime = current_regime(cur_vix)

    # Calibrate drift & vol from recent 252 days
    recent = df["log_return"].dropna().tail(252)
    annual_drift = float(recent.mean() * 252)
    annual_vol = float(recent.std() * np.sqrt(252))

    report = PredictionReport(
        as_of_date=current_date,
        current_price=current_price,
        current_vix=cur_vix,
        vix_regime=regime,
    )

    for name, days in horizons.items():
        results = []

        if method in ("empirical", "ensemble"):
            r = empirical_forecast(df["log_return"].dropna(), current_price, days)
            r.horizon_name = name
            results.append(r)

        if method in ("monte_carlo", "ensemble"):
            r = monte_carlo_forecast(
                current_price, annual_drift, annual_vol, days, n_simulations
            )
            r.horizon_name = name
            results.append(r)

        if method in ("regime", "ensemble"):
            r = regime_conditioned_forecast(df, current_price, regime, days)
            r.horizon_name = name
            results.append(r)

        # Ensemble: simple average of the individual forecasts
        if len(results) == 1:
            report.predictions[name] = results[0]
        else:
            avg = _average_results(results, name, days, current_price)
            report.predictions[name] = avg

    return report


def _average_results(
    results: list[PredictionResult],
    name: str,
    days: int,
    current_price: float,
) -> PredictionResult:
    """Average the scalar fields of multiple PredictionResults."""
    n = len(results)
    mean_p = sum(r.mean_price for r in results) / n
    median_p = sum(r.median_price for r in results) / n
    std_p = sum(r.std_price for r in results) / n
    prob_up = sum(r.prob_up for r in results) / n

    # merge percentile dicts
    all_cls = set()
    for r in results:
        all_cls.update(r.percentiles.keys())
    percentiles = {}
    for cl in sorted(all_cls):
        vals = [r.percentiles[cl] for r in results if cl in r.percentiles]
        lo = sum(v[0] for v in vals) / len(vals)
        hi = sum(v[1] for v in vals) / len(vals)
        percentiles[cl] = (lo, hi)

    return PredictionResult(
        horizon_name=name,
        horizon_days=days,
        current_price=current_price,
        mean_price=mean_p,
        median_price=median_p,
        std_price=std_p,
        percentiles=percentiles,
        prob_up=prob_up,
        prob_down=1 - prob_up,
    )


# ── Projected OHLC Candles from Monte Carlo ─────────────────────────────

def projected_candles(
    current_price: float,
    annual_drift: float,
    annual_vol: float,
    horizon_days: int,
    n_simulations: int = 10_000,
) -> list[dict]:
    """
    Run Monte Carlo GBM and produce synthetic OHLC candles (one per day)
    by taking percentile statistics across all simulated paths each day.

    Returns a list of dicts: {day, open, high, low, close, upper_90, lower_90}
    """
    dt = 1.0 / 252
    mu_dt = (annual_drift - 0.5 * annual_vol ** 2) * dt
    sigma_sqrt_dt = annual_vol * np.sqrt(dt)

    rng = np.random.default_rng()
    shocks = rng.normal(mu_dt, sigma_sqrt_dt, size=(n_simulations, horizon_days))
    log_paths = np.cumsum(shocks, axis=1)
    log_paths = np.hstack([np.zeros((n_simulations, 1)), log_paths])
    paths = current_price * np.exp(log_paths)  # shape (n_sims, horizon+1)

    candles = []
    for d in range(1, horizon_days + 1):
        today = paths[:, d]
        yesterday = paths[:, d - 1]
        candles.append({
            "day": d,
            "open": float(np.median(yesterday)),
            "high": float(np.percentile(today, 75)),
            "low": float(np.percentile(today, 25)),
            "close": float(np.median(today)),
            "upper_90": float(np.percentile(today, 95)),
            "lower_90": float(np.percentile(today, 5)),
        })
    return candles


def project_all_horizons(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
) -> dict:
    """
    Generate projected candles for every standard horizon.
    Returns {horizon_name: {info + candles}}.
    """
    last_row = df.iloc[-1]
    current_price = float(last_row[NIFTY_CLOSE_COL])
    cur_vix = float(last_row["vix_close"]) if "vix_close" in df.columns else None

    recent = df["log_return"].dropna().tail(252)
    annual_drift = float(recent.mean() * 252)
    annual_vol = float(recent.std() * np.sqrt(252))

    # If VIX is available, blend realised vol with implied vol
    if cur_vix is not None:
        implied_vol = cur_vix / 100.0
        annual_vol = 0.6 * annual_vol + 0.4 * implied_vol  # blend

    horizons = {"next_day": 1, "5_days": 5, "next_week": 5, "next_month": 21}
    result = {}
    for name, days in horizons.items():
        candles = projected_candles(
            current_price, annual_drift, annual_vol, days, n_simulations
        )
        result[name] = {
            "horizon_name": name,
            "horizon_days": days,
            "current_price": current_price,
            "annual_drift": round(annual_drift, 4),
            "annual_vol": round(annual_vol, 4),
            "vix": cur_vix,
            "candles": candles,
        }
    return result


# ── Historical Prediction Accuracy (Backtest the Predictor) ──────────────

def backtest_predictions(
    df: pd.DataFrame,
    horizons: dict[str, int] | None = None,
    lookback_days: int = 252,
    n_simulations: int = 5_000,
    step: int = 5,
) -> dict:
    """
    Walk through the last *lookback_days* of data, making predictions at
    each step point and comparing against actual outcomes.

    Returns a dict of {horizon_name: {metrics + detail_rows}}.
    """
    horizons = horizons or {"next_day": 1, "next_week": 5, "next_month": 21}
    min_warmup = 504  # need 2 years of history to calibrate

    if len(df) < min_warmup + lookback_days:
        lookback_days = max(50, len(df) - min_warmup)

    start_idx = len(df) - lookback_days
    if start_idx < min_warmup:
        start_idx = min_warmup

    max_horizon = max(horizons.values())
    end_idx = len(df) - max_horizon  # need room for actual outcome

    results = {name: [] for name in horizons}

    for i in range(start_idx, end_idx, step):
        hist = df.iloc[: i + 1]
        last = hist.iloc[-1]
        price_now = float(last[NIFTY_CLOSE_COL])
        cur_vix = float(last["vix_close"]) if "vix_close" in hist.columns else None

        recent = hist["log_return"].dropna().tail(252)
        if len(recent) < 50:
            continue
        annual_drift = float(recent.mean() * 252)
        annual_vol = float(recent.std() * np.sqrt(252))

        if cur_vix is not None:
            implied_vol = cur_vix / 100.0
            annual_vol = 0.6 * annual_vol + 0.4 * implied_vol

        for hname, hdays in horizons.items():
            future_idx = i + hdays
            if future_idx >= len(df):
                continue
            actual_price = float(df[NIFTY_CLOSE_COL].iloc[future_idx])

            # Quick MC forecast
            mc = monte_carlo_forecast(
                price_now, annual_drift, annual_vol, hdays, n_simulations
            )
            predicted_price = mc.mean_price
            predicted_median = mc.median_price
            prob_up = mc.prob_up

            # Direction accuracy
            actual_up = actual_price > price_now
            predicted_up = prob_up > 0.5

            pct_90 = mc.percentiles.get(0.90, (price_now, price_now))
            in_90_range = pct_90[0] <= actual_price <= pct_90[1]
            pct_error = (predicted_price - actual_price) / actual_price * 100

            results[hname].append({
                "date": str(last["Date"].date()) if hasattr(last["Date"], "date") else str(last["Date"]),
                "price_at_prediction": round(price_now, 2),
                "actual_price": round(actual_price, 2),
                "predicted_mean": round(predicted_price, 2),
                "predicted_median": round(predicted_median, 2),
                "predicted_prob_up": round(prob_up * 100, 2),
                "actual_direction_up": actual_up,
                "predicted_direction_up": predicted_up,
                "direction_correct": actual_up == predicted_up,
                "in_90_ci": in_90_range,
                "pct_error": round(pct_error, 2),
            })

    # Aggregate metrics
    summary = {}
    for hname, rows in results.items():
        if not rows:
            summary[hname] = {"n_samples": 0}
            continue
        n = len(rows)
        direction_acc = sum(1 for r in rows if r["direction_correct"]) / n * 100
        ci_coverage = sum(1 for r in rows if r["in_90_ci"]) / n * 100
        mae = np.mean([abs(r["pct_error"]) for r in rows])
        rmse = np.sqrt(np.mean([r["pct_error"] ** 2 for r in rows]))
        mean_bias = np.mean([r["pct_error"] for r in rows])

        summary[hname] = {
            "horizon_days": horizons[hname],
            "n_samples": n,
            "direction_accuracy_pct": round(direction_acc, 2),
            "ci_90_coverage_pct": round(ci_coverage, 2),
            "mae_pct": round(mae, 2),
            "rmse_pct": round(rmse, 2),
            "mean_bias_pct": round(mean_bias, 2),
            "details": rows,
        }
    return summary
