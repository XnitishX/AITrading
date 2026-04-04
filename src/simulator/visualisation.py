"""
Visualisation Helpers
─────────────────────
Matplotlib / Plotly charts for equity curves, probability distributions,
and prediction cone plots.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR

logger = logging.getLogger(__name__)


def plot_equity_curve(
    equity_df: pd.DataFrame,
    title: str = "Equity Curve",
    save_path: Optional[Path] = None,
) -> Path:
    """Plot equity curve with drawdown overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(equity_df["Date"], equity_df["equity"], linewidth=1.2, color="#1f77b4")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (₹)")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    ax2.fill_between(equity_df["Date"], equity_df["drawdown"], 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / "equity_curve.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Equity curve saved → %s", save_path)
    return save_path


def plot_prediction_distribution(
    prediction_result,
    title: str = "",
    save_path: Optional[Path] = None,
) -> Path:
    """Plot KDE of predicted future prices with percentile bands."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if prediction_result.kde_x is not None and prediction_result.kde_y is not None:
        ax.fill_between(prediction_result.kde_x, prediction_result.kde_y, alpha=0.3, color="#2ca02c")
        ax.plot(prediction_result.kde_x, prediction_result.kde_y, color="#2ca02c", linewidth=1.5)

    ax.axvline(prediction_result.current_price, color="blue", linestyle="--", label=f"Current: ₹{prediction_result.current_price:,.0f}")
    ax.axvline(prediction_result.mean_price, color="orange", linestyle="-.", label=f"Mean: ₹{prediction_result.mean_price:,.0f}")

    colors = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    for i, (cl, (lo, hi)) in enumerate(prediction_result.percentiles.items()):
        c = colors[i % len(colors)]
        ax.axvline(lo, color=c, linestyle=":", alpha=0.7)
        ax.axvline(hi, color=c, linestyle=":", alpha=0.7, label=f"{int(cl*100)}% CI: ₹{lo:,.0f} – ₹{hi:,.0f}")

    horizon = prediction_result.horizon_name or f"{prediction_result.horizon_days}d"
    title = title or f"Nifty Prediction – {horizon} Horizon"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Nifty Level")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    tag = f"prob_up={prediction_result.prob_up*100:.1f}%  prob_down={prediction_result.prob_down*100:.1f}%"
    ax.annotate(tag, xy=(0.02, 0.95), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / f"prediction_{horizon}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Prediction distribution saved → %s", save_path)
    return save_path


def plot_monte_carlo_paths(
    prediction_result,
    max_paths: int = 200,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot a sample of Monte Carlo price paths."""
    paths = prediction_result.simulated_paths
    if paths is None:
        raise ValueError("No simulated paths in this PredictionResult (use monte_carlo method).")

    fig, ax = plt.subplots(figsize=(14, 7))
    n = min(max_paths, paths.shape[0])
    rng = np.random.default_rng(42)
    indices = rng.choice(paths.shape[0], size=n, replace=False)

    for i in indices:
        ax.plot(paths[i], alpha=0.08, color="steelblue", linewidth=0.5)

    # Percentile fan
    for q in [5, 25, 50, 75, 95]:
        line = np.percentile(paths, q, axis=0)
        style = "-" if q == 50 else "--"
        ax.plot(line, style, color="darkred", linewidth=1.2, label=f"P{q}")

    horizon = prediction_result.horizon_name or f"{prediction_result.horizon_days}d"
    ax.set_title(f"Monte Carlo Simulation – {horizon} ({paths.shape[0]:,} paths)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Nifty Level")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / f"mc_paths_{horizon}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Monte Carlo paths saved → %s", save_path)
    return save_path


def plot_price_with_vix(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Dual-axis plot of Nifty price and VIX."""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(df["Date"], df["Close"], color="#1f77b4", linewidth=1, label="Nifty 50")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Nifty 50", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    if "vix_close" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df["Date"], df["vix_close"], color="#d62728", linewidth=0.8, alpha=0.7, label="India VIX")
        ax2.set_ylabel("India VIX", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("Nifty 50 & India VIX", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = save_path or OUTPUT_DIR / "nifty_vix.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Price + VIX chart saved → %s", save_path)
    return save_path
