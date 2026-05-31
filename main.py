"""
AITrading – Main Entry Point
─────────────────────────────
CLI application that orchestrates data download, backtesting, and
probability prediction for Nifty 50 + India VIX.

Usage:
    python main.py download        # Download data from Yahoo Finance
    python main.py sync            # Incremental sync from Yahoo Finance
    python main.py backtest        # Run backtests with built-in strategies
    python main.py predict         # Generate probability predictions
    python main.py all             # Run everything end-to-end
    python main.py visualise       # Generate charts from cached data
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOG_FORMAT, LOG_LEVEL, OUTPUT_DIR


def setup_logging(level: str = LOG_LEVEL) -> None:
    logging.basicConfig(level=getattr(logging, level), format=LOG_FORMAT)


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_download(args) -> None:
    """Download Nifty 50 and India VIX data from Yahoo Finance."""
    from src.data.yfinance_downloader import download_all

    print("⬇  Downloading data from Yahoo Finance …")
    paths = download_all()
    for name, p in paths.items():
        print(f"   {name:>8}: {p}")
    print("✓  Download complete.")

    # Rebuild master parquet
    from src.data.loader import build_master_dataframe
    print("   Rebuilding master dataframe …")
    df = build_master_dataframe(save=True)
    print(f"   Master: {len(df)} rows ({df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()})")
    print("✓  Done.")


def cmd_backtest(args) -> None:
    """Run backtests with built-in strategies."""
    from src.data.loader import load_master
    from src.simulator.backtester import (
        Backtester,
        rsi_mean_reversion_strategy,
        sma_crossover_strategy,
        vix_regime_strategy,
    )
    from src.simulator.visualisation import plot_equity_curve

    print("📊  Loading data …")
    df = load_master(rebuild=args.rebuild)
    print(f"   Loaded {len(df)} rows  ({df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()})")

    strategies = [
        sma_crossover_strategy(10, 50),
        sma_crossover_strategy(20, 100),
        rsi_mean_reversion_strategy(),
    ]
    if "vix_close" in df.columns:
        strategies.append(vix_regime_strategy())

    results = []
    for strat in strategies:
        bt = Backtester(
            df,
            strategy=strat,
            initial_capital=args.capital,
            stop_loss_pct=args.stop_loss / 100,
            take_profit_pct=args.take_profit / 100,
        )
        result = bt.run()
        results.append(result)
        print(f"\n{result.summary()}")

        # Save equity curve chart
        name_tag = getattr(strat, "__name__", "custom").replace("(", "_").replace(")", "").replace(",", "_")
        plot_equity_curve(
            result.equity_curve,
            title=f"Equity Curve – {result.strategy_name}",
            save_path=OUTPUT_DIR / f"equity_{name_tag}.png",
        )

    # Comparative summary
    print("\n\n═══ Strategy Comparison ═══")
    print(f"{'Strategy':<35} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>10}")
    print("─" * 75)
    for r in results:
        print(
            f"{r.strategy_name:<35} {r.total_return_pct:>+10.2f} "
            f"{r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>8.2f} "
            f"{r.win_rate_pct:>10.1f}"
        )


def cmd_predict(args) -> None:
    """Generate probability predictions for Nifty movements."""
    from src.data.loader import load_master
    from src.predictor.probability import predict
    from src.simulator.visualisation import (
        plot_monte_carlo_paths,
        plot_prediction_distribution,
    )

    print("🔮  Loading data …")
    df = load_master(rebuild=args.rebuild)
    print(f"   Loaded {len(df)} rows")

    print("   Running prediction engine …")
    report = predict(df, method=args.method, n_simulations=args.simulations)

    print(f"\n═══ Prediction Report (as of {report.as_of_date.date()}) ═══")
    print(f"   Current Nifty : ₹{report.current_price:,.2f}")
    if report.current_vix is not None:
        print(f"   Current VIX   : {report.current_vix:.2f}  (regime: {report.vix_regime})")

    summary = report.summary_df()
    print(f"\n{summary.to_string(index=False)}\n")

    # Detailed per-horizon output
    for name, pred in report.predictions.items():
        print(f"\n── {name} ({pred.horizon_days} trading days) ──")
        print(f"   P(Nifty ↑) = {pred.prob_up*100:.1f}%")
        print(f"   P(Nifty ↓) = {pred.prob_down*100:.1f}%")
        print(f"   Expected   = ₹{pred.mean_price:,.2f}  (std ₹{pred.std_price:,.2f})")
        for cl, (lo, hi) in pred.percentiles.items():
            print(f"   {int(cl*100)}% CI     : ₹{lo:,.2f} – ₹{hi:,.2f}")

        # Charts
        try:
            plot_prediction_distribution(pred)
        except Exception as e:
            logging.getLogger(__name__).warning("Could not plot distribution for %s: %s", name, e)

        if pred.simulated_paths is not None:
            try:
                plot_monte_carlo_paths(pred)
            except Exception as e:
                logging.getLogger(__name__).warning("Could not plot MC paths for %s: %s", name, e)

    print(f"\n✓  Charts saved to {OUTPUT_DIR}")


def cmd_visualise(args) -> None:
    """Generate charts from cached data."""
    from src.data.loader import load_master
    from src.simulator.visualisation import plot_price_with_vix

    df = load_master()
    plot_price_with_vix(df)
    print(f"✓  Charts saved to {OUTPUT_DIR}")


def cmd_all(args) -> None:
    """Run the full pipeline: download → backtest → predict."""
    cmd_download(args)
    cmd_backtest(args)
    cmd_predict(args)
    cmd_visualise(args)


def cmd_sync(args) -> None:
    """Sync data: fetch only new rows since last download."""
    from src.data.yfinance_downloader import sync_data
    from src.data.loader import build_master_dataframe

    print("🔄  Syncing data from Yahoo Finance …")
    result = sync_data()
    for name, info in result.items():
        print(f"   {name:>8}: {info['status']} (+{info['rows_added']} rows, total: {info['total_rows']}, last: {info['last_date']})")

    print("   Rebuilding master dataframe …")
    df = build_master_dataframe(save=True)
    print(f"   Master: {len(df)} rows ({df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()})")
    print("✓  Sync complete.")


def cmd_download_valuation(args) -> None:
    """Download / initialise the Nifty PE/PB/DivYield valuation CSV."""
    from src.data.valuation_scraper import download_history

    print("📊  Downloading Nifty valuation data (PE/PB/DivYield) …")
    df = download_history()
    print(f"   Saved {len(df)} row(s) → data/raw/nifty_pe_pb.csv")
    print(f"   Latest: PE={df['PE'].iloc[-1]:.2f}  PB={df['PB'].iloc[-1]:.2f}  DivYield={df['DivYield'].iloc[-1]:.2f}%")
    print("✓  Done.")


def cmd_sync_valuation(args) -> None:
    """Sync Nifty PE/PB/DivYield: append today's value if not already present."""
    from src.data.valuation_scraper import sync_data as val_sync

    print("🔄  Syncing valuation data from nifty-pe-ratio.com …")
    df = val_sync()
    latest = df.iloc[-1]
    print(
        f"   Latest: Date={latest['Date'].date() if hasattr(latest['Date'], 'date') else latest['Date']}  "
        f"PE={latest['PE']:.2f}  PB={latest['PB']:.2f}  DivYield={latest['DivYield']:.2f}%"
    )
    print(f"   Total rows: {len(df)}")
    print("✓  Sync complete.")


def cmd_download_crypto(args) -> None:
    """Download Crypto data (Bitcoin, etc.) from Yahoo Finance."""
    from src.data.crypto_downloader import download_all_crypto

    print("⬇  Downloading Crypto data from Yahoo Finance …")
    paths = download_all_crypto()
    for name, p in paths.items():
        print(f"   {name:>8}: {p}")
    print("✓  Crypto download complete.")


def cmd_sync_crypto(args) -> None:
    """Sync Crypto data: fetch only new rows since last download."""
    from src.data.crypto_downloader import sync_all_crypto

    print("🔄  Syncing Crypto data from Yahoo Finance …")
    result = sync_all_crypto()
    for name, info in result.items():
        print(f"   {name:>8}: {info['status']} (+{info['rows_added']} rows, total: {info['total_rows']}, last: {info['last_date']})")
    print("✓  Crypto sync complete.")


# ── Argument Parser ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AITrading",
        description="Nifty 50 Trading Simulator with VIX-based Probability Prediction",
    )
    parser.add_argument("--log-level", default=LOG_LEVEL, help="Logging level")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # download
    sub.add_parser("download", help="Download full data from Yahoo Finance")

    # sync
    sub.add_parser("sync", help="Sync latest data from Yahoo Finance (incremental)")

    # download-valuation
    sub.add_parser("download-valuation", help="Download Nifty PE/PB/DivYield from nifty-pe-ratio.com")

    # sync-valuation
    sub.add_parser("sync-valuation", help="Sync latest PE/PB/DivYield (append today if absent)")

    # download-crypto
    sub.add_parser("download-crypto", help="Download Crypto data (Bitcoin, etc.) from Yahoo Finance")

    # sync-crypto
    sub.add_parser("sync-crypto", help="Sync latest Crypto data (incremental)")

    # backtest
    bt = sub.add_parser("backtest", help="Run backtests")
    bt.add_argument("--capital", type=float, default=1_000_000, help="Initial capital (₹)")
    bt.add_argument("--stop-loss", type=float, default=2.0, help="Stop loss %%")
    bt.add_argument("--take-profit", type=float, default=4.0, help="Take profit %%")
    bt.add_argument("--rebuild", action="store_true", help="Rebuild data from raw CSVs")

    # predict
    pr = sub.add_parser("predict", help="Probability predictions")
    pr.add_argument("--method", choices=["empirical", "monte_carlo", "regime", "ensemble"], default="ensemble")
    pr.add_argument("--simulations", type=int, default=10_000, help="Monte Carlo simulations")
    pr.add_argument("--rebuild", action="store_true", help="Rebuild data from raw CSVs")

    # visualise
    sub.add_parser("visualise", help="Generate charts")

    # all
    a = sub.add_parser("all", help="Run full pipeline")
    a.add_argument("--capital", type=float, default=1_000_000)
    a.add_argument("--stop-loss", type=float, default=2.0)
    a.add_argument("--take-profit", type=float, default=4.0)
    a.add_argument("--method", default="ensemble")
    a.add_argument("--simulations", type=int, default=10_000)
    a.add_argument("--rebuild", action="store_true")

    return parser


# ── Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    commands = {
        "download": cmd_download,
        "sync": cmd_sync,
        "download-valuation": cmd_download_valuation,
        "sync-valuation": cmd_sync_valuation,
        "download-crypto": cmd_download_crypto,
        "sync-crypto": cmd_sync_crypto,
        "backtest": cmd_backtest,
        "predict": cmd_predict,
        "visualise": cmd_visualise,
        "all": cmd_all,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    fn = commands.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    fn(args)


if __name__ == "__main__":
    main()
