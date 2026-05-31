"""
FastAPI Backend – AITrading Web Application
────────────────────────────────────────────
REST API for:
  • Strategy CRUD
  • Backtest execution (single, walk-forward, Monte Carlo, sweep)
  • Strategy correlation analysis
  • Data inspection & quality
  • Copilot chat interaction
  • Probability predictions
  • Monthly return heatmap
  • CSV export
"""

import csv
import io
import json
import logging
import math
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_POSITION_SIZE,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    NIFTY_CLOSE_COL,
    OUTPUT_DIR,
    PREDICTION_HORIZONS,
)
from src.data.yfinance_downloader import sync_data as yf_sync_data, download_all as yf_download_all
from src.simulator.backtester import (
    Backtester,
    walk_forward_analysis,
    monte_carlo_trade_resample,
    parameter_sweep,
    strategy_correlation_matrix,
    sma_crossover_strategy,
    rsi_mean_reversion_strategy,
    macd_crossover_strategy,
    bollinger_band_strategy,
    atr_breakout_strategy,
    vix_regime_strategy,
    ema_crossover_strategy,
    stochastic_oscillator_strategy,
    mean_reversion_zscore_strategy,
    macd_histogram_strategy,
    composite_sniper_strategy,
)
from src.simulator.registry import get_strategy_fn, STRATEGY_REGISTRY, list_registered_strategies
from src.data.loader import load_master, build_master_dataframe, validate_data_quality
from src.predictor.probability import (
    predict as run_prediction,
    project_all_horizons,
    backtest_predictions,
)
from src.storage.database import (
    init_db,
    list_strategies,
    get_strategy,
    create_strategy,
    update_strategy,
    delete_strategy,
    save_backtest_run,
    list_backtest_runs,
    get_backtest_run,
    delete_backtest_run,
    delete_all_backtest_runs,
    count_backtest_runs,
    create_copilot_session,
    get_copilot_session,
    update_copilot_session,
)
from src.web.copilot import process_message, detect_intent, handle_llm_query

logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler replacing deprecated @app.on_event('startup')."""
    init_db()
    # Sync curated event catalog to DB
    try:
        from src.storage.database import sync_events_to_db
        sync_events_to_db()
    except Exception as e:
        logger.warning("Could not sync events to DB: %s", e)
    logger.info("AITrading Web App started.")
    try:
        _get_df()
        logger.info("Data loaded into memory.")
    except Exception as e:
        logger.warning("Could not preload data: %s", e)
    yield  # app is running
    logger.info("AITrading Web App shutting down.")


app = FastAPI(
    title="AITrading",
    version="2.0.0",
    description="Nifty 50 backtesting, prediction & analysis platform",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Data", "description": "Market data inspection, quality checks, and sync"},
        {"name": "Strategies", "description": "CRUD operations on trading strategies"},
        {"name": "Backtest", "description": "Run backtests and analyse results"},
        {"name": "Advanced", "description": "Walk-forward, Monte Carlo, correlation, parameter sweep"},
        {"name": "Predictions", "description": "Probability forecasts for Nifty 50"},
        {"name": "Copilot", "description": "AI chat assistant"},
        {"name": "Events", "description": "Macro-economic event metadata"},
        {"name": "Export", "description": "Download results as CSV"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request-ID Middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique X-Request-ID to every request/response for tracing."""
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = req_id
    logger.info("[%s] %s %s", req_id, request.method, request.url.path)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Cache dataframe in memory
_df_cache: dict = {"df": None}


def _safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, replacing NaN/Inf with None for JSON safety.

    Uses vectorized pandas ops instead of per-cell Python loops (~10× faster
    on 2000+ row DataFrames).
    """
    clean = df.copy()
    # Replace Inf/-Inf with NaN (vectorized on numeric cols only)
    numeric_cols = clean.select_dtypes(include="number").columns
    if len(numeric_cols):
        clean[numeric_cols] = clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # Compute NaN mask before dtype change, then convert to object so None is preserved
    mask = clean.notna()
    return clean.astype(object).where(mask, other=None).to_dict(orient="records")


def _get_df(rebuild: bool = False) -> pd.DataFrame:
    if _df_cache["df"] is None or rebuild:
        _df_cache["df"] = load_master(rebuild=rebuild)
    return _df_cache["df"]


# ── Pydantic Models ──────────────────────────────────────────────────────

class StrategyCreateReq(BaseModel):
    name: str
    type: str  # sma_crossover, rsi_mean_reversion, vix_regime
    params: dict
    description: str = ""


class StrategyUpdateReq(BaseModel):
    name: Optional[str] = None
    params: Optional[dict] = None
    description: Optional[str] = None
    is_active: Optional[int] = None


class BacktestReq(BaseModel):
    strategy_id: Optional[int] = None
    strategy_type: Optional[str] = None
    params: Optional[dict] = None
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    position_size_pct: float = DEFAULT_POSITION_SIZE * 100  # as percentage (1-100)
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT * 100
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT * 100
    trailing_stop_pct: float = 0.0  # trailing stop as percentage (0 = disabled)
    slippage_pct: float = 0.01      # slippage per side in % (default 0.01%)
    commission_pct: float = 0.02    # round-trip commission in % (default 0.02%)
    vol_target: float = 0.0         # annualised vol target for sizing (0 = disabled)
    cooldown_bars: int = 0           # min bars between exit and next entry (0 = disabled)
    max_holding_bars: int = 0        # max bars to hold a position (0 = unlimited)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    crypto_id: Optional[str] = None


class CopilotMessageReq(BaseModel):
    message: str
    session_id: Optional[int] = None
    use_llm: Optional[bool] = True  # whether to use LLM for unknown intents


class NLBacktestReq(BaseModel):
    """Natural language backtest request — forwarded to the LLM agent."""
    instruction: str


class WalkForwardReq(BaseModel):
    """Walk-forward analysis request."""
    strategy_type: str
    params: dict = Field(default_factory=dict)
    n_splits: int = Field(default=5, ge=2, le=20)
    train_pct: float = Field(default=0.70, ge=0.5, le=0.9)
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT * 100
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT * 100


class MonteCarloReq(BaseModel):
    """Monte Carlo trade resampling request."""
    run_id: int
    n_simulations: int = Field(default=1000, ge=100, le=50000)


class CorrelationReq(BaseModel):
    """Strategy correlation matrix request."""
    strategies: list[dict] = Field(
        ..., min_length=2,
        description='List of {"type": "sma_crossover", "params": {...}} objects'
    )
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SweepReq(BaseModel):
    """Parameter sensitivity sweep request."""
    strategy_type: str
    param_name: str
    param_values: list
    base_params: dict = Field(default_factory=dict)
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT * 100
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT * 100


# ── HTML Entry Point ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        )
    return HTMLResponse("<h1>AITrading – static/index.html not found</h1>")


# ── Data Endpoints ───────────────────────────────────────────────────────

@app.get("/api/data/summary", tags=["Data"])
async def data_summary():
    """Return summary statistics of the loaded dataset."""
    try:
        df = _get_df()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data not loaded: {e}")

    close = df[NIFTY_CLOSE_COL]
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "date_start": str(pd.Timestamp(df["Date"].iloc[0]).date()),
        "date_end": str(pd.Timestamp(df["Date"].iloc[-1]).date()),
        "nifty_last": round(float(close.iloc[-1]), 2),
        "nifty_min": round(float(close.min()), 2),
        "nifty_max": round(float(close.max()), 2),
        "nifty_mean": round(float(close.mean()), 2),
        "has_vix": "vix_close" in df.columns,
    }
    if "vix_close" in df.columns:
        vix = df["vix_close"].dropna()
        summary["vix_last"] = round(float(vix.iloc[-1]), 2) if len(vix) > 0 else None
        summary["vix_mean"] = round(float(vix.mean()), 2) if len(vix) > 0 else None
    return summary


@app.get("/api/data/chart", tags=["Data"])
async def data_chart(start: Optional[str] = None, end: Optional[str] = None):
    """Return OHLC + VIX data for charting (sampled for performance)."""
    df = _get_df()
    if start:
        df = df[df["Date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["Date"] <= pd.Timestamp(end)]

    # Sample if too large for the browser
    max_points = 2000
    step = max(1, len(df) // max_points)
    sampled = df.iloc[::step].copy()

    cols = ["Date", "Open", "High", "Low", "Close"]
    if "Volume" in sampled.columns:
        cols.append("Volume")
    if "vix_close" in sampled.columns:
        cols.append("vix_close")

    available_cols = [c for c in cols if c in sampled.columns]
    result = sampled[available_cols].copy()
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    return _safe_records(result)


@app.get("/api/data/preview", tags=["Data"])
async def data_preview(rows: int = 50):
    """Return the last N rows of the dataset for preview."""
    df = _get_df()
    preview = df.tail(rows).copy()
    display_cols = ["Date", "Open", "High", "Low", "Close"]
    if "Volume" in preview.columns:
        display_cols.append("Volume")
    if "vix_close" in preview.columns:
        display_cols.append("vix_close")
    if "simple_return" in preview.columns:
        display_cols.append("simple_return")
    if "rsi_14" in preview.columns:
        display_cols.append("rsi_14")

    available = [c for c in display_cols if c in preview.columns]
    result = preview[available].copy()
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    # Round numeric columns
    for col in result.columns:
        if result[col].dtype in ("float64", "float32"):
            result[col] = result[col].round(2)
    return _safe_records(result)


@app.post("/api/data/sync", tags=["Data"])
async def data_sync():
    """
    Sync data from Yahoo Finance: check the last data point in each CSV,
    fetch new data since then, rebuild the master parquet, and reload.
    """
    try:
        # Step 1: Incremental sync from yfinance
        sync_result = yf_sync_data()

        # Step 2: Rebuild master parquet with fresh data
        build_master_dataframe(save=True)

        # Step 3: Reload into memory
        _df_cache["df"] = None
        df = _get_df(rebuild=False)  # loads from fresh parquet

        # Build response
        resp = {
            "ok": True,
            "nifty": sync_result["nifty"],
            "vix": sync_result["vix"],
            "master_rows": len(df),
            "date_start": str(pd.Timestamp(df["Date"].iloc[0]).date()),
            "date_end": str(pd.Timestamp(df["Date"].iloc[-1]).date()),
        }
        logger.info("Data sync complete: %s", resp)
        return resp
    except Exception as e:
        logger.error("Data sync failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


@app.post("/api/data/download", tags=["Data"])
async def data_download():
    """
    Full re-download of all data from Yahoo Finance.
    Replaces existing CSVs entirely.
    """
    try:
        yf_download_all()
        build_master_dataframe(save=True)
        _df_cache["df"] = None
        df = _get_df(rebuild=False)
        return {
            "ok": True,
            "master_rows": len(df),
            "date_start": str(pd.Timestamp(df["Date"].iloc[0]).date()),
            "date_end": str(pd.Timestamp(df["Date"].iloc[-1]).date()),
        }
    except Exception as e:
        logger.error("Data download failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")


# ── Strategy Endpoints ───────────────────────────────────────────────────

@app.get("/api/strategies/registry", tags=["Strategies"])
async def api_strategy_registry():
    """List all registered strategy types with their parameter schemas."""
    return list_registered_strategies()


@app.get("/api/strategies", tags=["Strategies"])
async def api_list_strategies():
    strategies = list_strategies()
    for s in strategies:
        if isinstance(s["params"], str):
            s["params"] = json.loads(s["params"])
    return strategies


@app.get("/api/strategies/{strategy_id}", tags=["Strategies"])
async def api_get_strategy(strategy_id: int):
    s = get_strategy(strategy_id)
    if not s:
        raise HTTPException(status_code=404, detail="Strategy not found")
    if isinstance(s["params"], str):
        s["params"] = json.loads(s["params"])
    return s


@app.post("/api/strategies", tags=["Strategies"])
async def api_create_strategy(req: StrategyCreateReq):
    try:
        s = create_strategy(req.name, req.type, req.params, req.description)
        if isinstance(s["params"], str):
            s["params"] = json.loads(s["params"])
        return s
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/strategies/{strategy_id}", tags=["Strategies"])
async def api_update_strategy(strategy_id: int, req: StrategyUpdateReq):
    s = update_strategy(strategy_id, req.name, req.params, req.description, req.is_active)
    if not s:
        raise HTTPException(status_code=404, detail="Strategy not found")
    if isinstance(s["params"], str):
        s["params"] = json.loads(s["params"])
    return s


@app.delete("/api/strategies/{strategy_id}", tags=["Strategies"])
async def api_delete_strategy(strategy_id: int):
    if delete_strategy(strategy_id):
        return {"ok": True}
    raise HTTPException(status_code=404, detail="Strategy not found")


# ── Backtest Endpoints ───────────────────────────────────────────────────

def _validate_params(stype: str, params: dict):
    """Validate strategy parameters before building the strategy function.

    Raises ValueError with a clear message if any parameter is out of range.
    """
    def _pos(name, val):
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")

    def _lt(name_a, a, name_b, b):
        if a >= b:
            raise ValueError(f"{name_a} ({a}) must be < {name_b} ({b})")

    if stype == "sma_crossover":
        fw, sw = int(params.get("fast_window", 50)), int(params.get("slow_window", 200))
        _pos("fast_window", fw); _pos("slow_window", sw); _lt("fast_window", fw, "slow_window", sw)
    elif stype == "ema_crossover":
        fs, ss = int(params.get("fast_span", 12)), int(params.get("slow_span", 26))
        _pos("fast_span", fs); _pos("slow_span", ss); _lt("fast_span", fs, "slow_span", ss)
    elif stype == "rsi_mean_reversion":
        os_, ob = float(params.get("oversold", 30)), float(params.get("overbought", 70))
        _lt("oversold", os_, "overbought", ob)
        if not (0 <= os_ <= 100) or not (0 <= ob <= 100):
            raise ValueError(f"RSI thresholds must be in [0,100], got {os_}/{ob}")
    elif stype in ("macd_crossover", "macd_histogram"):
        fe, se, sp = int(params.get("fast_ema", 12)), int(params.get("slow_ema", 26)), int(params.get("signal_period", 9))
        _pos("fast_ema", fe); _pos("slow_ema", se); _pos("signal_period", sp); _lt("fast_ema", fe, "slow_ema", se)
    elif stype == "bollinger_band":
        w, ns = int(params.get("window", 20)), float(params.get("num_std", 2.0))
        _pos("window", w)
        if ns <= 0:
            raise ValueError(f"num_std must be > 0, got {ns}")
    elif stype == "atr_breakout":
        _pos("sma_window", int(params.get("sma_window", 20)))
        _pos("atr_period", int(params.get("atr_period", 14)))
        if float(params.get("atr_multiplier", 1.5)) <= 0:
            raise ValueError("atr_multiplier must be > 0")
    elif stype == "vix_regime":
        bb, sa = float(params.get("buy_below", 15)), float(params.get("sell_above", 25))
        _pos("buy_below", bb); _pos("sell_above", sa); _lt("buy_below", bb, "sell_above", sa)
    elif stype == "stochastic_oscillator":
        _pos("k_period", int(params.get("k_period", 14)))
        _pos("d_period", int(params.get("d_period", 3)))
        os_, ob = float(params.get("oversold", 20)), float(params.get("overbought", 80))
        _lt("oversold", os_, "overbought", ob)
    elif stype == "mean_reversion_zscore":
        _pos("lookback", int(params.get("lookback", 20)))
    elif stype == "composite_sniper":
        _pos("sma_period", int(params.get("sma_period", 50)))
        _pos("rsi_period", int(params.get("rsi_period", 14)))
        os_, ob = float(params.get("rsi_oversold", 35)), float(params.get("rsi_overbought", 65))
        _lt("rsi_oversold", os_, "rsi_overbought", ob)


def _build_strategy_fn(stype: str, params: dict):
    """Convert strategy type + params into a callable signal function.

    Uses the strategy registry pattern — all strategies auto-register
    via ``@register_strategy`` decorators in ``backtester.py``.
    Validates parameters first — raises ValueError on invalid input.
    """
    _validate_params(stype, params)
    return get_strategy_fn(stype, params)


def _run_single_backtest(stype: str, params: dict, strategy_id: Optional[int],
                         initial_capital: float, position_size_pct: float,
                         stop_loss_pct: float, take_profit_pct: float,
                         start_date: Optional[str], end_date: Optional[str],
                         trailing_stop_pct: float = 0.0,
                         slippage_pct: float = 0.01,
                         commission_pct: float = 0.02,
                         vol_target: float = 0.0,
                         cooldown_bars: int = 0,
                         max_holding_bars: int = 0,
                         crypto_id: Optional[str] = None) -> dict:
    """Run a single backtest and persist results."""
    if crypto_id:
        from config.settings import CRYPTO_RAW_DIR
        csv_path = CRYPTO_RAW_DIR / f"{crypto_id}.csv"
        if not csv_path.exists():
            raise ValueError(f"Crypto data for {crypto_id} not found.")
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df = _get_df()

    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]

    if len(df) < 100:
        raise ValueError(f"Not enough data: {len(df)} rows. Need at least 100.")

    strat_fn = _build_strategy_fn(stype, params)
    bt = Backtester(
        df, strat_fn,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct / 100,
        stop_loss_pct=stop_loss_pct / 100,
        take_profit_pct=take_profit_pct / 100,
        trailing_stop_pct=trailing_stop_pct / 100,
        slippage_pct=slippage_pct / 100,
        commission_pct=commission_pct / 100,
        vol_target=vol_target / 100 if vol_target > 0 else 0.0,
        cooldown_bars=cooldown_bars,
        max_holding_bars=max_holding_bars,
    )
    result = bt.run()

    # Serialise equity curve with min/max-preserving downsampling.
    # Naive iloc[::step] can miss the max-drawdown trough and equity peaks.
    # Instead, split into buckets of `step` bars and keep the row with the
    # highest and lowest equity in each bucket (plus first & last rows).
    eq = result.equity_curve.copy()
    max_points = 500
    if len(eq) > max_points:
        step = max(1, len(eq) // (max_points // 2))  # 2 rows per bucket
        bucket = np.arange(len(eq)) // step
        idx_min = eq.groupby(bucket)["equity"].idxmin()
        idx_max = eq.groupby(bucket)["equity"].idxmax()
        
        # Include all trade entry/exit bars so they are not missing from the downsampled graph
        trade_bars = set()
        for t in result.trades:
            if t.entry_bar is not None:
                trade_bars.add(t.entry_bar)
            if t.exit_bar is not None:
                trade_bars.add(t.exit_bar)
        trade_bars = {b for b in trade_bars if 0 <= b < len(eq)}
        
        keep = sorted(set(idx_min) | set(idx_max) | {0, len(eq) - 1} | trade_bars)
        eq_sampled = eq.iloc[keep].copy()
    else:
        eq_sampled = eq.copy()
    eq_sampled["Date"] = eq_sampled["Date"].dt.strftime("%Y-%m-%d")
    equity_records = _safe_records(eq_sampled)
    equity_json = json.dumps(equity_records)

    # Serialise all trades (complete history, no truncation)
    trades_list = []
    for t in result.trades:
        trade_rec = {
            "entry_date": str(pd.Timestamp(t.entry_date).date()) if t.entry_date is not None else "",
            "entry_price": round(t.entry_price, 2),
            "exit_date": str(pd.Timestamp(t.exit_date).date()) if t.exit_date is not None else "",
            "exit_price": round(t.exit_price, 2) if t.exit_price else 0,
            "direction": t.direction,
            "pnl": round(t.pnl, 2) if t.pnl else 0,
            "exit_reason": t.exit_reason or "",
            "mae": round(t.mae, 4) if t.mae is not None else None,
            "mfe": round(t.mfe, 4) if t.mfe is not None else None,
            "r_multiple": round(t.r_multiple, 2) if t.r_multiple is not None else None,
        }
        # Include duration if available
        if t.entry_bar is not None and t.exit_bar is not None:
            trade_rec["duration_bars"] = t.exit_bar - t.entry_bar
        trades_list.append(trade_rec)
    trades_json = json.dumps(trades_list)

    result_dict = {
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_return_pct": result.total_return_pct,
        "annual_return_pct": result.annual_return_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "calmar_ratio": result.calmar_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate_pct": result.win_rate_pct,
        "avg_trade_pnl": result.avg_trade_pnl,
        "profit_factor": result.profit_factor,
        "max_consecutive_losses": result.max_consecutive_losses,
        "max_consecutive_wins": result.max_consecutive_wins,
        "max_consecutive_win_days": result.max_consecutive_win_days,
        "max_consecutive_loss_days": result.max_consecutive_loss_days,
        "risk_reward_ratio": result.risk_reward_ratio,
        "max_drawdown_duration": result.max_drawdown_duration,
        "avg_trade_duration": result.avg_trade_duration,
        "exposure_time_pct": result.exposure_time_pct,
        "benchmark_return_pct": result.benchmark_return_pct,
        "alpha_pct": result.alpha_pct,
        "information_ratio": result.information_ratio,
        # New risk metrics
        "omega_ratio": result.omega_ratio,
        "tail_ratio": result.tail_ratio,
        "value_at_risk_95": result.value_at_risk_95,
        "cvar_95": result.cvar_95,
        "avg_mae_pct": result.avg_mae_pct,
        "avg_mfe_pct": result.avg_mfe_pct,
        "avg_r_multiple": result.avg_r_multiple,
    }
    # Sanitise NaN/Inf for JSON
    for k, v in result_dict.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            result_dict[k] = None

    run_id = save_backtest_run(
        strategy_id=strategy_id,
        strategy_name=result.strategy_name,
        params=params,
        result_dict=result_dict,
        equity_json=equity_json,
        trades_json=trades_json,
        data_start=str(pd.Timestamp(df["Date"].iloc[0]).date()),
        data_end=str(pd.Timestamp(df["Date"].iloc[-1]).date()),
    )

    return {
        "run_id": run_id,
        "strategy_name": result.strategy_name,
        **result_dict,
        "equity_curve": json.loads(equity_json),
        "trades": trades_list,
        "data_start": str(pd.Timestamp(df["Date"].iloc[0]).date()),
        "data_end": str(pd.Timestamp(df["Date"].iloc[-1]).date()),
    }


@app.post("/api/backtest", tags=["Backtest"])
async def api_run_backtest(req: BacktestReq):
    try:
        strategy_id = req.strategy_id
        stype = req.strategy_type
        params = req.params or {}

        # If strategy_id provided, load from DB
        if strategy_id:
            strat = get_strategy(strategy_id)
            if not strat:
                raise HTTPException(status_code=404, detail="Strategy not found")
            stype = strat["type"]
            params = json.loads(strat["params"]) if isinstance(strat["params"], str) else strat["params"]

        if not stype:
            raise HTTPException(status_code=400, detail="Provide strategy_id or strategy_type+params")

        result = _run_single_backtest(
            stype, params, strategy_id,
            req.initial_capital, req.position_size_pct, req.stop_loss_pct, req.take_profit_pct,
            req.start_date, req.end_date, req.trailing_stop_pct,
            req.slippage_pct, req.commission_pct, req.vol_target, req.cooldown_bars,
            req.max_holding_bars, req.crypto_id,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Backtest error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/optimize", tags=["Backtest"])
async def api_optimize_backtest(req: BacktestReq):
    try:
        stype = req.strategy_type
        if not stype:
            raise HTTPException(status_code=400, detail="Provide strategy_type")

        # Plausible Search Space definitions
        param_grids = {
            "sma_crossover": {
                "fast_window": [10, 20, 50],
                "slow_window": [100, 150, 200]
            },
            "rsi_mean_reversion": {
                "oversold": [20, 25, 30, 35],
                "overbought": [65, 70, 75, 80]
            },
            "macd_crossover": {
                "fast_ema": [8, 12, 16],
                "slow_ema": [Slow_Ema for Slow_Ema in [21, 26, 30]],
                "signal_period": [7, 9, 12]
            },
            "bollinger_band": {
                "window": [10, 20, 30],
                "num_std": [1.5, 2.0, 2.5]
            },
            "atr_breakout": {
                "sma_window": [10, 20],
                "atr_period": [10, 14],
                "atr_multiplier": [1.0, 1.5, 2.0]
            },
            "vix_regime": {
                "buy_below": [12, 15, 18],
                "sell_above": [20, 25, 30]
            },
            "ema_crossover": {
                "fast_span": [5, 10, 15, 20],
                "slow_span": [50, 100, 150]
            },
            "stochastic_oscillator": {
                "k_period": [10, 14, 21],
                "d_period": [3, 5],
                "oversold": [15, 20, 25],
                "overbought": [75, 80, 85]
            },
            "mean_reversion_zscore": {
                "lookback": [10, 20, 30],
                "entry_z": [-1.5, -2.0, -2.5],
                "exit_z": [-0.5, 0.0, 0.5]
            },
            "macd_histogram": {
                "fast_ema": [8, 12],
                "slow_ema": [24, 26],
                "signal_period": [7, 9]
            },
            "composite_sniper": {
                "sma_period": [20, 50],
                "rsi_period": [14],
                "rsi_oversold": [30, 35],
                "rsi_overbought": [65, 70],
                "vix_calm_threshold": [15, 20],
                "bb_window": [20],
                "bb_num_std": [2.0]
            }
        }

        grid = param_grids.get(stype)
        if not grid:
            raise HTTPException(status_code=400, detail=f"No optimization grid defined for {stype}")

        # Compute cartesian product of parameters
        import itertools
        keys = list(grid.keys())
        values = list(grid.values())
        combinations = [dict(zip(keys, prod)) for prod in itertools.product(*values)]

        best_score = -float('inf')
        best_params = None
        best_result = None
        runs_count = 0

        # We'll run the combinations
        for p in combinations:
            try:
                result = _run_single_backtest(
                    stype, p, None,
                    req.initial_capital, req.position_size_pct, req.stop_loss_pct, req.take_profit_pct,
                    req.start_date, req.end_date, req.trailing_stop_pct,
                    req.slippage_pct, req.commission_pct, req.vol_target, req.cooldown_bars,
                    req.max_holding_bars, req.crypto_id,
                )
                runs_count += 1
                
                # Metric to optimise: Sharpe Ratio, fallback to Total Return if Sharpe is None or equivalent.
                sharpe = result.get("sharpe_ratio") or 0.0
                total_return = result.get("total_return_pct") or -100.0
                
                # Combine Sharpe with a minor weight of total return for ties
                score = sharpe + (total_return * 0.001)
                if score > best_score:
                    best_score = score
                    best_params = p
                    best_result = result
            except Exception:
                continue

        if not best_result:
            raise HTTPException(status_code=500, detail="Optimization failed - all runs crashed.")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "evaluated_combinations": runs_count,
            "best_result": best_result
        }
    except Exception as e:
        logger.error("Optimizer error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/runs", tags=["Backtest"])
async def api_list_runs(limit: int = 50, offset: int = 0):
    """List backtest runs with pagination. Returns {total, limit, offset, runs}."""
    total = count_backtest_runs()
    runs = list_backtest_runs(limit, offset)
    for r in runs:
        if isinstance(r.get("params"), str):
            r["params"] = json.loads(r["params"])
    return {"total": total, "limit": limit, "offset": offset, "runs": runs}


@app.get("/api/backtest/runs/{run_id}", tags=["Backtest"])
async def api_get_run(run_id: int):
    r = get_backtest_run(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    if isinstance(r.get("equity_json"), str):
        r["equity_curve"] = json.loads(r["equity_json"])
    if isinstance(r.get("trades_json"), str):
        r["trades"] = json.loads(r["trades_json"])
    if isinstance(r.get("params"), str):
        r["params"] = json.loads(r["params"])
    return r


@app.delete("/api/backtest/runs/{run_id}", tags=["Backtest"])
async def api_delete_run(run_id: int):
    """Delete a single backtest run."""
    if delete_backtest_run(run_id):
        return {"ok": True}
    raise HTTPException(status_code=404, detail="Run not found")


@app.delete("/api/backtest/runs", tags=["Backtest"])
async def api_delete_all_runs():
    """Delete all backtest runs."""
    count = delete_all_backtest_runs()
    return {"ok": True, "deleted": count}


# ── Prediction Endpoint ──────────────────────────────────────────────────

@app.get("/api/predict", tags=["Predictions"])
async def api_predict(method: str = "ensemble", simulations: int = 10000):
    try:
        df = _get_df()
        report = run_prediction(df, method=method, n_simulations=simulations)
        result = {
            "as_of_date": str(pd.Timestamp(report.as_of_date).date()),
            "current_price": report.current_price,
            "current_vix": report.current_vix,
            "vix_regime": report.vix_regime,
            "predictions": {},
        }
        for name, pred in report.predictions.items():
            result["predictions"][name] = {
                "horizon_days": pred.horizon_days,
                "mean_price": round(pred.mean_price, 2),
                "median_price": round(pred.median_price, 2),
                "std_price": round(pred.std_price, 2),
                "prob_up": round(pred.prob_up * 100, 2),
                "prob_down": round(pred.prob_down * 100, 2),
                "percentiles": {
                    str(int(k * 100)): {"low": round(v[0], 2), "high": round(v[1], 2)}
                    for k, v in pred.percentiles.items()
                },
            }
        return result
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/candles", tags=["Predictions"])
async def api_predict_candles(simulations: int = 10000):
    """Projected OHLC candles for each horizon via Monte Carlo."""
    try:
        df = _get_df()
        return project_all_horizons(df, n_simulations=simulations)
    except Exception as e:
        logger.error("Projection error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/accuracy", tags=["Predictions"])
async def api_predict_accuracy(lookback: int = 252, step: int = 5, simulations: int = 3000):
    """Backtest prediction accuracy on historical data."""
    try:
        df = _get_df()
        return backtest_predictions(df, lookback_days=lookback, step=step, n_simulations=simulations)
    except Exception as e:
        logger.error("Accuracy backtest error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/candles", tags=["Data"])
async def api_data_candles(
    period: str = "daily",
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Return OHLCV data aggregated to daily / weekly / monthly candles."""
    df = _get_df()
    d = df.copy()
    if start:
        d = d[d["Date"] >= pd.Timestamp(start)]
    if end:
        d = d[d["Date"] <= pd.Timestamp(end)]

    if period in ("weekly", "monthly"):
        d = d.set_index("Date")
        rule = "W" if period == "weekly" else "ME"
        agg = d.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()
        if "Volume" in d.columns:
            agg["Volume"] = d["Volume"].resample(rule).sum()
        if "vix_close" in d.columns:
            agg["vix_close"] = d["vix_close"].resample(rule).last()
        agg = agg.reset_index()
        agg.rename(columns={"Date": "Date"}, inplace=True)
        d = agg
    else:
        cols = ["Date", "Open", "High", "Low", "Close"]
        if "Volume" in d.columns:
            cols.append("Volume")
        if "vix_close" in d.columns:
            cols.append("vix_close")
        d = d[[c for c in cols if c in d.columns]].copy()

    # Limit points for browser performance
    max_pts = 2500
    if len(d) > max_pts:
        step = max(1, len(d) // max_pts)
        d = d.iloc[::step]

    d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    return _safe_records(d)


# ── Data Quality Endpoint ────────────────────────────────────────────────

@app.get("/api/data/quality", tags=["Data"])
async def api_data_quality():
    """Run data quality checks on the loaded dataset."""
    try:
        df = _get_df()
        return validate_data_quality(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Monthly Return Heatmap ───────────────────────────────────────────────

@app.get("/api/data/heatmap", tags=["Data"])
async def api_monthly_heatmap():
    """Return monthly returns as a year×month matrix for heatmap display."""
    try:
        df = _get_df()
        d = df[["Date", "Close"]].copy()
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date")
        monthly = d["Close"].resample("ME").last().pct_change() * 100

        heatmap = {}
        for dt, ret in monthly.items():
            year = str(dt.year)
            month = dt.strftime("%b")
            if year not in heatmap:
                heatmap[year] = {}
            heatmap[year][month] = round(float(ret), 2) if not np.isnan(ret) else None

        # Also compute year totals
        yearly = d["Close"].resample("YE").last().pct_change() * 100
        for dt, ret in yearly.items():
            year = str(dt.year)
            if year in heatmap:
                heatmap[year]["Total"] = round(float(ret), 2) if not np.isnan(ret) else None

        return {
            "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Total"],
            "years": sorted(heatmap.keys()),
            "data": heatmap,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Advanced Backtest Endpoints ──────────────────────────────────────────

@app.post("/api/backtest/walk-forward", tags=["Advanced"])
async def api_walk_forward(req: WalkForwardReq):
    """Run walk-forward analysis to detect overfitting."""
    try:
        df = _get_df()
        _validate_params(req.strategy_type, req.params)

        def factory(**kw):
            return get_strategy_fn(req.strategy_type, kw or req.params)

        result = walk_forward_analysis(
            df,
            strategy_fn_factory=factory,
            strategy_params=req.params,
            n_splits=req.n_splits,
            train_pct=req.train_pct,
            initial_capital=req.initial_capital,
            stop_loss_pct=req.stop_loss_pct / 100,
            take_profit_pct=req.take_profit_pct / 100,
        )
        return result
    except Exception as e:
        logger.error("Walk-forward error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/monte-carlo", tags=["Advanced"])
async def api_monte_carlo(req: MonteCarloReq):
    """Run Monte Carlo trade resampling on a completed backtest run."""
    try:
        run = get_backtest_run(req.run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        trades_json = run.get("trades_json", "[]")
        if isinstance(trades_json, str):
            trades_data = json.loads(trades_json)
        else:
            trades_data = trades_json

        # Convert to lightweight Trade-like objects for the MC function
        from src.simulator.backtester import Trade
        trades = []
        for td in trades_data:
            t = Trade(
                entry_date=pd.Timestamp(td["entry_date"]) if td.get("entry_date") else None,
                entry_price=td.get("entry_price", 0),
                direction=td.get("direction", "long"),
            )
            t.pnl = td.get("pnl", 0)
            trades.append(t)

        initial_capital = run.get("initial_capital", DEFAULT_INITIAL_CAPITAL)
        if isinstance(run.get("result_json"), str):
            result_data = json.loads(run["result_json"])
            initial_capital = result_data.get("initial_capital", initial_capital)

        result = monte_carlo_trade_resample(trades, initial_capital, req.n_simulations)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Monte Carlo error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/correlation", tags=["Advanced"])
async def api_correlation(req: CorrelationReq):
    """Compute daily return correlation matrix across multiple strategies."""
    try:
        df = _get_df()
        if req.start_date:
            df = df[df["Date"] >= pd.Timestamp(req.start_date)]
        if req.end_date:
            df = df[df["Date"] <= pd.Timestamp(req.end_date)]

        result = strategy_correlation_matrix(df, req.strategies)
        return result
    except Exception as e:
        logger.error("Correlation error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/sweep", tags=["Advanced"])
async def api_parameter_sweep(req: SweepReq):
    """Vary one strategy parameter and return metrics for each value."""
    try:
        df = _get_df()
        results = parameter_sweep(
            df, req.strategy_type, req.param_name, req.param_values,
            base_params=req.base_params,
            stop_loss_pct=req.stop_loss_pct / 100,
            take_profit_pct=req.take_profit_pct / 100,
        )
        return {"param_name": req.param_name, "results": results}
    except Exception as e:
        logger.error("Sweep error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Background Backtest ──────────────────────────────────────────────────

_bg_tasks: dict = {}  # task_id → {"status": ..., "result": ..., "error": ...}


@app.post("/api/backtest/async", tags=["Backtest"])
async def api_run_backtest_async(req: BacktestReq, background_tasks: BackgroundTasks):
    """Launch a backtest in the background. Returns a task_id to poll."""
    task_id = str(uuid.uuid4())[:8]
    _bg_tasks[task_id] = {"status": "running", "result": None, "error": None}

    stype = req.strategy_type
    params = req.params or {}

    if req.strategy_id:
        strat = get_strategy(req.strategy_id)
        if not strat:
            raise HTTPException(status_code=404, detail="Strategy not found")
        stype = strat["type"]
        params = json.loads(strat["params"]) if isinstance(strat["params"], str) else strat["params"]

    if not stype:
        raise HTTPException(status_code=400, detail="Provide strategy_id or strategy_type+params")

    def _run_bg():
        try:
            result = _run_single_backtest(
                stype, params, req.strategy_id,
                req.initial_capital, req.position_size_pct,
                req.stop_loss_pct, req.take_profit_pct,
                req.start_date, req.end_date, req.trailing_stop_pct,
                req.slippage_pct, req.commission_pct, req.vol_target,
                req.cooldown_bars, req.max_holding_bars, req.crypto_id,
            )
            _bg_tasks[task_id]["status"] = "completed"
            _bg_tasks[task_id]["result"] = result
        except Exception as e:
            _bg_tasks[task_id]["status"] = "failed"
            _bg_tasks[task_id]["error"] = str(e)

    background_tasks.add_task(_run_bg)
    return {"task_id": task_id, "status": "running"}


@app.get("/api/backtest/async/{task_id}", tags=["Backtest"])
async def api_get_bg_task(task_id: str):
    """Check status of a background backtest task."""
    task = _bg_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **task}


# ── Export Endpoint ──────────────────────────────────────────────────────

@app.get("/api/backtest/runs/{run_id}/export", tags=["Export"])
async def api_export_run_csv(run_id: int):
    """Download backtest trades and summary as a CSV file."""
    r = get_backtest_run(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")

    # Build CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Header section — summary metrics
    writer.writerow(["# AITrading Backtest Export"])
    writer.writerow(["# Strategy", r.get("strategy_name", "")])
    writer.writerow(["# Run ID", run_id])

    result_json = r.get("result_json", "{}")
    if isinstance(result_json, str):
        metrics = json.loads(result_json)
    else:
        metrics = result_json or {}

    writer.writerow([])
    writer.writerow(["Metric", "Value"])
    for k, v in metrics.items():
        writer.writerow([k, v])

    # Trades section
    trades_json = r.get("trades_json", "[]")
    if isinstance(trades_json, str):
        trades = json.loads(trades_json)
    else:
        trades = trades_json or []

    writer.writerow([])
    if trades:
        headers = list(trades[0].keys())
        writer.writerow(headers)
        for t in trades:
            writer.writerow([t.get(h, "") for h in headers])

    output.seek(0)
    filename = f"backtest_{run_id}_{r.get('strategy_name', 'unknown').replace(' ', '_')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Copilot Endpoint ─────────────────────────────────────────────────────

@app.post("/api/copilot", tags=["Copilot"])
async def api_copilot(req: CopilotMessageReq):
    """Chat endpoint for the Copilot AI assistant."""
    try:
        # Get or create session
        session_id = req.session_id
        if not session_id:
            session_id = create_copilot_session()

        session = get_copilot_session(session_id)
        messages = session["messages"] if session else []
        messages.append({"role": "user", "content": req.message})

        # Process the message (with LLM fallback for unknown intents)
        result = process_message(req.message, use_llm=req.use_llm)

        # Handle actions that need data/backtest execution
        if result["action"] == "data_info":
            try:
                df = _get_df()
                close = df[NIFTY_CLOSE_COL]
                resp = (
                    f"### 📈 Dataset Summary\n\n"
                    f"- **Rows:** {len(df):,}\n"
                    f"- **Date Range:** {pd.Timestamp(df['Date'].iloc[0]).date()} → {pd.Timestamp(df['Date'].iloc[-1]).date()}\n"
                    f"- **Last Close:** ₹{close.iloc[-1]:,.2f}\n"
                    f"- **Min:** ₹{close.min():,.2f}\n"
                    f"- **Max:** ₹{close.max():,.2f}\n"
                    f"- **Mean:** ₹{close.mean():,.2f}\n"
                    f"- **VIX Available:** {'Yes ✅' if 'vix_close' in df.columns else 'No ❌'}\n"
                    f"- **Features:** {len(df.columns)} columns"
                )
                result["response"] = resp
            except Exception as e:
                result["response"] = f"⚠️ Could not load data: {e}"

        elif result["action"] == "run_backtest":
            strategy_name = result["params"].get("strategy_name", "")
            strategies = list_strategies()
            matched = None
            for s in strategies:
                if strategy_name.lower() in s["name"].lower():
                    matched = s
                    break

            if matched:
                try:
                    params = json.loads(matched["params"]) if isinstance(matched["params"], str) else matched["params"]
                    bt_result = _run_single_backtest(
                        matched["type"], params, matched["id"],
                        DEFAULT_INITIAL_CAPITAL, DEFAULT_POSITION_SIZE * 100,
                        DEFAULT_STOP_LOSS_PCT * 100, DEFAULT_TAKE_PROFIT_PCT * 100,
                        None, None,
                    )
                    result["response"] = (
                        f"### ✅ Backtest Complete: {bt_result['strategy_name']}\n\n"
                        f"| Metric | Value |\n|--------|-------|\n"
                        f"| Initial Capital | ₹{bt_result['initial_capital']:,.0f} |\n"
                        f"| Final Capital | ₹{bt_result['final_capital']:,.0f} |\n"
                        f"| Total Return | {bt_result['total_return_pct']:+.2f}% |\n"
                        f"| Benchmark (B&H) | {(bt_result.get('benchmark_return_pct') or 0):+.2f}% |\n"
                        f"| Alpha | {(bt_result.get('alpha_pct') or 0):+.2f}% |\n"
                        f"| Sharpe Ratio | {bt_result['sharpe_ratio']:.2f} |\n"
                        f"| Sortino Ratio | {(bt_result.get('sortino_ratio') or 0):.2f} |\n"
                        f"| Max Drawdown | {bt_result['max_drawdown_pct']:.2f}% |\n"
                        f"| Trades | {bt_result['total_trades']} |\n"
                        f"| Win Rate | {bt_result['win_rate_pct']:.1f}% |\n"
                        f"| Profit Factor | {bt_result['profit_factor']:.2f} |\n"
                        f"| Exposure Time | {(bt_result.get('exposure_time_pct') or 0):.1f}% |\n\n"
                        f"Run ID: **{bt_result['run_id']}** – check the Results tab for details."
                    )
                    result["backtest_run_id"] = bt_result["run_id"]
                except Exception as e:
                    result["response"] = f"⚠️ Backtest failed: {e}"
            else:
                strategy_list = ", ".join(s["name"] for s in strategies)
                result["response"] = (
                    f"Could not find strategy matching **\"{strategy_name}\"**.\n\n"
                    f"Available strategies: {strategy_list}\n\n"
                    f"Try: *\"backtest SMA Crossover (10/50)\"*"
                )

        elif result["action"] == "backtest_all":
            strategies = list_strategies()
            active = [s for s in strategies if s.get("is_active", 1)]
            if not active:
                result["response"] = "No active strategies found."
            else:
                lines = ["### 🚀 Backtesting All Active Strategies\n"]
                for s in active:
                    try:
                        params = json.loads(s["params"]) if isinstance(s["params"], str) else s["params"]
                        bt_result = _run_single_backtest(
                            s["type"], params, s["id"],
                            DEFAULT_INITIAL_CAPITAL, DEFAULT_POSITION_SIZE * 100,
                            DEFAULT_STOP_LOSS_PCT * 100, DEFAULT_TAKE_PROFIT_PCT * 100,
                            None, None,
                        )
                        lines.append(
                            f"**{bt_result['strategy_name']}**: Return {bt_result['total_return_pct']:+.2f}%, "
                            f"Sharpe {bt_result['sharpe_ratio']:.2f}, MaxDD {bt_result['max_drawdown_pct']:.2f}%"
                        )
                    except Exception as e:
                        lines.append(f"**{s['name']}**: ⚠️ Failed – {e}")
                lines.append("\n*Check the Results tab for full details.*")
                result["response"] = "\n".join(lines)

        elif result["action"] == "predict":
            try:
                df = _get_df()
                report = run_prediction(df, method="ensemble", n_simulations=10000)
                lines = [
                    f"### 🔮 Nifty Probability Forecast\n",
                    f"**As of:** {pd.Timestamp(report.as_of_date).date()}  |  **Current:** ₹{report.current_price:,.2f}",
                ]
                if report.current_vix:
                    lines.append(f"**VIX:** {report.current_vix:.2f} (regime: {report.vix_regime})\n")
                lines.append("| Horizon | Mean | P(Up) | P(Down) | 90% Range |")
                lines.append("|---------|------|-------|---------|-----------|")
                for name, pred in report.predictions.items():
                    pct90 = pred.percentiles.get(0.90, (0, 0))
                    lines.append(
                        f"| {name} | ₹{pred.mean_price:,.0f} | {pred.prob_up*100:.1f}% | "
                        f"{pred.prob_down*100:.1f}% | ₹{pct90[0]:,.0f} – ₹{pct90[1]:,.0f} |"
                    )
                result["response"] = "\n".join(lines)
            except Exception as e:
                result["response"] = f"⚠️ Prediction failed: {e}"

        if result["response"] is None:
            result["response"] = "Something went wrong. Please try again."

        messages.append({"role": "assistant", "content": result["response"]})
        update_copilot_session(session_id, messages)

        return {
            "session_id": session_id,
            "response": result["response"],
            "intent": result["intent"],
            "action": result.get("action"),
            "backtest_run_id": result.get("backtest_run_id"),
            "llm_used": result.get("llm_used", False),
        }

    except Exception as e:
        logger.error("Copilot error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Natural Language Backtest Endpoint ────────────────────────────────────

@app.post("/api/nl-backtest", tags=["Copilot"])
async def api_nl_backtest(req: NLBacktestReq):
    """
    Natural language backtest endpoint.

    Accepts a plain-English instruction like:
      "Run an SMA crossover with 20/100 windows and 3% stop loss on 2022 data"
    and forwards it to the LLM agent which parses, runs, and returns results.
    """
    try:
        llm_result = handle_llm_query(req.instruction)
        return {
            "success": llm_result["success"],
            "response": llm_result["response"],
            "intermediate_steps": llm_result.get("intermediate_steps", []),
        }
    except Exception as e:
        logger.error("NL-backtest error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/llm/status", tags=["Copilot"])
async def api_llm_status():
    """Check if LLM is available and configured."""
    try:
        from src.llm.config import is_llm_available, WATSONX_MODEL_ID, LLM_ENABLED
        available = is_llm_available()
        return {
            "llm_enabled": LLM_ENABLED,
            "llm_available": available,
            "model_id": WATSONX_MODEL_ID if available else None,
        }
    except ImportError:
        return {
            "llm_enabled": False,
            "llm_available": False,
            "model_id": None,
            "error": "LLM module not installed",
        }


# ── Market Events Endpoints ──────────────────────────────────────────────

@app.get("/api/events", tags=["Events"])
async def api_list_events(category: str = None, region: str = None, tag: str = None):
    """List market events, optionally filtered by category, region, or tag."""
    try:
        from src.data.events import (
            get_events_by_category, get_events_by_tag, get_all_events, CATEGORIES,
        )
        if tag:
            events = get_events_by_tag(tag)
        elif category:
            events = get_events_by_category(category)
        else:
            events = get_all_events()

        result = []
        for e in events:
            result.append({
                "name": e.name,
                "category": e.category,
                "start_date": str(e.start_date),
                "end_date": str(e.end_date) if e.end_date else None,
                "description": e.description,
                "source": e.source,
                "region": e.region,
                "impact": e.impact,
                "tags": e.tags,
            })
        return {"events": result, "total": len(result)}
    except Exception as e:
        logger.error("Events list error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/events/categories", tags=["Events"])
async def api_event_categories():
    """List available event categories with counts."""
    try:
        from src.data.events import get_event_summary
        return get_event_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EventAnalysisReq(BaseModel):
    category: Optional[str] = None
    tag: Optional[str] = None


@app.post("/api/events/analyze", tags=["Events"])
async def api_analyze_events(req: EventAnalysisReq):
    """Analyze how Nifty 50 performed during events of a specific category/tag."""
    try:
        from src.data.events import analyze_market_during_events
        df = _get_df()
        analysis = analyze_market_during_events(
            df, category=req.category, tag=req.tag, price_col=NIFTY_CLOSE_COL,
        )
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Event analysis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Leverage Simulator Endpoints ─────────────────────────────────────────────

from src.simulator.leverage_sim import (
    LeverageSimulator,
    TransactionCostModel,
    run_leverage_sweep,
    run_full_simulation,
    VIX_IV_REGIME_LABELS,
    get_repo_rate_series,
)
from config.settings import (
    LEVERAGE_DEFAULT_CAPITAL,
    LEVERAGE_RATIOS_DEFAULT,
    LEVERAGE_CALL_OTM_PCTS_DEFAULT,
    LEVERAGE_PUT_OTM_PCT_DEFAULT,
    LEVERAGE_LIQUID_FUND_SPREAD,
    LEVERAGE_DIVIDEND_YIELD,
)


class LeverageSimulateReq(BaseModel):
    leverage_ratio: float = Field(2.0, ge=0.1, le=10.0, description="Notional / capital")
    call_otm_pct: float = Field(0.05, ge=0.0, le=0.50, description="Call strike OTM fraction (0 = no calls)")
    put_otm_pct: float = Field(0.20, ge=0.0, le=0.50, description="Put strike OTM fraction (0 = no puts)")
    initial_capital: float = Field(3_000_000.0, gt=0)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    liquid_fund_spread: float = Field(0.005, ge=0.0, le=0.05)
    dividend_yield: float = Field(0.013, ge=0.0, le=0.10)
    # Realism controls
    use_vol_skew: bool = Field(True, description="Apply Nifty put skew premium (+15%) and call discount (-10%)")
    enable_transaction_costs: bool = Field(True, description="Include NSE F&O STT, brokerage, exchange charges")
    strike_round_increment: float = Field(50.0, ge=1.0, le=500.0, description="NSE strike rounding increment (50 for near-month)")
    margin_call_threshold_pct: float = Field(0.20, ge=0.0, le=1.0, description="Flag margin-call zone if equity < initial × this")


class LeverageSweepReq(BaseModel):
    leverage_ratios: list[float] = Field(default_factory=lambda: LEVERAGE_RATIOS_DEFAULT)
    call_otm_pcts: list[float] = Field(default_factory=lambda: LEVERAGE_CALL_OTM_PCTS_DEFAULT)
    put_otm_pct: float = Field(0.20, ge=0.0, le=0.50)
    initial_capital: float = Field(3_000_000.0, gt=0)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    liquid_fund_spread: float = Field(0.005, ge=0.0, le=0.05)
    dividend_yield: float = Field(0.013, ge=0.0, le=0.10)
    # Realism controls
    use_vol_skew: bool = Field(True)
    enable_transaction_costs: bool = Field(True)
    strike_round_increment: float = Field(50.0, ge=1.0, le=500.0)
    margin_call_threshold_pct: float = Field(0.20, ge=0.0, le=1.0)


@app.post("/api/leverage/simulate", tags=["Leverage"])
async def api_leverage_simulate(req: LeverageSimulateReq):
    """
    Run a single leveraged Nifty futures simulation.

    Returns full equity curve, monthly P&L breakdown, and all key metrics.
    Covered calls are sold monthly at the chosen OTM%; protective puts are
    bought quarterly at 20% OTM (LEAPS proxy). Capital earns liquid-fund rate.
    Carry cost is modelled from the historical RBI repo rate schedule.
    """
    try:
        df = _get_df()
        result = run_full_simulation(
            df=df,
            leverage_ratio=req.leverage_ratio,
            call_otm_pct=req.call_otm_pct,
            put_otm_pct=req.put_otm_pct,
            initial_capital=req.initial_capital,
            start_date=req.start_date,
            end_date=req.end_date,
            liquid_fund_spread=req.liquid_fund_spread,
            dividend_yield=req.dividend_yield,
            use_vol_skew=req.use_vol_skew,
            enable_transaction_costs=req.enable_transaction_costs,
            strike_round_increment=req.strike_round_increment,
            margin_call_threshold_pct=req.margin_call_threshold_pct,
        )
        # Determine if VIX data coverage is partial (warn in response)
        vix_warning = None
        if result.vix_data_coverage_pct < 50:
            vix_warning = (
                f"VIX data covers only {result.vix_data_coverage_pct:.0f}% of the simulation period. "
                "Options pricing falls back to 'normal' IV regime (17%) for missing dates. "
                "Re-run 'python main.py download' to extend VIX coverage."
            )

        # Downsample equity curve for response (keep ≤ 2000 points)
        ec = result.equity_curve
        if len(ec) > 2000:
            step = max(1, len(ec) // 2000)
            ec = ec.iloc[::step]

        return {
            "leverage_ratio": result.leverage_ratio,
            "call_otm_pct": result.call_otm_pct,
            "put_otm_pct": result.put_otm_pct,
            "initial_capital": result.initial_capital,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "final_capital": result.final_capital,
            "total_return_pct": result.total_return_pct,
            "annual_return_pct": result.annual_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "total_carry_cost_pct": result.total_carry_cost_pct,
            "total_put_cost_pct": result.total_put_cost_pct,
            "total_call_income_pct": result.total_call_income_pct,
            "total_liquid_fund_income_pct": result.total_liquid_fund_income_pct,
            "total_futures_pnl_pct": result.total_futures_pnl_pct,
            "put_payout_events": result.put_payout_events,
            "call_cap_events": result.call_cap_events,
            "vix_data_coverage_pct": result.vix_data_coverage_pct,
            "vix_warning": vix_warning,
            "margin_call_triggered": result.margin_call_triggered,
            "margin_call_date": result.margin_call_date,
            "total_transaction_cost_pct": result.total_transaction_cost_pct,
            "equity_curve": _safe_records(ec),
            "monthly_breakdown": _safe_records(result.monthly_breakdown),
        }
    except Exception as e:
        logger.error("Leverage simulate error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/leverage/sweep", tags=["Leverage"])
async def api_leverage_sweep(req: LeverageSweepReq):
    """
    Run a parameter sweep across leverage ratios × covered-call OTM strikes.

    Returns a flat list of summary rows (no equity curves) suitable for
    rendering as a 2D heatmap (leverage × CC-strike → metric).
    Protective puts at put_otm_pct are applied to every combination.
    """
    try:
        if len(req.leverage_ratios) > 10:
            raise HTTPException(status_code=400, detail="Max 10 leverage ratios per sweep")
        if len(req.call_otm_pcts) > 8:
            raise HTTPException(status_code=400, detail="Max 8 call OTM values per sweep")

        df = _get_df()
        rows = run_leverage_sweep(
            df=df,
            leverage_ratios=req.leverage_ratios,
            call_otm_pcts=req.call_otm_pcts,
            put_otm_pct=req.put_otm_pct,
            initial_capital=req.initial_capital,
            start_date=req.start_date,
            end_date=req.end_date,
            liquid_fund_spread=req.liquid_fund_spread,
            dividend_yield=req.dividend_yield,
            use_vol_skew=req.use_vol_skew,
            enable_transaction_costs=req.enable_transaction_costs,
            strike_round_increment=req.strike_round_increment,
            margin_call_threshold_pct=req.margin_call_threshold_pct,
        )
        return {
            "count": len(rows),
            "leverage_ratios": req.leverage_ratios,
            "call_otm_pcts": req.call_otm_pcts,
            "put_otm_pct": req.put_otm_pct,
            "results": [
                {
                    "leverage_ratio": r.leverage_ratio,
                    "call_otm_pct": r.call_otm_pct,
                    "final_capital": r.final_capital,
                    "total_return_pct": r.total_return_pct,
                    "annual_return_pct": r.annual_return_pct,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "sharpe_ratio": r.sharpe_ratio,
                    "sortino_ratio": r.sortino_ratio,
                    "calmar_ratio": r.calmar_ratio,
                    "total_carry_cost_pct": r.total_carry_cost_pct,
                    "total_put_cost_pct": r.total_put_cost_pct,
                    "total_call_income_pct": r.total_call_income_pct,
                    "total_liquid_fund_income_pct": r.total_liquid_fund_income_pct,
                    "put_payout_events": r.put_payout_events,
                    "call_cap_events": r.call_cap_events,
                    "margin_call_triggered": r.margin_call_triggered,
                    "total_transaction_cost_pct": r.total_transaction_cost_pct,
                }
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Leverage sweep error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/leverage/assumptions", tags=["Leverage"])
async def api_leverage_assumptions():
    """
    Return the model assumptions used by the leverage simulator.

    Includes the VIX → implied-vol regime table and a sample of the
    RBI repo rate history used for carry cost and liquid fund income.
    """
    try:
        from src.simulator.leverage_sim import _RBI_REPO_RATES
        # Build repo rate records for the chart
        repo_records = [
            {"date": date_str, "repo_rate_pct": rate}
            for date_str, rate in _RBI_REPO_RATES
        ]
        return {
            "vix_iv_regimes": VIX_IV_REGIME_LABELS,
            "rbi_repo_rates": repo_records,
            "model_notes": [
                "All capital is deployed as futures margin and simultaneously earns liquid-fund rate (repo − 0.5%).",
                "Futures carry cost = (repo_rate − dividend_yield) / 252 per day on notional.",
                "Covered calls: monthly (every 21 trading days), settled at each roll date.",
                "Protective puts: quarterly rolling (every 63 trading days) at chosen OTM%.",
                "Vol skew: put IV ≈ ATM IV × 1.15, call IV ≈ ATM IV × 0.90 (Nifty historical skew).",
                "Transaction costs: STT (0.01% futures sell, 0.05% option premium), ₹20 brokerage+GST, exchange charges.",
                "Strikes rounded to nearest ₹50 increment (valid NSE near-month option strikes).",
                "Black-Scholes pricing uses VIX-regime IV buckets with skew adjustments.",
                "Margin call zone flag: triggers when equity < initial_capital × margin_call_threshold (default 20%).",
                "Post-2016 dates where VIX data is absent use IV = 17% (normal regime fallback).",
                "Dividend yield fixed at 1.3% p.a. (Nifty 50 historical average).",
                "Simulation is in capital units — lot size changes do not affect results.",
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/leverage/valuation-context", tags=["Leverage"])
async def api_leverage_valuation_context():
    """
    Return current Nifty valuation context (PE, PB, DivYield) with
    leverage insights for the Leverage simulator page.

    Scrapes today's values from nifty-pe-ratio.com and enriches with:
    - PE/PB/DivYield zone classifications (Undervalued → Overvalued)
    - Equity Risk Premium (earnings yield minus risk-free rate)
    - Recommended maximum leverage based on PE zone
    - Historical PE statistics if local data available

    Returns HTTP 503 if the source site is unreachable and no cached
    data exists.
    """
    try:
        from src.data.valuation_scraper import get_valuation_context
        return get_valuation_context()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/valuation/sync", tags=["Leverage"])
async def api_valuation_sync():
    """
    Trigger a sync of the Nifty PE/PB/DivYield CSV (appends today if absent).

    Returns the latest scraped values and total row count.
    """
    try:
        from src.data.valuation_scraper import sync_data
        df = sync_data()
        latest = df.iloc[-1]
        return {
            "status": "ok",
            "total_rows": len(df),
            "latest": {
                "date": str(latest["Date"].date() if hasattr(latest["Date"], "date") else latest["Date"]),
                "pe": float(latest["PE"]),
                "pb": float(latest["PB"]),
                "div_yield": float(latest["DivYield"]),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Retirement Monte Carlo Simulator ─────────────────────────────────────────

class RetirementSimReq(BaseModel):
    """Request body for the retirement Monte Carlo simulation endpoint."""
    initial_corpus: float = Field(10_000_000.0, gt=0)
    equity_pct: float = Field(60.0, ge=0.0, le=100.0)
    monthly_withdrawal: float = Field(50_000.0, gt=0)
    inflation_rate_pct: float = Field(6.0, ge=0.0, le=30.0)
    debt_instrument: str = Field("liquid_fund")
    future_debt_rate_override: Optional[float] = Field(None, ge=0.0, le=30.0)
    swp_replenish_years: int = Field(5, ge=1, le=20)
    emergency_months_threshold: int = Field(12, ge=1, le=120)
    replenish_mode: str = Field("fill_years", description="fill_years | pct_equity | rebalance")
    replenish_pct: float = Field(20.0, ge=1.0, le=100.0, description="% used for pct_equity/rebalance modes")
    total_years: int = Field(30, ge=1, le=50)
    tax_enabled: bool = Field(True)
    income_tax_bracket_pct: float = Field(30.0, ge=0.0, le=42.0)
    n_simulations: int = Field(1000, ge=100, le=5000)
    random_seed: Optional[int] = Field(None)
    starting_year: Optional[int] = Field(None, ge=2000, le=2030)
    compute_swr: bool = Field(True, description="Compute SWR (adds ~3000 sub-sims; uncheck for speed)")
    compute_tax_drag: bool = Field(True, description="Compute tax drag (adds ~200 sub-sims)")
    # FIRE mode: multi-phase accumulation before full retirement
    fire_mode: str = Field("classic", description="classic | coast | barista")
    phase1_years: int = Field(0, ge=0, le=40, description="Duration of Phase 1 (pre-retirement) in years")
    barista_monthly_withdrawal: float = Field(20_000.0, ge=0.0, description="Monthly corpus draw during Barista Phase 1 (₹)")
    monthly_contribution: float = Field(0.0, ge=0.0, description="Monthly contribution to equity during Phase 1 (₹)")
    phase1_equity_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Equity % during Phase 1 (None = same as equity_pct)")
    # Leverage: replace debt/cash bucket with leveraged Nifty futures position
    use_leverage: bool = Field(False, description="Replace cash/debt bucket with leveraged Nifty futures. False = classic cash view (default).")
    leverage_ratio: float = Field(2.0, ge=1.01, le=5.0, description="Notional / own margin capital (1.01–5.0×)")
    leverage_borrow_rate_pct: float = Field(9.0, ge=0.0, le=30.0, description="Gross annual borrowing rate for leveraged position (%)")


class RetirementOptimizeReq(BaseModel):
    """Request body for the equity% optimization sweep endpoint."""
    initial_corpus: float = Field(10_000_000.0, gt=0)
    monthly_withdrawal: float = Field(50_000.0, gt=0)
    inflation_rate_pct: float = Field(6.0, ge=0.0, le=30.0)
    debt_instrument: str = Field("liquid_fund")
    future_debt_rate_override: Optional[float] = Field(None, ge=0.0, le=30.0)
    swp_replenish_years: int = Field(5, ge=1, le=20)
    emergency_months_threshold: int = Field(12, ge=1, le=120)
    total_years: int = Field(30, ge=1, le=50)
    tax_enabled: bool = Field(True)
    income_tax_bracket_pct: float = Field(30.0, ge=0.0, le=42.0)
    equity_pct_range: Optional[list[float]] = Field(None)
    n_sims_per_point: int = Field(500, ge=100, le=2000)
    random_seed: Optional[int] = Field(None)


@app.post("/api/swp/simulate", tags=["SWP"])
async def api_retirement_simulate(req: RetirementSimReq):
    """
    Run a Monte Carlo retirement simulation using the bucket strategy.

    Returns percentile fan-chart data, survival rates by year,
    corpus depletion histogram, and summary metrics.

    Note: n_simulations > 2000 is capped at 2000 to keep response time reasonable.
    """
    try:
        from src.simulator.retirement_sim import RetirementParams, run_monte_carlo

        n_sims = min(req.n_simulations, 2000)
        capped = n_sims < req.n_simulations

        params = RetirementParams(
            initial_corpus=req.initial_corpus,
            equity_pct=req.equity_pct,
            monthly_withdrawal=req.monthly_withdrawal,
            inflation_rate_pct=req.inflation_rate_pct,
            debt_instrument=req.debt_instrument,
            future_debt_rate_override=req.future_debt_rate_override,
            swp_replenish_years=req.swp_replenish_years,
            emergency_months_threshold=req.emergency_months_threshold,
            replenish_mode=req.replenish_mode,
            replenish_pct=req.replenish_pct,
            total_years=req.total_years,
            tax_enabled=req.tax_enabled,
            income_tax_bracket_pct=req.income_tax_bracket_pct,
            n_simulations=n_sims,
            random_seed=req.random_seed,
            starting_year=req.starting_year,
            fire_mode=req.fire_mode,
            phase1_years=req.phase1_years,
            barista_monthly_withdrawal=req.barista_monthly_withdrawal,
            monthly_contribution=req.monthly_contribution,
            phase1_equity_pct=req.phase1_equity_pct,
            use_leverage=req.use_leverage,
            leverage_ratio=req.leverage_ratio,
            leverage_borrow_rate_pct=req.leverage_borrow_rate_pct,
        )

        result = run_monte_carlo(
            params, _get_df(),
            compute_swr=req.compute_swr,
            compute_tax_drag=req.compute_tax_drag,
        )

        return {
            "n_simulations": result.n_simulations,
            "n_survived": result.n_survived,
            "capped_at_2000": capped,
            "portfolio_percentiles": result.portfolio_percentiles,
            "survival_by_year": result.survival_by_year,
            "ruin_histogram": result.ruin_histogram,
            "summary": {
                "survival_20yr_pct": result.survival_20yr_pct,
                "survival_30yr_pct": result.survival_30yr_pct,
                "median_final_corpus": result.median_final_corpus,
                "mean_final_corpus": result.mean_final_corpus,
                "safe_withdrawal_rate_pct": result.safe_withdrawal_rate_pct,
                "actual_withdrawal_rate_pct": round(params.monthly_withdrawal * 12 / params.initial_corpus * 100, 2),
                "debt_months_coverage": round(
                    params.initial_corpus * (1.0 - params.equity_pct / 100.0) / params.monthly_withdrawal, 1
                ),
                "median_total_tax": result.median_total_tax,
                "tax_drag_pct": result.tax_drag_pct,
                "median_emergency_count": result.median_emergency_count,
                "median_ruin_year": result.median_ruin_year,
                "corpus_needed_at_swr": (
                    round(params.monthly_withdrawal * 12 / result.safe_withdrawal_rate_pct * 100, 0)
                    if result.safe_withdrawal_rate_pct and result.safe_withdrawal_rate_pct > 0 else None
                ),
                "portfolio_cagr_pct": (
                    round(((result.median_final_corpus / params.initial_corpus)
                           ** (1.0 / params.total_years) - 1.0) * 100, 2)
                    if result.median_final_corpus > 0 else None
                ),
                # Leverage settings
                "use_leverage": params.use_leverage,
                "leverage_ratio": params.leverage_ratio if params.use_leverage else None,
                "leverage_borrow_rate_pct": params.leverage_borrow_rate_pct if params.use_leverage else None,
                "effective_nifty_exposure_pct": (
                    round(params.equity_pct + (100.0 - params.equity_pct) * params.leverage_ratio, 1)
                    if params.use_leverage else params.equity_pct
                ),
                # FIRE phase metrics
                "fire_mode": params.fire_mode,
                "phase1_years": params.phase1_years,
                "corpus_at_phase2_start_median": result.corpus_at_phase2_start_median,
                "corpus_at_phase2_start_p10": result.corpus_at_phase2_start_p10,
                "corpus_at_phase2_start_p90": result.corpus_at_phase2_start_p90,
                "phase2_survival_pct": result.phase2_survival_pct,
                "expected_phase2_starting_withdrawal": result.expected_phase2_starting_withdrawal,
                "phase1_portfolio_cagr_pct": result.phase1_portfolio_cagr_pct,
                "net_phase1_contribution_total": result.net_phase1_contribution_total,
            },
            "sequence_risk_data": result.sequence_risk_data[:200],
            "yearly_stats": result.yearly_stats,
            "actual_start_year": result.actual_start_year,
            "debt_medians": result.debt_medians,
            "params": {
                "initial_corpus": params.initial_corpus,
                "equity_pct": params.equity_pct,
                "monthly_withdrawal": params.monthly_withdrawal,
                "inflation_rate_pct": params.inflation_rate_pct,
                "debt_instrument": params.debt_instrument,
                "total_years": params.total_years,
                "tax_enabled": params.tax_enabled,
            },
        }
    except Exception as exc:
        logger.error("Retirement simulation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/swp/optimize", tags=["SWP"])
async def api_retirement_optimize(req: RetirementOptimizeReq):
    """
    Run an equity% optimization sweep to find the allocation that maximises
    survival probability, median corpus, or a balanced score.

    Sweeps equity_pct from 10 to 90 (step 5) unless equity_pct_range is provided.
    """
    try:
        from src.simulator.retirement_sim import RetirementParams, run_optimization_sweep

        base_params = RetirementParams(
            initial_corpus=req.initial_corpus,
            equity_pct=50.0,
            monthly_withdrawal=req.monthly_withdrawal,
            inflation_rate_pct=req.inflation_rate_pct,
            debt_instrument=req.debt_instrument,
            future_debt_rate_override=req.future_debt_rate_override,
            swp_replenish_years=req.swp_replenish_years,
            emergency_months_threshold=req.emergency_months_threshold,
            total_years=req.total_years,
            tax_enabled=req.tax_enabled,
            income_tax_bracket_pct=req.income_tax_bracket_pct,
            n_simulations=req.n_sims_per_point,
            random_seed=req.random_seed,
        )

        result = run_optimization_sweep(
            base_params, _get_df(),
            equity_pct_range=req.equity_pct_range,
            n_sims_per_point=req.n_sims_per_point,
        )

        def _point_to_dict(p):
            return {
                "equity_pct": p.equity_pct,
                "debt_pct": round(100.0 - p.equity_pct, 1),
                "survival_30yr_pct": p.survival_30yr_pct,
                "median_final_corpus": p.median_final_corpus,
                "safe_withdrawal_rate_pct": p.safe_withdrawal_rate_pct,
                "tax_drag_pct": p.tax_drag_pct,
                "median_emergency_count": p.median_emergency_count,
            }

        return {
            "points": [_point_to_dict(p) for p in result.points],
            "optimal_survival": _point_to_dict(result.optimal_survival),
            "optimal_median_corpus": _point_to_dict(result.optimal_median_corpus),
            "optimal_balanced": _point_to_dict(result.optimal_balanced),
        }
    except Exception as exc:
        logger.error("Retirement optimization error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/swp/debt-rates", tags=["SWP"])
async def api_retirement_debt_rates():
    """
    Return historical debt return rates for all three instrument types,
    indexed by RBI rate-change events.  Used by the UI chart.
    """
    try:
        from src.data.debt_rates import get_historical_rates_df, list_instruments

        instruments = ["liquid_fund", "short_term_debt", "bank_fd"]
        result = {}
        for inst in instruments:
            df = get_historical_rates_df(inst)
            result[inst] = {
                "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
                "repo_rates": df["repo_rate_pct"].tolist(),
                "instrument_returns": df["instrument_return_pct"].tolist(),
            }

        return {
            "instruments": result,
            "metadata": list_instruments(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Crypto Leverage Simulator Endpoints ─────────────────────────────────────

class CryptoSimReq(BaseModel):
    crypto_id: str = "bitcoin"
    leverage_ratio: float = 2.0
    borrow_rate_pct: float = 10.0
    initial_capital: float = 1_000_000.0
    horizon_years: int = 10
    n_simulations: int = 1000
    block_size_months: int = 6
    liquidation_threshold_pct: float = 0.0


class CryptoSweepReq(BaseModel):
    crypto_id: str = "bitcoin"
    borrow_rate_pct: float = 10.0
    initial_capital: float = 1_000_000.0
    horizon_years: int = 10
    n_simulations: int = 1000
    block_size_months: int = 6
    liquidation_threshold_pct: float = 0.0
    leverage_ratios: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


@app.get("/api/crypto/available", tags=["Crypto"])
async def api_crypto_available():
    """
    Returns available crypto ids, ticker identifiers, and date ranges.
    """
    try:
        from config.settings import CRYPTO_TICKERS, CRYPTO_RAW_DIR
        import pandas as pd
        
        available = []
        for cid, ticker in CRYPTO_TICKERS.items():
            csv_path = CRYPTO_RAW_DIR / f"{cid}.csv"
            date_range = "No data"
            rows = 0
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, usecols=["Date"])
                    if not df.empty:
                        rows = len(df)
                        start_date = str(pd.to_datetime(df["Date"].iloc[0]).date())
                        end_date = str(pd.to_datetime(df["Date"].iloc[-1]).date())
                        date_range = f"{start_date} to {end_date}"
                except Exception:
                    pass
            available.append({
                "id": cid,
                "ticker": ticker,
                "date_range": date_range,
                "rows_count": rows,
            })
        return {"cryptos": available}
    except Exception as exc:
        logger.error("Error getting available cryptos: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/crypto/sync", tags=["Crypto"])
async def api_crypto_sync():
    """
    Sync all crypto data from Yahoo Finance: check the last data point in each CSV,
    fetch new data since then, and append.
    """
    try:
        from src.data.crypto_downloader import sync_all_crypto
        result = sync_all_crypto()
        total_rows_added = sum(info.get("rows_added", 0) for info in result.values())
        return {
            "status": "ok",
            "rows_added": total_rows_added,
            "details": result
        }
    except Exception as exc:
        logger.error("Crypto sync error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/crypto/{crypto_id}/chart", tags=["Crypto"])
async def api_crypto_chart(crypto_id: str):
    """
    Returns chart data for a specific crypto.
    """
    try:
        from config.settings import CRYPTO_RAW_DIR
        import pandas as pd
        csv_path = CRYPTO_RAW_DIR / f"{crypto_id.strip()}.csv"
        print(f"DEBUG: Checking path: {csv_path}, exists: {csv_path.exists()}")
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"Crypto data not found for {crypto_id}")
        
        df = pd.read_csv(csv_path)
        if df.empty:
            raise HTTPException(status_code=404, detail="Crypto data is empty")
        
        # Format for lightweight UI chart, maybe downsample slightly if needed
        # Just return daily Close
        dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").tolist()
        prices = df["Close"].tolist()
        
        return {
            "dates": dates,
            "prices": prices,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting crypto chart: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/crypto/simulate", tags=["Crypto"])
async def api_crypto_simulate(req: CryptoSimReq):
    """
    Run Monte Carlo simulations for a single crypto leverage set of parameters.
    """
    try:
        from src.simulator.crypto_leverage_sim import CryptoLeverageParams, run_monte_carlo_crypto
        
        params = CryptoLeverageParams(
            crypto_id=req.crypto_id,
            leverage_ratio=req.leverage_ratio,
            borrow_rate_pct=req.borrow_rate_pct,
            initial_capital=req.initial_capital,
            horizon_years=req.horizon_years,
            n_simulations=req.n_simulations,
            block_size_months=req.block_size_months,
            liquidation_threshold_pct=req.liquidation_threshold_pct,
        )
        
        result = run_monte_carlo_crypto(params)
        return result.model_dump()
    except Exception as exc:
        logger.error("Crypto simulation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/crypto/sweep", tags=["Crypto"])
async def api_crypto_sweep(req: CryptoSweepReq):
    """
    Run leverage sweeps for a given crypto.
    """
    try:
        from src.simulator.crypto_leverage_sim import CryptoLeverageParams, run_crypto_sweep
        
        params_base = CryptoLeverageParams(
            crypto_id=req.crypto_id,
            leverage_ratio=1.0, # dummy value, overridden by sweep
            borrow_rate_pct=req.borrow_rate_pct,
            initial_capital=req.initial_capital,
            horizon_years=req.horizon_years,
            n_simulations=req.n_simulations,
            block_size_months=req.block_size_months,
            liquidation_threshold_pct=req.liquidation_threshold_pct,
        )
        
        sweep_points = run_crypto_sweep(params_base, leverage_ratios=req.leverage_ratios)
        return [pt.model_dump() for pt in sweep_points]
    except Exception as exc:
        logger.error("Crypto sweep error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
