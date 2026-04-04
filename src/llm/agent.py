"""
LLM-Powered Trading Agent
──────────────────────────
A LangChain agent that interprets natural-language backtest
instructions and executes them against the AITrading engine.

Uses the modern langchain 1.2+ `create_agent` API with ChatWatsonx
(IBM WatsonX AI) as the chat model.

The agent has access to tools that can:
  • Query market data (date range, latest prices, indicators)
  • List available strategy types and their parameters
  • Run backtests with specific strategy configurations
  • Compare backtest results
  • Generate probability predictions

Example user queries the agent can handle:
  "Run an SMA crossover with 20 and 100 day windows and 3% stop loss"
  "Which strategy performed best in the last 2 years?"
  "Backtest RSI with oversold at 25 and overbought at 75, then compare with default RSI"
  "What was the max drawdown of the MACD strategy?"
  "Test a bollinger band strategy on data from 2020 to 2023"
"""

import json
import logging
import traceback
from typing import Optional

import pandas as pd

from langchain_core.tools import tool

from src.llm.config import get_llm, is_llm_available

logger = logging.getLogger(__name__)

# ── Lazy data / backtest imports ──────────────────────────────────────────
# These are imported inside functions to avoid circular imports and to
# defer heavy pandas/numpy loading until actually needed.


def _get_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load the master dataframe, optionally filtered by date range."""
    from src.data.loader import load_master
    df = load_master()
    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    return df


# ── Tool Functions ────────────────────────────────────────────────────────

@tool
def data_summary(query: str) -> str:
    """Get summary of available Nifty 50 market data including date range, price statistics, and whether VIX data is available. Input: any text (ignored)."""
    try:
        from config.settings import NIFTY_CLOSE_COL
        df = _get_data()
        close = df[NIFTY_CLOSE_COL]
        has_vix = "vix_close" in df.columns

        result = {
            "total_rows": len(df),
            "date_start": str(df["Date"].iloc[0].date()),
            "date_end": str(df["Date"].iloc[-1].date()),
            "last_close": round(float(close.iloc[-1]), 2),
            "min_close": round(float(close.min()), 2),
            "max_close": round(float(close.max()), 2),
            "mean_close": round(float(close.mean()), 2),
            "has_vix": has_vix,
            "columns": list(df.columns[:20]),  # first 20 columns
        }
        if has_vix:
            vix = df["vix_close"].dropna()
            result["last_vix"] = round(float(vix.iloc[-1]), 2)
            result["mean_vix"] = round(float(vix.mean()), 2)

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting data summary: {e}"


@tool
def list_strategy_types(query: str) -> str:
    """List all available trading strategy types and their configurable parameters with defaults and valid ranges. Input: any text (ignored)."""
    strategies = {
        "sma_crossover": {
            "description": "Simple Moving Average Crossover (Golden/Death Cross)",
            "params": {
                "fast_window": "Fast SMA period (default: 50, range: 5-100)",
                "slow_window": "Slow SMA period (default: 200, range: 20-500)",
            },
        },
        "ema_crossover": {
            "description": "Exponential Moving Average Crossover",
            "params": {
                "fast_span": "Fast EMA span (default: 12)",
                "slow_span": "Slow EMA span (default: 26)",
            },
        },
        "rsi_mean_reversion": {
            "description": "RSI Mean Reversion - buy oversold, sell overbought",
            "params": {
                "oversold": "Buy threshold (default: 30, range: 10-45)",
                "overbought": "Sell threshold (default: 70, range: 55-90)",
            },
        },
        "macd_crossover": {
            "description": "MACD Signal Line Crossover",
            "params": {
                "fast_ema": "Fast EMA period (default: 12)",
                "slow_ema": "Slow EMA period (default: 26)",
                "signal_period": "Signal line period (default: 9)",
            },
        },
        "macd_histogram": {
            "description": "MACD Histogram Reversal",
            "params": {
                "fast_ema": "Fast EMA period (default: 12)",
                "slow_ema": "Slow EMA period (default: 26)",
                "signal_period": "Signal line period (default: 9)",
            },
        },
        "bollinger_band": {
            "description": "Bollinger Band Mean Reversion",
            "params": {
                "window": "SMA period (default: 20)",
                "num_std": "Number of standard deviations (default: 2.0)",
            },
        },
        "atr_breakout": {
            "description": "ATR Breakout / Chandelier-style",
            "params": {
                "sma_window": "SMA period (default: 20)",
                "atr_period": "ATR period (default: 14)",
                "atr_multiplier": "ATR multiplier (default: 1.5)",
            },
        },
        "vix_regime": {
            "description": "VIX Regime Filter - long in calm markets, flat in fearful",
            "params": {
                "buy_below": "VIX threshold to go long (default: 15)",
                "sell_above": "VIX threshold to exit (default: 25)",
            },
        },
        "stochastic_oscillator": {
            "description": "Stochastic Oscillator %K/%D Crossover",
            "params": {
                "k_period": "%K period (default: 14)",
                "d_period": "%D smoothing period (default: 3)",
                "oversold": "Oversold level (default: 20)",
                "overbought": "Overbought level (default: 80)",
            },
        },
        "mean_reversion_zscore": {
            "description": "Z-Score Mean Reversion",
            "params": {
                "lookback": "Lookback period (default: 20)",
                "entry_z": "Entry z-score threshold (default: -2.0)",
                "exit_z": "Exit z-score threshold (default: 0.0)",
            },
        },
        "composite_sniper": {
            "description": "Multi-factor Composite (trend + RSI + Bollinger + VIX)",
            "params": {
                "sma_period": "Trend SMA (default: 50)",
                "rsi_period": "RSI period (default: 14)",
                "rsi_oversold": "RSI oversold (default: 35)",
                "rsi_overbought": "RSI overbought (default: 65)",
                "vix_calm_threshold": "VIX calm threshold (default: 20)",
                "bb_window": "Bollinger window (default: 20)",
                "bb_num_std": "Bollinger std devs (default: 2.0)",
            },
        },
    }
    return json.dumps(strategies, indent=2)


@tool
def run_backtest(params_json: str) -> str:
    """Run a backtest with a specific strategy configuration. Input must be a JSON string with fields: strategy_type (required, e.g. "sma_crossover"), params (optional dict), initial_capital (optional, default 1000000), stop_loss_pct (optional, default 2.0), take_profit_pct (optional, default 4.0), start_date (optional YYYY-MM-DD), end_date (optional YYYY-MM-DD). Example: {"strategy_type": "rsi_mean_reversion", "params": {"oversold": 25, "overbought": 75}}"""
    try:
        # Parse JSON input — handle common LLM formatting issues
        params_json = params_json.strip()
        # Sometimes the LLM wraps in extra text; extract JSON
        if not params_json.startswith("{"):
            # Try to find JSON within the text
            start = params_json.find("{")
            end = params_json.rfind("}") + 1
            if start >= 0 and end > start:
                params_json = params_json[start:end]
            else:
                return f"Error: Could not parse JSON from input: {params_json[:200]}"

        config = json.loads(params_json)
        strategy_type = config.get("strategy_type")
        if not strategy_type:
            return "Error: 'strategy_type' is required. Use list_strategy_types to see options."

        strategy_params = config.get("params", {})
        initial_capital = float(config.get("initial_capital", 1_000_000))
        stop_loss_pct = float(config.get("stop_loss_pct", 2.0))
        take_profit_pct = float(config.get("take_profit_pct", 4.0))
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        trailing_stop_pct = float(config.get("trailing_stop_pct", 0.0))
        slippage_pct = float(config.get("slippage_pct", 0.01))
        commission_pct = float(config.get("commission_pct", 0.02))

        from config.settings import DEFAULT_POSITION_SIZE
        from src.simulator.backtester import Backtester

        # Import the strategy builder from api module
        from src.web.api import _build_strategy_fn

        df = _get_data(start_date, end_date)

        if len(df) < 100:
            return f"Error: Not enough data ({len(df)} rows). Need at least 100 rows."

        strat_fn = _build_strategy_fn(strategy_type, strategy_params)
        bt = Backtester(
            df, strat_fn,
            initial_capital=initial_capital,
            position_size_pct=DEFAULT_POSITION_SIZE,
            stop_loss_pct=stop_loss_pct / 100,
            take_profit_pct=take_profit_pct / 100,
            trailing_stop_pct=trailing_stop_pct / 100,
            slippage_pct=slippage_pct / 100,
            commission_pct=commission_pct / 100,
        )
        result = bt.run()

        # Also save to DB for persistence
        from src.storage.database import save_backtest_run
        import math

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
            "risk_reward_ratio": result.risk_reward_ratio,
            "max_drawdown_duration": result.max_drawdown_duration,
            "avg_trade_duration": result.avg_trade_duration,
            "exposure_time_pct": result.exposure_time_pct,
            "benchmark_return_pct": result.benchmark_return_pct,
            "alpha_pct": result.alpha_pct,
            "information_ratio": result.information_ratio,
        }
        # Sanitise NaN/Inf
        for k, v in result_dict.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                result_dict[k] = None

        run_id = save_backtest_run(
            strategy_id=None,
            strategy_name=result.strategy_name,
            params=strategy_params,
            result_dict=result_dict,
            equity_json="[]",
            trades_json="[]",
            data_start=str(df["Date"].iloc[0].date()),
            data_end=str(df["Date"].iloc[-1].date()),
        )

        # Return a concise summary for the LLM
        summary = {
            "run_id": run_id,
            "strategy_name": result.strategy_name,
            "period": f"{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}",
            "initial_capital": f"₹{result.initial_capital:,.0f}",
            "final_capital": f"₹{result.final_capital:,.0f}",
            "total_return": f"{result.total_return_pct:+.2f}%",
            "annual_return": f"{result.annual_return_pct:+.2f}%",
            "benchmark_return": f"{result.benchmark_return_pct:+.2f}%",
            "alpha": f"{result.alpha_pct:+.2f}%",
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "sortino_ratio": round(result.sortino_ratio, 2),
            "max_drawdown": f"{result.max_drawdown_pct:.2f}%",
            "total_trades": result.total_trades,
            "win_rate": f"{result.win_rate_pct:.1f}%",
            "profit_factor": round(result.profit_factor, 2),
            "avg_trade_pnl": f"₹{result.avg_trade_pnl:,.2f}",
            "exposure_time": f"{result.exposure_time_pct:.1f}%",
        }
        return json.dumps(summary, indent=2)

    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}. Input was: {params_json[:200]}"
    except Exception as e:
        logger.error("Backtest tool error: %s\n%s", e, traceback.format_exc())
        return f"Error running backtest: {e}"


@tool
def get_recent_results(query: str) -> str:
    """Get recent backtest results to compare strategies. Input: any text (ignored). Returns JSON array of recent run summaries."""
    try:
        from src.storage.database import list_backtest_runs
        runs = list_backtest_runs(limit=10)
        if not runs:
            return "No backtest results found. Run a backtest first."

        results = []
        for r in runs:
            results.append({
                "run_id": r["id"],
                "strategy": r["strategy_name"],
                "total_return": f"{r['total_return']:+.2f}%",
                "sharpe": round(r["sharpe_ratio"], 2),
                "sortino": round(r.get("sortino_ratio", 0) or 0, 2),
                "max_drawdown": f"{r['max_drawdown']:.2f}%",
                "alpha": f"{(r.get('alpha', 0) or 0):+.2f}%",
                "trades": r["total_trades"],
                "win_rate": f"{r['win_rate']:.1f}%",
                "run_date": r["run_at"][:10],
            })
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error getting results: {e}"


@tool
def list_event_categories(query: str) -> str:
    """List all available event categories and a count of events in each. Categories include: war, oil_shock, financial_crisis, pandemic, india_budget, rbi_policy, fed_meeting, india_election, us_election, policy_reform, trade_war, terror_attack, geopolitical, natural_disaster, corporate_crisis. Input: any text (ignored)."""
    try:
        from src.data.events import get_event_summary
        summary = get_event_summary()
        # Return concise version
        result = {}
        for cat, info in summary.items():
            result[cat] = {
                "description": info["description"],
                "count": info["count"],
                "events": [e["name"] for e in info["events"]],
            }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error listing event categories: {e}"


@tool
def analyze_events(params_json: str) -> str:
    """Analyze how the Nifty 50 market performed during specific types of events.
Input must be a JSON string with at least one of:
  - category: event category (e.g. "war", "oil_shock", "financial_crisis", "pandemic", "india_budget", "rbi_policy", "fed_meeting", "india_election", "us_election", "policy_reform", "trade_war", "terror_attack", "geopolitical", "natural_disaster", "corporate_crisis")
  - tag: a tag to search for (e.g. "oil", "india", "fed", "covid", "modi", "tariffs")
Example: {"category": "war"}
Example: {"tag": "oil"}
Returns: Statistics for each event period and aggregate comparison vs non-event periods."""
    try:
        params_json = params_json.strip()
        if not params_json.startswith("{"):
            start = params_json.find("{")
            end = params_json.rfind("}") + 1
            if start >= 0 and end > start:
                params_json = params_json[start:end]
            else:
                return f"Error: Could not parse JSON from input: {params_json[:200]}"

        config = json.loads(params_json)
        category = config.get("category")
        tag = config.get("tag")

        if not category and not tag:
            return "Error: Provide 'category' or 'tag'. Use list_event_categories to see options."

        from src.data.events import analyze_market_during_events
        from config.settings import NIFTY_CLOSE_COL

        df = _get_data()
        analysis = analyze_market_during_events(
            df, category=category, tag=tag, price_col=NIFTY_CLOSE_COL,
        )

        if "error" in analysis:
            return f"Error: {analysis['error']}"

        return json.dumps(analysis, indent=2)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}. Input was: {params_json[:200]}"
    except Exception as e:
        logger.error("Event analysis tool error: %s\n%s", e, traceback.format_exc())
        return f"Error analysing events: {e}"


@tool
def search_event_metadata(query: str) -> str:
    """Search the curated event catalog by keyword. Finds events matching the search term in name, description, category, or tags.
Input: a search keyword or phrase (e.g. "war", "oil", "covid", "budget 2020", "modi", "demonetisation").
Returns: List of matching events with dates, category, impact, and description."""
    try:
        from src.data.events import search_events
        events = search_events(query)
        if not events:
            return f"No events found matching '{query}'. Try broader terms like 'war', 'oil', 'crisis', 'election', 'budget'."

        results = []
        for e in events:
            results.append({
                "name": e.name,
                "category": e.category,
                "dates": f"{e.start_date} to {e.end_date}" if e.end_date else str(e.start_date),
                "impact": e.impact,
                "region": e.region,
                "description": e.description,
                "tags": e.tags,
            })
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching events: {e}"


@tool
def predict_market(query: str) -> str:
    """Run probability predictions for where Nifty 50 will be in the next day, week, and month. Uses ensemble method with Monte Carlo. Input: any text (ignored)."""
    try:
        from src.predictor.probability import predict as run_prediction
        df = _get_data()
        report = run_prediction(df, method="ensemble", n_simulations=5000)

        result = {
            "as_of_date": str(report.as_of_date.date()),
            "current_price": f"₹{report.current_price:,.2f}",
            "current_vix": round(report.current_vix, 2) if report.current_vix else None,
            "vix_regime": report.vix_regime,
            "predictions": {},
        }
        for name, pred in report.predictions.items():
            pct90 = pred.percentiles.get(0.90, (0, 0))
            result["predictions"][name] = {
                "horizon_days": pred.horizon_days,
                "mean_price": f"₹{pred.mean_price:,.2f}",
                "prob_up": f"{pred.prob_up * 100:.1f}%",
                "prob_down": f"{pred.prob_down * 100:.1f}%",
                "90pct_range": f"₹{pct90[0]:,.0f} – ₹{pct90[1]:,.0f}",
            }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error running predictions: {e}"


# ── Agent Setup ───────────────────────────────────────────────────────────

TOOLS = [
    data_summary, list_strategy_types, run_backtest, get_recent_results,
    predict_market, list_event_categories, analyze_events, search_event_metadata,
]

SYSTEM_PROMPT = """You are AITrading Copilot, an expert trading strategy assistant for the Nifty 50 Indian stock market index.

You help users by:
1. Understanding their natural-language backtest instructions
2. Choosing the right strategy type and parameters
3. Running backtests and explaining results clearly
4. Comparing strategies and making recommendations
5. Providing market predictions
6. Analysing how markets behaved during specific macro events (wars, oil shocks, budgets, elections, crises, etc.)

EVENT ANALYSIS CAPABILITIES:
- You have access to a curated catalog of 60+ major market events (2007-2026) from reputable sources (RBI, Ministry of Finance India, US Federal Reserve, ECI, OPEC, IMF, Reuters, Bloomberg)
- Event categories: war, oil_shock, financial_crisis, pandemic, india_budget, rbi_policy, fed_meeting, india_election, us_election, policy_reform, trade_war, terror_attack, geopolitical, natural_disaster, corporate_crisis
- Use analyze_events with {"category": "war"} to see market performance during all wars
- Use analyze_events with {"tag": "oil"} to see performance during oil-related events
- Use search_event_metadata to find specific events by keyword
- When users ask about historical events and market behaviour, use these tools to provide data-driven answers
- Always include probability of decline, avg returns, and comparison with non-event periods

IMPORTANT RULES:
- When running a backtest, ALWAYS use the run_backtest tool with properly formatted JSON
- Strategy types are: sma_crossover, ema_crossover, rsi_mean_reversion, macd_crossover, macd_histogram, bollinger_band, atr_breakout, vix_regime, stochastic_oscillator, mean_reversion_zscore, composite_sniper
- If the user mentions "moving average" without specifying SMA/EMA, use sma_crossover
- If the user asks about data, use data_summary first
- If the user wants to compare, use get_recent_results
- Present results clearly with key metrics: return, Sharpe, max drawdown, win rate
- Amounts are in Indian Rupees (₹)
- Be concise and action-oriented in your responses
"""


def create_trading_agent():
    """
    Create and return the LLM-powered trading agent.

    Uses langchain 1.2+ create_agent with ChatWatsonx.
    Returns None if LLM is not available.
    """
    llm = get_llm()
    if llm is None:
        logger.warning("LLM not available — agent cannot be created")
        return None

    try:
        from langchain.agents import create_agent

        agent = create_agent(
            model=llm,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
        logger.info("Trading agent created successfully (create_agent)")
        return agent
    except Exception as e:
        logger.error("Failed to create trading agent: %s\n%s", e, traceback.format_exc())
        return None


# Singleton agent
_agent_instance = None


def get_agent():
    """Get or create the singleton trading agent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_trading_agent()
    return _agent_instance


def run_agent_query(user_message: str) -> dict:
    """
    Run a natural language query through the trading agent.

    Returns:
        dict with keys: success, response, intermediate_steps (optional)
    """
    agent = get_agent()
    if agent is None:
        return {
            "success": False,
            "response": (
                "LLM is not available. Please check your WatsonX credentials in `.env`.\n\n"
                "Required variables:\n"
                "- `WATSONX_APIKEY`\n"
                "- `WATSONX_PROJECT_ID`\n\n"
                "Falling back to rule-based copilot."
            ),
        }

    try:
        # The modern create_agent returns a CompiledStateGraph
        # which accepts {"messages": [{"role": "user", "content": ...}]}
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
        )

        # Extract the final AI message from the response
        messages = result.get("messages", [])
        if messages:
            # Last message should be the AI response
            final_msg = messages[-1]
            output = getattr(final_msg, "content", str(final_msg))
        else:
            output = "I couldn't process that request."

        # Extract tool call info from intermediate messages
        intermediate_steps = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    intermediate_steps.append({
                        "tool": tc.get("name", "unknown"),
                        "input": str(tc.get("args", ""))[:200],
                    })
            elif hasattr(msg, "name") and msg.name:  # ToolMessage
                intermediate_steps.append({
                    "tool": msg.name,
                    "output": str(getattr(msg, "content", ""))[:500],
                })

        return {
            "success": True,
            "response": output,
            "intermediate_steps": intermediate_steps,
        }
    except Exception as e:
        logger.error("Agent query failed: %s\n%s", e, traceback.format_exc())
        return {
            "success": False,
            "response": f"The AI agent encountered an error: {str(e)}\n\nPlease try rephrasing your question.",
        }
