"""
Copilot Engine
──────────────
An AI assistant that can:
  • Interpret natural-language strategy descriptions
  • Create and configure strategies from chat messages
  • Trigger backtests and report results
  • Answer questions about data and performance

Uses a hybrid approach:
  1. Rule-based pattern matching for quick, well-defined actions
  2. IBM WatsonX LLM (via LangChain) for complex natural-language queries
     that the rule engine can't handle — e.g. "run a 20/100 SMA backtest
     with 3% stop loss on 2022 data and compare with RSI"

LLM configuration follows the IBM_RAG project pattern:
  - Credentials in .env (WATSONX_APIKEY, WATSONX_PROJECT_ID)
  - Uses langchain_ibm.WatsonxLLM with granite model
  - Falls back to rule-based if LLM is unavailable
"""

import json
import logging
import re
from typing import Optional

from src.storage.database import (
    create_strategy,
    delete_strategy,
    get_strategy,
    get_strategy_by_name,
    list_strategies,
    list_backtest_runs,
    save_backtest_run,
)

logger = logging.getLogger(__name__)


# ── Intent Detection ─────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "list_strategies": [
        r"(?:list|show|get|display)\s+(?:all\s+)?(?:strategies|strats)",
        r"what\s+strategies",
        r"available\s+strategies",
    ],
    "create_sma": [
        r"(?:create|add|new|make)\s+(?:an?\s+)?sma\s*(?:crossover)?\s*(?:strategy)?.*?(\d+).*?(\d+)",
        r"sma\s+(\d+)\s*[/,]\s*(\d+)",
        r"moving\s+average.*?(\d+).*?(\d+)",
    ],
    "create_rsi": [
        r"(?:create|add|new|make)\s+(?:an?\s+)?rsi\s*(?:mean\s*reversion)?\s*(?:strategy)?",
        r"rsi.*?(?:oversold|buy)\s*(?:at|=|:)?\s*(\d+).*?(?:overbought|sell)\s*(?:at|=|:)?\s*(\d+)",
        r"rsi\s+(\d+)\s*[/,]\s*(\d+)",
    ],
    "create_vix": [
        r"(?:create|add|new|make)\s+(?:an?\s+)?vix\s*(?:regime)?\s*(?:strategy)?",
        r"vix.*?(?:buy|below)\s*(?:at|=|:)?\s*(\d+\.?\d*).*?(?:sell|above)\s*(?:at|=|:)?\s*(\d+\.?\d*)",
    ],
    "create_ema": [
        r"ema\s+(\d+)\s*[/,]\s*(\d+)",
        r"exponential\s+moving\s+average.*?(\d+).*?(\d+)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?ema\s*(?:crossover)?\s*(?:strategy)?.*?(\d+).*?(\d+)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?ema\s*(?:crossover)?\s*(?:strategy)?",
    ],
    "create_macd_hist": [
        r"(?:create|add|new|make)\s+(?:an?\s+)?macd\s*hist(?:ogram)?\s*(?:strategy)?",
        r"macd\s+hist",
    ],
    "create_macd": [
        r"macd\s+(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?macd\s*(?:crossover)?\s*(?:strategy)?",
    ],
    "create_bollinger": [
        r"(?:bollinger|bb)\s+(\d+)\s*[/,\s]\s*(\d+\.?\d*)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?(?:bollinger|bb)\s*(?:band)?.*?(\d+).*?(\d+\.?\d*)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?(?:bollinger|bb)\s*(?:band)?\s*(?:strategy)?",
    ],
    "create_atr": [
        r"atr\s+(\d+)\s*[/,]\s*(\d+)\s*[/,]\s*(\d+\.?\d*)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?atr\s*(?:breakout)?.*?(\d+).*?(\d+).*?(\d+\.?\d*)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?atr\s*(?:breakout)?\s*(?:strategy)?",
    ],
    "create_stochastic": [
        r"stoch(?:astic)?\s+(\d+)\s*[/,]\s*(\d+)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?stoch(?:astic)?\s*(?:oscillator)?.*?(\d+)\s*[/,]\s*(\d+)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?stoch(?:astic)?\s*(?:oscillator)?\s*(?:strategy)?",
    ],
    "create_zscore": [
        r"z[\-\s]?score\s+(\d+)\s*[/,\s]\s*([\-\d.]+)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?(?:z[\-\s]?score|mean\s*reversion).*?(\d+).*?([\-\d.]+)",
        r"(?:create|add|new|make)\s+(?:an?\s+)?(?:z[\-\s]?score|mean\s*reversion)\s*(?:strategy)?",
    ],
    "create_sniper": [
        r"(?:create|add|new|make)\s+(?:an?\s+)?(?:sniper|composite)\s*(?:strategy)?",
    ],
    "delete_strategy": [
        r"(?:delete|remove)\s+(?:strategy\s+)?#?(\d+)",
        r"(?:delete|remove)\s+(?:strategy\s+)?['\"]?(.+?)['\"]?\s*$",
        r"drop\s+strategy\s+#?(\d+)",
        r"drop\s+strategy\s+['\"]?(.+?)['\"]?\s*$",
    ],
    "run_backtest": [
        r"(?:run|execute|start|do)\s+(?:a\s+)?(?:back\s*test|bt)",
        r"backtest\s+(?:strategy|strat)",
        r"test\s+(?:the\s+)?strategy",
        r"backtest\s+['\"]?(.+?)['\"]?\s*$",
    ],
    "backtest_all": [
        r"(?:run|execute|backtest)\s+all",
        r"test\s+all\s+strategies",
        r"backtest\s+everything",
    ],
    "show_results": [
        r"(?:show|list|get|display)\s+(?:backtest\s+)?results",
        r"(?:recent|last|previous)\s+(?:backtest|results|runs)",
        r"how\s+did\s+(?:it|they)\s+(?:do|perform)",
    ],
    "compare_strategies": [
        r"compare\s+(?:(?:strategy|strat|sma|rsi|ema|macd|bollinger|atr|vix|stoch|zscore|sniper)\s+.+?\s+(?:and|vs|versus|with|,)\s+.+?)\s*$",
        r"compare\s+(.+?)\s+(?:and|vs|versus|with)\s+(.+?)\s+(?:strat|backtest|result)",
        r"(?:compare|diff)\s+strategies",
    ],
    "data_info": [
        r"(?:show|tell|display|what)\s+(?:me\s+)?(?:about\s+)?(?:the\s+)?data",
        r"data\s+(?:info|summary|overview|stats)",
        r"how\s+much\s+data",
        r"date\s+range",
    ],
    "list_events": [
        r"(?:list|show|get|display)\s+(?:all\s+)?(?:events|market\s+events)",
        r"what\s+events",
        r"event\s+(?:catalog|categories|list)",
    ],
    "analyze_events": [
        r"(?:during|when\s+there\s+(?:was|were)|times?\s+(?:of|there\s+was))\s+(?:a\s+)?(?:war|oil|crisis|pandemic|budget|election|terror)",
        r"(?:market|nifty)\s+(?:during|in)\s+(?:war|oil|crisis|pandemic|budget|election|terror)",
        r"(?:how|what)\s+(?:does|did|is)\s+(?:the\s+)?(?:market|nifty)\s+(?:perform|do|behave|react)\s+(?:during|in|when)",
        r"(?:probability|chance)\s+(?:of)\s+(?:market|nifty)\s+(?:drop|fall|crash|declin).*?(?:during|war|crisis|oil|pandemic|budget|election)",
        r"(?:event|macro)\s+(?:analysis|impact)",
        r"(?:analyse|analyze)\s+(?:events?|market)\s+(?:during|for)",
    ],
    "predict": [
        r"(?:predict|forecast)\s+(?:nifty|market)",
        r"where\s+(?:will|would)\s+nifty\s+be",
        r"probability\s+(?:of|for)\s+(?:nifty|market)\s+(?:going|moving|being)",
    ],
    "help": [
        r"^(?:help|what\s+can\s+you\s+do|\?|commands)$",
        r"how\s+(?:do\s+I|to|can\s+I)",
    ],
}


def detect_intent(message: str) -> tuple[str, dict]:
    """
    Parse user message and return (intent, extracted_params).
    """
    msg = message.strip().lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, msg)
            if match:
                params = {}
                groups = match.groups()
                if intent == "create_sma" and len(groups) >= 2:
                    params = {"fast_window": int(groups[0]), "slow_window": int(groups[1])}
                elif intent == "create_rsi" and len(groups) >= 2:
                    params = {"oversold": float(groups[0]), "overbought": float(groups[1])}
                elif intent == "create_vix" and len(groups) >= 2:
                    params = {"buy_below": float(groups[0]), "sell_above": float(groups[1])}
                elif intent == "create_ema" and len(groups) >= 2:
                    params = {"fast_span": int(groups[0]), "slow_span": int(groups[1])}
                elif intent == "create_macd" and len(groups) >= 3:
                    params = {"fast_ema": int(groups[0]), "slow_ema": int(groups[1]), "signal_period": int(groups[2])}
                elif intent == "create_bollinger" and len(groups) >= 2:
                    params = {"window": int(groups[0]), "num_std": float(groups[1])}
                elif intent == "create_atr" and len(groups) >= 3:
                    params = {"sma_window": int(groups[0]), "atr_period": int(groups[1]), "atr_multiplier": float(groups[2])}
                elif intent == "create_stochastic" and len(groups) >= 2:
                    params = {"k_period": int(groups[0]), "d_period": int(groups[1])}
                elif intent == "create_zscore" and len(groups) >= 2:
                    params = {"lookback": int(groups[0]), "entry_z": float(groups[1])}
                elif intent == "delete_strategy" and len(groups) >= 1 and groups[0]:
                    # Could be an ID (digits) or a name
                    val = groups[0].strip().strip("'\"")
                    if val.isdigit():
                        params = {"strategy_id": int(val)}
                    else:
                        params = {"strategy_name": val}
                elif intent == "compare_strategies" and len(groups) >= 2:
                    params = {"strategy_a": groups[0].strip().strip("'\""), "strategy_b": groups[1].strip().strip("'\"")}
                elif intent == "run_backtest" and len(groups) >= 1 and groups[0]:
                    params = {"strategy_name": groups[0].strip().strip("'\"") }
                return intent, params

    return "unknown", {}


# ── Action Handlers ──────────────────────────────────────────────────────

def handle_list_strategies(**kwargs) -> str:
    strategies = list_strategies()
    if not strategies:
        return "No strategies found. Try creating one first!\n\nExample: *Create an SMA crossover strategy with 10/50 windows*"

    lines = ["### 📋 Saved Strategies\n"]
    lines.append("| # | Name | Type | Parameters | Active |")
    lines.append("|---|------|------|------------|--------|")
    for s in strategies:
        params = json.loads(s["params"]) if isinstance(s["params"], str) else s["params"]
        p_str = ", ".join(f"{k}={v}" for k, v in params.items())
        active = "✅" if s["is_active"] else "❌"
        lines.append(f"| {s['id']} | {s['name']} | {s['type']} | {p_str} | {active} |")

    return "\n".join(lines)


def handle_create_sma(fast_window: int = 10, slow_window: int = 50, **kwargs) -> str:
    name = f"SMA Crossover ({fast_window}/{slow_window})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']}). Use a different name or parameters."

    params = {"fast_window": fast_window, "slow_window": slow_window}
    strat = create_strategy(
        name=name,
        stype="sma_crossover",
        params=params,
        description=f"Buy when {fast_window}-day SMA crosses above {slow_window}-day SMA.",
    )
    return (
        f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n"
        f"- Type: SMA Crossover\n"
        f"- Fast window: {fast_window} days\n"
        f"- Slow window: {slow_window} days\n\n"
        f"Say **\"backtest {strat['name']}\"** to run it!"
    )


def handle_create_rsi(oversold: float = 30, overbought: float = 70, **kwargs) -> str:
    name = f"RSI Mean Reversion ({int(oversold)}/{int(overbought)})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."

    params = {"oversold": oversold, "overbought": overbought}
    strat = create_strategy(
        name=name,
        stype="rsi_mean_reversion",
        params=params,
        description=f"Buy when RSI < {oversold}, sell when RSI > {overbought}.",
    )
    return (
        f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n"
        f"- Oversold threshold: {oversold}\n"
        f"- Overbought threshold: {overbought}\n\n"
        f"Say **\"backtest {strat['name']}\"** to run it!"
    )


def handle_create_vix(buy_below: float = 15, sell_above: float = 25, **kwargs) -> str:
    name = f"VIX Regime ({buy_below}/{sell_above})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."

    params = {"buy_below": buy_below, "sell_above": sell_above}
    strat = create_strategy(
        name=name,
        stype="vix_regime",
        params=params,
        description=f"Buy when VIX < {buy_below}, sell when VIX > {sell_above}.",
    )
    return (
        f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n"
        f"- Buy below VIX: {buy_below}\n"
        f"- Sell above VIX: {sell_above}"
    )


def handle_create_ema(fast_span: int = 12, slow_span: int = 26, **kwargs) -> str:
    name = f"EMA Crossover ({fast_span}/{slow_span})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"fast_span": fast_span, "slow_span": slow_span}
    strat = create_strategy(name=name, stype="ema_crossover", params=params,
                            description=f"EMA Crossover: buy when {fast_span}-EMA crosses above {slow_span}-EMA.")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n- Fast EMA: {fast_span}\n- Slow EMA: {slow_span}"


def handle_create_macd(fast_ema: int = 12, slow_ema: int = 26, signal_period: int = 9, **kwargs) -> str:
    name = f"MACD Crossover ({fast_ema}/{slow_ema}/{signal_period})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"fast_ema": fast_ema, "slow_ema": slow_ema, "signal_period": signal_period}
    strat = create_strategy(name=name, stype="macd_crossover", params=params,
                            description=f"MACD {fast_ema}/{slow_ema}/{signal_period} signal-line crossover.")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n- MACD: {fast_ema}/{slow_ema}/{signal_period}"


def handle_create_bollinger(window: int = 20, num_std: float = 2.0, **kwargs) -> str:
    name = f"Bollinger Bands ({window}/{num_std}σ)"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"window": window, "num_std": num_std}
    strat = create_strategy(name=name, stype="bollinger_band", params=params,
                            description=f"Bollinger Bands: {window}-day SMA ± {num_std}σ mean reversion.")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n- Window: {window}\n- Std devs: {num_std}"


def handle_create_atr(sma_window: int = 20, atr_period: int = 14, atr_multiplier: float = 1.5, **kwargs) -> str:
    name = f"ATR Breakout ({sma_window}/{atr_period}/{atr_multiplier})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"sma_window": sma_window, "atr_period": atr_period, "atr_multiplier": atr_multiplier}
    strat = create_strategy(name=name, stype="atr_breakout", params=params,
                            description=f"ATR Breakout: {sma_window}-SMA ± {atr_multiplier}×ATR({atr_period}).")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n- SMA: {sma_window}, ATR: {atr_period}, Multiplier: {atr_multiplier}"


def handle_create_stochastic(k_period: int = 14, d_period: int = 3, **kwargs) -> str:
    name = f"Stochastic ({k_period}/{d_period})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"k_period": k_period, "d_period": d_period, "oversold": 20, "overbought": 80}
    strat = create_strategy(name=name, stype="stochastic_oscillator", params=params,
                            description=f"Stochastic %K({k_period})/%D({d_period}) crossover.")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n- %K: {k_period}, %D: {d_period}"


def handle_create_zscore(lookback: int = 20, entry_z: float = -2.0, **kwargs) -> str:
    name = f"Z-Score ({lookback}/{entry_z})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"lookback": lookback, "entry_z": entry_z, "exit_z": 0.0}
    strat = create_strategy(name=name, stype="mean_reversion_zscore", params=params,
                            description=f"Z-Score mean reversion: {lookback}-day, entry at z={entry_z}.")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n- Lookback: {lookback}, Entry z: {entry_z}"


def handle_create_macd_hist(fast_ema: int = 12, slow_ema: int = 26, signal_period: int = 9, **kwargs) -> str:
    name = f"MACD Histogram ({fast_ema}/{slow_ema}/{signal_period})"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"fast_ema": fast_ema, "slow_ema": slow_ema, "signal_period": signal_period}
    strat = create_strategy(name=name, stype="macd_histogram", params=params,
                            description=f"MACD Histogram reversal {fast_ema}/{slow_ema}/{signal_period}.")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\n- MACD Hist: {fast_ema}/{slow_ema}/{signal_period}"


def handle_create_sniper(**kwargs) -> str:
    name = "Composite Sniper (Custom)"
    existing = get_strategy_by_name(name)
    if existing:
        return f"Strategy **{name}** already exists (ID: {existing['id']})."
    params = {"sma_period": 50, "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65,
              "vix_calm_threshold": 20, "bb_window": 20, "bb_num_std": 2.0}
    strat = create_strategy(name=name, stype="composite_sniper", params=params,
                            description="Multi-factor sniper: trend + pullback + value + regime filter.")
    return f"✅ Created strategy **{strat['name']}** (ID: {strat['id']})\n\nMulti-factor composite with trend, RSI, Bollinger, and VIX filters."


def handle_delete_strategy(strategy_id: int = None, strategy_name: str = None, **kwargs) -> str:
    """Delete a strategy by ID or name."""
    if strategy_id:
        strat = get_strategy(strategy_id)
        if not strat:
            return f"No strategy found with ID **{strategy_id}**."
        delete_strategy(strategy_id)
        return f"🗑️ Deleted strategy **{strat['name']}** (ID: {strategy_id})."
    elif strategy_name:
        # Fuzzy match by name
        strategies = list_strategies()
        matched = None
        for s in strategies:
            if strategy_name.lower() in s["name"].lower():
                matched = s
                break
        if matched:
            delete_strategy(matched["id"])
            return f"🗑️ Deleted strategy **{matched['name']}** (ID: {matched['id']})."
        else:
            return f"No strategy found matching **\"{strategy_name}\"**. Use `list strategies` to see available ones."
    else:
        return "Please specify which strategy to delete. Examples:\n- *\"delete strategy 5\"*\n- *\"delete SMA Crossover\"*"


def handle_show_results(**kwargs) -> str:
    runs = list_backtest_runs(limit=10)
    if not runs:
        return "No backtest results yet. Try running a backtest first!"

    lines = ["### 📊 Recent Backtest Results\n"]
    lines.append("| # | Strategy | Return% | Sharpe | Sortino | MaxDD% | Alpha% | Trades | WinRate% | Run Date |")
    lines.append("|---|----------|---------|--------|---------|--------|--------|--------|----------|----------|")
    for r in runs:
        sortino = r.get('sortino_ratio', 0) or 0
        alpha = r.get('alpha', 0) or 0
        lines.append(
            f"| {r['id']} | {r['strategy_name']} | {r['total_return']:+.2f}% | "
            f"{r['sharpe_ratio']:.2f} | {sortino:.2f} | {r['max_drawdown']:.2f}% | "
            f"{alpha:+.2f}% | {r['total_trades']} | "
            f"{r['win_rate']:.1f}% | {r['run_at'][:10]} |"
        )
    return "\n".join(lines)


def handle_compare_strategies(strategy_a: str = "", strategy_b: str = "", **kwargs) -> str:
    """Compare the latest backtest runs for two strategies side-by-side."""
    if not strategy_a or not strategy_b:
        return ("Please name two strategies to compare.\n\n"
                "Example: *\"compare SMA and RSI\"*")

    runs = list_backtest_runs(limit=200)  # grab enough to find both
    if not runs:
        return "No backtest results yet. Run backtests first!"

    def _find_run(name: str):
        name_l = name.lower()
        for r in runs:
            if name_l in (r.get("strategy_name") or "").lower():
                return r
        return None

    run_a = _find_run(strategy_a)
    run_b = _find_run(strategy_b)

    if not run_a and not run_b:
        return f"Could not find backtest runs matching **\"{strategy_a}\"** or **\"{strategy_b}\"**. Run backtests first."
    if not run_a:
        return f"Could not find a backtest run matching **\"{strategy_a}\"**."
    if not run_b:
        return f"Could not find a backtest run matching **\"{strategy_b}\"**."

    def _val(r, key, fmt=".2f"):
        v = r.get(key, 0) or 0
        return f"{v:{fmt}}"

    name_a = run_a.get("strategy_name", strategy_a)
    name_b = run_b.get("strategy_name", strategy_b)

    lines = [
        f"### 🔍 Strategy Comparison\n",
        f"| Metric | {name_a} | {name_b} |",
        f"|--------|{'---' * len(name_a)}|{'---' * len(name_b)}|",
        f"| Total Return | {_val(run_a, 'total_return', '+.2f')}% | {_val(run_b, 'total_return', '+.2f')}% |",
        f"| Annual Return | {_val(run_a, 'annual_return', '+.2f')}% | {_val(run_b, 'annual_return', '+.2f')}% |",
        f"| Sharpe Ratio | {_val(run_a, 'sharpe_ratio')} | {_val(run_b, 'sharpe_ratio')} |",
        f"| Sortino Ratio | {_val(run_a, 'sortino_ratio')} | {_val(run_b, 'sortino_ratio')} |",
        f"| Max Drawdown | {_val(run_a, 'max_drawdown')}% | {_val(run_b, 'max_drawdown')}% |",
        f"| Alpha | {_val(run_a, 'alpha', '+.2f')}% | {_val(run_b, 'alpha', '+.2f')}% |",
        f"| Profit Factor | {_val(run_a, 'profit_factor')} | {_val(run_b, 'profit_factor')} |",
        f"| Win Rate | {_val(run_a, 'win_rate', '.1f')}% | {_val(run_b, 'win_rate', '.1f')}% |",
        f"| Total Trades | {_val(run_a, 'total_trades', '.0f')} | {_val(run_b, 'total_trades', '.0f')} |",
        f"| Risk/Reward | {_val(run_a, 'risk_reward_ratio')} | {_val(run_b, 'risk_reward_ratio')} |",
        f"| Exposure | {_val(run_a, 'exposure_time', '.1f')}% | {_val(run_b, 'exposure_time', '.1f')}% |",
    ]
    return "\n".join(lines)


def handle_help(**kwargs) -> str:
    return """### 🤖 AITrading Copilot – Commands

I can help you create, manage, and backtest trading strategies. Try saying:

**Strategy Management:**
- *"List all strategies"*
- *"Delete strategy 5"* or *"Delete SMA Crossover"*
- *"Create SMA crossover 15/60"* — Simple Moving Average
- *"Create EMA crossover 12/26"* — Exponential Moving Average
- *"Create RSI strategy oversold 25 overbought 75"*
- *"Create MACD 12/26/9"* — MACD signal crossover
- *"Create MACD histogram"* — MACD histogram reversal
- *"Create Bollinger 20 2.0"* — Bollinger band mean reversion
- *"Create ATR breakout 20 14 1.5"* — ATR breakout strategy
- *"Create Stochastic 14/3"* — Stochastic oscillator
- *"Create Z-Score 20 -2.0"* — Z-Score mean reversion
- *"Create VIX regime buy below 12 sell above 28"*
- *"Create sniper strategy"* — Multi-factor composite

**Backtesting:**
- *"Run backtest SMA Crossover (10/50)"*  — test a specific strategy
- *"Backtest all strategies"*  — run all active strategies
- *"Show backtest results"*  — view recent results
- *"Compare SMA and RSI"*  — side-by-side comparison

**Data & Predictions:**
- *"Show me the data"*  — data summary
- *"Predict where Nifty will be"*  — probability forecast

**Event Analysis:**
- *"List all events"*  — see the curated event catalog
- *"How does the market perform during wars?"*
- *"What's the probability of a drop during oil shocks?"*
- *"Analyze market during financial crises"*
- *"Market behaviour during RBI policy changes"*
- *"How did Nifty react to elections?"*

**Tips:**
- You can use natural language — I'll figure out what you mean!
- All strategies and results are saved and persist between sessions.
"""


def handle_list_events(**kwargs) -> str:
    """List available event categories and event counts."""
    try:
        from src.data.events import CATEGORIES, get_events_by_category
        lines = ["### 📅 Market Event Catalog\n"]
        lines.append("| Category | Description | Events |")
        lines.append("|----------|-------------|--------|")
        total = 0
        for cat, desc in CATEGORIES.items():
            count = len(get_events_by_category(cat))
            total += count
            lines.append(f"| `{cat}` | {desc} | {count} |")
        lines.append(f"\n**Total: {total} curated events** (2007-2026)")
        lines.append("\n💡 Try: *\"How does the market perform during wars?\"* or *\"Analyze events during oil shocks\"*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing events: {e}"


def handle_analyze_events_rule(message: str = "", **kwargs) -> str:
    """Rule-based handler that extracts category from the message and runs event analysis."""
    try:
        from src.data.events import analyze_market_during_events, CATEGORIES
        from src.data.loader import load_master
        from config.settings import NIFTY_CLOSE_COL

        msg_lower = message.lower()

        # Map common words to categories
        keyword_map = {
            "war": "war",
            "conflict": "war",
            "military": "war",
            "oil": "oil_shock",
            "crude": "oil_shock",
            "energy": "oil_shock",
            "opec": "oil_shock",
            "crisis": "financial_crisis",
            "crash": "financial_crisis",
            "recession": "financial_crisis",
            "lehman": "financial_crisis",
            "pandemic": "pandemic",
            "covid": "pandemic",
            "corona": "pandemic",
            "budget": "india_budget",
            "union budget": "india_budget",
            "rbi": "rbi_policy",
            "repo rate": "rbi_policy",
            "monetary policy": "rbi_policy",
            "fed": "fed_meeting",
            "fomc": "fed_meeting",
            "federal reserve": "fed_meeting",
            "election": "india_election",
            "vote": "india_election",
            "modi": "india_election",
            "us election": "us_election",
            "trump": "us_election",
            "obama": "us_election",
            "biden": "us_election",
            "terror": "terror_attack",
            "attack": "terror_attack",
            "geopolit": "geopolitical",
            "sanction": "geopolitical",
            "tariff": "trade_war",
            "trade war": "trade_war",
            "demonetis": "policy_reform",
            "gst": "policy_reform",
            "reform": "policy_reform",
            "earthquake": "natural_disaster",
            "flood": "natural_disaster",
            "cyclone": "natural_disaster",
            "disaster": "natural_disaster",
            "corporate": "corporate_crisis",
            "fraud": "corporate_crisis",
            "adani": "corporate_crisis",
            "satyam": "corporate_crisis",
        }

        detected_category = None
        for keyword, cat in keyword_map.items():
            if keyword in msg_lower:
                detected_category = cat
                break

        if not detected_category:
            return (
                "I can analyse market behaviour during specific events. Please mention an event type:\n\n"
                "- **war** / **oil shock** / **financial crisis** / **pandemic**\n"
                "- **budget** / **RBI policy** / **Fed meeting** / **election**\n"
                "- **trade war** / **terror attack** / **natural disaster**\n\n"
                "Example: *\"How does the market perform during wars?\"*"
            )

        df = load_master()
        analysis = analyze_market_during_events(df, category=detected_category, price_col=NIFTY_CLOSE_COL)

        if "error" in analysis:
            return f"⚠️ {analysis['error']}"

        agg = analysis["aggregate"]
        comp = analysis["comparison"]
        cat_desc = CATEGORIES.get(detected_category, detected_category)

        lines = [f"### 📊 Market Analysis During: **{cat_desc}**\n"]
        lines.append(f"*{analysis['query']['events_with_data']} events analysed out of {analysis['query']['events_found']} in catalog*\n")

        # Aggregate stats
        lines.append("**Aggregate Statistics Across All Event Periods:**")
        lines.append(f"- Average period return: **{agg['avg_period_return_pct']:+.2f}%**")
        lines.append(f"- Median period return: **{agg['median_period_return_pct']:+.2f}%**")
        lines.append(f"- % of periods with negative return: **{agg['pct_periods_negative']:.1f}%**")
        lines.append(f"- Best period return: **{agg['best_return_pct']:+.2f}%**")
        lines.append(f"- Worst period return: **{agg['worst_return_pct']:+.2f}%**")
        lines.append(f"- Worst drawdown: **{agg['worst_drawdown_pct']:.2f}%**")
        lines.append(f"- Avg annualised volatility: **{agg['avg_annualised_vol_pct']:.2f}%**")

        # Comparison
        lines.append("\n**During Events vs Normal Periods:**")
        lines.append(f"- Event days ann. return: **{comp['during_events']['annualised_return_pct']:+.2f}%** "
                     f"({comp['during_events']['trading_days']} trading days)")
        lines.append(f"- Normal days ann. return: **{comp['outside_events']['annualised_return_pct']:+.2f}%** "
                     f"({comp['outside_events']['trading_days']} trading days)")
        lines.append(f"- Event days volatility: **{comp['during_events']['annualised_vol_pct']:.2f}%**")
        lines.append(f"- Normal days volatility: **{comp['outside_events']['annualised_vol_pct']:.2f}%**")

        # Individual events table
        lines.append("\n**Per-Event Breakdown:**")
        lines.append("| Event | Return% | MaxDD% | Vol% | Days |")
        lines.append("|-------|---------|--------|------|------|")
        for ep in analysis["event_periods"]:
            lines.append(f"| {ep['name'][:35]} | {ep['return_pct']:+.2f}% | {ep['max_drawdown_pct']:.2f}% | {ep['annualised_vol_pct']:.1f}% | {ep['trading_days']} |")

        return "\n".join(lines)

    except Exception as e:
        logger.error("Event analysis error: %s", e)
        return f"Error analysing events: {e}"


def handle_unknown(**kwargs) -> str:
    return (
        "I'm not sure what you mean. Try asking me to:\n\n"
        "- **List strategies** — see all saved strategies\n"
        "- **Create SMA 10/50** — create a new strategy\n"
        "- **Backtest all** — run backtests\n"
        "- **Show results** — view past performance\n"
        "- **Help** — see all commands\n\n"
        "💡 *Tip: I also support natural language! Try something like:\n"
        "\"Run an SMA crossover with 20 and 100 day windows with 3% stop loss\"*"
    )


def handle_llm_query(message: str, **kwargs) -> dict:
    """
    Forward a complex natural-language query to the LLM agent.

    Returns a dict with: success, response, intermediate_steps
    """
    try:
        from src.llm.agent import run_agent_query
        return run_agent_query(message)
    except ImportError:
        return {
            "success": False,
            "response": (
                "LLM module not available. Install dependencies:\n"
                "`pip install langchain-ibm ibm_watsonx_ai python-dotenv`"
            ),
        }
    except Exception as e:
        logger.error("LLM query failed: %s", e)
        return {
            "success": False,
            "response": f"LLM error: {e}",
        }


# ── Router ────────────────────────────────────────────────────────────────

HANDLERS = {
    "list_strategies": handle_list_strategies,
    "create_sma": handle_create_sma,
    "create_rsi": handle_create_rsi,
    "create_vix": handle_create_vix,
    "create_ema": handle_create_ema,
    "create_macd": handle_create_macd,
    "create_bollinger": handle_create_bollinger,
    "create_atr": handle_create_atr,
    "create_stochastic": handle_create_stochastic,
    "create_zscore": handle_create_zscore,
    "create_macd_hist": handle_create_macd_hist,
    "create_sniper": handle_create_sniper,
    "delete_strategy": handle_delete_strategy,
    "compare_strategies": handle_compare_strategies,
    "show_results": handle_show_results,
    "list_events": handle_list_events,
    "help": handle_help,
    "unknown": handle_unknown,
}


def process_message(message: str, use_llm: bool = True) -> dict:
    """
    Process a copilot chat message and return a response.

    Uses a two-tier approach:
      1. Rule-based pattern matching for well-defined actions
      2. LLM agent for complex/unknown queries (if use_llm=True)

    Returns
    -------
    dict with keys:  intent, params, response, action (optional), llm_used (bool)
    """
    intent, params = detect_intent(message)
    logger.info("Copilot intent: %s, params: %s", intent, params)

    # Special intents that need async backtest execution
    if intent in ("run_backtest", "backtest_all", "predict", "data_info"):
        return {
            "intent": intent,
            "params": params,
            "response": None,  # filled by the API endpoint
            "action": intent,
            "llm_used": False,
        }

    # Event analysis: run the handler inline (needs the original message for keyword extraction)
    if intent == "analyze_events":
        response = handle_analyze_events_rule(message=message)
        return {
            "intent": intent,
            "params": params,
            "response": response,
            "action": None,
            "llm_used": False,
        }

    # If intent is unknown and LLM is enabled, try the LLM agent
    if intent == "unknown" and use_llm:
        try:
            from src.llm.config import is_llm_available
            if is_llm_available():
                logger.info("Rule engine returned 'unknown' — forwarding to LLM agent")
                llm_result = handle_llm_query(message)
                if llm_result["success"]:
                    return {
                        "intent": "llm_agent",
                        "params": params,
                        "response": llm_result["response"],
                        "action": None,
                        "llm_used": True,
                        "intermediate_steps": llm_result.get("intermediate_steps", []),
                    }
                else:
                    # LLM failed — include its error but also the rule-based fallback
                    logger.warning("LLM agent failed: %s", llm_result["response"])
        except Exception as e:
            logger.warning("LLM fallback failed: %s", e)

    handler = HANDLERS.get(intent, handle_unknown)
    response = handler(**params)

    return {
        "intent": intent,
        "params": params,
        "response": response,
        "action": None,
        "llm_used": False,
    }
