"""
Strategy Sweep – backtest many parameter combos via the REST API,
rank by total return, and persist the top 10 as saved strategies.

Covers all 10 strategies × multiple param variations × risk profiles.
Full-period: 2008-04-01 → latest available data.
"""

import requests, json, time

BASE = "http://localhost:8000"
START_DATE = "2008-04-01"  # VIX data starts ~Mar 2008

# ── Strategy Grid ────────────────────────────────────────────────────────
STRATEGIES = [
    # ── SMA Crossover ────────────────────────────────────────────────
    {"name": "SMA(10/50)",    "type": "sma_crossover", "params": {"fast_window": 10,  "slow_window": 50}},
    {"name": "SMA(20/50)",    "type": "sma_crossover", "params": {"fast_window": 20,  "slow_window": 50}},
    {"name": "SMA(20/100)",   "type": "sma_crossover", "params": {"fast_window": 20,  "slow_window": 100}},
    {"name": "SMA(50/200)",   "type": "sma_crossover", "params": {"fast_window": 50,  "slow_window": 200}},
    {"name": "SMA(50/100)",   "type": "sma_crossover", "params": {"fast_window": 50,  "slow_window": 100}},
    {"name": "SMA(100/200)",  "type": "sma_crossover", "params": {"fast_window": 100, "slow_window": 200}},

    # ── EMA Crossover (NEW) ──────────────────────────────────────────
    {"name": "EMA(5/13)",     "type": "ema_crossover", "params": {"fast_span": 5,  "slow_span": 13}},
    {"name": "EMA(8/21)",     "type": "ema_crossover", "params": {"fast_span": 8,  "slow_span": 21}},
    {"name": "EMA(9/21)",     "type": "ema_crossover", "params": {"fast_span": 9,  "slow_span": 21}},
    {"name": "EMA(12/26)",    "type": "ema_crossover", "params": {"fast_span": 12, "slow_span": 26}},
    {"name": "EMA(20/50)",    "type": "ema_crossover", "params": {"fast_span": 20, "slow_span": 50}},
    {"name": "EMA(50/200)",   "type": "ema_crossover", "params": {"fast_span": 50, "slow_span": 200}},

    # ── RSI Mean Reversion ───────────────────────────────────────────
    {"name": "RSI(20/80)",    "type": "rsi_mean_reversion", "params": {"oversold": 20, "overbought": 80}},
    {"name": "RSI(25/75)",    "type": "rsi_mean_reversion", "params": {"oversold": 25, "overbought": 75}},
    {"name": "RSI(30/70)",    "type": "rsi_mean_reversion", "params": {"oversold": 30, "overbought": 70}},
    {"name": "RSI(35/65)",    "type": "rsi_mean_reversion", "params": {"oversold": 35, "overbought": 65}},
    {"name": "RSI(40/60)",    "type": "rsi_mean_reversion", "params": {"oversold": 40, "overbought": 60}},

    # ── MACD Crossover ───────────────────────────────────────────────
    {"name": "MACD(5/35/5)",  "type": "macd_crossover", "params": {"fast_ema": 5,  "slow_ema": 35, "signal_period": 5}},
    {"name": "MACD(8/21/5)",  "type": "macd_crossover", "params": {"fast_ema": 8,  "slow_ema": 21, "signal_period": 5}},
    {"name": "MACD(12/26/9)", "type": "macd_crossover", "params": {"fast_ema": 12, "slow_ema": 26, "signal_period": 9}},
    {"name": "MACD(15/35/9)", "type": "macd_crossover", "params": {"fast_ema": 15, "slow_ema": 35, "signal_period": 9}},

    # ── MACD Histogram (NEW) ────────────────────────────────────────
    {"name": "MACDHist(5/13/5)",  "type": "macd_histogram", "params": {"fast_ema": 5,  "slow_ema": 13, "signal_period": 5}},
    {"name": "MACDHist(8/21/5)",  "type": "macd_histogram", "params": {"fast_ema": 8,  "slow_ema": 21, "signal_period": 5}},
    {"name": "MACDHist(12/26/9)", "type": "macd_histogram", "params": {"fast_ema": 12, "slow_ema": 26, "signal_period": 9}},
    {"name": "MACDHist(15/35/9)", "type": "macd_histogram", "params": {"fast_ema": 15, "slow_ema": 35, "signal_period": 9}},

    # ── Bollinger Bands ──────────────────────────────────────────────
    {"name": "BB(10,1.5s)",   "type": "bollinger_band", "params": {"window": 10, "num_std": 1.5}},
    {"name": "BB(20,1.5s)",   "type": "bollinger_band", "params": {"window": 20, "num_std": 1.5}},
    {"name": "BB(20,2.0s)",   "type": "bollinger_band", "params": {"window": 20, "num_std": 2.0}},
    {"name": "BB(20,2.5s)",   "type": "bollinger_band", "params": {"window": 20, "num_std": 2.5}},
    {"name": "BB(30,2.0s)",   "type": "bollinger_band", "params": {"window": 30, "num_std": 2.0}},
    {"name": "BB(50,2.0s)",   "type": "bollinger_band", "params": {"window": 50, "num_std": 2.0}},

    # ── ATR Breakout ─────────────────────────────────────────────────
    {"name": "ATR(20,14,1.0)", "type": "atr_breakout", "params": {"sma_window": 20, "atr_period": 14, "atr_multiplier": 1.0}},
    {"name": "ATR(20,14,1.5)", "type": "atr_breakout", "params": {"sma_window": 20, "atr_period": 14, "atr_multiplier": 1.5}},
    {"name": "ATR(20,14,2.0)", "type": "atr_breakout", "params": {"sma_window": 20, "atr_period": 14, "atr_multiplier": 2.0}},
    {"name": "ATR(50,14,1.5)", "type": "atr_breakout", "params": {"sma_window": 50, "atr_period": 14, "atr_multiplier": 1.5}},
    {"name": "ATR(50,14,2.0)", "type": "atr_breakout", "params": {"sma_window": 50, "atr_period": 14, "atr_multiplier": 2.0}},
    {"name": "ATR(20,21,1.5)", "type": "atr_breakout", "params": {"sma_window": 20, "atr_period": 21, "atr_multiplier": 1.5}},

    # ── VIX Regime ───────────────────────────────────────────────────
    {"name": "VIX(12/25)",    "type": "vix_regime", "params": {"buy_below": 12, "sell_above": 25}},
    {"name": "VIX(15/25)",    "type": "vix_regime", "params": {"buy_below": 15, "sell_above": 25}},
    {"name": "VIX(15/30)",    "type": "vix_regime", "params": {"buy_below": 15, "sell_above": 30}},
    {"name": "VIX(18/30)",    "type": "vix_regime", "params": {"buy_below": 18, "sell_above": 30}},
    {"name": "VIX(20/35)",    "type": "vix_regime", "params": {"buy_below": 20, "sell_above": 35}},

    # ── Stochastic Oscillator (NEW) ──────────────────────────────────
    {"name": "Stoch(5,3,20/80)",   "type": "stochastic_oscillator", "params": {"k_period": 5,  "d_period": 3, "oversold": 20, "overbought": 80}},
    {"name": "Stoch(14,3,20/80)",  "type": "stochastic_oscillator", "params": {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}},
    {"name": "Stoch(14,3,30/70)",  "type": "stochastic_oscillator", "params": {"k_period": 14, "d_period": 3, "oversold": 30, "overbought": 70}},
    {"name": "Stoch(21,5,20/80)",  "type": "stochastic_oscillator", "params": {"k_period": 21, "d_period": 5, "oversold": 20, "overbought": 80}},

    # ── Mean Reversion Z-Score (NEW) ─────────────────────────────────
    {"name": "ZScore(10,-2.0)",  "type": "mean_reversion_zscore", "params": {"lookback": 10, "entry_z": -2.0, "exit_z": 0.0}},
    {"name": "ZScore(20,-1.5)",  "type": "mean_reversion_zscore", "params": {"lookback": 20, "entry_z": -1.5, "exit_z": 0.0}},
    {"name": "ZScore(20,-2.0)",  "type": "mean_reversion_zscore", "params": {"lookback": 20, "entry_z": -2.0, "exit_z": 0.0}},
    {"name": "ZScore(30,-2.5)",  "type": "mean_reversion_zscore", "params": {"lookback": 30, "entry_z": -2.5, "exit_z": 0.0}},
    {"name": "ZScore(50,-2.0)",  "type": "mean_reversion_zscore", "params": {"lookback": 50, "entry_z": -2.0, "exit_z": 0.0}},

    # ── Composite Sniper (Multi-Factor) ──────────────────────────────
    {"name": "Sniper(50,35/65,20)",  "type": "composite_sniper", "params": {"sma_period": 50, "rsi_oversold": 35, "rsi_overbought": 65, "vix_calm_threshold": 20, "bb_window": 20, "bb_num_std": 2.0}},
    {"name": "Sniper(50,30/70,20)",  "type": "composite_sniper", "params": {"sma_period": 50, "rsi_oversold": 30, "rsi_overbought": 70, "vix_calm_threshold": 20, "bb_window": 20, "bb_num_std": 2.0}},
    {"name": "Sniper(100,35/65,22)", "type": "composite_sniper", "params": {"sma_period": 100, "rsi_oversold": 35, "rsi_overbought": 65, "vix_calm_threshold": 22, "bb_window": 20, "bb_num_std": 2.0}},
    {"name": "Sniper(50,35/65,25)",  "type": "composite_sniper", "params": {"sma_period": 50, "rsi_oversold": 35, "rsi_overbought": 65, "vix_calm_threshold": 25, "bb_window": 20, "bb_num_std": 1.5}},
]

# ── Risk management settings to sweep ───────────────────────────────────
RISK_PROFILES = [
    {"label": "tight",   "stop_loss_pct": 2,  "take_profit_pct": 4,  "trailing_stop_pct": 1.5, "cooldown_bars": 3,  "max_holding_bars": 60},
    {"label": "medium",  "stop_loss_pct": 3,  "take_profit_pct": 6,  "trailing_stop_pct": 2.5, "cooldown_bars": 2,  "max_holding_bars": 0},
    {"label": "wide",    "stop_loss_pct": 5,  "take_profit_pct": 10, "trailing_stop_pct": 4.0, "cooldown_bars": 0,  "max_holding_bars": 0},
    {"label": "trail",   "stop_loss_pct": 5,  "take_profit_pct": 0,  "trailing_stop_pct": 3.0, "cooldown_bars": 5,  "max_holding_bars": 120},  # trailing stop only, no TP cap + cooldown
]

# Annualised volatility target for position sizing (0 = disabled)
VOL_TARGET_PCT = 15  # 15% annualised vol target (Kelly-inspired)

def run_backtest(strat, risk):
    """POST a backtest request and return the result dict (or None on failure)."""
    payload = {
        "strategy_type": strat["type"],
        "params": strat["params"],
        "initial_capital": 1_000_000,
        "stop_loss_pct": risk["stop_loss_pct"],
        "take_profit_pct": risk["take_profit_pct"],
        "trailing_stop_pct": risk.get("trailing_stop_pct", 0),
        "cooldown_bars": risk.get("cooldown_bars", 0),
        "max_holding_bars": risk.get("max_holding_bars", 0),
        "vol_target": VOL_TARGET_PCT,
        "start_date": START_DATE,
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE}/api/backtest", json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                data["_strat_name"] = strat["name"]
                data["_risk_label"] = risk["label"]
                data["_type"] = strat["type"]
                data["_params"] = strat["params"]
                return data
            else:
                print(f"  FAIL {r.status_code}: {strat['name']} / {risk['label']}")
                return None
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            print(f"  ERROR: {strat['name']} / {risk['label']} => {e}")
            return None


def main():
    total = len(STRATEGIES) * len(RISK_PROFILES)
    print(f"=== Strategy Sweep: {len(STRATEGIES)} strategies x {len(RISK_PROFILES)} risk profiles = {total} backtests ===")
    print(f"=== Period: {START_DATE} -> latest available ===\n")

    results = []
    for i, strat in enumerate(STRATEGIES):
        for risk in RISK_PROFILES:
            tag = f"[{len(results)+1}/{total}]"
            print(f"{tag} {strat['name']:22s} / {risk['label']:6s} ... ", end="", flush=True)
            res = run_backtest(strat, risk)
            if res:
                ret = res.get("total_return_pct", 0) or 0
                dd  = res.get("max_drawdown_pct", 0) or 0
                sr  = res.get("sharpe_ratio", 0) or 0
                wr  = res.get("win_rate_pct", 0) or 0
                trades = res.get("total_trades", 0) or 0
                print(f"Return: {ret:+8.2f}%  DD: {dd:6.2f}%  Sharpe: {sr:5.2f}  WinRate: {wr:5.1f}%  Trades: {trades}")
                results.append(res)
            else:
                print("SKIPPED")

    # ── Rank by Sharpe ratio (risk-adjusted) then total return ────────────
    results.sort(key=lambda x: (x.get("sharpe_ratio") or 0, x.get("total_return_pct") or 0), reverse=True)

    print(f"\n{'='*150}")
    print(f"{'RANK':<5} {'STRATEGY':<24} {'RISK':<8} {'RETURN%':>9} {'ANNUAL%':>9} {'DD%':>8} {'DDdur':>6} {'SHARPE':>8} {'SORTINO':>8} {'R/R':>6} {'ALPHA%':>8} {'WIN%':>7} {'TRADES':>7} {'PF':>7}")
    print(f"{'='*150}")
    for rank, r in enumerate(results[:30], 1):
        print(f"{rank:<5} {r['_strat_name']:<24} {r['_risk_label']:<8} "
              f"{(r.get('total_return_pct') or 0):>+8.2f}% "
              f"{(r.get('annual_return_pct') or 0):>+8.2f}% "
              f"{(r.get('max_drawdown_pct') or 0):>7.2f}% "
              f"{(r.get('max_drawdown_duration') or 0):>5} "
              f"{(r.get('sharpe_ratio') or 0):>7.2f} "
              f"{(r.get('sortino_ratio') or 0):>7.2f} "
              f"{(r.get('risk_reward_ratio') or 0):>5.2f} "
              f"{(r.get('alpha_pct') or 0):>+7.2f}% "
              f"{(r.get('win_rate_pct') or 0):>6.1f}% "
              f"{(r.get('total_trades') or 0):>6} "
              f"{(r.get('profit_factor') or 0):>6.2f}")

    # ── Save Top 10 as strategies in the database ────────────────────────
    print(f"\n--- Saving Top 10 strategies to database ---")
    saved = 0
    seen_names = set()
    for r in results:
        if saved >= 10:
            break
        full_name = f"TOP{saved+1}: {r['_strat_name']} ({r['_risk_label']})"
        if full_name in seen_names:
            continue
        seen_names.add(full_name)

        ret = r.get('total_return_pct') or 0
        annual = r.get('annual_return_pct') or 0
        dd  = r.get('max_drawdown_pct') or 0
        sr  = r.get('sharpe_ratio') or 0
        pf  = r.get('profit_factor') or 0
        wr  = r.get('win_rate_pct') or 0

        desc = (
            f"Return: {ret:+.2f}%, Annual: {annual:+.2f}%, MaxDD: {dd:.2f}%, "
            f"Sharpe: {sr:.2f}, WinRate: {wr:.1f}%, PF: {pf:.2f}, "
            f"Risk: {r['_risk_label']}"
        )

        # Merge risk params into strategy params
        merged_params = dict(r["_params"])
        risk_match = [p for p in RISK_PROFILES if p["label"] == r["_risk_label"]][0]
        merged_params["stop_loss_pct"] = risk_match["stop_loss_pct"]
        merged_params["take_profit_pct"] = risk_match["take_profit_pct"]
        merged_params["trailing_stop_pct"] = risk_match.get("trailing_stop_pct", 0)

        payload = {
            "name": full_name,
            "type": r["_type"],
            "params": merged_params,
            "description": desc,
        }
        resp = requests.post(f"{BASE}/api/strategies", json=payload, timeout=10)
        if resp.status_code == 200:
            saved += 1
            print(f"  ✓ Saved #{saved}: {full_name} — {desc}")
        else:
            print(f"  ✗ Failed to save {full_name}: {resp.text}")

    print(f"\n=== Done! {saved} top strategies saved. ===")


if __name__ == "__main__":
    main()
