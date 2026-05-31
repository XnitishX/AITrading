---
name: AITrading Workspace Instructions
description: >
  Workspace instructions for AITrading – Nifty 50 trading simulator with 
  probability prediction and LLM agent integration. 
  Use when: working on any AITrading features (data pipeline, backtesting, 
  web UI, MCP server, LLM agent, visualizations, tests).
---

# AITrading Workspace Instructions

## Project Overview

**AITrading** is a Python-based trading simulation and probability prediction system for the Nifty 50 index and India VIX volatility index. The system combines data ingestion, backtesting, natural language interfaces (LLM agent + web UI), and predictive modeling with Copilot integration via MCP.

### Key Components

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **data/** | Download, load, validate, and engineer features from Yahoo Finance | `loader.py`, `yfinance_downloader.py` |
| **llm/** | LangChain agent interface with IBM WatsonX AI for natural language queries | `agent.py`, `config.py` |
| **simulator/** | Backtesting engine with strategy registry and performance metrics | `backtester.py`, `registry.py`, `visualisation.py` |
| **storage/** | SQLite persistence layer (strategies, results, sessions) | `database.py` |
| **web/** | FastAPI application + Jinja2 templates + Copilot integration | `api.py`, `copilot.py` |
| **mcp_server.py** | MCP service exposing analysis tools to external clients | Top-level in `src/` |

---

## Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Essential Commands

| Goal | Command | Notes |
|------|---------|-------|
| Download fresh data | `python main.py download` | Fetches Nifty50 + VIX from Yahoo Finance (CSV) |
| Run backtests | `python main.py backtest` | Executes all registered strategies, saves results |
| Generate predictions | `python main.py predict` | Computes up/down probabilities for 1d/1w/1m |
| Sync incremental data | `python main.py sync` | Appends new data to existing CSV files |
| Visualize results | `python main.py visualise` | Generates equity curves + drawdown plots |
| Run all (end-to-end) | `python main.py all` | download → backtest → predict → visualize |
| **Web UI** | `python run_web.py` | Starts FastAPI at `http://localhost:8000` |
| **MCP Server** | `python run_mcp_server.py` | Exposes tools to Copilot Chat; run in separate terminal |
| **HTTP API Server** | `python run_api_server.py` | Standalone HTTP API (no Copilot needed); runs on `http://localhost:8001` |
| **Unit tests** | `pytest tests/ -v` | Core feature/data validation tests |

---

## Architecture & Design Patterns

### Module Organization

The codebase follows a **feature-based organization**:
- **src/data/** — Pure data loading and feature engineering (no side effects)
- **src/simulator/** — Strategy execution and backtesting logic
- **src/llm/** — LLM integration + tool definitions for agent workflows
- **src/storage/** — Database I/O layer (queries isolate from schema)
- **src/web/** — FastAPI routes + Copilot integration
- **config/** — Centralized settings validated via Pydantic at import

### Key Design Decisions

1. **Vectorized Backtester**: All computations use NumPy/Pandas for performance (not event-by-event loops).
2. **Strategy Registry**: Strategies registered in `registry.py`; new strategies added via `@register_strategy` decorator.
3. **Lazy LLM Imports**: LangChain imports deferred to `llm/agent.py` to avoid slow startup unless needed.
4. **Dataframe Pipeline**: Features chained via column additions (`add_returns()` → `add_sma()` → `add_rsi()`).
5. **Event-Driven Trade Tracking**: Backtester tracks MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion) per trade.
6. **Configuration as Code**: All parameters (SMA windows, RSI periods, position sizing) live in `config/settings.py`.

### Data Flow

```
[Yahoo Finance]
       ↓
[yfinance_downloader.py] → data/raw/*.csv
       ↓
[loader.py] → Merge + Feature Engineering
       ↓
[backtester.py] → Apply Signals + Track Trades
       ↓
[storage/database.py] → SQLite Results
       ↓
[visualisation.py] + [web/api.py] → Charts & APIs
```

---

## Development Workflow

### Adding a New Strategy

1. **Define strategy function** in `src/simulator/backtester.py` or a new module.
2. **Register it** using the `@register_strategy` decorator.
3. **Add parameters** to `config/settings.py` if strategy-specific tuning is needed.
4. **Test with**: `python main.py backtest`
5. **Store results** in SQLite via `storage/database.py`.

### Modifying Features or Adding Indicators

1. **Edit** `src/data/loader.py` (all feature engineering happens here).
2. **Update tests** in `tests/test_core.py` (test your new feature).
3. **Run tests**: `pytest tests/ -v`
4. **Run backtest** to validate impact: `python main.py backtest`

### Adding Web Endpoints

1. **Define route** in `src/web/api.py` (FastAPI).
2. **Reuse existing tools** from `src/llm/agent.py` or `src/simulator/backtester.py`.
3. **Test locally**: `python run_web.py` then browse to `http://localhost:8000`.

### Running the LLM Agent

1. **Configure credentials** in `.env` (required: `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`).
2. **Start MCP server**: `python run_mcp_server.py` (exposes tools to Copilot).
3. **Query via web UI** or Copilot Chat (tools registered in MCP).

---

## Configuration & Settings

All parameters consolidated in [config/settings.py](../config/settings.py):

### Path Configuration
- `RAW_DATA_DIR` = `data/raw/` (CSV input)
- `PROCESSED_DATA_DIR` = `data/processed/` (Parquet output)
- `OUTPUT_DIR` = `output/` (Results, charts)
- `LOG_DIR` = `logs/`

### Strategy Parameters
- **Initial Capital**: `INITIAL_CAPITAL = 100000`
- **Position Sizing**: `POSITION_SIZE_PERCENT`
- **Stop-Loss / Take-Profit**: `STOPLOSS_PERCENT`, `TAKEPROFIT_PERCENT`

### Technical Indicators
- **SMA Windows**: `SMA_PERIODS = [5, 10, 21, 63, 126, 252]`
- **RSI Period**: `RSI_PERIOD = 14`
- **MACD**: `MACD_FAST = 12, MACD_SLOW = 26, MACD_SIGNAL = 9`

### Prediction Horizons
- **1-day** (`PREDICTION_HORIZON_1D = 1`)
- **1-week** (`PREDICTION_HORIZON_1W = 5`)
- **1-month** (`PREDICTION_HORIZON_1M = 21`)

### Logging
- **Log Level**: `LOG_LEVEL = 'INFO'` (set to `'DEBUG'` for verbose output)
- **Format**: `LOG_FORMAT` includes timestamp, module, level

---

## Debugging & Troubleshooting

### Data Issues

**Problem**: "No data found" or CSV files missing.  
**Solution**: Run `python main.py download` to fetch from Yahoo Finance.

**Problem**: Feature columns missing (e.g., no `RSI` in backtest).  
**Solution**: Check that `loader.build_master_dataframe()` is called before backtesting. Verify `src/data/loader.py` adds the required columns.

### Backtest Issues

**Problem**: Strategy produces no trades.  
**Solution**: Check strategy logic in `backtester.py`. Verify signals are generated correctly (add print statements or debug with `pytest`).

**Problem**: Results not saved to database.  
**Solution**: Ensure `storage/database.py` is called after backtest. Check SQLite connection string in `config/settings.py`.

### LLM/MCP Issues

**Problem**: LangChain imports fail.  
**Solution**: Run `pip install langchain langchain-ibm` (check `requirements.txt`).

**Problem**: MCP tools not available in Copilot.  
**Solution**: Restart the MCP server (`python run_mcp_server.py`). Verify IBM WatsonX credentials in `.env`.

### Performance Issues

**Problem**: Backtest is slow.  
**Solution**: Vectorization is already optimized; check symbol count. For large datasets, filter date ranges in `config/settings.py`.

---

## Testing

### Test Framework & Location

- **Framework**: pytest
- **Test File**: [tests/test_core.py](../tests/test_core.py)

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_core.py::test_add_returns -v

# With coverage
pytest tests/ --cov=src
```

### Key Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Config** | `test_config_imports` | Verify Pydantic validation |
| **Features** | `test_add_returns`, `test_add_rolling_features`, `test_add_rsi`, `test_add_macd` | Feature engineering correctness |
| **Data** | `test_load_master_dataframe`, `test_data_quality` | Data loading and validation |

### Writing New Tests

1. **Add test function** to `tests/test_core.py` starting with `test_`.
2. **Use fixtures** for common setup (data loading, config).
3. **Assert outcomes** (column existence, value ranges, no NaN where unexpected).
4. **Run locally** before pushing: `pytest tests/test_core.py::your_test -v`.

---

## Common Tasks for AI Agents

### Task: Implement a New Backtest Strategy

1. **Open** [src/simulator/backtester.py](../src/simulator/backtester.py)
2. **Define function** with signature: `def strategy_name(df: pd.DataFrame, **params) -> pd.Series:`
3. **Return Series** with `1` (buy), `-1` (sell), `0` (hold).
4. **Decorate** with `@register_strategy(name="Strategy Name")`
5. **Test**: `python main.py backtest` should include your strategy.

### Task: Add a Feature or Indicator

1. **Open** [src/data/loader.py](../src/data/loader.py)
2. **Add function**: `def add_your_indicator(df: pd.DataFrame) -> pd.DataFrame:`
3. **Call in `build_master_dataframe()`** to apply during load.
4. **Test**: `pytest tests/test_core.py -v`
5. **Validate** with backtest: `python main.py backtest`

### Task: Create a New Web Endpoint

1. **Open** [src/web/api.py](../src/web/api.py)
2. **Add route** using `@app.get()` or `@app.post()`.
3. **Reuse tools** from `src/llm/agent.py` or `src/simulator/backtester.py`.
4. **Test locally**: `python run_web.py` then visit `http://localhost:8000/docs` (auto-generated API docs).

### Task: Fix Test Failures

1. **Identify failing test**: `pytest tests/ -v` shows test name and error.
2. **Read error stack trace** to locate issue.
3. **Check data assumptions** (e.g., expected column names, NaN handling).
4. **Update test or fix code** as appropriate.
5. **Re-run**: `pytest tests/ -v`

---

## Code Conventions

### Naming Conventions

- **Modules/Files**: `snake_case.py` (e.g., `yfinance_downloader.py`)
- **Functions**: `snake_case()` (e.g., `add_sma_indicators()`)
- **Classes**: `PascalCase` (e.g., `AITradingAnalyzer`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `INITIAL_CAPITAL`)

### Import Organization

1. **Standard library** (e.g., `os`, `sys`, `pathlib`)
2. **Third-party** (e.g., `numpy`, `pandas`, `fastapi`)
3. **Local modules** (e.g., `from src.data import loader`)

Separate each group with a blank line.

### Docstring Format

Use **Google-style docstrings** for functions and classes:

```python
def backtest_strategy(df: pd.DataFrame, strategy_name: str) -> dict:
    """
    Execute a backtesting run for a given strategy.
    
    Args:
        df: Master dataframe with OHLCV and indicators.
        strategy_name: Name of registered strategy.
    
    Returns:
        Dictionary with keys: metrics, trades, equity_curve.
    
    Raises:
        ValueError: If strategy_name not registered.
    """
```

### Type Hints

Use **full type hints** for function signatures:

```python
def add_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add SMA to dataframe."""
```

---

## Important Files & Where to Make Changes

| Task | File |
|------|------|
| Add/modify data features | [src/data/loader.py](../src/data/loader.py) |
| Add/modify strategy logic | [src/simulator/backtester.py](../src/simulator/backtester.py) |
| Adjust parameters (SMA, RSI, position size, etc.) | [config/settings.py](../config/settings.py) |
| Add LLM tools or agent logic | [src/llm/agent.py](../src/llm/agent.py) |
| Add web endpoints or UI logic | [src/web/api.py](../src/web/api.py), `static/index.html` |
| Query or persist results | [src/storage/database.py](../src/storage/database.py) |
| Fix backtester bugs | [src/simulator/backtester.py](../src/simulator/backtester.py) |
| Write/fix tests | [tests/test_core.py](../tests/test_core.py) |

---

## Frequently Asked Questions

**Q: How do I add a new symbol (e.g., BankNifty instead of Nifty50)?**  
A: Edit `config/settings.py` to change the Yahoo Finance ticker, then run `python main.py download`.

**Q: Can I create custom strategies without editing core files?**  
A: Yes, define a function matching the signature in `backtester.py` and decorate with `@register_strategy()`.

**Q: How do I integrate with Copilot Chat?**  
A: Run `python run_mcp_server.py`. Tools are auto-registered via MCP. Query from Copilot Chat or web UI.

**Q: What's the difference between `.sync()` and `.download()`?**  
A: `.download()` fetches full history; `.sync()` appends only new data. Use `.sync()` for daily updates.

**Q: How is test coverage measured?**  
A: Run `pytest tests/ --cov=src` to see coverage by module.

---

## Links & References

- **Entry Point**: [main.py](../main.py) — CLI commands
- **Architecture**: [MCP_SERVER_SETUP.md](../MCP_SERVER_SETUP.md), [GIT_SETUP_INSTRUCTIONS.md](../GIT_SETUP_INSTRUCTIONS.md)
- **Data**: [data/raw/](../data/raw/) (CSV storage), [data/processed/](../data/processed/) (Parquet)
- **Results**: [output/](../output/) (charts, metrics), `data/aitrading.db` (SQLite)
- **Logs**: [logs/](../logs/)

---

## Next Steps for AI Agents

When augmenting or extending AITrading:

1. **Understand the current state** by reading [main.py](../main.py) and [src/simulat/backtester.py](../src/simulator/backtester.py).
2. **Respect naming conventions** and design patterns (see "Architecture & Design Patterns").
3. **Update tests** whenever you modify data loading or feature engineering; run `pytest tests/ -v`.
4. **Use configuration files** for parameters, not hard-coded values.
5. **Document new endpoints** and tools (add docstrings to functions and classes).
6. **Test locally** before committing (run backtest, check logs, validate outputs).

---

**Last Updated**: April 2026  
**Maintained By**: AITrading Development Team

---

## Using MCP Tools in Copilot Chat

To use AITrading's MCP tools from Copilot Chat (or any LLM agent):

1. **Start the MCP server**
    ```bash
    python run_mcp_server.py
    ```
    Leave this terminal running.

2. **Open Copilot Chat in VS Code**
    - Make sure you have the latest Copilot Chat extension.
    - MCP integration is automatic if the server is running on localhost.

3. **Type a tool prompt in chat**
    - Examples:
      - `gap_up_analysis()`
      - `gap_up_analysis(vix_threshold=15, gap_percent=1.0)`
      - `sma_crossover_analysis(sma_short=50, sma_long=200)`
      - `data_summary()`
      - Or ask: "How many days did Nifty gap up?"
    - Copilot will route the request to the MCP server and show a tool result.

4. **Add new tools**
    - Define a function in `src/mcp_server.py` with `@mcp.tool()`.
    - Restart the MCP server to register new tools.

If you do not see tool results, check:
 - MCP server is running and not blocked by firewall
 - Copilot Chat is up to date
 - No port conflicts (default is localhost)

**Tip:** You can use both natural language and direct tool calls. The agent will pick the best tool automatically if your question matches a tool's description.
---

## Standalone HTTP API (No Copilot Dependency)

If you want to use the AITrading tools **independently without Copilot Chat**, use the HTTP API server:

1. **Start the HTTP API server**
   ```bash
   python run_api_server.py
   ```
   Server runs on `http://localhost:8001` with interactive API docs at `/docs`

2. **Call tools via HTTP**
   
   **PowerShell Example:**
   ```powershell
   $response = Invoke-RestMethod -Uri "http://localhost:8001/gap_up_analysis?vix_threshold=15&gap_percent=1.0" -Method Get
   Write-Host $response.result
   ```
   
   **cURL Example:**
   ```bash
   curl "http://localhost:8001/gap_up_analysis?vix_threshold=15&gap_percent=1.0"
   ```
   
   **Python Example:**
   ```python
   import requests
   response = requests.get("http://localhost:8001/gap_up_analysis", 
                          params={"vix_threshold": 15, "gap_percent": 1.0})
   print(response.json()["result"])
   ```

3. **Available endpoints**
   | Endpoint | Purpose | Example |
   |----------|---------|---------|
   | `/health` | Server health check | `http://localhost:8001/health` |
   | `/tools` | List available tools | `http://localhost:8001/tools` |
   | `/gap_up_analysis` | Gap-up detection | `?vix_threshold=15&gap_percent=1.0` |
   | `/sma_crossover_analysis` | SMA crossovers | `?sma_short=50&sma_long=200` |
   | `/vix_analysis` | VIX analysis | `?vix_threshold=15&lookback_days=30` |
   | `/price_pattern` | Price patterns | `?pattern=higher_highs&lookback_days=60` |
   | `/data_summary` | Market summary | No parameters |
   | `/custom_query` | Custom pandas query | `?query=df[df['Close']>25000]` |

4. **Example scripts**
   - **Python client**: `examples/api_client.py` — Run all examples
   - **PowerShell**: `examples/api_examples.ps1` — Windows PowerShell queries
   - **cmd batch**: `examples/curl_examples.cmd` — Windows cmd with curl

**Use HTTP API when:**
- You want standalone local deployment
- Copilot Chat is not available
- You need programmatic access from other tools/languages
- You want an API-first approach

**Use MCP when:**
- You primarily work in VS Code with Copilot Chat
- You want natural language integration
- You prefer agent-based workflows