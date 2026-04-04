# MCP Server Setup for AITrading

This document explains how to run AITrading as a local MCP (Model Context Protocol) server and configure it with Copilot Chat.

## What is MCP?

The Model Context Protocol (MCP) allows Copilot Chat to interact directly with your project. Instead of manually explaining your data structure and functions, Copilot can call tools exposed by the MCP server to analyze data, run queries, and answer questions about market patterns.

## Quick Start

### 1. Install MCP Package

```bash
pip install -r requirements.txt
```

### 2. Run the MCP Server

```bash
cd C:\Workspace\AITrading
python -m src.mcp_server
```

You should see:
```
Starting AITrading MCP Server...
Server is running. Press Ctrl+C to stop.
```

### 3. Configure Copilot Chat (VS Code)

1. Open VS Code Settings (Ctrl+,)
2. Search for "copilot" in settings
3. Find **"Copilot: MCP Servers"**
4. Click "Edit in settings.json"
5. Add the AITrading MCP server configuration:

```json
"github.copilot.mcp.servers": {
  "aitrading": {
    "command": "python",
    "args": ["-m", "src.mcp_server"],
    "cwd": "C:\\Workspace\\AITrading"
  }
}
```

### 4. Start Using in Copilot Chat

Now you can ask questions like:

- "Find days with more than 1% gap-up when VIX > 15"
- "Show me all 50-day SMA crossover points above 200-day SMA"
- "Analyze VIX levels in the last 30 days"
- "Find all higher highs patterns in the last 60 days"
- "What dates had outside bar patterns?"

## Available Tools

### gap_up_analysis
Find days with gap-up moves when VIX is above a threshold.

**Parameters:**
- `vix_threshold` (float, default=15): VIX level threshold
- `gap_percent` (float, default=1%): Gap-up percentage threshold

**Example:**
```
"Find gap-ups > 1.5% when VIX > 20"
```

### sma_crossover_analysis
Find SMA crossover points (short SMA crossing long SMA from below - bullish signal).

**Parameters:**
- `sma_short` (int, default=50): Short-term SMA period
- `sma_long` (int, default=200): Long-term SMA period

**Example:**
```
"Show me 20-day SMA crossing 100-day SMA from below"
```

### vix_analysis
Analyze VIX levels and market conditions in a lookback period.

**Parameters:**
- `vix_threshold` (float, default=15): VIX level to analyze around
- `lookback_days` (int, default=30): Days to look back

**Example:**
```
"Analyze VIX levels for the last 60 days, threshold 18"
```

### price_pattern
Find price patterns like higher highs, lower lows, outside bars, inside bars.

**Parameters:**
- `pattern` (string, required): One of: 'higher_highs', 'lower_lows', 'outside_bar', 'inside_bar'
- `lookback_days` (int, default=60): Days to analyze

**Example:**
```
"Find all inside bar patterns in the last 30 days"
```

### data_summary
Get summary statistics of the loaded dataset.

**Example:**
```
"Give me a summary of the current data"
```

### custom_query
Execute custom pandas DataFrame queries for advanced analysis.

**Parameters:**
- `query` (string): Python pandas query using 'df' variable

**Example:**
```
"Show me rows where Close > 19000 and VIX_Close < 15"
```

## Querying Data

When asking questions, you can be specific about:

- **Time ranges**: "in the last 30 days", "since January 2024"
- **Conditions**: "when VIX > 15", "during gap-ups", "on higher highs"
- **Metrics**: Gap percentage, SMA values, VIX levels, price patterns
- **Thresholds**: Customize all parameters to your needs

## Data Available

The MCP server loads data from your master dataframe which includes:

- **Date**: Trading date
- **Open, High, Low, Close**: Nifty 50 price points
- **Volume**: Trading volume
- **VIX_Close**: India VIX close value
- **Return_Pct**: Daily return percentage
- **Prev_Close**: Previous day's close
- **Multiple SMAs**: Various pre-calculated moving averages
- **Technical Indicators**: Any pre-calculated indicators in your data

## Troubleshooting

### Server Won't Start
- Check that your Python environment is activated
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Look for error messages in the terminal

### Copilot Isn't Using the Server
- Restart VS Code after updating settings.json
- Check that the server is running before asking questions
- Verify the path in configuration is correct

### Data Not Updating
- Restart the MCP server after running a data download
- Connect the server after you've synced new data: `python main.py sync`

## Integration with Your Workflow

1. **Download/Sync new data:**
   ```bash
   python main.py sync
   ```

2. **Start the MCP server:**
   ```bash
   python -m src.mcp_server
   ```

3. **Ask Copilot questions** about market patterns, gaps, crossovers, etc.

4. **Use insights** for backtesting or strategy development

## Advanced Usage

You can extend the MCP server by:

1. Adding more tools in `src/mcp_server.py`
2. Registering custom strategies or pattern detection
3. Creating composite queries that combine multiple data sources
4. Building trading signal detectors

Example: Add a new tool to detect "V-bottom" reversal patterns by editing `src/mcp_server.py` and adding a new `@self.server.call_tool` decorated function.

## Next Steps

- Run `python main.py download` to ensure you have latest data
- Start the MCP server
- Go to VS Code and try: "Find dates with more than 1% gap-up when VIX > 15"
- Explore different queries and patterns
