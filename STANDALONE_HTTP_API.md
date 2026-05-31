# Standalone HTTP API Setup Complete

## What You Now Have

You have **TWO independent ways** to use AITrading tools:

### Option 1: MCP Server (for Copilot Chat)
**Use when:** Working in VS Code with Copilot Chat
```bash
python run_mcp_server.py
```
Then type tool prompts in Copilot Chat (e.g., `gap_up_analysis()`)

### Option 2: HTTP API Server (Standalone)
**Use when:** You want independent local deployment without Copilot
```bash
python run_api_server.py
```
Server runs on `http://localhost:8001`

---

## Quick Start: HTTP API Server

### 1. Start the API Server
```bash
python run_api_server.py
```

### 2. Make Requests

**PowerShell:**
```powershell
$response = Invoke-RestMethod -Uri "http://localhost:8001/gap_up_analysis?vix_threshold=15&gap_percent=1.0" -Method Get
Write-Host $response.result
```

**Python:**
```python
import requests
response = requests.get("http://localhost:8001/gap_up_analysis", 
                       params={"vix_threshold": 15, "gap_percent": 1.0})
print(response.json()["result"])
```

**cURL / cmd:**
```cmd
curl "http://localhost:8001/gap_up_analysis?vix_threshold=15&gap_percent=1.0"
```

---

## Available Tools

| Tool | Endpoint | Purpose |
|------|----------|---------|
| `gap_up_analysis` | `/gap_up_analysis` | Find gap-ups when VIX is high |
| `sma_crossover_analysis` | `/sma_crossover_analysis` | Detect SMA crossover signals |
| `vix_analysis` | `/vix_analysis` | Analyze VIX market conditions |
| `price_pattern` | `/price_pattern` | Identify price patterns (higher_highs, lower_lows, etc.) |
| `data_summary` | `/data_summary` | Get market statistics |
| `custom_query` | `/custom_query` | Execute custom pandas queries |

---

## Example Scripts

Run these to test the API:

### Python
```bash
python examples/api_client.py
```

### PowerShell
```powershell
.\examples\api_examples.ps1
```

### cmd
```cmd
examples\curl_examples.cmd
```

---

## API Endpoints Reference

| Endpoint | Parameters | Example |
|----------|------------|---------|
| `/health` | None | `http://localhost:8001/health` |
| `/tools` | None | `http://localhost:8001/tools` |
| `/gap_up_analysis` | `vix_threshold=15, gap_percent=1.0` | `?vix_threshold=15&gap_percent=1.0` |
| `/sma_crossover_analysis` | `sma_short=50, sma_long=200` | `?sma_short=50&sma_long=200` |
| `/vix_analysis` | `vix_threshold=15, lookback_days=30` | `?vix_threshold=15&lookback_days=30` |
| `/price_pattern` | `pattern, lookback_days=60` | `?pattern=higher_highs&lookback_days=60` |
| `/data_summary` | None | No parameters |
| `/custom_query` | `query` | `?query=df[df['Close']>25000]` |

---

## Interactive API Documentation

When the API server is running, visit:
- **API Docs:** http://localhost:8001/docs (Swagger UI)
- **ReDoc:** http://localhost:8001/redoc

---

## Summary

You now have:

✓ **MCP Server** for Copilot Chat integration  
✓ **HTTP API Server** for standalone deployment  
✓ **Example scripts** for Python, PowerShell, and cmd  
✓ **Full documentation** in `.github/copilot-instructions.md`  

**No Copilot dependency required!** Run the HTTP API anytime, anywhere, from any language or tool that supports HTTP requests.
