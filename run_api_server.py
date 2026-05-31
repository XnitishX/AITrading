#!/usr/bin/env python
"""
Standalone HTTP API Server for AITrading MCP Tools
Runs independently without Copilot dependency
Usage: python run_api_server.py
Access: http://localhost:8001
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.mcp_server import AITradingAnalyzer

# Initialize analyzer
analyzer = AITradingAnalyzer()

# Create FastAPI app
app = FastAPI(
    title="AITrading MCP Tools API",
    description="Standalone HTTP API for AITrading analysis tools",
    version="1.0"
)

# Add CORS middleware for local cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "data_rows": len(analyzer.df) if analyzer.df is not None else 0}


@app.get("/tools")
def list_tools():
    """List available tools."""
    return {
        "tools": [
            "gap_up_analysis",
            "sma_crossover_analysis",
            "vix_analysis",
            "price_pattern",
            "data_summary",
            "custom_query"
        ]
    }


@app.get("/gap_up_analysis")
def gap_up_analysis(vix_threshold: float = 15, gap_percent: float = 1.0):
    """Find days with gap-up moves when VIX is above threshold."""
    try:
        result = analyzer.gap_up_analysis(vix_threshold, gap_percent)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sma_crossover_analysis")
def sma_crossover_analysis(sma_short: int = 50, sma_long: int = 200):
    """Find SMA crossover points."""
    try:
        result = analyzer.sma_crossover_analysis(sma_short, sma_long)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/vix_analysis")
def vix_analysis(vix_threshold: float = 15, lookback_days: int = 30):
    """Analyze VIX levels and market conditions."""
    try:
        result = analyzer.vix_analysis(vix_threshold, lookback_days)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/price_pattern")
def price_pattern(pattern: str, lookback_days: int = 60):
    """Find price patterns: 'higher_highs', 'lower_lows', 'outside_bar', 'inside_bar'."""
    try:
        result = analyzer.price_pattern(pattern, lookback_days)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/data_summary")
def data_summary():
    """Get summary statistics of the loaded dataset."""
    try:
        result = analyzer.data_summary()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/custom_query")
def custom_query(query: str):
    """Execute custom pandas query using 'df' variable."""
    try:
        result = analyzer.custom_query(query)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def main():
    """Run the API server."""
    print("\n" + "="*70)
    print("🚀 AITrading Standalone HTTP API Server")
    print("="*70)
    print("\n📍 Server running at: http://localhost:8001")
    print("📚 API Docs at: http://localhost:8001/docs")
    print("🔧 Check tools at: http://localhost:8001/tools")
    print("❤️  Health check: http://localhost:8001/health")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )


if __name__ == "__main__":
    main()
