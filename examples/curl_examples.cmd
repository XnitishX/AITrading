@REM AITrading HTTP API - curl examples for Windows cmd
@REM Usage: Run these commands one at a time in cmd or PowerShell
@REM Make sure python run_api_server.py is running first!

@echo off
echo.
echo ============================================================
echo AITrading HTTP API - cURL Examples
echo ============================================================
echo.
echo Server URL: http://localhost:8001
echo API Docs: http://localhost:8001/docs
echo.

REM Health check
echo 1. Health Check:
curl http://localhost:8001/health
echo.

REM List tools
echo 2. Available Tools:
curl http://localhost:8001/tools
echo.

REM Gap-up analysis
echo 3. Gap-up Analysis (VIX > 15, Gap > 1 percent):
curl "http://localhost:8001/gap_up_analysis?vix_threshold=15^&gap_percent=1.0"
echo.

REM SMA crossover
echo 4. SMA Crossover Analysis (50/200):
curl "http://localhost:8001/sma_crossover_analysis?sma_short=50^&sma_long=200"
echo.

REM VIX analysis
echo 5. VIX Analysis (Last 30 days):
curl "http://localhost:8001/vix_analysis?vix_threshold=15^&lookback_days=30"
echo.

REM Data summary
echo 6. Data Summary:
curl http://localhost:8001/data_summary
echo.

REM Price pattern
echo 7. Price Pattern - Higher Highs (60-day lookback):
curl "http://localhost:8001/price_pattern?pattern=higher_highs^&lookback_days=60"
echo.

echo ============================================================
echo Done!
echo ============================================================
echo.
