# AITrading HTTP API - PowerShell Examples
# Usage: Run these commands one at a time in PowerShell
# Make sure python run_api_server.py is running first!

$BASE_URL = "http://localhost:8001"

Write-Host ""
Write-Host "============================================================"
Write-Host "AITrading HTTP API - PowerShell Examples"
Write-Host "============================================================"
Write-Host ""
Write-Host "Server URL: $BASE_URL"
Write-Host "API Docs: $BASE_URL/docs"
Write-Host ""

# Health check
Write-Host "1. Health Check:"
$response = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get
$response | ConvertTo-Json | Write-Host
Write-Host ""

# List tools
Write-Host "2. Available Tools:"
$response = Invoke-RestMethod -Uri "$BASE_URL/tools" -Method Get
$response | ConvertTo-Json | Write-Host
Write-Host ""

# Gap-up analysis
Write-Host "3. Gap-up Analysis (VIX > 15, Gap > 1%):"
$response = Invoke-RestMethod -Uri "$BASE_URL/gap_up_analysis?vix_threshold=15&gap_percent=1.0" -Method Get
Write-Host $response.result
Write-Host ""

# SMA crossover
Write-Host "4. SMA Crossover Analysis (50/200):"
$response = Invoke-RestMethod -Uri "$BASE_URL/sma_crossover_analysis?sma_short=50&sma_long=200" -Method Get
Write-Host $response.result
Write-Host ""

# VIX analysis
Write-Host "5. VIX Analysis (Last 30 days):"
$response = Invoke-RestMethod -Uri "$BASE_URL/vix_analysis?vix_threshold=15&lookback_days=30" -Method Get
Write-Host $response.result
Write-Host ""

# Data summary
Write-Host "6. Data Summary:"
$response = Invoke-RestMethod -Uri "$BASE_URL/data_summary" -Method Get
Write-Host $response.result
Write-Host ""

# Price pattern
Write-Host "7. Price Pattern - Higher Highs (60-day lookback):"
$response = Invoke-RestMethod -Uri "$BASE_URL/price_pattern?pattern=higher_highs&lookback_days=60" -Method Get
Write-Host $response.result
Write-Host ""

Write-Host "============================================================"
Write-Host "Done!"
Write-Host "============================================================"
Write-Host ""
