#!/usr/bin/env python
"""
Example: Call the AITrading HTTP API from Python
Usage: python examples/api_client.py
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def call_api(endpoint, params=None):
    """Make an API call and print the result."""
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ {endpoint}")
        print(f"Result:\n{data['result']}\n")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AITrading HTTP API Client Examples")
    print("="*70)
    
    # Health check
    print("\n1. Health Check:")
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))
    
    # List tools
    print("\n2. Available Tools:")
    response = requests.get(f"{BASE_URL}/tools")
    print(json.dumps(response.json(), indent=2))
    
    # Gap-up analysis
    print("\n3. Gap-up Analysis:")
    call_api("/gap_up_analysis", {"vix_threshold": 15, "gap_percent": 1.0})
    
    # SMA Crossover
    print("4. SMA Crossover Analysis:")
    call_api("/sma_crossover_analysis", {"sma_short": 50, "sma_long": 200})
    
    # VIX Analysis
    print("5. VIX Analysis:")
    call_api("/vix_analysis", {"vix_threshold": 15, "lookback_days": 30})
    
    # Data Summary
    print("6. Data Summary:")
    call_api("/data_summary")
    
    # Price Pattern
    print("7. Price Pattern (Higher Highs):")
    call_api("/price_pattern", {"pattern": "higher_highs", "lookback_days": 60})
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")
