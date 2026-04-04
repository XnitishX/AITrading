#!/usr/bin/env python
"""
Quick runner for AITrading MCP Server
Usage: python run_mcp_server.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
from src.mcp_server import AITradingServer


async def main():
    """Run the MCP server."""
    print("\n" + "="*60)
    print("🚀 AITrading MCP Server")
    print("="*60)
    print("\nServer is starting...")
    print("\n📖 Setup guide: See MCP_SERVER_SETUP.md")
    print("\n" + "="*60 + "\n")
    
    server = AITradingServer()
    try:
        await server.run()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
