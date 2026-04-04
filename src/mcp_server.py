"""
MCP Server for AITrading Project
─────────────────────────────────
Exposes AITrading data and analysis functions as an MCP server.
Can be used by Copilot Chat or other MCP clients.

Usage:
    python -m src.mcp_server
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from config.settings import LOG_FORMAT, LOG_LEVEL
from src.data.loader import build_master_dataframe

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.warning("MCP library not found. Install with: pip install mcp")
    FastMCP = None


class AITradingAnalyzer:
    """Analysis engine for AITrading data."""

    def __init__(self):
        self.df = None
        self._load_data()

    def _load_data(self):
        """Load the master dataframe."""
        try:
            self.df = build_master_dataframe(save=False)
            logger.info(f"Loaded {len(self.df)} rows of data")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.df = pd.DataFrame()

    def gap_up_analysis(self, vix_threshold: float = 15, gap_percent: float = 1.0) -> str:
        """Find days with gap-up moves when VIX is above threshold."""
        try:
            if self.df.empty:
                return "No data available"
            
            df = self.df.copy()
            
            # Calculate gap percentage
            df['Gap_Pct'] = ((df['Open'] - df['Prev_Close']) / df['Prev_Close'] * 100).fillna(0)
            
            # Filter conditions
            mask = (df['VIX_Close'] > vix_threshold) & (df['Gap_Pct'] > gap_percent)
            result_df = df[mask][['Date', 'Open', 'Prev_Close', 'Gap_Pct', 'VIX_Close', 'Close', 'Volume']].copy()
            
            if result_df.empty:
                return f"No gap-ups >{gap_percent}% found when VIX>{vix_threshold}"
            
            # Format result
            text = f"Found {len(result_df)} instances of gap-up >{gap_percent}% when VIX>{vix_threshold}:\n\n"
            for idx, row in result_df.iterrows():
                text += f"📅 {row['Date'].date()}: Gap {row['Gap_Pct']:+.2f}% (VIX={row['VIX_Close']:.2f}, Open={row['Open']:.2f}, Close={row['Close']:.2f})\n"
            
            return text
        except Exception as e:
            return f"Error: {str(e)}"

    def sma_crossover_analysis(self, sma_short: int = 50, sma_long: int = 200) -> str:
        """Find SMA crossover points (short SMA crossing long SMA from below)."""
        try:
            if self.df.empty:
                return "No data available"
            
            df = self.df.copy()
            
            # Calculate SMAs
            df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
            df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
            
            # Detect crossovers
            df['SMA_Diff'] = df['SMA_Short'] - df['SMA_Long']
            df['Prev_SMA_Diff'] = df['SMA_Diff'].shift(1)
            df['Crossover'] = (df['Prev_SMA_Diff'] < 0) & (df['SMA_Diff'] > 0)
            
            crossovers = df[df['Crossover']][['Date', 'Close', 'SMA_Short', 'SMA_Long', 'VIX_Close']].copy()
            
            if crossovers.empty:
                return f"No bullish crossovers ({sma_short}SMA > {sma_long}SMA) found"
            
            # Format result
            text = f"Found {len(crossovers)} bullish SMA crossovers ({sma_short} crossing {sma_long} from below):\n\n"
            for idx, row in crossovers.iterrows():
                text += f"📅 {row['Date'].date()}: SMA({sma_short})={row['SMA_Short']:.2f}, SMA({sma_long})={row['SMA_Long']:.2f}, Close={row['Close']:.2f}, VIX={row['VIX_Close']:.2f}\n"
            
            return text
        except Exception as e:
            return f"Error: {str(e)}"

    def vix_analysis(self, vix_threshold: float = 15, lookback_days: int = 30) -> str:
        """Analyze VIX levels and market conditions in a lookback period."""
        try:
            if self.df.empty:
                return "No data available"
            
            df = self.df.copy()
            df = df.tail(lookback_days).copy()
            
            high_vix = df[df['VIX_Close'] > vix_threshold]
            low_vix = df[df['VIX_Close'] <= vix_threshold]
            
            text = f"VIX Analysis (Last {lookback_days} days, threshold={vix_threshold}):\n\n"
            text += f"Days with VIX > {vix_threshold}: {len(high_vix)}\n"
            text += f"Days with VIX ≤ {vix_threshold}: {len(low_vix)}\n"
            text += f"VIX Range: {df['VIX_Close'].min():.2f} - {df['VIX_Close'].max():.2f}\n"
            text += f"VIX Mean: {df['VIX_Close'].mean():.2f}\n\n"
            
            if not high_vix.empty:
                text += f"High VIX Days (> {vix_threshold}):\n"
                text += high_vix[['Date', 'VIX_Close', 'Close', 'Return_Pct']].to_string()
            
            return text
        except Exception as e:
            return f"Error: {str(e)}"

    def price_pattern(self, pattern: str, lookback_days: int = 60) -> str:
        """Find price patterns (e.g., 'higher_highs', 'lower_lows', 'outside_bar')."""
        try:
            if self.df.empty:
                return "No data available"
            
            df = self.df.copy()
            df = df.tail(lookback_days).copy()
            
            results = []
            
            if pattern == 'higher_highs':
                for i in range(2, len(df)):
                    if df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i-1]['High'] > df.iloc[i-2]['High']:
                        results.append(i)
            
            elif pattern == 'lower_lows':
                for i in range(2, len(df)):
                    if df.iloc[i]['Low'] < df.iloc[i-1]['Low'] and df.iloc[i-1]['Low'] < df.iloc[i-2]['Low']:
                        results.append(i)
            
            elif pattern == 'outside_bar':
                for i in range(1, len(df)):
                    if df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['Low'] < df.iloc[i-1]['Low']:
                        results.append(i)
            
            elif pattern == 'inside_bar':
                for i in range(1, len(df)):
                    if df.iloc[i]['High'] < df.iloc[i-1]['High'] and df.iloc[i]['Low'] > df.iloc[i-1]['Low']:
                        results.append(i)
            
            if not results:
                return f"No '{pattern}' patterns found in last {lookback_days} days"
            
            text = f"Found {len(results)} '{pattern}' instances in last {lookback_days} days:\n\n"
            for idx in results[-10:]:  # Show last 10
                row = df.iloc[idx]
                text += f"📅 {row['Date'].date()}: High={row['High']:.2f}, Low={row['Low']:.2f}, Close={row['Close']:.2f}\n"
            
            return text
        except Exception as e:
            return f"Error: {str(e)}"

    def data_summary(self) -> str:
        """Get summary statistics of the loaded dataset."""
        try:
            if self.df.empty:
                return "No data available"
            
            df = self.df
            text = f"""
Data Summary:
─────────────
Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}
Total Rows: {len(df)}
Nifty 50 Close: {df['Close'].iloc[-1]:.2f} (Range: {df['Close'].min():.2f} - {df['Close'].max():.2f})
Current VIX: {df['VIX_Close'].iloc[-1]:.2f} (Range: {df['VIX_Close'].min():.2f} - {df['VIX_Close'].max():.2f})
Avg Daily Return: {df['Return_Pct'].mean():.3f}%
Volatility (Std Dev): {df['Return_Pct'].std():.3f}%
"""
            return text
        except Exception as e:
            return f"Error: {str(e)}"

    def custom_query(self, query: str) -> str:
        """Execute a custom pandas DataFrame query."""
        try:
            if self.df.empty:
                return "No data available"
            
            df = self.df
            result = eval(query)
            
            if isinstance(result, pd.DataFrame):
                return result.tail(20).to_string()
            else:
                return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


def create_mcp_server():
    """Create and configure the MCP server."""
    if FastMCP is None:
        print("Error: MCP library not installed.")
        print("Install with: pip install mcp")
        sys.exit(1)

    mcp = FastMCP("AITrading")
    analyzer = AITradingAnalyzer()

    @mcp.tool()
    def gap_up_analysis(vix_threshold: float = 15, gap_percent: float = 1.0) -> str:
        """Find days with gap-up moves when VIX is above threshold."""
        return analyzer.gap_up_analysis(vix_threshold, gap_percent)

    @mcp.tool()
    def sma_crossover_analysis(sma_short: int = 50, sma_long: int = 200) -> str:
        """Find SMA crossover points (short SMA crossing long SMA from below)."""
        return analyzer.sma_crossover_analysis(sma_short, sma_long)

    @mcp.tool()
    def vix_analysis(vix_threshold: float = 15, lookback_days: int = 30) -> str:
        """Analyze VIX levels and market conditions."""
        return analyzer.vix_analysis(vix_threshold, lookback_days)

    @mcp.tool()
    def price_pattern(pattern: str, lookback_days: int = 60) -> str:
        """Find price patterns: 'higher_highs', 'lower_lows', 'outside_bar', 'inside_bar'."""
        return analyzer.price_pattern(pattern, lookback_days)

    @mcp.tool()
    def data_summary() -> str:
        """Get summary statistics of the loaded dataset."""
        return analyzer.data_summary()

    @mcp.tool()
    def custom_query(query: str) -> str:
        """Execute custom pandas query using 'df' variable."""
        return analyzer.custom_query(query)

    return mcp


async def main():
    """Run the MCP server."""
    mcp = create_mcp_server()
    logger.info("Starting AITrading MCP Server...")
    async with mcp:
        logger.info("Server is running. Press Ctrl+C to stop.")
        await mcp.wait()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
